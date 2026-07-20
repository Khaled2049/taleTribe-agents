"""Tests for creditProxy HTTP error classification and retry bounds.

Covers the classification gaps found while debugging a real prod incident
(a 402 from a credit reservation failure wasn't being classified as
InsufficientCreditsError) and the retry policy that guards against
retrying non-idempotent / non-retryable failures forever.
"""

import asyncio
import os

import httpx
import pytest
from tenacity import stop_after_attempt, stop_after_delay, stop_any

os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from agents.storyAgent.llm_provider import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    BackendUnavailableError,
    BillingCommitError,
    CreditProxyProvider,
    InsufficientCreditsError,
    InvalidRequestError,
    LLMProviderError,
    LLMTimeoutError,
    ProviderAuthError,
    ProviderNotFoundError,
    RateLimitedError,
    _classify_http_error,
)

# --- _classify_http_error ----------------------------------------------------


class TestClassifyHttpError:
    def test_402_is_insufficient_credits(self):
        err = _classify_http_error(402, "unable to reserve credits (ref req_1)", None)
        assert isinstance(err, InsufficientCreditsError)

    def test_insufficient_credits_body_text_also_matches(self):
        # Fallback path for any caller that DOES echo the reason in the body.
        err = _classify_http_error(500, "insufficient credits", None)
        assert isinstance(err, InsufficientCreditsError)

    def test_401_is_provider_auth_error(self):
        err = _classify_http_error(401, "generation failed (ref llm_1)", None)
        assert isinstance(err, ProviderAuthError)

    def test_403_is_provider_auth_error(self):
        err = _classify_http_error(
            403, "user_id does not match firebase identity", None
        )
        assert isinstance(err, ProviderAuthError)

    def test_429_is_rate_limited(self):
        err = _classify_http_error(429, "platform daily request limit reached", None)
        assert isinstance(err, RateLimitedError)

    def test_404_with_model_config_is_provider_not_found(self):
        err = _classify_http_error(
            404,
            "generation failed (ref llm_2)",
            {"provider": "openai", "model": "gpt-9"},
        )
        assert isinstance(err, ProviderNotFoundError)
        assert err.provider == "openai"
        assert err.model == "gpt-9"

    def test_404_without_model_config_is_backend_unavailable(self):
        err = _classify_http_error(404, "not found", None)
        assert isinstance(err, BackendUnavailableError)

    def test_400_is_invalid_request(self):
        err = _classify_http_error(400, "prompt must be <= 8000 characters", None)
        assert isinstance(err, InvalidRequestError)

    def test_409_is_billing_commit_error(self):
        err = _classify_http_error(409, "commit reservation failed (ref req_3)", None)
        assert isinstance(err, BillingCommitError)

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_5xx_is_backend_unavailable(self, status):
        err = _classify_http_error(status, "internal error", None)
        assert isinstance(err, BackendUnavailableError)

    def test_unrecognized_status_is_generic_provider_error(self):
        err = _classify_http_error(418, "I'm a teapot", None)
        assert type(err) is LLMProviderError


# --- retry policy shape (no network involved) --------------------------------


def test_retry_is_bounded_by_attempts_and_wall_clock_time():
    """Regression guard: retries must never be unbounded. Two independent
    caps — attempt count AND elapsed time — either of which stops retrying."""
    retrying = CreditProxyProvider.generate_content_async.retry
    assert isinstance(retrying.stop, stop_any)
    assert any(isinstance(s, stop_after_attempt) for s in retrying.stop.stops)
    assert any(isinstance(s, stop_after_delay) for s in retrying.stop.stops)


def test_retry_only_targets_backend_unavailable_error():
    retrying = CreditProxyProvider.generate_content_async.retry
    assert retrying.retry.exception_types == BackendUnavailableError


# --- retry behavior against a mocked transport --------------------------------


def _provider_with_transport(
    handler: "httpx.MockTransport | callable",
) -> CreditProxyProvider:
    provider = CreditProxyProvider(base_url="http://creditproxy.test")
    transport = (
        handler
        if isinstance(handler, httpx.MockTransport)
        else httpx.MockTransport(handler)
    )
    provider._client = httpx.AsyncClient(transport=transport)
    return provider


@pytest.fixture(autouse=True)
def _no_real_gcp_metadata_call(monkeypatch):
    """CreditProxyProvider probes the GCE metadata server on every call; keep
    tests hermetic and fast by short-circuiting it like local dev would see."""

    async def fake_token(_audience: str) -> str:
        return ""

    monkeypatch.setattr(
        "agents.storyAgent.llm_provider._gcp_id_token_async", fake_token
    )


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Skip real backoff sleeps so retry tests run in milliseconds, not seconds."""

    async def fast_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)


@pytest.mark.asyncio
async def test_retries_on_backend_unavailable_up_to_three_attempts():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="usage service unreachable")

    provider = _provider_with_transport(handler)
    with pytest.raises(BackendUnavailableError):
        await provider.generate_content_async("hello")
    assert calls == 3


@pytest.mark.asyncio
async def test_recovers_after_transient_failure_within_retry_budget():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(503, text="temporarily down")
        return httpx.Response(200, json={"response": {"output": "hello world"}})

    provider = _provider_with_transport(handler)
    result = await provider.generate_content_async("hello")
    assert result == "hello world"
    assert calls == 2


@pytest.mark.asyncio
async def test_network_error_is_retried_as_backend_unavailable():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider_with_transport(handler)
    with pytest.raises(BackendUnavailableError):
        await provider.generate_content_async("hello")
    assert calls == 3


@pytest.mark.asyncio
async def test_timeout_is_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMTimeoutError):
        await provider.generate_content_async("hello")
    assert calls == 1


@pytest.mark.asyncio
async def test_insufficient_credits_is_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(402, text="unable to reserve credits (ref req_1)")

    provider = _provider_with_transport(handler)
    with pytest.raises(InsufficientCreditsError):
        await provider.generate_content_async("hello")
    assert calls == 1


@pytest.mark.asyncio
async def test_billing_commit_error_is_never_retried():
    """A 409 must never be retried: the LLM call already succeeded upstream,
    so retrying would generate (and pay for) new content and reserve credits
    a second time on top of an already-failed billing commit."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(409, text="commit reservation failed (ref req_2)")

    provider = _provider_with_transport(handler)
    with pytest.raises(BillingCommitError):
        await provider.generate_content_async("hello")
    assert calls == 1


@pytest.mark.asyncio
async def test_invalid_request_is_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="prompt must be <= 8000 characters")

    provider = _provider_with_transport(handler)
    with pytest.raises(InvalidRequestError):
        await provider.generate_content_async("hello")
    assert calls == 1


@pytest.mark.asyncio
async def test_rate_limited_is_not_retried():
    """429s are surfaced immediately rather than retried blindly — retrying
    without a real Retry-After signal risks compounding provider throttling."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="rate limit exceeded")

    provider = _provider_with_transport(handler)
    with pytest.raises(RateLimitedError):
        await provider.generate_content_async("hello")
    assert calls == 1


# --- credit balance / purchase ------------------------------------------------


@pytest.mark.asyncio
async def test_get_balance_returns_usage_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/users/user123/balance"
        return httpx.Response(
            200, json={"user_id": "user123", "available_credits": 1749}
        )

    provider = _provider_with_transport(handler)
    data = await provider.get_balance("user123")
    assert data["available_credits"] == 1749


@pytest.mark.asyncio
async def test_get_balance_forwards_firebase_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-Firebase-Token") == "fb-token"
        return httpx.Response(200, json={"user_id": "user123", "available_credits": 0})

    provider = _provider_with_transport(handler)
    await provider.get_balance("user123", firebase_token="fb-token")


@pytest.mark.asyncio
async def test_get_balance_maps_5xx_to_backend_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="usage down")

    provider = _provider_with_transport(handler)
    with pytest.raises(BackendUnavailableError):
        await provider.get_balance("user123")


@pytest.mark.asyncio
async def test_purchase_credits_posts_amount_and_returns_new_balance():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/credits/purchase"
        import json as _json

        body = _json.loads(request.content)
        assert body == {"user_id": "user123", "credits": 10000}
        return httpx.Response(
            200,
            json={
                "user_id": "user123",
                "purchased_credits": 10000,
                "available_credits": 11749,
            },
        )

    provider = _provider_with_transport(handler)
    data = await provider.purchase_credits("user123", 10000)
    assert data["available_credits"] == 11749


@pytest.mark.asyncio
async def test_purchase_credits_disabled_maps_to_backend_unavailable():
    # gateway returns 503 when the usage purchase API is disabled
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unable to purchase credits (ref req_1)")

    provider = _provider_with_transport(handler)
    with pytest.raises(BackendUnavailableError):
        await provider.purchase_credits("user123", 10000)


# --- per-tool max_output_tokens ----------------------------------------------


@pytest.mark.asyncio
async def test_generate_forwards_max_output_tokens():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={"response": {"output": "ok"}})

    provider = _provider_with_transport(handler)
    await provider.generate_content_async("hello", max_output_tokens=1024)
    assert captured["max_output_tokens"] == 1024


@pytest.mark.asyncio
async def test_generate_defaults_max_output_tokens():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={"response": {"output": "ok"}})

    provider = _provider_with_transport(handler)
    await provider.generate_content_async("hello")
    assert captured["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_structured_content_forwards_max_output_tokens():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={"response": {"output": "[]"}})

    provider = _provider_with_transport(handler)
    await provider.generate_structured_content(
        "sys", "user", {"type": "array"}, max_output_tokens=512
    )
    assert captured["max_output_tokens"] == 512
