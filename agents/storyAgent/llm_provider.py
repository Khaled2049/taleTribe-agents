"""LLM provider abstraction — all AI calls route through creditProxy."""

import json
import logging
import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# Per-request BYOK config. Set in server.py before each agent call.
# ContextVar is async-safe: each asyncio Task sees its own copy.
_byok_config: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "byok_config", default=None
)
_firebase_token: ContextVar[Optional[str]] = ContextVar("firebase_token", default=None)


class LLMProviderError(Exception):
    """Base class for all typed LLM provider errors."""


class InsufficientCreditsError(LLMProviderError):
    """Platform credit balance exhausted."""


class ProviderAuthError(LLMProviderError):
    """AI provider rejected the API key."""


class ProviderNotFoundError(LLMProviderError):
    """AI provider or model not found."""

    def __init__(self, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model
        label = f"{provider}/{model}" if provider and model else model or provider
        super().__init__(label or "unknown model")


class BackendUnavailableError(LLMProviderError):
    """creditProxy is unreachable or returned an unexpected error."""


class RateLimitedError(LLMProviderError):
    """AI provider rate limit reached."""


class LLMTimeoutError(LLMProviderError):
    """AI request timed out."""


class InvalidRequestError(LLMProviderError):
    """creditProxy rejected the request itself (bad params) — not a provider
    or credits issue. Retrying with the same payload will never succeed."""


class BillingCommitError(LLMProviderError):
    """The LLM call succeeded but committing the reservation afterward
    failed — content was generated (and paid for at the provider) but never
    billed to the user. Must NOT be retried: a retry would generate new
    content and reserve credits a second time for a billing-side failure."""


def _classify_http_error(
    status: int, body: str, config: Optional[Dict[str, str]]
) -> LLMProviderError:
    """Map a creditProxy HTTP error to a typed exception.

    Status codes are trusted over body text: the gateway deliberately never
    echoes upstream error bodies to callers (see creditProxy/cmd/gateway),
    so body-text matching alone would miss real failures. Body matching is
    kept only as a fallback for providers/paths that do return descriptive
    text.
    """
    body_lower = body.lower()
    if status == 402 or "insufficient credits" in body_lower:
        return InsufficientCreditsError(body)
    if status in (401, 403) or "unauthorized" in body_lower:
        return ProviderAuthError(body)
    if status == 429 or "rate limit" in body_lower or "too many requests" in body_lower:
        return RateLimitedError(body)
    if status == 404:
        provider = (config.get("provider", "") if config else "") or ""
        model = (config.get("model", "") if config else "") or ""
        if model or provider:
            return ProviderNotFoundError(provider, model)
        return BackendUnavailableError(f"creditProxy error 404: {body}")
    if status == 400:
        return InvalidRequestError(body)
    if status == 409:
        return BillingCommitError(body)
    if status in (500, 502, 503, 504):
        return BackendUnavailableError(f"creditProxy error {status}: {body}")
    return LLMProviderError(f"creditProxy error {status}: {body}")


async def _gcp_id_token_async(audience: str) -> str:
    """Fetch a GCP OIDC identity token from the metadata server (async).

    Returns '' when not running on GCP (local dev / tests) so callers can
    skip adding the Authorization header without any special-casing.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/"
                f"service-accounts/default/identity?audience={audience}",
                headers={"Metadata-Flavor": "Google"},
            )
            resp.raise_for_status()
            return resp.text
    except Exception:
        return ""  # metadata server unreachable — not on GCP


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_content_async(self, prompt: str) -> str:
        """Generate text content from a prompt."""

    @abstractmethod
    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        """Generate structured (JSON array) content from system+user prompts."""


class CreditProxyProvider(LLMProvider):
    """Routes LLM calls through the creditProxy gateway (fully async).

    Platform requests use the gateway's default provider.
    BYOK requests read per-request config from the _byok_config ContextVar
    (set by server.py before each agent call — async-safe, no tool changes needed).
    """

    def __init__(self, base_url: str, platform_user_id: str = "platform"):
        self.base_url = base_url.rstrip("/")
        self.platform_user_id = platform_user_id
        # Shared async client — reused across all requests in this process.
        # Timeout covers the full LLM round-trip (up to 300 s).
        self._client = httpx.AsyncClient(timeout=300.0)

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Idempotent."""
        await self._client.aclose()

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        config = _byok_config.get()
        user_id = (
            config.get("user_id", self.platform_user_id)
            if config
            else self.platform_user_id
        )
        provider = config.get("provider", "") if config else ""
        api_key = config.get("api_key", "") if config else ""
        model = config.get("model", "") if config else ""
        if api_key:
            logger.info(
                "[LLM] BYOK  provider=%s model=%s user=%s", provider, model, user_id
            )
        else:
            logger.info("[LLM] platform  user=%s proxy=%s", user_id, self.base_url)
        return {
            "user_id": user_id,
            "prompt": prompt,
            "byok_provider": provider,
            "byok_api_key": api_key,
            "byok_model": model,
            "max_output_tokens": 8192,
        }

    # Only BackendUnavailableError (network failure, or a creditProxy 5xx) is
    # retried — every other typed error is either not the caller's fault to
    # fix by retrying (insufficient credits, bad request) or would cause a
    # double-charge/double-generation if retried (BillingCommitError). Bounded
    # two ways so a slow-but-not-quite-failing backend can't retry forever:
    # at most 3 attempts, AND at most 30s of total retry-loop time, whichever
    # comes first. Jittered backoff avoids a thundering herd against
    # creditProxy when many requests fail at once.
    @retry(
        stop=stop_after_attempt(3) | stop_after_delay(30),
        wait=wait_exponential_jitter(initial=2, max=10),
        retry=retry_if_exception_type(BackendUnavailableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def generate_content_async(self, prompt: str) -> str:
        payload = self._build_payload(prompt)
        headers: Dict[str, str] = {}
        if token := await _gcp_id_token_async(self.base_url):
            headers["Authorization"] = f"Bearer {token}"
        if firebase_token := _firebase_token.get():
            headers["X-Firebase-Token"] = firebase_token
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]["output"]
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise BackendUnavailableError(f"creditProxy unreachable: {e}") from e
        except httpx.HTTPStatusError as e:
            raise _classify_http_error(
                e.response.status_code, e.response.text, _byok_config.get()
            ) from e

    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        schema_hint = json.dumps(response_schema, indent=2)
        combined = (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            f"IMPORTANT: Respond with ONLY a valid JSON array matching this schema:\n{schema_hint}\n"
            "Do not include any text before or after the JSON array."
        )
        try:
            text = await self.generate_content_async(combined)
            text = text.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "")
            json_data = json.loads(text)
            if isinstance(json_data, list):
                return [str(item) for item in json_data]
            if isinstance(json_data, dict):
                for value in json_data.values():
                    if isinstance(value, list):
                        return [str(item) for item in value]
            return []
        except (json.JSONDecodeError, KeyError):
            logger.warning("CreditProxyProvider: failed to parse structured response")
            return []


def get_llm_provider(
    project_id: Optional[str] = None, location: str = "us-central1"
) -> LLMProvider:
    """Return a CreditProxyProvider. CREDIT_PROXY_URL must be set.

    Provider/model selection is configured entirely in creditProxy via LLM_PROVIDER:
      mock     — canned responses, no API key needed
      ollama   — local LLM via OLLAMA_BASE_URL
      gemini   — GEMINI_API_KEY (platform default)
      BYOK     — per-request key forwarded via byok_* fields in the payload
    """
    credit_proxy_url = os.getenv("CREDIT_PROXY_URL")
    if not credit_proxy_url:
        raise ValueError(
            "CREDIT_PROXY_URL must be set. "
            "Run creditProxy via docker compose and set CREDIT_PROXY_URL=http://localhost:8080."
        )
    logger.info("Using CreditProxy Provider (%s)", credit_proxy_url)
    return CreditProxyProvider(base_url=credit_proxy_url)
