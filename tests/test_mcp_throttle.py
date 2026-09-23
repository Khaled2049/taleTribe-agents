"""Per-IP throttling of the unauthenticated MCP OAuth endpoints.

Built as its own module because the caps are set from env at create_app() time
and this suite needs them low, while test_mcp_oauth_flow.py needs them high.
"""

import os
from unittest.mock import patch

import pytest

os.environ["CREDIT_PROXY_URL"] = "http://localhost:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["USE_MOCK"] = "true"
os.environ["ENABLE_MCP"] = "true"
os.environ["MCP_CONSENT_URL"] = "https://consent.example/mcp-connect"
os.environ["MCP_ISSUER_URL"] = "http://localhost:8000"
os.environ["MCP_REGISTER_REQUESTS_PER_MINUTE_PER_IP"] = "2"
os.environ["MCP_OAUTH_REQUESTS_PER_MINUTE_PER_IP"] = "3"
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:9999")

from fastapi.testclient import TestClient  # noqa: E402

import mcp_server.app as mcp_app_module  # noqa: E402
from mcp_server.throttle import (  # noqa: E402
    MAX_OAUTH_BODY_BYTES,
    OAuthThrottleMiddleware,
    client_ip,
)
from tests.mcp_fakes import FakeFirestoreClient  # noqa: E402

fake_db = FakeFirestoreClient()

with patch.object(mcp_app_module, "_make_firestore_client", return_value=fake_db):
    from server import create_app

    app = create_app()

REGISTRATION = {
    "client_name": "Throttle Test Client",
    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
    "token_endpoint_auth_method": "none",
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _register(client, ip: str):
    return client.post("/register", json=REGISTRATION, headers={"X-Forwarded-For": ip})


# ---------------------------------------------------------------------------
# /register — the unauthenticated Firestore write
# ---------------------------------------------------------------------------


def test_register_throttled_after_budget(client):
    """Cap is 2/min: the third unauthenticated registration is refused."""
    assert _register(client, "10.0.0.1").status_code == 201
    assert _register(client, "10.0.0.1").status_code == 201

    resp = _register(client, "10.0.0.1")
    assert resp.status_code == 429
    assert resp.json()["error"] == "temporarily_unavailable"
    assert resp.headers["retry-after"] == "60"


def test_throttle_is_per_ip_not_global(client):
    """One abusive caller must not lock everyone else out."""
    for _ in range(3):
        _register(client, "10.0.0.2")
    assert _register(client, "10.0.0.2").status_code == 429
    assert _register(client, "10.0.0.3").status_code == 201


def test_no_client_document_written_when_throttled(client):
    """The point of the guard: a refused call must not reach Firestore."""
    before = sum(1 for path in fake_db.docs if path.startswith("mcpOauthClients/"))
    for _ in range(6):
        _register(client, "10.0.0.4")
    after = sum(1 for path in fake_db.docs if path.startswith("mcpOauthClients/"))
    assert after - before == 2  # the budget, not the 6 attempts


# ---------------------------------------------------------------------------
# The rest of the flow, and what must stay open
# ---------------------------------------------------------------------------


def test_authorize_and_token_share_the_looser_budget(client):
    """Cap is 3/min across /authorize and /token combined."""
    ip = "10.0.0.5"
    for _ in range(3):
        resp = client.post("/token", data={}, headers={"X-Forwarded-For": ip})
        assert resp.status_code != 429  # 400s are fine; not throttled yet
    assert (
        client.get("/authorize", params={}, headers={"X-Forwarded-For": ip}).status_code
        == 429
    )


def test_discovery_documents_are_never_throttled(client):
    """A client that cannot read discovery cannot start the flow at all."""
    ip = "10.0.0.6"
    for _ in range(25):
        resp = client.get(
            "/.well-known/oauth-authorization-server",
            headers={"X-Forwarded-For": ip},
        )
        assert resp.status_code == 200


def test_mcp_endpoint_not_covered_by_oauth_throttle(client):
    """/mcp has its own per-user tool limiter; it must still answer 401."""
    ip = "10.0.0.7"
    for _ in range(10):
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={
                "Accept": "application/json, text/event-stream",
                "X-Forwarded-For": ip,
            },
        )
        assert resp.status_code == 401


def test_health_still_reachable_under_throttle(client):
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# IP extraction
# ---------------------------------------------------------------------------


def _request(forwarded=None, peer=("10.1.1.1", 1234)):
    from starlette.requests import Request

    headers = [(b"x-forwarded-for", forwarded)] if forwarded is not None else []
    return Request({"type": "http", "headers": headers, "client": peer})


def test_client_ip_takes_entry_appended_by_trusted_proxy():
    request = _request(b"203.0.113.9, 70.41.3.18, 150.172.238.178")
    assert client_ip(request, 1) == "150.172.238.178"
    assert client_ip(request, 2) == "70.41.3.18"


def test_client_ip_ignores_forwarded_header_with_zero_hops():
    assert client_ip(_request(b"203.0.113.9"), 0) == "10.1.1.1"


def test_client_ip_falls_back_to_peer_when_chain_is_short():
    assert client_ip(_request(b"203.0.113.9"), 2) == "10.1.1.1"
    assert client_ip(_request(b" , "), 1) == "10.1.1.1"


def test_client_ip_falls_back_to_socket_then_unknown():
    assert client_ip(_request()) == "10.1.1.1"
    assert client_ip(_request(peer=None)) == "unknown"


def test_forged_forwarded_prefixes_do_not_create_new_buckets(client):
    real = "10.0.3.1"
    statuses = [
        _register(client, f"{forged}, {real}").status_code
        for forged in ("198.51.100.1", "198.51.100.2", "198.51.100.3")
    ]
    assert statuses == [201, 201, 429]


def test_consent_routes_key_on_trusted_entry(client):
    real = "10.0.3.2"
    statuses = [
        client.get(
            "/oauth/txn/missing", headers={"X-Forwarded-For": f"198.51.100.{i}, {real}"}
        ).status_code
        for i in range(4)
    ]
    assert statuses == [404, 404, 404, 429]


async def test_register_total_budget_caps_distinct_clients():
    from starlette.responses import Response

    passed = []

    async def inner_app(scope, receive, send):
        passed.append(scope["path"])
        await Response(status_code=201)(scope, receive, send)

    middleware = OAuthThrottleMiddleware(
        inner_app,
        register_per_minute=5,
        oauth_per_minute=5,
        register_total_per_minute=2,
    )

    async def call(path, ip):
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": path,
                "headers": [(b"x-forwarded-for", ip.encode())],
                "client": ("10.9.9.9", 1),
            },
            receive,
            send,
        )
        return sent[0]["status"]

    assert [await call("/register", f"203.0.113.{i}") for i in range(3)] == [
        201,
        201,
        429,
    ]
    assert await call("/token", "203.0.113.50") == 201
    assert passed == ["/register", "/register", "/token"]


async def test_non_http_scopes_pass_through():
    """Lifespan/websocket scopes have no path or headers to inspect."""
    seen = []

    async def inner_app(scope, receive, send):
        seen.append(scope["type"])

    middleware = OAuthThrottleMiddleware(
        inner_app, register_per_minute=1, oauth_per_minute=1
    )
    await middleware({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"]


def _client_docs():
    return sum(1 for path in fake_db.docs if path.startswith("mcpOauthClients/"))


async def test_starlette_enforces_urlencoded_form_limits():
    import starlette
    from starlette.formparsers import MultiPartException
    from starlette.requests import Request

    version = tuple(int(p) for p in starlette.__version__.split(".")[:3])
    assert version >= (1, 3, 1)

    async def receive():
        return {"type": "http.request", "body": b"a=1&b=2", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"content-type", b"application/x-www-form-urlencoded")],
        },
        receive,
    )
    with pytest.raises(MultiPartException):
        await request.form(max_fields=1)


def test_oversized_declared_body_rejected_before_parsing(client):
    before = _client_docs()
    resp = client.post(
        "/register",
        content=b"{" + b" " * (MAX_OAUTH_BODY_BYTES + 1) + b"}",
        headers={"Content-Type": "application/json", "X-Forwarded-For": "10.0.1.1"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "invalid_request"
    assert _client_docs() == before


def test_oversized_token_form_rejected(client):
    resp = client.post(
        "/token",
        content=b"&".join(b"f%d=x" % i for i in range(5000)),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Forwarded-For": "10.0.1.2",
        },
    )
    assert resp.status_code == 413


def test_body_at_limit_still_reaches_handler(client):
    resp = client.post(
        "/token",
        content=b"grant_type=x&pad=" + b"a" * (MAX_OAUTH_BODY_BYTES - 17),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Forwarded-For": "10.0.1.3",
        },
    )
    assert resp.status_code not in (413, 429)


async def _run_middleware(chunks, headers, path="/token"):
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    parsed = []

    async def inner_app(scope, receive, send):
        form = await Request(scope, receive).form()
        parsed.append(len(form))
        await JSONResponse({"ok": True})(scope, receive, send)

    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    consumed = 0

    async def receive():
        nonlocal consumed
        consumed += 1
        return messages[consumed - 1]

    sent = []

    async def send(message):
        sent.append(message)

    middleware = OAuthThrottleMiddleware(
        inner_app, register_per_minute=10, oauth_per_minute=10, max_body_bytes=100
    )
    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [
                (b"content-type", b"application/x-www-form-urlencoded"),
                *headers,
            ],
            "client": ("10.2.2.2", 1234),
        },
        receive,
        send,
    )
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, parsed, consumed


async def test_streamed_body_without_length_is_cut_off():
    chunks = [b"a=" + b"x" * 60] * 10
    status, parsed, consumed = await _run_middleware(chunks, [])
    assert status == 413
    assert parsed == []
    assert consumed == 2


async def test_streamed_body_within_limit_passes():
    status, parsed, _ = await _run_middleware([b"a=1&", b"b=2"], [])
    assert status == 200
    assert parsed == [2]


async def test_malformed_content_length_rejected():
    status, parsed, consumed = await _run_middleware(
        [b"a=1"], [(b"content-length", b"nope")]
    )
    assert status == 413
    assert parsed == []
    assert consumed == 0


async def test_unthrottled_paths_have_no_body_cap():
    status, parsed, _ = await _run_middleware(
        [b"a=" + b"x" * 500], [], path="/somewhere-else"
    )
    assert status == 200
    assert parsed == [1]
