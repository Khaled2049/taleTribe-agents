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
from mcp_server.throttle import OAuthThrottleMiddleware, client_ip  # noqa: E402
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


def test_client_ip_prefers_first_forwarded_entry():
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"203.0.113.9, 70.41.3.18, 150.172.238.178")],
        "client": ("10.1.1.1", 1234),
    }
    assert client_ip(Request(scope)) == "203.0.113.9"


def test_client_ip_falls_back_to_socket_then_unknown():
    from starlette.requests import Request

    assert (
        client_ip(Request({"type": "http", "headers": [], "client": ("10.1.1.1", 80)}))
        == "10.1.1.1"
    )
    assert (
        client_ip(Request({"type": "http", "headers": [], "client": None})) == "unknown"
    )


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
