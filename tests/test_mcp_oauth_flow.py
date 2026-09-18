"""End-to-end OAuth 2.1 flow tests over the mounted MCP app (TestClient)."""

import base64
import hashlib
import os
import secrets
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

os.environ["CREDIT_PROXY_URL"] = "http://localhost:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["USE_MOCK"] = "true"
os.environ["MAX_REQUESTS_PER_MINUTE_PER_USER"] = "1000"
os.environ["MCP_MAX_REQUESTS_PER_MINUTE_PER_USER"] = "1000"
# This module registers a fresh client in most tests, all from one TestClient
# "IP"; the per-IP OAuth throttle is exercised in test_mcp_throttle.py instead.
os.environ["MCP_REGISTER_REQUESTS_PER_MINUTE_PER_IP"] = "1000"
os.environ["MCP_OAUTH_REQUESTS_PER_MINUTE_PER_IP"] = "1000"
os.environ["ENABLE_MCP"] = "true"
# Writes on, so this module can exercise the stories:write grant end to end.
os.environ["ENABLE_MCP_WRITES"] = "true"
os.environ["MCP_CONSENT_URL"] = "https://consent.example/mcp-connect"
# RFC 8414 requires HTTPS issuers; the SDK carves out localhost for testing.
os.environ["MCP_ISSUER_URL"] = "http://localhost:8000"
# Prevent the sync Firestore client from ever reaching a real project; the
# actual client is replaced with a fake below.
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:9999")
# Force the production-style Firebase verification path even when the
# developer's .env sets FIREBASE_AUTH_EMULATOR_HOST (load_dotenv never
# overrides pre-set env vars, and "" is falsy for the branch check).
os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = ""

from fastapi.testclient import TestClient  # noqa: E402

import mcp_server.app as mcp_app_module  # noqa: E402
from tests.mcp_fakes import FakeFirestoreClient  # noqa: E402

fake_db = FakeFirestoreClient()
# The rollout allowlist is on by default, so the flow's test user has to be
# approved or every consent in this module would 403. test_access_gate.py
# covers the refusal path.
fake_db.seed("mcpAccess/user-a", {"status": "granted"})
fake_db.seed("mcpAccess/emu-user", {"status": "granted"})

with patch.object(mcp_app_module, "_make_firestore_client", return_value=fake_db):
    from server import create_app

    app = create_app()

REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"


@pytest.fixture(scope="module")
def client():
    # One lifespan for the whole module: the MCP session manager's run() can
    # only be entered once per instance.
    with TestClient(app) as test_client:
        yield test_client


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _register(client, scope: str | None = None) -> str:
    """Register a client. `scope` omitted means the server's default_scopes."""
    body = {
        "client_name": "Flow Test Client",
        "redirect_uris": [REDIRECT_URI],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    if scope is not None:
        body["scope"] = scope
    resp = client.post("/register", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["client_id"]


def _authorize(
    client,
    client_id: str,
    challenge: str,
    resource: str | None = None,
    scope: str = "stories:read",
) -> str:
    """Run /authorize; return the consent txn id."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "state-123",
        "scope": scope,
    }
    if resource:
        params["resource"] = resource
    resp = client.get(
        "/authorize",
        params=params,
        follow_redirects=False,
    )
    assert resp.status_code in (302, 307), resp.text
    location = resp.headers["location"]
    assert location.startswith("https://consent.example/mcp-connect?txn=")
    return parse_qs(urlparse(location).query)["txn"][0]


def _approve(client, txn_id: str, uid: str = "user-a") -> str:
    """Consent-page approval with a stubbed Firebase verification; return code."""
    with patch(
        "mcp_server.oauth_routes.google_id_token.verify_firebase_token",
        return_value={"sub": uid},
    ):
        resp = client.post(
            "/oauth/complete",
            json={"txn_id": txn_id, "approve": True, "id_token": "stub-token"},
        )
    assert resp.status_code == 200, resp.text
    redirect_url = resp.json()["redirect_url"]
    assert redirect_url.startswith(REDIRECT_URI)
    query = parse_qs(urlparse(redirect_url).query)
    assert query["state"] == ["state-123"]
    return query["code"][0]


def _exchange(
    client, client_id: str, code: str, verifier: str, resource: str | None = None
):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    if resource:
        data["resource"] = resource
    return client.post("/token", data=data)


# ---------------------------------------------------------------------------
# Discovery metadata
# ---------------------------------------------------------------------------


def test_authorization_server_metadata(client):
    resp = client.get("/.well-known/oauth-authorization-server")
    assert resp.status_code == 200
    body = resp.json()
    assert body["issuer"].rstrip("/") == "http://localhost:8000"
    assert body["authorization_endpoint"].endswith("/authorize")
    assert body["token_endpoint"].endswith("/token")
    assert body["registration_endpoint"].endswith("/register")
    assert body["code_challenge_methods_supported"] == ["S256"]


def test_metadata_advertises_only_public_client_auth(client):
    """The document must describe the server that exists.

    The SDK hard-codes the two client_secret methods here, but every
    registration is downgraded to a public client, so advertising them would
    invite a client to authenticate in a way this server never checks.
    """
    body = client.get("/.well-known/oauth-authorization-server").json()
    assert body["token_endpoint_auth_methods_supported"] == ["none"]
    assert body["revocation_endpoint_auth_methods_supported"] == ["none"]


def test_protected_resource_metadata(client):
    resp = client.get("/.well-known/oauth-protected-resource/mcp")
    assert resp.status_code == 200
    body = resp.json()
    assert body["resource"].rstrip("/").endswith("/mcp")
    assert any(
        issuer.rstrip("/") == "http://localhost:8000"
        for issuer in body["authorization_servers"]
    )


def test_health_reports_mcp_available(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["services"]["mcp"] == "available"


# ---------------------------------------------------------------------------
# Registration + authorization + token
# ---------------------------------------------------------------------------


def test_registration_issues_no_client_secret(client):
    """A secret we would have to store in the clear is never minted at all."""
    resp = client.post(
        "/register",
        json={"redirect_uris": [REDIRECT_URI], "client_name": "Secretless"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # Absent from the response, so it never reaches the wire or a config file.
    assert body.get("client_secret") is None
    assert body["token_endpoint_auth_method"] == "none"
    # ...and absent from Firestore, which is the point.
    stored = fake_db.collection("mcpOauthClients").document(body["client_id"]).get()
    assert "client_secret" not in stored.to_dict()


def test_token_exchange_ignores_a_posted_client_secret(client):
    """A client that asks for client_secret_post must not be broken by this.

    Real clients register without naming an auth method, which the SDK would
    default to client_secret_post. They may then send a `client_secret` form
    field out of habit or from a cached registration. ClientAuthenticator skips
    the comparison when the stored client has no secret, so the flow still
    completes — this pins that, because getting it wrong would lock out every
    existing connector.
    """
    verifier, challenge = _pkce_pair()
    resp = client.post(
        "/register",
        json={
            "redirect_uris": [REDIRECT_URI],
            "client_name": "Secret Poster",
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    client_id = resp.json()["client_id"]
    txn_id = _authorize(client, client_id, challenge)
    code = _approve(client, txn_id)

    token_resp = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
            "client_secret": "a-secret-the-server-never-issued",
        },
    )
    assert token_resp.status_code == 200, token_resp.text
    assert token_resp.json()["access_token"].startswith("mcp_at_")


def test_registration_rejects_non_loopback_http(client):
    resp = client.post(
        "/register",
        json={
            "client_name": "Evil",
            "redirect_uris": ["http://evil.com/cb"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_redirect_uri"


def test_full_authorization_code_flow(client):
    verifier, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)

    # Consent page can inspect the pending txn.
    txn_info = client.get(f"/oauth/txn/{txn_id}").json()
    assert txn_info["client_name"] == "Flow Test Client"
    assert txn_info["redirect_host"] == "claude.ai"

    code = _approve(client, txn_id)
    resp = _exchange(client, client_id, code, verifier)
    assert resp.status_code == 200, resp.text
    token = resp.json()
    assert token["access_token"].startswith("mcp_at_")
    assert token["refresh_token"].startswith("mcp_rt_")
    assert token["token_type"].lower() == "bearer"

    # Refresh grant rotates the pair.
    refresh_resp = client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": token["refresh_token"],
            "client_id": client_id,
        },
    )
    assert refresh_resp.status_code == 200, refresh_resp.text
    rotated = refresh_resp.json()
    assert rotated["access_token"] != token["access_token"]


def test_full_flow_with_resource_indicator(client):
    """RFC 8707: real clients (Claude Code) always send `resource`.

    The round-trip through Firestore must hand the SDK a plain str — an AnyUrl
    fails AuthorizationCode validation and surfaces as a bare 500 at /token.
    """
    resource = "https://mcp.example.com/mcp"
    verifier, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge, resource=resource)
    code = _approve(client, txn_id)

    resp = _exchange(client, client_id, code, verifier, resource=resource)
    assert resp.status_code == 200, resp.text
    assert resp.json()["access_token"].startswith("mcp_at_")


def test_token_exchange_rejects_wrong_verifier(client):
    verifier, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    code = _approve(client, txn_id)

    wrong = _exchange(client, client_id, code, "wrong-" + verifier)
    assert wrong.status_code == 400
    assert wrong.json()["error"] == "invalid_grant"


def test_authorization_code_is_single_use(client):
    verifier, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    code = _approve(client, txn_id)

    assert _exchange(client, client_id, code, verifier).status_code == 200
    replay = _exchange(client, client_id, code, verifier)
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"


def test_consent_txn_is_single_use(client):
    _, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    _approve(client, txn_id)

    assert client.get(f"/oauth/txn/{txn_id}").status_code == 404
    with patch(
        "mcp_server.oauth_routes.google_id_token.verify_firebase_token",
        return_value={"sub": "user-a"},
    ):
        replay = client.post(
            "/oauth/complete",
            json={"txn_id": txn_id, "approve": True, "id_token": "stub"},
        )
    assert replay.status_code == 404


def test_deny_redirects_with_access_denied(client):
    _, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    resp = client.post("/oauth/complete", json={"txn_id": txn_id, "approve": False})
    assert resp.status_code == 200
    assert "error=access_denied" in resp.json()["redirect_url"]


def test_emulator_tokens_accepted_only_with_emulator_env(client):
    """Dev mode: unsigned Firebase-emulator JWTs are decoded without verification."""
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=")
    payload = base64.urlsafe_b64encode(b'{"sub":"emu-user"}').rstrip(b"=")
    unsigned = (header + b"." + payload + b".").decode("ascii")

    _, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    with patch.dict(os.environ, {"FIREBASE_AUTH_EMULATOR_HOST": "localhost:9099"}):
        resp = client.post(
            "/oauth/complete",
            json={"txn_id": txn_id, "approve": True, "id_token": unsigned},
        )
    assert resp.status_code == 200, resp.text
    assert "code=" in resp.json()["redirect_url"]


def test_invalid_firebase_token_rejected(client):
    _, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    with patch(
        "mcp_server.oauth_routes.google_id_token.verify_firebase_token",
        side_effect=ValueError("bad token"),
    ):
        resp = client.post(
            "/oauth/complete",
            json={"txn_id": txn_id, "approve": True, "id_token": "garbage"},
        )
    assert resp.status_code == 401
    # Txn is still pending: the user can retry with a valid session.
    assert client.get(f"/oauth/txn/{txn_id}").status_code == 200


# ---------------------------------------------------------------------------
# Resource server enforcement
# ---------------------------------------------------------------------------

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}


def test_mcp_requires_bearer_token(client):
    resp = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
    assert resp.status_code == 401
    www_auth = resp.headers.get("WWW-Authenticate", "")
    assert "resource_metadata" in www_auth


def test_mcp_rejects_garbage_token(client):
    resp = client.post(
        "/mcp",
        json=INITIALIZE,
        headers={**MCP_HEADERS, "Authorization": "Bearer mcp_at_garbage"},
    )
    assert resp.status_code == 401


def test_mcp_accepts_valid_token(client):
    verifier, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    code = _approve(client, txn_id)
    token = _exchange(client, client_id, code, verifier).json()

    resp = client.post(
        "/mcp",
        json=INITIALIZE,
        headers={**MCP_HEADERS, "Authorization": f"Bearer {token['access_token']}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["serverInfo"]["name"] == "TheTaleTribe"


def test_refresh_reuse_kills_the_grant_end_to_end(client):
    """A replayed refresh token must invalidate the successor, over real HTTP.

    Exercises the path the SDK actually takes: /token rejects a rotated refresh
    token inside load_refresh_token and never reaches exchange_refresh_token, so
    detection has to live there. Ends by proving the thief's access token stops
    working at /mcp — the whole point of the exercise.
    """
    verifier, challenge = _pkce_pair()
    client_id = _register(client)
    txn_id = _authorize(client, client_id, challenge)
    code = _approve(client, txn_id)
    original = _exchange(client, client_id, code, verifier).json()

    def _refresh(refresh_token: str):
        return client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
        )

    # The thief rotates first and walks away with a working pair.
    stolen = _refresh(original["refresh_token"])
    assert stolen.status_code == 200, stolen.text
    stolen_tokens = stolen.json()
    assert (
        client.post(
            "/mcp",
            json=INITIALIZE,
            headers={
                **MCP_HEADERS,
                "Authorization": f"Bearer {stolen_tokens['access_token']}",
            },
        ).status_code
        == 200
    )

    # The real client refreshes with the token it still thinks is current.
    replay = _refresh(original["refresh_token"])
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"

    # That replay burned the whole lineage, including the thief's pair.
    burned = _refresh(stolen_tokens["refresh_token"])
    assert burned.status_code == 400
    assert burned.json()["error"] == "invalid_grant"

    resp = client.post(
        "/mcp",
        json=INITIALIZE,
        headers={
            **MCP_HEADERS,
            "Authorization": f"Bearer {stolen_tokens['access_token']}",
        },
    )
    assert resp.status_code == 401


def test_agent_execute_still_requires_internal_token(client):
    resp = client.post(
        "/agent/execute",
        json={"action": "generate_story", "parameters": {}, "user_id": "u"},
    )
    # Outside production the OIDC guard is a no-op, but the request must never
    # be treated as an MCP route; it still hits the guarded host endpoint.
    assert resp.status_code in (200, 401, 422, 500)
    assert "jsonrpc" not in resp.text


# ---------------------------------------------------------------------------
# Write scope
# ---------------------------------------------------------------------------


def test_protected_resource_metadata_advertises_write(client):
    """The SDK builds this document from required_scopes, which is the list
    RequireAuthMiddleware demands a token ALREADY has — so stories:write can
    never appear there. Without the shadow route in oauth_routes.py the client
    SDK would copy scopes_supported into its registration verbatim and could
    never request write at all, leaving the write tools unreachable."""
    resp = client.get("/.well-known/oauth-protected-resource/mcp")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["scopes_supported"]) == {"stories:read", "stories:write"}
    assert body["resource"].rstrip("/").endswith("/mcp")
    assert body["authorization_servers"]


def test_register_defaults_to_read_only(client):
    """Omitting `scope` must not hand out write by accident."""
    resp = client.post(
        "/register",
        json={
            "client_name": "Defaulting Client",
            "redirect_uris": [REDIRECT_URI],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["scope"] == "stories:read"


def test_register_rejects_unknown_scope(client):
    resp = client.post(
        "/register",
        json={
            "client_name": "Greedy Client",
            "redirect_uris": [REDIRECT_URI],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "scope": "stories:admin",
        },
    )
    assert resp.status_code == 400, resp.text


def test_authorize_rejects_write_for_a_read_only_registration(client):
    """Write cannot be self-granted after the fact: /authorize validates the
    request against the CLIENT's registered ceiling, not the server's."""
    client_id = _register(client, scope="stories:read")
    _verifier, challenge = _pkce_pair()
    resp = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "state-123",
            "scope": "stories:read stories:write",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (302, 307)
    location = resp.headers["location"]
    assert location.startswith(REDIRECT_URI)
    assert "error=invalid_scope" in location


def test_write_scoped_flow_end_to_end(client):
    client_id = _register(client, scope="stories:read stories:write")
    verifier, challenge = _pkce_pair()
    txn_id = _authorize(
        client, client_id, challenge, scope="stories:read stories:write"
    )
    code = _approve(client, txn_id)
    resp = _exchange(client, client_id, code, verifier)
    assert resp.status_code == 200, resp.text
    granted = set(resp.json()["scope"].split())
    assert granted == {"stories:read", "stories:write"}


def test_consent_page_sees_the_write_scope(client):
    """The consent copy in mcpConsentScopes.ts keys off exactly this list."""
    client_id = _register(client, scope="stories:read stories:write")
    _verifier, challenge = _pkce_pair()
    txn_id = _authorize(
        client, client_id, challenge, scope="stories:read stories:write"
    )
    resp = client.get(f"/oauth/txn/{txn_id}")
    assert resp.status_code == 200
    assert set(resp.json()["scopes"]) == {"stories:read", "stories:write"}


# ---------------------------------------------------------------------------
# Rollout allowlist
# ---------------------------------------------------------------------------


def test_consent_refused_for_a_user_not_on_the_allowlist(client):
    """A non-approved user must never receive a grant at all — refusing at
    consent is what stops them holding a token that only fails later."""
    client_id = _register(client)
    _verifier, challenge = _pkce_pair()
    txn_id = _authorize(client, client_id, challenge)
    with patch(
        "mcp_server.oauth_routes.google_id_token.verify_firebase_token",
        return_value={"sub": "stranger-uid"},
    ):
        resp = client.post(
            "/oauth/complete",
            json={"txn_id": txn_id, "approve": True, "id_token": "stub-token"},
        )
    assert resp.status_code == 403
    # The app wraps HTTPException detail in its standard error envelope.
    assert "request access" in resp.json()["error"]["message"].lower()


def test_revoked_status_is_refused_like_an_absent_record(client):
    fake_db.seed("mcpAccess/revoked-uid", {"status": "revoked"})
    client_id = _register(client)
    _verifier, challenge = _pkce_pair()
    txn_id = _authorize(client, client_id, challenge)
    with patch(
        "mcp_server.oauth_routes.google_id_token.verify_firebase_token",
        return_value={"sub": "revoked-uid"},
    ):
        resp = client.post(
            "/oauth/complete",
            json={"txn_id": txn_id, "approve": True, "id_token": "stub-token"},
        )
    assert resp.status_code == 403
