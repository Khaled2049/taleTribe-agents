"""Unit tests for mcp_server.oauth_store and mcp_server.oauth_provider."""

from datetime import datetime, timedelta, timezone

import pytest
from mcp.server.auth.provider import (
    AuthorizationParams,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from mcp_server.oauth_provider import (
    FirestoreOAuthProvider,
)
from mcp_server.oauth_store import (
    TXN_STATUS_COMPLETED,
    OAuthStore,
    hash_token,
)
from tests.mcp_fakes import FakeFirestoreClient

CONSENT_URL = "https://thetaletribe.web.app/mcp-connect"


def _store() -> tuple[OAuthStore, FakeFirestoreClient]:
    db = FakeFirestoreClient()
    return OAuthStore(db), db


def _provider(store: OAuthStore) -> FirestoreOAuthProvider:
    return FirestoreOAuthProvider(
        store,
        consent_url=CONSENT_URL,
        access_token_ttl_seconds=3600,
        refresh_token_ttl_seconds=2592000,
    )


def _client(client_id: str = "client-1", **overrides) -> OAuthClientInformationFull:
    payload = {
        "client_id": client_id,
        "client_name": "Test MCP Client",
        "redirect_uris": [AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    payload.update(overrides)
    return OAuthClientInformationFull.model_validate(payload)


def _params(**overrides) -> AuthorizationParams:
    payload = {
        "state": "xyz-state",
        "scopes": ["stories:read"],
        "code_challenge": "a" * 43,
        "redirect_uri": AnyUrl("https://claude.ai/api/mcp/auth_callback"),
        "redirect_uri_provided_explicitly": True,
        "resource": None,
    }
    payload.update(overrides)
    return AuthorizationParams(**payload)


# ---------------------------------------------------------------------------
# Store primitives
# ---------------------------------------------------------------------------


def test_expired_client_is_not_returned():
    """An abandoned registration stops being usable before TTL collects it."""
    store, _ = _store()
    store.save_client("c1", {"client_id": "c1"}, ttl_seconds=-1)
    assert store.get_client("c1") is None
    # Renewal must not resurrect it: expiry is checked before the sliding write.
    assert store.get_client("c1", renew_ttl_seconds=3600) is None


def test_client_expiry_slides_forward_on_use():
    """Registered with a short window; first use promotes it to the full one."""
    store, db = _store()
    store.save_client("c1", {"client_id": "c1"}, 60)

    assert store.get_client("c1", renew_ttl_seconds=3600) == {"client_id": "c1"}

    data, _ = db.docs["mcpOauthClients/c1"]
    remaining = data["expiresAt"] - datetime.now(timezone.utc)
    assert timedelta(minutes=50) < remaining <= timedelta(hours=1)


def test_legacy_client_without_expiry_is_backfilled():
    """Records registered before expiry existed stay usable and gain a window."""
    store, db = _store()
    db.seed("mcpOauthClients/legacy", {"client_id": "legacy"})

    assert store.get_client("legacy", renew_ttl_seconds=3600) == {"client_id": "legacy"}

    data, _ = db.docs["mcpOauthClients/legacy"]
    assert data["expiresAt"] > datetime.now(timezone.utc)


def test_txn_expiry_checked_at_read():
    store, _ = _store()
    txn_id = store.create_txn({"clientId": "c1"}, ttl_seconds=-1)
    assert store.get_pending_txn(txn_id) is None
    assert store.finish_txn(txn_id, TXN_STATUS_COMPLETED) is None


def test_expired_code_cannot_be_consumed():
    store, _ = _store()
    code_hash = hash_token("code-2")
    store.save_code(code_hash, {"clientId": "c1", "subject": "u1"}, ttl_seconds=-1)
    assert store.consume_code(code_hash) is None


def _mint_pair(
    store: OAuthStore,
    access="at-1",
    refresh="rt-1",
    access_ttl=3600,
    family_id="fam-1",
):
    store.save_token_pair(
        access_hash=hash_token(access),
        refresh_hash=hash_token(refresh),
        uid="user-a",
        client_id="c1",
        scopes=["stories:read"],
        family_id=family_id,
        access_ttl_seconds=access_ttl,
        refresh_ttl_seconds=2592000,
    )


def test_expired_access_token_rejected_at_read():
    store, _ = _store()
    _mint_pair(store, access_ttl=-1)
    assert store.get_token(hash_token("at-1")) is None
    assert store.get_token(hash_token("rt-1")) is not None


def test_refresh_rotation_revokes_predecessor_pair():
    store, _ = _store()
    _mint_pair(store)
    consumed = store.consume_refresh_token(hash_token("rt-1"))
    assert consumed["uid"] == "user-a"
    # Both halves of the old pair are dead; a second rotation attempt fails.
    assert store.get_token(hash_token("rt-1")) is None
    assert store.get_token(hash_token("at-1")) is None
    assert store.consume_refresh_token(hash_token("rt-1")) is None


def test_consume_refresh_rejects_access_tokens():
    store, _ = _store()
    _mint_pair(store)
    assert store.consume_refresh_token(hash_token("at-1")) is None


def test_revoke_token_kills_both_halves():
    store, _ = _store()
    _mint_pair(store)
    store.revoke_token(hash_token("at-1"))
    assert store.get_token(hash_token("at-1")) is None
    assert store.get_token(hash_token("rt-1")) is None


def test_include_revoked_distinguishes_replay_from_unknown():
    """The signal reuse detection runs on: rotated-away vs. never-existed."""
    store, _ = _store()
    _mint_pair(store)
    store.consume_refresh_token(hash_token("rt-1"))

    assert store.get_token(hash_token("rt-1")) is None
    replayed = store.get_token(hash_token("rt-1"), include_revoked=True)
    assert replayed["revoked"] is True
    assert replayed["familyId"] == "fam-1"
    # A token we never issued stays invisible either way.
    assert store.get_token(hash_token("never-issued"), include_revoked=True) is None


def test_include_revoked_still_hides_expired_tokens():
    """An expired replay carries no signal, so it must not resurrect a record."""
    store, _ = _store()
    _mint_pair(store, access_ttl=-1)
    assert store.get_token(hash_token("at-1"), include_revoked=True) is None


def test_revoke_token_family_kills_live_descendants_only():
    store, _ = _store()
    _mint_pair(store, access="at-1", refresh="rt-1", family_id="fam-1")
    _mint_pair(store, access="at-2", refresh="rt-2", family_id="fam-1")
    # An unrelated grant must survive.
    _mint_pair(store, access="at-9", refresh="rt-9", family_id="fam-other")
    store.consume_refresh_token(hash_token("rt-1"))  # rotates gen 1 away

    # Only the two still-live gen-2 documents are rewritten.
    assert store.revoke_token_family("fam-1", 2592000) == 2

    for token in ("at-1", "rt-1", "at-2", "rt-2"):
        assert store.get_token(hash_token(token)) is None
    assert store.get_token(hash_token("rt-9")) is not None


def test_every_revocation_path_stamps_revoked_at():
    """revokedAt is what makes a reuse alert triageable; no path may skip it."""
    store, _ = _store()

    _mint_pair(store, access="at-1", refresh="rt-1")
    store.consume_refresh_token(hash_token("rt-1"))  # rotation + paired revoke
    for token in ("rt-1", "at-1"):
        record = store.get_token(hash_token(token), include_revoked=True)
        assert record["revokedAt"] is not None

    _mint_pair(store, access="at-2", refresh="rt-2")
    store.revoke_token(hash_token("at-2"))  # RFC 7009
    for token in ("at-2", "rt-2"):
        record = store.get_token(hash_token(token), include_revoked=True)
        assert record["revokedAt"] is not None

    _mint_pair(store, access="at-3", refresh="rt-3", family_id="fam-3")
    store.revoke_token_family("fam-3", 2592000)  # reuse detection
    for token in ("at-3", "rt-3"):
        record = store.get_token(hash_token(token), include_revoked=True)
        assert record["revokedAt"] is not None


def test_mint_refused_after_family_revoked_with_no_live_tokens():
    store, _ = _store()
    _mint_pair(store, access="at-1", refresh="rt-1", family_id="fam-r")
    store.consume_refresh_token(hash_token("rt-1"))
    assert store.revoke_token_family("fam-r", 2592000) == 0

    assert (
        store.save_token_pair(
            access_hash=hash_token("at-2"),
            refresh_hash=hash_token("rt-2"),
            uid="user-a",
            client_id="c1",
            scopes=["stories:read"],
            family_id="fam-r",
            access_ttl_seconds=3600,
            refresh_ttl_seconds=2592000,
        )
        is False
    )
    assert store.get_token(hash_token("at-2")) is None
    assert store.get_token(hash_token("rt-2")) is None


@pytest.mark.parametrize("family_exists", [True, False])
def test_revocation_during_mint_commit_is_not_lost(family_exists):
    store, db = _store()
    if family_exists:
        _mint_pair(store, access="at-1", refresh="rt-1", family_id="fam-c")
        store.consume_refresh_token(hash_token("rt-1"))
    fired = []

    def revoke_once():
        if not fired:
            fired.append(True)
            store.revoke_token_family("fam-c", 2592000)

    db.before_commit = revoke_once
    saved = store.save_token_pair(
        access_hash=hash_token("at-2"),
        refresh_hash=hash_token("rt-2"),
        uid="user-a",
        client_id="c1",
        scopes=["stories:read"],
        family_id="fam-c",
        access_ttl_seconds=3600,
        refresh_ttl_seconds=2592000,
    )
    assert fired and saved is False
    assert "mcpOauthTokens/" + hash_token("at-2") not in db.docs
    assert "mcpOauthTokens/" + hash_token("rt-2") not in db.docs


def test_revoked_family_rejects_tokens_the_walk_missed():
    store, db = _store()
    _mint_pair(store, access="at-1", refresh="rt-1", family_id="fam-w")
    db.collection("mcpOauthFamilies").document("fam-w").set(
        {"revoked": True}, merge=True
    )
    assert store.get_token(hash_token("at-1")) is None
    assert store.get_token(hash_token("rt-1")) is None
    assert store.get_token(hash_token("rt-1"), include_revoked=True) is not None


# ---------------------------------------------------------------------------
# Provider: registration hardening
# ---------------------------------------------------------------------------


async def test_register_client_accepts_https_and_loopback():
    store, _ = _store()
    provider = _provider(store)
    client = _client(
        redirect_uris=[
            AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            AnyUrl("http://localhost:33418/cb"),
            AnyUrl("http://127.0.0.1:6274/oauth/callback"),
        ]
    )
    await provider.register_client(client)
    loaded = await provider.get_client("client-1")
    assert loaded is not None
    assert loaded.client_name == "Test MCP Client"


# ---------------------------------------------------------------------------
# Provider: authorization flow
# ---------------------------------------------------------------------------


async def _authorize(provider, client, params=None):
    consent = await provider.authorize(client, params or _params())
    assert consent.startswith(f"{CONSENT_URL}?txn=")
    return consent.split("txn=")[1]


async def _granted_code(provider, client) -> str:
    txn_id = await _authorize(provider, client)
    redirect = await provider.complete_authorization(txn_id, uid="user-a")
    query = redirect.split("?", 1)[1]
    return dict(part.split("=", 1) for part in query.split("&"))["code"]


async def test_code_exchange_mints_tokens_bound_to_uid():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)

    loaded = await provider.load_authorization_code(client, code)
    assert loaded is not None
    assert loaded.subject == "user-a"
    assert loaded.code_challenge == "a" * 43

    token = await provider.exchange_authorization_code(client, loaded)
    assert token.access_token.startswith("mcp_at_")
    assert token.refresh_token.startswith("mcp_rt_")
    assert token.expires_in == 3600

    access = await provider.load_access_token(token.access_token)
    assert access is not None
    assert access.subject == "user-a"
    assert access.scopes == ["stories:read"]


async def test_code_without_expiry_is_rejected(monkeypatch):
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    monkeypatch.setattr(store, "load_code", lambda _: {"clientId": client.client_id})
    assert await provider.load_authorization_code(client, "malformed-code") is None


async def test_code_bound_to_registered_client():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    other = _client(client_id="client-2")
    assert await provider.load_authorization_code(other, code) is None


async def test_replay_between_rotation_and_mint_kills_the_successor():
    import anyio

    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    loaded = await provider.load_authorization_code(client, code)
    first = await provider.exchange_authorization_code(client, loaded)
    stolen = await provider.load_refresh_token(client, first.refresh_token)

    consume = store.consume_refresh_token

    def consume_then_replay(token_hash):
        record = consume(token_hash)
        replayed = anyio.from_thread.run(
            provider.load_refresh_token, client, first.refresh_token
        )
        assert replayed is None
        return record

    store.consume_refresh_token = consume_then_replay
    with pytest.raises(TokenError) as exc:
        await provider.exchange_refresh_token(client, stolen, ["stories:read"])
    assert exc.value.error == "invalid_grant"
    assert not any(
        path.startswith("mcpOauthTokens/") and not data.get("revoked")
        for path, (data, _) in store._db.docs.items()
    )


async def test_refresh_reuse_spares_other_grants():
    """One compromised lineage must not log the user out of everything."""
    store, _ = _store()
    provider = _provider(store)
    client = _client()

    async def _grant():
        code = await _granted_code(provider, client)
        loaded = await provider.load_authorization_code(client, code)
        return await provider.exchange_authorization_code(client, loaded)

    compromised = await _grant()
    healthy = await _grant()

    stale = await provider.load_refresh_token(client, compromised.refresh_token)
    await provider.exchange_refresh_token(client, stale, ["stories:read"])
    await provider.load_refresh_token(client, compromised.refresh_token)  # replay

    assert await provider.load_access_token(healthy.access_token) is not None
    assert await provider.load_refresh_token(client, healthy.refresh_token) is not None


async def test_refresh_reuse_survives_a_token_without_a_family():
    """Tokens predating family tracking must log, not crash."""
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    store.save_token_pair(
        access_hash=hash_token("legacy-at"),
        refresh_hash=hash_token("legacy-rt"),
        uid="user-a",
        client_id=client.client_id,
        scopes=["stories:read"],
        family_id="",
        access_ttl_seconds=3600,
        refresh_ttl_seconds=2592000,
    )
    store.consume_refresh_token(hash_token("legacy-rt"))

    assert await provider.load_refresh_token(client, "legacy-rt") is None


async def test_refresh_never_widens_scope():
    """Now load-bearing: stories:write is a real scope, so a refresh that could
    widen into it would be a genuine privilege escalation rather than a
    hypothetical one."""
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    loaded = await provider.load_authorization_code(client, code)
    first = await provider.exchange_authorization_code(client, loaded)
    refresh = await provider.load_refresh_token(client, first.refresh_token)
    rotated = await provider.exchange_refresh_token(
        client, refresh, ["stories:read", "stories:write"]
    )
    access = await provider.load_access_token(rotated.access_token)
    assert access.scopes == ["stories:read"]


async def test_revoke_token_via_provider():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    loaded = await provider.load_authorization_code(client, code)
    token = await provider.exchange_authorization_code(client, loaded)
    access = await provider.load_access_token(token.access_token)
    await provider.revoke_token(access)
    assert await provider.load_access_token(token.access_token) is None
    assert await provider.load_refresh_token(client, token.refresh_token) is None
