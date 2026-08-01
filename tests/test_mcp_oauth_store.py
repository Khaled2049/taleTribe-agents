"""Unit tests for mcp_server.oauth_store and mcp_server.oauth_provider."""

from datetime import datetime, timedelta, timezone

import pytest
from mcp.server.auth.provider import (
    AuthorizationParams,
    RegistrationError,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from mcp_server.oauth_provider import (
    RETRY_WINDOW_SECONDS,
    FirestoreOAuthProvider,
    TxnNotFoundError,
    seconds_since_rotation,
)
from mcp_server.oauth_store import (
    TXN_STATUS_COMPLETED,
    OAuthStore,
    as_utc,
    hash_token,
    new_secret,
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


def test_hash_token_is_deterministic_and_not_identity():
    assert hash_token("abc") == hash_token("abc")
    assert hash_token("abc") != "abc"
    assert len(hash_token("abc")) == 64


def test_new_secret_has_prefix_and_entropy():
    a, b = new_secret("mcp_at"), new_secret("mcp_at")
    assert a.startswith("mcp_at_") and b.startswith("mcp_at_")
    assert a != b


def test_client_roundtrip():
    store, _ = _store()
    store.save_client("c1", {"client_id": "c1", "client_name": "X"}, 3600)
    # createdAt/expiresAt are internal: the SDK's model would reject them.
    assert store.get_client("c1") == {"client_id": "c1", "client_name": "X"}
    assert store.get_client("missing") is None


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


def test_client_renewal_throttled_before_half_life():
    """A client refreshing tokens must not cost a Firestore write per call."""
    store, db = _store()
    store.save_client("c1", {"client_id": "c1"}, 3600)
    _, version_after_save = db.docs["mcpOauthClients/c1"]

    for _ in range(5):
        assert store.get_client("c1", renew_ttl_seconds=3600) is not None

    _, version_now = db.docs["mcpOauthClients/c1"]
    assert version_now == version_after_save, "expiry rewritten inside its half-life"


def test_legacy_client_without_expiry_is_backfilled():
    """Records registered before expiry existed stay usable and gain a window."""
    store, db = _store()
    db.seed("mcpOauthClients/legacy", {"client_id": "legacy"})

    assert store.get_client("legacy", renew_ttl_seconds=3600) == {"client_id": "legacy"}

    data, _ = db.docs["mcpOauthClients/legacy"]
    assert data["expiresAt"] > datetime.now(timezone.utc)


def test_txn_lifecycle_single_use():
    store, _ = _store()
    txn_id = store.create_txn({"clientId": "c1"}, ttl_seconds=600)
    assert store.get_pending_txn(txn_id)["clientId"] == "c1"
    assert store.finish_txn(txn_id, TXN_STATUS_COMPLETED)["clientId"] == "c1"
    # No longer pending, cannot be finished twice.
    assert store.get_pending_txn(txn_id) is None
    assert store.finish_txn(txn_id, TXN_STATUS_COMPLETED) is None


def test_txn_expiry_checked_at_read():
    store, _ = _store()
    txn_id = store.create_txn({"clientId": "c1"}, ttl_seconds=-1)
    assert store.get_pending_txn(txn_id) is None
    assert store.finish_txn(txn_id, TXN_STATUS_COMPLETED) is None


def test_code_single_use():
    store, _ = _store()
    code_hash = hash_token("code-1")
    store.save_code(code_hash, {"clientId": "c1", "subject": "u1"}, ttl_seconds=300)
    assert store.load_code(code_hash)["subject"] == "u1"
    assert store.consume_code(code_hash)["subject"] == "u1"
    assert store.consume_code(code_hash) is None
    assert store.load_code(code_hash) is None


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


def test_token_pair_readable_until_revoked():
    store, _ = _store()
    _mint_pair(store)
    assert store.get_token(hash_token("at-1"))["type"] == "access"
    assert store.get_token(hash_token("rt-1"))["type"] == "refresh"


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


def test_token_pair_carries_family_id():
    store, _ = _store()
    _mint_pair(store, family_id="fam-xyz")
    assert store.get_token(hash_token("at-1"))["familyId"] == "fam-xyz"
    assert store.get_token(hash_token("rt-1"))["familyId"] == "fam-xyz"


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
    assert store.revoke_token_family("fam-1") == 2

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
    store.revoke_token_family("fam-3")  # reuse detection
    for token in ("at-3", "rt-3"):
        record = store.get_token(hash_token(token), include_revoked=True)
        assert record["revokedAt"] is not None


def test_as_utc_normalizes_naive_and_rejects_non_datetimes():
    naive = datetime(2026, 7, 1, 12, 0, 0)
    assert as_utc(naive) == datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    aware = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert as_utc(aware) is aware

    assert as_utc(None) is None
    assert as_utc("2026-07-01") is None


def test_revoke_token_family_ignores_missing_family():
    store, _ = _store()
    _mint_pair(store)
    assert store.revoke_token_family("") == 0
    assert store.revoke_token_family("fam-unknown") == 0
    assert store.get_token(hash_token("rt-1")) is not None


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


async def test_register_client_downgrades_to_a_public_client():
    """The provider must strip the secret from both the record and the response.

    The SDK serializes the same object it hands us as the 201 body, so mutating
    it here is what keeps the secret off the wire as well as out of Firestore.
    """
    store, db = _store()
    provider = _provider(store)
    client_info = OAuthClientInformationFull(
        client_id="client-secretless",
        client_secret="a-secret-the-sdk-generated",
        client_secret_expires_at=99999999,
        token_endpoint_auth_method="client_secret_post",
        redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
    )
    await provider.register_client(client_info)

    assert client_info.client_secret is None
    assert client_info.client_secret_expires_at is None
    assert client_info.token_endpoint_auth_method == "none"

    stored = db.docs["mcpOauthClients/client-secretless"][0]
    assert "client_secret" not in stored
    assert stored["token_endpoint_auth_method"] == "none"

    # And it round-trips back through the SDK's model without the secret.
    loaded = await provider.get_client("client-secretless")
    assert loaded is not None and loaded.client_secret is None


async def test_register_client_rejects_plain_http():
    store, _ = _store()
    provider = _provider(store)
    client = _client(redirect_uris=[AnyUrl("http://evil.com/cb")])
    with pytest.raises(RegistrationError) as exc:
        await provider.register_client(client)
    assert exc.value.error == "invalid_redirect_uri"


# ---------------------------------------------------------------------------
# Provider: authorization flow
# ---------------------------------------------------------------------------


async def _authorize(provider, client, params=None):
    consent = await provider.authorize(client, params or _params())
    assert consent.startswith(f"{CONSENT_URL}?txn=")
    return consent.split("txn=")[1]


async def test_authorize_then_complete_yields_code_redirect():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    txn_id = await _authorize(provider, client)

    info = await provider.get_txn_info(txn_id)
    assert info["client_name"] == "Test MCP Client"
    assert info["redirect_host"] == "claude.ai"
    assert info["scopes"] == ["stories:read"]

    redirect = await provider.complete_authorization(txn_id, uid="user-a")
    assert redirect.startswith("https://claude.ai/api/mcp/auth_callback?")
    assert "code=mcp_ac_" in redirect
    assert "state=xyz-state" in redirect
    # A txn is single-use.
    with pytest.raises(TxnNotFoundError):
        await provider.complete_authorization(txn_id, uid="user-a")


async def test_deny_authorization_redirects_with_error():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    txn_id = await _authorize(provider, client)
    redirect = await provider.deny_authorization(txn_id)
    assert "error=access_denied" in redirect
    assert "state=xyz-state" in redirect


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


async def test_code_reuse_is_rejected():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    loaded = await provider.load_authorization_code(client, code)
    await provider.exchange_authorization_code(client, loaded)
    with pytest.raises(TokenError) as exc:
        await provider.exchange_authorization_code(client, loaded)
    assert exc.value.error == "invalid_grant"


async def test_code_bound_to_registered_client():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    other = _client(client_id="client-2")
    assert await provider.load_authorization_code(other, code) is None


async def test_refresh_rotation_via_provider():
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    loaded = await provider.load_authorization_code(client, code)
    first = await provider.exchange_authorization_code(client, loaded)

    refresh = await provider.load_refresh_token(client, first.refresh_token)
    assert refresh is not None and refresh.subject == "user-a"

    second = await provider.exchange_refresh_token(client, refresh, ["stories:read"])
    assert second.access_token != first.access_token

    # Old pair is fully dead; the new access token carries the same uid.
    # Checked at the store, not via load_refresh_token: presenting a rotated
    # token is exactly what reuse detection reacts to, and this test is about
    # the happy path. See test_refresh_reuse_revokes_the_whole_family.
    assert await provider.load_access_token(first.access_token) is None
    assert store.get_token(hash_token(first.refresh_token)) is None
    new_access = await provider.load_access_token(second.access_token)
    assert new_access.subject == "user-a"

    with pytest.raises(TokenError):
        await provider.exchange_refresh_token(client, refresh, ["stories:read"])


async def test_refresh_reuse_revokes_the_whole_family():
    """Replaying a rotated refresh token must kill the successor pair too.

    Without this, rotation only shortens the thief's window: the loser of the
    race silently re-authenticates and the theft leaves no mark. Here the thief
    has already rotated and holds `second`; the real client then presents its
    now-stale token, and both are destroyed.
    """
    store, _ = _store()
    provider = _provider(store)
    client = _client()
    code = await _granted_code(provider, client)
    loaded = await provider.load_authorization_code(client, code)
    first = await provider.exchange_authorization_code(client, loaded)

    stolen = await provider.load_refresh_token(client, first.refresh_token)
    second = await provider.exchange_refresh_token(client, stolen, ["stories:read"])
    assert await provider.load_access_token(second.access_token) is not None

    # The rightful client now replays the token it still believes is current.
    assert await provider.load_refresh_token(client, first.refresh_token) is None

    # The thief's freshly minted pair died with it — the race has no winner.
    assert await provider.load_access_token(second.access_token) is None
    assert store.get_token(hash_token(second.refresh_token)) is None


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


def test_reuse_triage_separates_a_retry_from_a_stolen_token():
    """The one fact that tells the two apart after the fact: how long ago."""
    now = datetime.now(timezone.utc)

    lost_response = seconds_since_rotation({"revokedAt": now - timedelta(seconds=2)})
    assert lost_response is not None
    assert lost_response <= RETRY_WINDOW_SECONDS

    stolen = seconds_since_rotation({"revokedAt": now - timedelta(hours=3)})
    assert stolen is not None
    assert stolen > RETRY_WINDOW_SECONDS

    # Naive timestamps must not blow up the alert path.
    naive = seconds_since_rotation({"revokedAt": now.replace(tzinfo=None)})
    assert naive is not None and naive >= 0.0

    # Clock skew must never produce a negative age.
    assert seconds_since_rotation({"revokedAt": now + timedelta(seconds=5)}) == 0.0

    # Records predating revokedAt report "unknown" rather than guessing.
    assert seconds_since_rotation({}) is None


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
