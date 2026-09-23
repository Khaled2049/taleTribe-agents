"""OAuth 2.1 authorization-server provider backed by Firestore.

Implements the `mcp` SDK's OAuthAuthorizationServerProvider protocol. The SDK
owns the spec-sensitive endpoint behavior (/authorize, /token, /register,
/revoke, PKCE verification, redirect_uri validation against the registration);
this class only persists state and delegates login to TheTaleTribe's web app
via the consent-page handoff (see oauth_routes.py).

All Firestore work happens in oauth_store.py (sync) and is bridged with
anyio.to_thread so the event loop is never blocked.
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import anyio.to_thread
import structlog
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import build_metadata
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthToken,
    ProtectedResourceMetadata,
)
from pydantic import AnyHttpUrl, AnyUrl

from mcp_server.oauth_store import (
    TXN_STATUS_COMPLETED,
    TXN_STATUS_DENIED,
    OAuthStore,
    as_utc,
    hash_token,
    new_family_id,
    new_secret,
)

logger = structlog.get_logger(__name__)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# A replay this soon after rotation is almost certainly the same client retrying
# a refresh whose response was lost in flight, not a stolen token — see
# _handle_refresh_reuse. Purely a triage hint for whoever reads the alert: the
# family is revoked either way, because the two are indistinguishable to us.
RETRY_WINDOW_SECONDS = 30.0


def seconds_since_rotation(record: dict[str, Any]) -> Optional[float]:
    """Age of a revocation, or None for records written before revokedAt existed."""
    revoked_at = as_utc(record.get("revokedAt"))
    if revoked_at is None:
        return None
    delta = (datetime.now(timezone.utc) - revoked_at).total_seconds()
    return max(0.0, delta)


class TxnNotFoundError(Exception):
    """The consent transaction is missing, expired, or already used."""


async def _run_sync(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


PUBLIC_CLIENT_AUTH_METHOD = "none"


def _downgrade_to_public_client(client_info: OAuthClientInformationFull) -> None:
    """Refuse to issue or store a client secret. Every MCP client here is public.

    Left alone, the SDK mints a 32-byte secret for any registration that doesn't
    explicitly ask for `token_endpoint_auth_method="none"` (register.py defaults
    a missing value to `client_secret_post`). We would then have to persist that
    secret in Firestore *in the clear*, because ClientAuthenticator compares the
    stored value directly with hmac.compare_digest — hashing it at rest would
    mean reimplementing the SDK's client authentication.

    It buys nothing here. PKCE (S256) is mandatory, authorization codes are
    single-use and bound to the challenge, and redirect_uris are pinned at
    registration, so the secret is never the thing standing between an attacker
    and a token. Meanwhile the clients that register against this server —
    Claude Desktop, Claude Code, MCP Inspector — are installed applications that
    cannot keep a secret anyway. RFC 6749 §2.1 calls them public clients, and
    that is what we record them as.

    Mutating `client_info` rather than only stripping the stored copy is
    deliberate: the SDK serializes this same object as the 201 response *after*
    calling us, so the secret is never written to Firestore, never sent over the
    wire, and never lands in a client's config file. RFC 7591 §3.2.1 explicitly
    lets the server replace requested metadata. A client that ignores the
    response and posts a secret anyway still authenticates, because
    ClientAuthenticator skips the comparison entirely when the stored client
    has no secret.
    """
    client_info.client_secret = None
    client_info.client_secret_expires_at = None
    client_info.token_endpoint_auth_method = PUBLIC_CLIENT_AUTH_METHOD


def public_client_metadata(
    *,
    issuer_url: AnyHttpUrl,
    registration_options: ClientRegistrationOptions,
    revocation_options: RevocationOptions,
) -> dict[str, Any]:
    """The SDK's AS metadata, corrected to advertise public-client auth.

    build_metadata() hard-codes `token_endpoint_auth_methods_supported` to the
    two client_secret variants. Since every registration is downgraded to a
    public client, advertising only those would describe a server that does not
    exist and could turn away a client that checks its registered method against
    the list (RFC 8414 §2). Everything else is the SDK's own output, so this
    inherits future changes to the document instead of forking it.
    """
    metadata = build_metadata(
        issuer_url=issuer_url,
        service_documentation_url=None,
        client_registration_options=registration_options,
        revocation_options=revocation_options,
    )
    document = metadata.model_dump(mode="json", exclude_none=True)
    for field in (
        "token_endpoint_auth_methods_supported",
        "revocation_endpoint_auth_methods_supported",
    ):
        if field in document:
            document[field] = [PUBLIC_CLIENT_AUTH_METHOD]
    return document


def public_resource_metadata(
    *,
    issuer_url: AnyHttpUrl,
    scopes_supported: list[str],
) -> dict[str, Any]:
    """The RFC 9728 protected-resource document, with scopes we choose.

    The SDK builds this itself from AuthSettings.required_scopes
    (fastmcp/server.py create_protected_resource_routes). That welds the one
    field telling a client which scopes it MAY request to the list
    RequireAuthMiddleware demands it ALREADY has — and that middleware is
    conjunctive over the whole /mcp mount, so a scope in that list is required
    of every caller. stories:write can never go there without making write
    access mandatory and read-only connections impossible.

    Left alone, the consequence is that no SDK client ever asks for write:
    get_client_metadata_scopes() in the client copies scopes_supported verbatim
    into its registration and its /authorize request, and validate_scope() then
    rejects anything outside the client's own registered ceiling. The write
    tools would be unreachable.

    So this document is served in place of the SDK's, from a FastAPI route that
    wins over the mount — the same technique, and the same class of reason, as
    public_client_metadata above.
    """
    metadata = ProtectedResourceMetadata(
        resource=AnyHttpUrl(f"{str(issuer_url).rstrip('/')}/mcp"),
        authorization_servers=[issuer_url],
        scopes_supported=list(scopes_supported),
    )
    return metadata.model_dump(mode="json", exclude_none=True)


def _validate_redirect_uris(uris: list[Any]) -> None:
    """Allow https anywhere and http only on loopback (Claude Code/Desktop, Inspector)."""
    for uri in uris:
        parsed = urlparse(str(uri))
        if parsed.scheme == "https":
            continue
        if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
            continue
        raise RegistrationError(
            error="invalid_redirect_uri",
            error_description=(
                f"redirect_uri {uri} is not allowed: must be https, "
                "or http on localhost/127.0.0.1"
            ),
        )


class FirestoreOAuthProvider:
    """Authorization server state machine over the mcpOauth* collections."""

    def __init__(
        self,
        store: OAuthStore,
        *,
        consent_url: str,
        access_token_ttl_seconds: int,
        refresh_token_ttl_seconds: int,
        txn_ttl_seconds: int = 600,
        code_ttl_seconds: int = 300,
        client_ttl_seconds: int = 90 * 24 * 3600,
        unused_client_ttl_seconds: int = 7 * 24 * 3600,
    ) -> None:
        self._store = store
        self._consent_url = consent_url.rstrip("/")
        self._access_ttl = access_token_ttl_seconds
        self._refresh_ttl = refresh_token_ttl_seconds
        self._txn_ttl = txn_ttl_seconds
        self._code_ttl = code_ttl_seconds
        # A client record must outlive any refresh token issued to it, or a
        # client holding a live refresh token would fail to renew because its
        # registration had been garbage-collected underneath it.
        self._client_ttl = max(client_ttl_seconds, refresh_token_ttl_seconds * 2)
        self._unused_client_ttl = min(unused_client_ttl_seconds, self._client_ttl)

    # ------------------------------------------------------------------
    # Client registration (RFC 7591)
    # ------------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        # Every use of a client lands here, so this is also where the sliding
        # expiry is renewed: a client in active use never gets collected.
        record = await _run_sync(
            self._store.get_client, client_id, renew_ttl_seconds=self._client_ttl
        )
        if record is None:
            return None
        return OAuthClientInformationFull.model_validate(record)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        _validate_redirect_uris(client_info.redirect_uris or [])
        _downgrade_to_public_client(client_info)
        record = client_info.model_dump(mode="json", exclude_none=True)
        await _run_sync(
            self._store.save_client,
            client_info.client_id,
            record,
            self._unused_client_ttl,
        )
        logger.info(
            "mcp_oauth_client_registered",
            client_id=client_info.client_id,
            client_name=client_info.client_name,
            redirect_uris=[str(u) for u in client_info.redirect_uris or []],
        )

    # ------------------------------------------------------------------
    # Authorization (consent-page handoff)
    # ------------------------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        txn = {
            "clientId": client.client_id,
            "clientName": client.client_name,
            "redirectUri": str(params.redirect_uri),
            "redirectUriProvidedExplicitly": params.redirect_uri_provided_explicitly,
            "codeChallenge": params.code_challenge,
            "state": params.state,
            "scopes": params.scopes or [],
            "resource": str(params.resource) if params.resource else None,
        }
        txn_id = await _run_sync(self._store.create_txn, txn, self._txn_ttl)
        logger.info(
            "mcp_oauth_authorize_started",
            client_id=client.client_id,
            txn_id=txn_id,
        )
        return f"{self._consent_url}?txn={txn_id}"

    async def get_txn_info(self, txn_id: str) -> dict[str, Any]:
        """Consent-page view of a pending transaction (no secrets exposed)."""
        txn = await _run_sync(self._store.get_pending_txn, txn_id)
        if txn is None:
            raise TxnNotFoundError(txn_id)
        return {
            "client_name": txn.get("clientName") or "Unknown application",
            "redirect_host": urlparse(txn["redirectUri"]).netloc,
            "scopes": txn.get("scopes") or [],
        }

    async def complete_authorization(self, txn_id: str, uid: str) -> str:
        """User approved on the consent page: mint a code, return the client redirect."""
        txn = await _run_sync(self._store.finish_txn, txn_id, TXN_STATUS_COMPLETED)
        if txn is None:
            raise TxnNotFoundError(txn_id)
        code = new_secret("mcp_ac")
        await _run_sync(
            self._store.save_code,
            hash_token(code),
            {
                "clientId": txn["clientId"],
                "redirectUri": txn["redirectUri"],
                "redirectUriProvidedExplicitly": txn["redirectUriProvidedExplicitly"],
                "codeChallenge": txn["codeChallenge"],
                "scopes": txn["scopes"],
                "resource": txn.get("resource"),
                "subject": uid,
            },
            self._code_ttl,
        )
        logger.info(
            "mcp_oauth_authorization_granted",
            client_id=txn["clientId"],
            txn_id=txn_id,
        )
        return construct_redirect_uri(txn["redirectUri"], code=code, state=txn["state"])

    async def deny_authorization(self, txn_id: str) -> str:
        """User denied on the consent page: return the error redirect."""
        txn = await _run_sync(self._store.finish_txn, txn_id, TXN_STATUS_DENIED)
        if txn is None:
            raise TxnNotFoundError(txn_id)
        logger.info(
            "mcp_oauth_authorization_denied",
            client_id=txn["clientId"],
            txn_id=txn_id,
        )
        return construct_redirect_uri(
            txn["redirectUri"], error="access_denied", state=txn["state"]
        )

    # ------------------------------------------------------------------
    # Authorization codes
    # ------------------------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        record = await _run_sync(self._store.load_code, hash_token(authorization_code))
        if record is None or record.get("clientId") != client.client_id:
            return None
        return AuthorizationCode(
            code=authorization_code,
            scopes=record.get("scopes") or [],
            expires_at=self._epoch(record),
            client_id=record["clientId"],
            code_challenge=record["codeChallenge"],
            redirect_uri=AnyUrl(record["redirectUri"]),
            redirect_uri_provided_explicitly=record["redirectUriProvidedExplicitly"],
            # RFC 8707 resource indicator: the SDK types this as a plain str,
            # not AnyUrl (unlike redirect_uri above).
            resource=record.get("resource"),
            subject=record.get("subject"),
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        record = await _run_sync(
            self._store.consume_code, hash_token(authorization_code.code)
        )
        if record is None or record.get("clientId") != client.client_id:
            raise TokenError(
                error="invalid_grant",
                error_description="authorization code is invalid, expired, or already used",
            )
        # A fresh authorization starts a new lineage; every rotation below
        # inherits this id, so reuse detection can reach the whole grant.
        return await self._mint_token_pair(
            uid=record["subject"],
            client_id=client.client_id,
            scopes=record.get("scopes") or [],
            family_id=new_family_id(),
        )

    # ------------------------------------------------------------------
    # Refresh tokens (rotating)
    # ------------------------------------------------------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        """Load a refresh token, treating a replayed one as a compromise.

        Reuse detection has to happen *here*, not in exchange_refresh_token: the
        SDK's token handler rejects the request as soon as this returns None and
        never reaches the exchange, so a replay would otherwise be
        indistinguishable from a typo and leave no trace.
        """
        record = await _run_sync(
            self._store.get_token, hash_token(refresh_token), include_revoked=True
        )
        if (
            record is None
            or record.get("type") != "refresh"
            or record.get("clientId") != client.client_id
        ):
            return None
        if record.get("revoked"):
            await self._handle_refresh_reuse(record, client)
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=record["clientId"],
            scopes=record.get("scopes") or [],
            expires_at=self._epoch(record),
            subject=record.get("uid"),
        )

    async def _handle_refresh_reuse(
        self, record: dict[str, Any], client: OAuthClientInformationFull
    ) -> None:
        """A refresh token we already rotated away came back. Kill the lineage.

        Rotation on its own only shortens a thief's window: whoever refreshes
        second gets `invalid_grant`, silently re-runs the OAuth flow, and the
        theft leaves no mark. Revoking the whole family on reuse is what turns
        rotation into detection — the thief's freshly minted pair dies too, so
        the race has no winner (OAuth 2.1 §4.14.2, RFC 9700).

        The accepted cost is a false positive: a client that retries a refresh
        because the response was lost in flight replays a token that really was
        consumed, and gets logged out. The spec takes that trade deliberately,
        and here it costs the user one re-consent. That is cheap on a read-only
        grant and merely inconvenient on a write one — no content is lost,
        since the reuse revokes tokens, not stories.

        Because those two cases are indistinguishable at the moment of the
        replay, the log carries the one fact that separates them after the
        fact — how long ago the token was rotated away. Seconds means a retry;
        hours means someone kept a copy.
        """
        family_id = record.get("familyId")
        revoked = await _run_sync(
            self._store.revoke_token_family, family_id or "", self._refresh_ttl
        )
        age = seconds_since_rotation(record)
        logger.warning(
            "mcp_oauth_refresh_token_reuse",
            client_id=client.client_id,
            uid=record.get("uid"),
            family_id=family_id,
            tokens_revoked=revoked,
            # Tokens predating family tracking can't have their lineage walked;
            # the replayed token itself is already revoked either way.
            family_tracked=bool(family_id),
            seconds_since_rotation=round(age, 3) if age is not None else None,
            # Triage hint only — never a reason to skip revoking.
            likely_client_retry=age is not None and age <= RETRY_WINDOW_SECONDS,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        record = await _run_sync(
            self._store.consume_refresh_token, hash_token(refresh_token.token)
        )
        if record is None or record.get("clientId") != client.client_id:
            raise TokenError(
                error="invalid_grant",
                error_description="refresh token is invalid, expired, or already used",
            )
        granted = record.get("scopes") or []
        # Defense in depth: never widen scope on refresh.
        requested = [s for s in (scopes or granted) if s in granted] or granted
        return await self._mint_token_pair(
            uid=record["uid"],
            client_id=client.client_id,
            scopes=requested,
            # Stay in the same lineage. A token predating family tracking has no
            # id to inherit, so it starts one rather than dropping out of it.
            family_id=record.get("familyId") or new_family_id(),
        )

    # ------------------------------------------------------------------
    # Access tokens
    # ------------------------------------------------------------------

    async def load_access_token(self, token: str) -> AccessToken | None:
        record = await _run_sync(self._store.get_token, hash_token(token))
        if record is None or record.get("type") != "access":
            return None
        return AccessToken(
            token=token,
            client_id=record["clientId"],
            scopes=record.get("scopes") or [],
            expires_at=self._epoch(record),
            subject=record.get("uid"),
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        await _run_sync(self._store.revoke_token, hash_token(token.token))
        logger.info("mcp_oauth_token_revoked", client_id=token.client_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _mint_token_pair(
        self, *, uid: str, client_id: str, scopes: list[str], family_id: str
    ) -> OAuthToken:
        access = new_secret("mcp_at")
        refresh = new_secret("mcp_rt")
        saved = await _run_sync(
            self._store.save_token_pair,
            access_hash=hash_token(access),
            refresh_hash=hash_token(refresh),
            uid=uid,
            client_id=client_id,
            scopes=scopes,
            family_id=family_id,
            access_ttl_seconds=self._access_ttl,
            refresh_ttl_seconds=self._refresh_ttl,
        )
        if not saved:
            logger.warning(
                "mcp_oauth_mint_refused_revoked_family",
                client_id=client_id,
                uid=uid,
                family_id=family_id,
            )
            raise TokenError(
                error="invalid_grant",
                error_description="this authorization has been revoked",
            )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self._access_ttl,
            scope=" ".join(scopes) if scopes else None,
            refresh_token=refresh,
        )

    @staticmethod
    def _epoch(record: dict[str, Any]) -> Optional[int]:
        expires_at = as_utc(record.get("expiresAt"))
        return int(expires_at.timestamp()) if expires_at is not None else None
