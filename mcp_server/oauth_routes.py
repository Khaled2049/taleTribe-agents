"""Consent-page handoff endpoints for the MCP OAuth flow.

These two routes are the bridge between the SDK-generated /authorize endpoint
(which redirects the user to the frontend consent page with a txn id) and the
NovelSync web app (which authenticates the user with Firebase and posts the
resulting ID token back here).

They live on the host FastAPI app rather than the mounted MCP Starlette app
because they aren't part of the MCP protocol surface and need FastAPI's request
validation. That placement has nothing to do with CORS: `add_middleware` wraps
the entire ASGI stack, mounts included, so the CORS middleware would cover these
routes either way. What does matter is that the browser calls them
cross-origin — the frontend origin must be in CORS_ORIGINS.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import anyio.to_thread
import structlog
from fastapi import APIRouter, HTTPException, Request
from google.auth import jwt as google_jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field

from mcp_server.access import AccessGate
from mcp_server.oauth_provider import FirestoreOAuthProvider, TxnNotFoundError
from mcp_server.throttle import client_ip
from rate_limit import PerUserRateLimiter

logger = structlog.get_logger(__name__)


class CompleteRequest(BaseModel):
    txn_id: str = Field(min_length=1, max_length=256)
    approve: bool
    id_token: Optional[str] = None


def _verify_firebase_uid(
    raw_token: str,
    *,
    project_id: str,
    environment: str,
    auth_request: google_requests.Request,
) -> str:
    """Firebase ID token -> uid. Emulator tokens (unsigned) accepted only in dev."""
    if environment != "production" and os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
        claims = google_jwt.decode(raw_token, verify=False)
    else:
        claims = google_id_token.verify_firebase_token(
            raw_token, auth_request, audience=project_id
        )
    uid = (claims or {}).get("sub") or (claims or {}).get("user_id")
    if not uid:
        raise ValueError("Firebase token has no subject")
    return uid


def build_oauth_router(
    provider: FirestoreOAuthProvider,
    *,
    project_id: str,
    environment: str,
    auth_request: google_requests.Request,
    ip_rate_limiter: PerUserRateLimiter,
    as_metadata: dict,
    resource_metadata: dict,
    access_gate: AccessGate,
) -> APIRouter:
    # No prefix: this router also serves a /.well-known document, which must sit
    # at the domain root. Paths are spelled out rather than sharing an /oauth
    # prefix so that stays obvious.
    router = APIRouter(tags=["MCP OAuth"])

    @router.get("/.well-known/oauth-authorization-server", include_in_schema=False)
    async def authorization_server_metadata():
        """Overrides the SDK's copy, which is mounted below this router.

        FastAPI matches declared routes before mounts, so this wins. It exists
        because the SDK hard-codes client_secret auth into the document while
        this server registers every client as public — see
        oauth_provider.public_client_metadata. Deliberately not throttled, for
        the same reason the mount exempts /.well-known/*: a client that cannot
        read discovery cannot start the flow at all.
        """
        return as_metadata

    @router.get("/.well-known/oauth-protected-resource/mcp", include_in_schema=False)
    async def protected_resource_metadata():
        """Overrides the SDK's copy, which is mounted below this router.

        The SDK builds this document's `scopes_supported` from
        AuthSettings.required_scopes — the list RequireAuthMiddleware enforces
        conjunctively. So the field that tells a client which scopes it MAY ask
        for is welded to the list of scopes it MUST already have, and
        stories:write cannot go in the second without 403-ing every read-only
        token. Left alone, no SDK client would ever request write access:
        get_client_metadata_scopes() copies scopes_supported straight into its
        registration and /authorize request.

        Same technique and same class of reason as the authorization-server
        document above. Deliberately not throttled, for the same reason.
        """
        return resource_metadata

    @router.get("/oauth/txn/{txn_id}")
    async def get_txn(txn_id: str, request: Request):
        if not await ip_rate_limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many requests")
        try:
            return await provider.get_txn_info(txn_id)
        except TxnNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Connection request not found or expired. "
                "Restart the connection from your MCP client.",
            )

    @router.post("/oauth/complete")
    async def complete(body: CompleteRequest, request: Request):
        if not await ip_rate_limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many requests")

        if not body.approve:
            try:
                redirect_url = await provider.deny_authorization(body.txn_id)
            except TxnNotFoundError:
                raise HTTPException(status_code=404, detail="Unknown transaction")
            return {"redirect_url": redirect_url}

        if not body.id_token:
            raise HTTPException(status_code=400, detail="id_token is required")
        try:
            # verify_firebase_token fetches Google's signing certs over the
            # network, so it must not run on the event loop. Catch broadly:
            # a bad token raises ValueError, but a cert-fetch failure raises
            # google.auth.exceptions.TransportError, which is not a ValueError
            # and would otherwise escape as a 500.
            uid = await anyio.to_thread.run_sync(
                functools.partial(
                    _verify_firebase_uid,
                    body.id_token,
                    project_id=project_id,
                    environment=environment,
                    auth_request=auth_request,
                )
            )
        except Exception as exc:
            logger.warning(
                "mcp_oauth_bad_id_token",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=401, detail="Invalid Firebase ID token")

        # Rollout allowlist. Refusing here means a non-approved user never gets
        # an MCP token at all, rather than getting one that fails on first use.
        # The gate is re-checked per tool call too (see tools._authorized_uid),
        # because this one only runs at grant time.
        if not await anyio.to_thread.run_sync(access_gate.is_allowed, uid):
            logger.warning("mcp_access_denied_at_consent", uid=uid)
            raise HTTPException(
                status_code=403,
                detail="MCP access is currently limited to approved accounts. "
                "You can request access from your NovelSync profile.",
            )

        try:
            redirect_url = await provider.complete_authorization(body.txn_id, uid)
        except TxnNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown transaction")
        return {"redirect_url": redirect_url}

    return router
