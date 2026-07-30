"""Assembly of the NovelSync MCP server (FastMCP + embedded OAuth 2.1 AS).

build_mcp_bundle() is the single entry point server.py uses. The returned
bundle carries everything the host FastAPI app needs to mount: the FastMCP
instance (whose streamable_http_app serves /mcp, /authorize, /token,
/register, /revoke and both /.well-known documents), the OAuth provider
(needed by the consent-handoff routes), and the shared Firestore client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import structlog
from google.cloud import firestore
from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl

from config import Settings
from mcp_server.oauth_provider import FirestoreOAuthProvider, public_client_metadata
from mcp_server.oauth_store import OAuthStore
from mcp_server.tools import register_tools
from rate_limit import PerUserRateLimiter

logger = structlog.get_logger(__name__)

MCP_SCOPES = ["stories:read"]


@dataclass
class McpBundle:
    mcp: FastMCP
    provider: FirestoreOAuthProvider
    db: Any
    tool_rate_limiter: PerUserRateLimiter
    # Served in place of the SDK's own /.well-known/oauth-authorization-server,
    # which advertises client_secret auth this server does not accept.
    as_metadata: dict


def _make_firestore_client(project_id: str) -> firestore.Client:
    """Same construction style as StoryContextBuilder.__init__ — ADC credentials;
    the google-cloud-firestore library picks up FIRESTORE_EMULATOR_HOST itself."""
    emulator = os.getenv("FIRESTORE_EMULATOR_HOST", "")
    if emulator:
        logger.info("mcp_firestore_emulator", host=emulator)
    return firestore.Client(project=project_id) if project_id else firestore.Client()


def build_mcp_bundle(settings: Settings) -> McpBundle:
    db = _make_firestore_client(settings.google_cloud_project)
    store = OAuthStore(db)
    provider = FirestoreOAuthProvider(
        store,
        consent_url=settings.resolved_mcp_consent_url,
        access_token_ttl_seconds=settings.mcp_access_token_ttl_seconds,
        refresh_token_ttl_seconds=settings.mcp_refresh_token_ttl_seconds,
    )

    issuer = settings.resolved_mcp_issuer_url
    # Named locals so the served metadata is built from the same options the
    # server actually enforces, rather than a second copy that can drift.
    registration_options = ClientRegistrationOptions(
        enabled=True,
        valid_scopes=MCP_SCOPES,
        default_scopes=MCP_SCOPES,
    )
    revocation_options = RevocationOptions(enabled=True)

    mcp = FastMCP(
        "NovelSync",
        instructions=(
            "Read-only access to the connected user's NovelSync stories: "
            "list stories, read chapters, and inspect characters, places, "
            "and plots. All access is scoped to stories the user owns."
        ),
        auth_server_provider=provider,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(issuer),
            resource_server_url=AnyHttpUrl(f"{issuer}/mcp"),
            required_scopes=MCP_SCOPES,
            client_registration_options=registration_options,
            revocation_options=revocation_options,
        ),
        # Cloud Run: min_instances=0 and no session affinity, so sessions must
        # not live in one worker's memory.
        stateless_http=True,
        json_response=True,
        # v1.29 defaults to a localhost-only Host allowlist; Cloud Run owns the
        # Host header at the infra layer, so the guard would 421 every request.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )

    tool_rate_limiter = PerUserRateLimiter(
        settings.mcp_max_requests_per_minute_per_user
    )
    register_tools(mcp, db=db, rate_limiter=tool_rate_limiter)

    logger.info(
        "mcp_server_built",
        issuer=issuer,
        consent_url=settings.resolved_mcp_consent_url,
    )
    return McpBundle(
        mcp=mcp,
        provider=provider,
        db=db,
        tool_rate_limiter=tool_rate_limiter,
        as_metadata=public_client_metadata(
            issuer_url=AnyHttpUrl(issuer),
            registration_options=registration_options,
            revocation_options=revocation_options,
        ),
    )
