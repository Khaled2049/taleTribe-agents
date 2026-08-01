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
from mcp_server.access import AccessGate
from mcp_server.oauth_provider import (
    FirestoreOAuthProvider,
    public_client_metadata,
    public_resource_metadata,
)
from mcp_server.oauth_store import OAuthStore
from mcp_server.tools import register_tools
from rate_limit import PerUserRateLimiter

logger = structlog.get_logger(__name__)

MCP_READ_SCOPE = "stories:read"
MCP_WRITE_SCOPE = "stories:write"
MCP_REQUIRED_SCOPES = [MCP_READ_SCOPE]
MCP_VALID_SCOPES = [MCP_READ_SCOPE, MCP_WRITE_SCOPE]
MCP_DEFAULT_SCOPES = [MCP_READ_SCOPE]


@dataclass
class McpBundle:
    mcp: FastMCP
    provider: FirestoreOAuthProvider
    db: Any
    tool_rate_limiter: PerUserRateLimiter
    write_rate_limiter: PerUserRateLimiter
    access_gate: AccessGate
    as_metadata: dict
    resource_metadata: dict


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
    writes_enabled = settings.resolved_mcp_writes_enabled
    # With writes off the scope is not merely unused: /register rejects it and
    # discovery stops advertising it, so the flag is a real kill switch rather
    # than a check somewhere downstream.
    valid_scopes = MCP_VALID_SCOPES if writes_enabled else MCP_REQUIRED_SCOPES

    # Named locals so the served metadata is built from the same options the
    # server actually enforces, rather than a second copy that can drift.
    registration_options = ClientRegistrationOptions(
        enabled=True,
        valid_scopes=valid_scopes,
        default_scopes=MCP_DEFAULT_SCOPES,
    )
    revocation_options = RevocationOptions(enabled=True)

    mcp = FastMCP(
        "NovelSync",
        instructions=(
            "Access to the connected user's NovelSync stories: list stories, "
            "read chapters, and inspect characters, places, and plots. With "
            "write access granted, also create new stories, append chapters "
            "to them, and edit existing chapters paragraph by paragraph. All "
            "access is scoped to stories the user owns."
        ),
        auth_server_provider=provider,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(issuer),
            resource_server_url=AnyHttpUrl(f"{issuer}/mcp"),
            required_scopes=MCP_REQUIRED_SCOPES,
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
    write_rate_limiter = PerUserRateLimiter(settings.mcp_max_writes_per_minute_per_user)
    access_gate = AccessGate(
        db,
        enabled=settings.enable_mcp_access_allowlist,
        cache_ttl_seconds=settings.mcp_access_cache_ttl_seconds,
    )
    register_tools(
        mcp,
        db=db,
        rate_limiter=tool_rate_limiter,
        write_rate_limiter=write_rate_limiter,
        enable_writes=writes_enabled,
        access_gate=access_gate,
    )

    logger.info(
        "mcp_server_built",
        issuer=issuer,
        consent_url=settings.resolved_mcp_consent_url,
        writes_enabled=writes_enabled,
        access_allowlist=access_gate.enabled,
    )
    return McpBundle(
        mcp=mcp,
        provider=provider,
        db=db,
        tool_rate_limiter=tool_rate_limiter,
        write_rate_limiter=write_rate_limiter,
        access_gate=access_gate,
        as_metadata=public_client_metadata(
            issuer_url=AnyHttpUrl(issuer),
            registration_options=registration_options,
            revocation_options=revocation_options,
        ),
        resource_metadata=public_resource_metadata(
            issuer_url=AnyHttpUrl(issuer),
            scopes_supported=valid_scopes,
        ),
    )
