"""Application settings via pydantic-settings.

Reads from environment variables (and optionally a .env file loaded by server.py).
Instantiated once inside create_app() so tests can monkeypatch env vars before creation.
"""

import json
import logging
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Required in all environments
    google_cloud_project: str

    # Runtime environment
    environment: str = "development"

    # OIDC / service-to-service auth (required in production)
    agent_service_url: str = ""
    firebase_functions_service_account: str = ""
    allowed_service_accounts: str = ""  # comma-separated list

    # CORS (JSON array string, e.g. '["https://example.com"]')
    cors_origins: str = "[]"

    # Rate limiting
    max_requests_per_minute_per_user: int = 20

    # GCP / Vertex
    vertex_ai_location: str = "us-central1"

    # Feature flags
    enable_local_image_generation: bool = True
    assistant_api_enabled: bool = False
    assistant_edit_proposals_enabled: bool = False
    assistant_research_enabled: bool = False
    assistant_legacy_fallback_enabled: bool = True
    assistant_stream_spike_enabled: bool = False

    # Ceiling on one tool result before it re-enters the prompt. read_chapter's
    # schema allows a 20 000-char window, so without this a run that stays
    # inside its step ceiling can still grow an expensive message list. The
    # executors clamp the windows their arguments imply against it, rather than
    # generating the result first and trimming it afterwards.
    assistant_max_tool_result_chars: int = 8000

    @model_validator(mode="after")
    def check_assistant_spike(self) -> "Settings":
        if self.assistant_stream_spike_enabled and (
            self.environment not in {"development", "test"}
            or not self.assistant_api_enabled
        ):
            raise ValueError(
                "ASSISTANT_STREAM_SPIKE_ENABLED requires development/test "
                "and ASSISTANT_API_ENABLED=true"
            )
        return self

    # MCP server (OAuth 2.1 authorization server + owner-scoped story tools)
    enable_mcp: bool = True
    mcp_issuer_url: str = ""  # defaults to agent_service_url / localhost (see property)
    mcp_consent_url: str = ""  # frontend consent page (required in production)
    mcp_max_requests_per_minute_per_user: int = 60
    # Story/chapter creation over MCP. Off by default: it is the only path in
    # this service that mutates user content, and the Admin SDK bypasses every
    # limit in firestore.rules, so mcp_server/writes.py re-implements them.
    enable_mcp_writes: bool = False
    # Second, much tighter bucket applied only to the write tools, on top of
    # mcp_max_requests_per_minute_per_user. Also what keeps the soft story cap
    # honest: the frontend's storyCountTrigger is eventually consistent, so a
    # burst faster than this could overshoot MAX_STORIES_PER_USER.
    mcp_max_writes_per_minute_per_user: int = 6
    # Owner-controlled rollout allowlist: only users with
    # mcpAccess/{uid}.status == "granted" may connect or call tools. ON by
    # default while MCP is being trialled — turning it OFF is how the feature
    # goes generally available, and deletes nothing.
    enable_mcp_access_allowlist: bool = True
    # How long an allowlist decision is cached per instance. This is also the
    # worst-case delay before a revocation disconnects someone, so it trades
    # Firestore reads against how sharp the "off" switch feels.
    mcp_access_cache_ttl_seconds: int = 60
    # Per-IP caps on the unauthenticated OAuth endpoints. /register writes a
    # Firestore document per call with no credential required, so it gets a
    # tighter bucket than the rest of the flow.
    mcp_register_requests_per_minute_per_ip: int = 5
    mcp_oauth_requests_per_minute_per_ip: int = 30
    mcp_access_token_ttl_seconds: int = 3600
    mcp_refresh_token_ttl_seconds: int = 2592000  # 30 days

    # Credit proxy
    credit_proxy_url: str = ""

    # Firestore emulator (auto-set to localhost:8080 in non-production if blank)
    firestore_emulator_host: str = ""

    # PostgreSQL story workspace + pgvector. When empty, legacy Firestore AI
    # behavior remains available for Firestore stories during the rollout.
    story_data_database_url: str = ""
    indexing_worker_enabled: bool = False
    indexing_worker_interval_seconds: float = 2.0

    # story-data's HTTP API, which the MCP read tools go through. Separate from
    # story_data_database_url above: the indexing worker owns rows directly,
    # while MCP is a caller like any other and must go through the service so
    # ownership and validation stay in one place.
    story_data_url: str = ""
    # Shared secret that lets MCP assert X-User-ID. story-data ignores the
    # header without it.
    story_data_service_token: str = ""

    # Server port
    port: int = 8000

    @field_validator("max_requests_per_minute_per_user", mode="before")
    @classmethod
    def clamp_rpm(cls, v: object) -> int:
        try:
            return max(0, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 20

    @field_validator("mcp_max_writes_per_minute_per_user", mode="before")
    @classmethod
    def clamp_write_rpm(cls, v: object) -> int:
        try:
            return max(0, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 6

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors(cls, v: object) -> str:
        """Ensure cors_origins is a valid JSON list; fall back to '[]' (logged)."""
        raw = str(v) if v is not None else "[]"
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                logger.warning(
                    "cors_origins_invalid_shape: CORS_ORIGINS=%r is not a JSON list; "
                    "browser requests will be blocked",
                    raw,
                )
                return "[]"
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "cors_origins_parse_failed: CORS_ORIGINS=%r is not valid JSON (%s); "
                "browser requests will be blocked",
                raw,
                exc,
            )
            return "[]"
        return raw

    @model_validator(mode="after")
    def warn_empty_cors_in_production(self) -> "Settings":
        if self.environment == "production" and not self.parsed_cors_origins:
            logger.warning(
                "cors_origins_empty_in_production: no CORS origins configured; "
                "browser-side callers will be blocked"
            )
        return self

    @model_validator(mode="after")
    def check_production_fields(self) -> "Settings":
        if self.environment == "production":
            if not self.agent_service_url.strip():
                raise ValueError(
                    "AGENT_SERVICE_URL must be set when ENVIRONMENT=production "
                    "(OIDC token audience for Firebase Functions → agents calls)"
                )
            if (
                not self.firebase_functions_service_account.strip()
                and not self.allowed_service_accounts.strip()
            ):
                raise ValueError(
                    "FIREBASE_FUNCTIONS_SERVICE_ACCOUNT or ALLOWED_SERVICE_ACCOUNTS must be set "
                    "when ENVIRONMENT=production (trusted OIDC caller allowlist)"
                )
            if self.enable_mcp and not self.mcp_consent_url.strip():
                raise ValueError(
                    "MCP_CONSENT_URL must be set when ENVIRONMENT=production and "
                    "ENABLE_MCP=true (the frontend consent page the OAuth "
                    "authorization flow redirects to)"
                )
            if self.enable_mcp and not self.story_data_url.strip():
                raise ValueError(
                    "STORY_DATA_URL must be set when ENVIRONMENT=production and "
                    "ENABLE_MCP=true (the MCP read tools serve story content "
                    "from story-data; without it every read fails)"
                )
        return self

    @model_validator(mode="after")
    def check_mcp_write_backend(self) -> "Settings":
        """Refuse the split-brain configuration.

        The MCP read tools go through story-data; the write tools still write
        Firestore. Enabling writes while reads come from PostgreSQL would let a
        client create a chapter and then be told it does not exist, so the
        combination is rejected outright rather than left as a footgun. Lifted
        when the write tools are ported.
        """
        if self.enable_mcp_writes and self.story_data_url.strip():
            raise ValueError(
                "ENABLE_MCP_WRITES cannot be enabled yet: the MCP read tools "
                "read story-data (STORY_DATA_URL) while the write tools still "
                "write Firestore, so writes would be invisible to reads. Port "
                "the write tools first, or unset ENABLE_MCP_WRITES."
            )
        return self

    # ------------------------------------------------------------------
    # Derived helpers (computed from raw fields)
    # ------------------------------------------------------------------

    @property
    def oidc_audience(self) -> Optional[str]:
        """OIDC audience for service-to-service auth; None outside production."""
        if self.environment != "production":
            return None
        return self.agent_service_url.strip().rstrip("/") or None

    @property
    def allowed_callers(self) -> frozenset:
        """Frozenset of trusted caller service account emails."""
        if self.environment != "production":
            return frozenset()
        raw_list = self.allowed_service_accounts.strip()
        if raw_list:
            return frozenset(
                part.strip() for part in raw_list.split(",") if part.strip()
            )
        single = self.firebase_functions_service_account.strip()
        if single:
            return frozenset({single})
        return frozenset()

    @property
    def resolved_mcp_issuer_url(self) -> str:
        """OAuth issuer / MCP resource base URL (no trailing slash)."""
        raw = self.mcp_issuer_url.strip() or self.agent_service_url.strip()
        return (raw or f"http://localhost:{self.port}").rstrip("/")

    @property
    def resolved_mcp_writes_enabled(self) -> bool:
        """Writes can never be on without the MCP server that hosts them."""
        return self.enable_mcp and self.enable_mcp_writes

    @property
    def resolved_mcp_consent_url(self) -> str:
        """Consent page URL; defaults to the local Vite dev server outside production."""
        raw = self.mcp_consent_url.strip()
        return (raw or "http://localhost:5173/mcp-connect").rstrip("/")

    @property
    def parsed_cors_origins(self) -> list:
        """cors_origins as a Python list."""
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, ValueError):
            return []
