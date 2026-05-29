"""Application settings via pydantic-settings.

Reads from environment variables (and optionally a .env file loaded by server.py).
Instantiated once inside create_app() so tests can monkeypatch env vars before creation.
"""
import json
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


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

    # Credit proxy
    credit_proxy_url: str = ""

    # Firestore emulator (auto-set to localhost:8080 in non-production if blank)
    firestore_emulator_host: str = ""

    # Server port
    port: int = 8000

    @field_validator("max_requests_per_minute_per_user", mode="before")
    @classmethod
    def clamp_rpm(cls, v: object) -> int:
        try:
            return max(0, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 20

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors(cls, v: object) -> str:
        """Ensure cors_origins is a valid JSON list; fall back to '[]'."""
        raw = str(v) if v is not None else "[]"
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return "[]"
        except (json.JSONDecodeError, ValueError):
            return "[]"
        return raw

    @model_validator(mode="after")
    def check_production_fields(self) -> "Settings":
        if self.environment == "production":
            if not self.agent_service_url.strip():
                raise ValueError(
                    "AGENT_SERVICE_URL must be set when ENVIRONMENT=production "
                    "(OIDC token audience for Firebase Functions → agents calls)"
                )
            if not self.firebase_functions_service_account.strip() and not self.allowed_service_accounts.strip():
                raise ValueError(
                    "FIREBASE_FUNCTIONS_SERVICE_ACCOUNT or ALLOWED_SERVICE_ACCOUNTS must be set "
                    "when ENVIRONMENT=production (trusted OIDC caller allowlist)"
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
            return frozenset(part.strip() for part in raw_list.split(",") if part.strip())
        single = self.firebase_functions_service_account.strip()
        if single:
            return frozenset({single})
        return frozenset()

    @property
    def parsed_cors_origins(self) -> list:
        """cors_origins as a Python list."""
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, ValueError):
            return []
