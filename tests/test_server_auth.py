"""Tests for production OIDC / auth configuration via Settings and _verify_internal_token."""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("MAX_REQUESTS_PER_MINUTE_PER_USER", "1000")
os.environ.setdefault("ENABLE_LOCAL_IMAGE_GENERATION", "false")

from config import Settings
from server import _verify_internal_token, create_app

TRUSTED_SA = "trusted@project.iam.gserviceaccount.com"
AGENT_URL = "https://agents.example.run.app"


# ------------------------------------------------------------------
# Settings.oidc_audience  (replaces _normalize_service_url / _production_oidc_audience)
# ------------------------------------------------------------------

def test_oidc_audience_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", f"{AGENT_URL}/")
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    s = Settings()
    assert s.oidc_audience == AGENT_URL


def test_oidc_audience_none_outside_production(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    s = Settings()
    assert s.oidc_audience is None

    monkeypatch.setenv("ENVIRONMENT", "development")
    s = Settings()
    assert s.oidc_audience is None


def test_oidc_audience_requires_agent_service_url_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    monkeypatch.delenv("AGENT_SERVICE_URL", raising=False)

    with pytest.raises((ValueError, ValidationError), match="AGENT_SERVICE_URL"):
        Settings()


def test_oidc_audience_returns_normalized_url(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", f"{AGENT_URL}/")
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    s = Settings()
    assert s.oidc_audience == AGENT_URL


# ------------------------------------------------------------------
# Settings.allowed_callers  (replaces _parse_allowed_service_accounts / _production_allowed_callers)
# ------------------------------------------------------------------

def test_allowed_callers_from_single_sa(monkeypatch):
    monkeypatch.delenv("ALLOWED_SERVICE_ACCOUNTS", raising=False)
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    s = Settings()
    assert s.allowed_callers == frozenset({TRUSTED_SA})


def test_allowed_callers_from_comma_list(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_SERVICE_ACCOUNTS",
        f"{TRUSTED_SA}, other@project.iam.gserviceaccount.com ",
    )
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", "ignored@example.com")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    s = Settings()
    assert s.allowed_callers == frozenset({TRUSTED_SA, "other@project.iam.gserviceaccount.com"})


def test_allowed_callers_requires_config_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    monkeypatch.delenv("ALLOWED_SERVICE_ACCOUNTS", raising=False)
    monkeypatch.delenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", raising=False)

    with pytest.raises((ValueError, ValidationError), match="FIREBASE_FUNCTIONS_SERVICE_ACCOUNT|ALLOWED_SERVICE_ACCOUNTS"):
        Settings()


def test_allowed_callers_empty_outside_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    s = Settings()
    assert s.allowed_callers == frozenset()


# ------------------------------------------------------------------
# create_app() validation (production config guard-rails)
# ------------------------------------------------------------------

def test_create_app_fails_in_production_without_agent_service_url(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    monkeypatch.delenv("AGENT_SERVICE_URL", raising=False)

    with pytest.raises((ValueError, ValidationError), match="AGENT_SERVICE_URL"):
        create_app()


def test_create_app_fails_in_production_without_caller_allowlist(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    monkeypatch.delenv("ALLOWED_SERVICE_ACCOUNTS", raising=False)
    monkeypatch.delenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", raising=False)

    with pytest.raises((ValueError, ValidationError), match="FIREBASE_FUNCTIONS_SERVICE_ACCOUNT|ALLOWED_SERVICE_ACCOUNTS"):
        create_app()


def test_create_app_sets_production_auth_state(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)

    app = create_app()
    assert app.state.oidc_audience == AGENT_URL
    assert app.state.allowed_callers == frozenset({TRUSTED_SA})


# ------------------------------------------------------------------
# _verify_internal_token
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_rejects_caller_not_on_allowlist():
    request = MagicMock()
    request.app.state.oidc_audience = AGENT_URL
    request.app.state.allowed_callers = frozenset({TRUSTED_SA})
    request.headers.get.return_value = "Bearer fake-token"

    with patch(
        "server.google_id_token.verify_oauth2_token",
        return_value={"email": "evil@other.iam.gserviceaccount.com"},
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_internal_token(request)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_accepts_allowlisted_caller():
    request = MagicMock()
    request.app.state.oidc_audience = AGENT_URL
    request.app.state.allowed_callers = frozenset({TRUSTED_SA})
    request.headers.get.return_value = "Bearer fake-token"

    with patch(
        "server.google_id_token.verify_oauth2_token",
        return_value={"email": TRUSTED_SA},
    ):
        await _verify_internal_token(request)
