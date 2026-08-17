"""Tests for production OIDC / auth configuration via Settings and _verify_internal_token."""

import os
from types import SimpleNamespace
from unittest.mock import patch

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


def require_production_env(monkeypatch):
    """Set what a production Settings needs beyond the field under test.

    ENABLE_MCP defaults on, and production then requires MCP_CONSENT_URL and
    STORY_DATA_URL. These tests are about OIDC audience and the caller
    allowlist, so they supply those explicitly rather than inheriting whichever
    values another test module happened to leak into os.environ.
    """
    monkeypatch.setenv("MCP_CONSENT_URL", "https://consent.example/mcp-connect")
    monkeypatch.setenv("STORY_DATA_URL", "http://story-data.internal:8084")
    # Set to "false" rather than deleted: create_app() calls load_dotenv, which
    # fills in anything absent from os.environ, and the developer .env in this
    # repo turns writes on. With STORY_DATA_URL set that is a rejected
    # combination, and it is not what these tests exercise.
    monkeypatch.setenv("ENABLE_MCP_WRITES", "false")


# ------------------------------------------------------------------
# Settings.oidc_audience  (replaces _normalize_service_url / _production_oidc_audience)
# ------------------------------------------------------------------


def test_oidc_audience_strips_trailing_slash(monkeypatch):
    require_production_env(monkeypatch)
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
    require_production_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", f"{AGENT_URL}/")
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    s = Settings()
    assert s.oidc_audience == AGENT_URL


# ------------------------------------------------------------------
# Settings.allowed_callers  (replaces _parse_allowed_service_accounts / _production_allowed_callers)
# ------------------------------------------------------------------


def test_allowed_callers_from_single_sa(monkeypatch):
    require_production_env(monkeypatch)
    monkeypatch.delenv("ALLOWED_SERVICE_ACCOUNTS", raising=False)
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    s = Settings()
    assert s.allowed_callers == frozenset({TRUSTED_SA})


def test_allowed_callers_from_comma_list(monkeypatch):
    require_production_env(monkeypatch)
    monkeypatch.setenv(
        "ALLOWED_SERVICE_ACCOUNTS",
        f"{TRUSTED_SA}, other@project.iam.gserviceaccount.com ",
    )
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", "ignored@example.com")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    s = Settings()
    assert s.allowed_callers == frozenset(
        {TRUSTED_SA, "other@project.iam.gserviceaccount.com"}
    )


def test_allowed_callers_requires_config_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    monkeypatch.delenv("ALLOWED_SERVICE_ACCOUNTS", raising=False)
    monkeypatch.delenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", raising=False)

    with pytest.raises(
        (ValueError, ValidationError),
        match="FIREBASE_FUNCTIONS_SERVICE_ACCOUNT|ALLOWED_SERVICE_ACCOUNTS",
    ):
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

    with pytest.raises(
        (ValueError, ValidationError),
        match="FIREBASE_FUNCTIONS_SERVICE_ACCOUNT|ALLOWED_SERVICE_ACCOUNTS",
    ):
        create_app()


def test_create_app_sets_production_auth_state(monkeypatch):
    require_production_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)

    app = create_app()
    assert app.state.oidc_audience == AGENT_URL
    assert app.state.allowed_callers == frozenset({TRUSTED_SA})


# ------------------------------------------------------------------
# _verify_internal_token
# ------------------------------------------------------------------


def _fake_request(allowed_callers: frozenset) -> SimpleNamespace:
    """Build a request stub with the exact app.state attrs _verify_internal_token reads."""
    state = SimpleNamespace(
        oidc_audience=AGENT_URL,
        allowed_callers=allowed_callers,
        google_auth_request=object(),  # opaque — verify_oauth2_token is patched
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state),
        headers={"Authorization": "Bearer fake-token"},
    )


@pytest.mark.asyncio
async def test_verify_rejects_caller_not_on_allowlist():
    request = _fake_request(frozenset({TRUSTED_SA}))

    with patch(
        "server.google_id_token.verify_oauth2_token",
        return_value={"email": "evil@other.iam.gserviceaccount.com"},
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_internal_token(request)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_accepts_allowlisted_caller():
    request = _fake_request(frozenset({TRUSTED_SA}))

    with patch(
        "server.google_id_token.verify_oauth2_token",
        return_value={"email": TRUSTED_SA},
    ):
        await _verify_internal_token(request)
