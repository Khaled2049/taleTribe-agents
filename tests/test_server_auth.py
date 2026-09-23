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


@pytest.fixture(autouse=True)
def _secure_by_default(monkeypatch):
    monkeypatch.setenv("ALLOW_INSECURE_LOCAL_AUTH", "false")
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("HOST", raising=False)


def require_production_env(monkeypatch):
    """Set what a production Settings needs beyond the field under test.

    ENABLE_MCP defaults on, and production then requires MCP_CONSENT_URL and
    STORY_DATA_URL. These tests are about OIDC audience and the caller
    allowlist, so they supply those explicitly rather than inheriting whichever
    values another test module happened to leak into os.environ.
    """
    monkeypatch.setenv("MCP_CONSENT_URL", "https://consent.example/mcp-connect")
    monkeypatch.setenv("STORY_DATA_URL", "http://story-data.internal:8084")
    # Pinned rather than inherited: create_app() calls load_dotenv, so whatever
    # the developer's .env says about writes would otherwise leak in. These
    # tests are not about the write flag either way.
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


# ------------------------------------------------------------------
# Environment validation and the explicit local-auth switch
# ------------------------------------------------------------------


@pytest.mark.parametrize("value", ["prod", "Production", "PRODUCTION", "staging", ""])
def test_unknown_environment_values_are_rejected(monkeypatch, value):
    monkeypatch.setenv("ENVIRONMENT", value)
    with pytest.raises(ValidationError, match="environment"):
        Settings()


def test_missing_environment_does_not_disable_auth(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    s = Settings()
    assert s.environment == "development"
    assert s.insecure_local_auth is False
    assert s.oidc_audience is None


def _unconfigured_request(insecure_local_auth: bool) -> SimpleNamespace:
    state = SimpleNamespace(
        oidc_audience=None,
        allowed_callers=frozenset(),
        insecure_local_auth=insecure_local_auth,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state), headers={})


@pytest.mark.asyncio
async def test_unconfigured_auth_rejects_internal_calls():
    with pytest.raises(HTTPException) as exc_info:
        await _verify_internal_token(_unconfigured_request(False))
    assert exc_info.value.status_code == 401
    assert "not configured" in exc_info.value.detail["message"]


@pytest.mark.asyncio
async def test_explicit_local_auth_switch_skips_verification():
    await _verify_internal_token(_unconfigured_request(True))


@pytest.mark.parametrize("environment", ["development", "test"])
def test_local_auth_switch_allowed_outside_production(monkeypatch, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("ALLOW_INSECURE_LOCAL_AUTH", "true")
    assert Settings().insecure_local_auth is True


def test_local_auth_switch_refused_in_production(monkeypatch):
    require_production_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
    monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    monkeypatch.setenv("ALLOW_INSECURE_LOCAL_AUTH", "true")
    with pytest.raises(ValidationError, match="ALLOW_INSECURE_LOCAL_AUTH"):
        Settings()


def test_local_auth_switch_refused_on_cloud_run(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ALLOW_INSECURE_LOCAL_AUTH", "true")
    monkeypatch.setenv("K_SERVICE", "taletribe-agents")
    with pytest.raises(ValidationError, match="Cloud Run"):
        Settings()


def test_create_app_without_switch_rejects_internal_routes(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ENABLE_MCP", "false")
    app = create_app()
    assert app.state.insecure_local_auth is False
    with TestClient(app) as test_client:
        resp = test_client.post(
            "/agent/execute",
            json={"action": "generateNextLines", "parameters": {}},
            headers={"X-User-ID": "someone-else"},
        )
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "environment,host,expected",
    [
        ("production", None, "0.0.0.0"),
        ("development", None, "127.0.0.1"),
        ("test", None, "127.0.0.1"),
        ("development", "0.0.0.0", "0.0.0.0"),
    ],
)
def test_bind_host_defaults_to_loopback_outside_production(
    monkeypatch, environment, host, expected
):
    if environment == "production":
        require_production_env(monkeypatch)
        monkeypatch.setenv("AGENT_SERVICE_URL", AGENT_URL)
        monkeypatch.setenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", TRUSTED_SA)
    monkeypatch.setenv("ENVIRONMENT", environment)
    if host is not None:
        monkeypatch.setenv("HOST", host)
    assert Settings().bind_host == expected


def test_emulator_tokens_refused_without_local_auth_switch(monkeypatch):
    from mcp_server.oauth_routes import _verify_firebase_uid

    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "localhost:9099")
    with patch(
        "mcp_server.oauth_routes.google_id_token.verify_firebase_token",
        side_effect=ValueError("signature required"),
    ) as verify:
        with pytest.raises(ValueError, match="signature required"):
            _verify_firebase_uid(
                "unsigned.token.",
                project_id="test-project",
                allow_emulator_tokens=False,
                auth_request=object(),
            )
    verify.assert_called_once()
