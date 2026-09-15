"""The app must build and serve normally with ENABLE_MCP=false."""

import os

os.environ["CREDIT_PROXY_URL"] = "http://localhost:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["USE_MOCK"] = "true"
os.environ["ENABLE_MCP"] = "false"
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:9999")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from config import Settings  # noqa: E402
from server import create_app  # noqa: E402

app = create_app()


def test_health_reports_mcp_unavailable():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["services"]["mcp"] == "unavailable"
    assert app.state.mcp_available is False


def test_mcp_routes_absent_when_disabled():
    with TestClient(app) as client:
        assert client.post("/mcp", json={}).status_code == 404
        assert client.get("/.well-known/oauth-authorization-server").status_code == 404
        assert client.get("/oauth/txn/x").status_code == 404


def test_protected_resource_metadata_absent_when_disabled():
    with TestClient(app) as client:
        resp = client.get("/.well-known/oauth-protected-resource/mcp")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Write flag defaults (Settings only — no app build needed)
# ---------------------------------------------------------------------------


def test_writes_off_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_MCP_WRITES", raising=False)
    monkeypatch.setenv("ENABLE_MCP", "true")
    assert Settings().enable_mcp_writes is False


def test_writes_require_mcp_enabled(monkeypatch):
    """A write flag left on must not resurrect writes when MCP itself is off."""
    monkeypatch.setenv("ENABLE_MCP", "false")
    monkeypatch.setenv("ENABLE_MCP_WRITES", "true")
    # No backend is needed: with its host disabled, the write flag registers
    # nothing and must not make this otherwise valid configuration fail.
    monkeypatch.setenv("STORY_DATA_URL", "")
    settings = Settings()
    assert settings.enable_mcp_writes is True
    assert settings.resolved_mcp_writes_enabled is False


def test_writes_require_story_data_url(monkeypatch):
    """Writes go to story-data, so the flag alone would register four tools
    that fail on their first call — which the model reports as an outage."""
    monkeypatch.setenv("ENABLE_MCP", "true")
    monkeypatch.setenv("ENABLE_MCP_WRITES", "true")
    monkeypatch.setenv("STORY_DATA_URL", "")
    with pytest.raises(ValidationError, match="ENABLE_MCP_WRITES requires"):
        Settings()


def test_write_rate_limit_clamps_garbage(monkeypatch):
    monkeypatch.setenv("MCP_MAX_WRITES_PER_MINUTE_PER_USER", "not-a-number")
    assert Settings().mcp_max_writes_per_minute_per_user == 6
    monkeypatch.setenv("MCP_MAX_WRITES_PER_MINUTE_PER_USER", "-5")
    assert Settings().mcp_max_writes_per_minute_per_user == 0
