"""The app must build and serve normally with ENABLE_MCP=false."""

import os

os.environ["CREDIT_PROXY_URL"] = "http://localhost:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["USE_MOCK"] = "true"
os.environ["ENABLE_MCP"] = "false"
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:9999")

from fastapi.testclient import TestClient  # noqa: E402

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
