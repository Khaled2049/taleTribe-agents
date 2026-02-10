"""Tests for FastAPI server endpoints."""
import os

from fastapi.testclient import TestClient

# Set environment for testing
os.environ["USE_MOCK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

from server import app

client = TestClient(app)


class TestHealth:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_healthy_status(self):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_contains_project_id(self):
        """Test that health check includes project ID."""
        response = client.get("/health")
        data = response.json()
        assert "project_id" in data
        assert data["project_id"] == "test-project"

    def test_health_check_contains_services(self):
        """Test that health check includes service status."""
        response = client.get("/health")
        data = response.json()
        assert "services" in data
        assert isinstance(data["services"], dict)


class TestAgentExecution:
    """Tests for agent execution endpoint."""

    def test_agent_execute_invalid_action(self):
        """Test agent execution with invalid action."""
        response = client.post(
            "/agent/execute",
            json={"action": "invalidAction", "parameters": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        # Invalid action should fail
        if not data["success"]:
            assert "error" in data

    def test_agent_execute_missing_required_fields(self):
        """Test agent execution with missing required fields."""
        response = client.post(
            "/agent/execute",
            json={},
        )
        # Should return 422 (validation error) or 200 with error
        assert response.status_code in [200, 422]

    def test_agent_execute_returns_success_field(self):
        """Test that agent execution returns success field."""
        response = client.post(
            "/agent/execute",
            json={"action": "generateNextLines", "parameters": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data


class TestDocsEndpoints:
    """Tests for documentation endpoints."""

    def test_swagger_docs_available(self):
        """Test that Swagger docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """Test that ReDoc docs are available."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema_available(self):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
