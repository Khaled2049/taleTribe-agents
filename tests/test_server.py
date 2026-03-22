"""Tests for FastAPI server endpoints."""
import os
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

# Set environment for testing before importing app
os.environ["USE_MOCK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

from server import app

client = TestClient(app)


class TestHealth:
    def test_health_check_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_contains_project_id(self):
        response = client.get("/health")
        data = response.json()
        assert data["project_id"] == "test-project"


class TestAgentExecution:
    def test_agent_execute_success(self):
        app.state.agent.execute_agent = AsyncMock(return_value={"storyId": "s1"})

        response = client.post(
            "/agent/execute",
            json={"action": "generateStory", "parameters": {"storyId": "s1"}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["storyId"] == "s1"

    def test_agent_execute_unknown_action_validation_error(self):
        response = client.post(
            "/agent/execute",
            json={"action": "invalidAction", "parameters": {}},
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_agent_execute_invalid_parameters_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={"action": "generateNextLines", "parameters": {"storyId": "s1"}},
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_chat_with_context_accepts_legacy_context_payload(self):
        app.state.agent.execute_agent = AsyncMock(return_value={"response": "ok"})

        response = client.post(
            "/agent/execute",
            json={
                "action": "chatWithContext",
                "parameters": {
                    "storyId": "s1",
                    "message": "hello",
                    "context": {
                        "story": {"title": "Joy of Santa Fe"},
                        "chapters": [],
                    },
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        app.state.agent.execute_agent.assert_awaited_once_with(
            "chatWithContext",
            {
                "storyId": "s1",
                "message": "hello",
                "context": {
                    "story": {"title": "Joy of Santa Fe"},
                    "chapters": [],
                },
            },
        )


class TestDocsEndpoints:
    def test_openapi_schema_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
