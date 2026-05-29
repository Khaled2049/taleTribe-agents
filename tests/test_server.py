"""Tests for FastAPI server endpoints."""
import os
from unittest.mock import ANY, AsyncMock

from fastapi.testclient import TestClient

# Set environment for testing before importing app
os.environ["CREDIT_PROXY_URL"] = "http://localhost:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["MAX_REQUESTS_PER_MINUTE_PER_USER"] = "1000"

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
            json={"action": "generateStory", "parameters": {"storyId": "s1"}, "user_id": "u1"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["storyId"] == "s1"

    def test_agent_execute_missing_user_id_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={"action": "generateStory", "parameters": {"storyId": "s1"}},
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

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
                "user_id": "u1",
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
            background_tasks=ANY,
            user_id="u1",
        )

    def test_enhance_wizard_input_success(self):
        app.state.agent.execute_agent = AsyncMock(return_value={"enhanced": "In a world..."})

        response = client.post(
            "/agent/execute",
            json={
                "action": "enhanceWizardInput",
                "parameters": {
                    "type": "premise",
                    "data": {"title": "The Last Lantern", "premise": "A girl finds a magic lamp"},
                    "userId": "user-1",
                },
                "user_id": "u-session",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["enhanced"] == "In a world..."
        app.state.agent.execute_agent.assert_awaited_once_with(
            "enhanceWizardInput",
            {
                "type": "premise",
                "data": {"title": "The Last Lantern", "premise": "A girl finds a magic lamp"},
                "userId": "user-1",
            },
            background_tasks=ANY,
            user_id="u-session",
        )

    def test_enhance_wizard_input_invalid_type_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={
                "action": "enhanceWizardInput",
                "parameters": {
                    "type": "unknown",
                    "data": {"title": "The Last Lantern"},
                    "userId": "user-1",
                },
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"


class TestGenerateStoryChoices:
    def test_opening_mode_success(self):
        mock_result = {
            "storyId": "s1",
            "openingScene": "The rain had been falling for three days...",
            "choices": [
                {"label": "Elena discovers the hidden letter", "sceneText": "She found it..."},
                {"label": "A stranger arrives at the inn", "sceneText": "The door swung..."},
                {"label": "The market erupts in chaos", "sceneText": "First came the sound..."},
            ],
        }
        app.state.agent.execute_agent = AsyncMock(return_value=mock_result)

        response = client.post(
            "/agent/execute",
            json={
                "action": "generateStoryChoices",
                "parameters": {"storyId": "s1", "mode": "opening"},
                "user_id": "u1",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["openingScene"] == "The rain had been falling for three days..."
        assert len(data["data"]["choices"]) == 3
        app.state.agent.execute_agent.assert_awaited_once_with(
            "generateStoryChoices",
            {"storyId": "s1", "mode": "opening", "currentContent": "", "turnCount": 0},
            background_tasks=ANY,
            user_id="u1",
        )

    def test_continuation_mode_success(self):
        mock_result = {
            "storyId": "s1",
            "choices": [
                {"label": "Confront Marcus directly", "sceneText": "She stepped forward..."},
                {"label": "Follow the shadow into alley", "sceneText": "The figure vanished..."},
                {"label": "Return to the archive", "sceneText": "The old building..."},
            ],
        }
        app.state.agent.execute_agent = AsyncMock(return_value=mock_result)

        response = client.post(
            "/agent/execute",
            json={
                "action": "generateStoryChoices",
                "parameters": {
                    "storyId": "s1",
                    "mode": "continuation",
                    "currentContent": "<p>Some prose.</p>",
                    "chapterId": "ch1",
                },
                "user_id": "u1",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["choices"]) == 3
        assert "openingScene" not in data["data"]
        app.state.agent.execute_agent.assert_awaited_once_with(
            "generateStoryChoices",
            {
                "storyId": "s1",
                "mode": "continuation",
                "currentContent": "<p>Some prose.</p>",
                "chapterId": "ch1",
                "turnCount": 0,
            },
            background_tasks=ANY,
            user_id="u1",
        )

    def test_missing_mode_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={
                "action": "generateStoryChoices",
                "parameters": {"storyId": "s1"},
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_missing_story_id_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={
                "action": "generateStoryChoices",
                "parameters": {"mode": "opening"},
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_invalid_mode_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={
                "action": "generateStoryChoices",
                "parameters": {"storyId": "s1", "mode": "invalid"},
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_current_content_defaults_to_empty_string(self):
        app.state.agent.execute_agent = AsyncMock(return_value={"storyId": "s1", "choices": []})

        client.post(
            "/agent/execute",
            json={
                "action": "generateStoryChoices",
                "parameters": {"storyId": "s1", "mode": "opening"},
                "user_id": "u1",
            },
        )

        _, called_params = app.state.agent.execute_agent.await_args.args
        assert called_params["currentContent"] == ""


class TestDocsEndpoints:
    def test_openapi_schema_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
