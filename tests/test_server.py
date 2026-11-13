"""Tests for FastAPI server endpoints."""

import os
from unittest.mock import ANY, AsyncMock

import pytest
from fastapi.testclient import TestClient

# Set environment for testing before importing app
os.environ["CREDIT_PROXY_URL"] = "http://localhost:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["MAX_REQUESTS_PER_MINUTE_PER_USER"] = "1000"

from agents.storyAgent.llm_provider import _usage_feature  # noqa: E402
from server import app

client = TestClient(app)


class TestHealth:
    def test_health_check_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200


class TestAgentExecution:

    def test_agent_execute_missing_user_id_returns_422(self):
        response = client.post(
            "/agent/execute",
            json={
                "action": "enhanceWizardInput",
                "parameters": {"type": "premise", "data": {"title": "T"}},
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    @pytest.mark.parametrize(
        "action",
        [
            "chatWithContext",
            "generateNextLines",
            "enhanceText",
            "brainstormIdeas",
            "generateStoryChoices",
            "summarizeChapter",
        ],
    )
    def test_a_retired_action_is_rejected_as_unknown(self, action):
        """A removed action must read as unknown, not as a route that silently
        accepts and does nothing."""
        app.state.agent.execute_agent = AsyncMock(return_value={"response": "ok"})

        response = client.post(
            "/agent/execute",
            json={
                "action": action,
                "parameters": {"storyId": "s1"},
                "user_id": "u1",
            },
        )

        assert response.status_code == 422
        app.state.agent.execute_agent.assert_not_awaited()

    def test_enhance_wizard_input_success(self):
        features = []

        async def run(*_args, **_kwargs):
            features.append(_usage_feature.get())
            return {"enhanced": "In a world..."}

        app.state.agent.execute_agent = AsyncMock(side_effect=run)

        response = client.post(
            "/agent/execute",
            json={
                "action": "enhanceWizardInput",
                "parameters": {
                    "type": "premise",
                    "data": {
                        "title": "The Last Lantern",
                        "premise": "A girl finds a magic lamp",
                    },
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
                "data": {
                    "title": "The Last Lantern",
                    "premise": "A girl finds a magic lamp",
                },
                "userId": "user-1",
            },
            background_tasks=ANY,
            user_id="u-session",
        )
        # The action's LLM calls are reported under its feature.
        assert features == ["wizard_enhance"]

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
                "user_id": "u1",
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"


class TestCredits:
    def test_balance_success(self):
        app.state.agent.llm_provider.get_balance = AsyncMock(
            return_value={"user_id": "u1", "available_credits": 1749}
        )

        response = client.post(
            "/credits/balance",
            json={"user_id": "u1"},
            headers={"X-Firebase-Token": "fb-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["available_credits"] == 1749
        app.state.agent.llm_provider.get_balance.assert_awaited_once_with(
            "u1", "fb-token"
        )

    def test_balance_missing_user_id_returns_422(self):
        response = client.post("/credits/balance", json={})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_purchase_success(self):
        app.state.agent.llm_provider.purchase_credits = AsyncMock(
            return_value={
                "user_id": "u1",
                "purchased_credits": 10000,
                "available_credits": 11749,
            }
        )

        response = client.post(
            "/credits/purchase", json={"user_id": "u1", "credits": 10000}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["available_credits"] == 11749
        app.state.agent.llm_provider.purchase_credits.assert_awaited_once_with(
            "u1", 10000, None
        )

    def test_purchase_rejects_non_tier_amount(self):
        app.state.agent.llm_provider.purchase_credits = AsyncMock()

        response = client.post(
            "/credits/purchase", json={"user_id": "u1", "credits": 12345}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"
        # never reached the provider for an invalid amount
        app.state.agent.llm_provider.purchase_credits.assert_not_awaited()

    def test_purchase_backend_unavailable_returns_503(self):
        from agents.storyAgent.llm_provider import BackendUnavailableError

        app.state.agent.llm_provider.purchase_credits = AsyncMock(
            side_effect=BackendUnavailableError("purchase disabled")
        )

        response = client.post(
            "/credits/purchase", json={"user_id": "u1", "credits": 10000}
        )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "BACKEND_UNAVAILABLE"

    def test_purchase_rate_limited_returns_429(self):
        from agents.storyAgent.llm_provider import RateLimitedError

        # creditProxy returns 429 once the per-user daily purchase cap is hit.
        app.state.agent.llm_provider.purchase_credits = AsyncMock(
            side_effect=RateLimitedError("daily purchase limit reached")
        )

        response = client.post(
            "/credits/purchase", json={"user_id": "u1", "credits": 10000}
        )

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "RATE_LIMITED"


class TestAiProviders:
    def test_catalog_success(self):
        app.state.agent.llm_provider.get_provider_catalog = AsyncMock(
            return_value={"version": 1, "providers": [{"id": "gemini"}]}
        )

        response = client.post(
            "/ai/providers", headers={"X-Firebase-Token": "fb-token"}
        )

        assert response.status_code == 200
        assert response.json()["data"]["providers"][0]["id"] == "gemini"
        app.state.agent.llm_provider.get_provider_catalog.assert_awaited_once_with(
            "fb-token"
        )

    def test_validation_forwards_user_key_and_model(self):
        app.state.agent.llm_provider.validate_provider = AsyncMock(
            return_value={
                "valid": True,
                "provider": "openai",
                "model": "gpt-4o-mini",
            }
        )

        response = client.post(
            "/ai/providers/validate",
            json={
                "user_id": "u1",
                "provider": "openai",
                "api_key": "secret",
                "model": "gpt-4o-mini",
            },
            headers={"X-Firebase-Token": "fb-token"},
        )

        assert response.status_code == 200
        assert response.json()["data"]["valid"] is True
        app.state.agent.llm_provider.validate_provider.assert_awaited_once_with(
            "u1", "openai", "secret", "gpt-4o-mini", "fb-token"
        )
