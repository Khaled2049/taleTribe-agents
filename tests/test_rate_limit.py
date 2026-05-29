"""Tests for per-user rate limiting."""
import os
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from rate_limit import PerUserRateLimiter
from server import create_app


@pytest.mark.asyncio
async def test_allow_respects_limit():
    limiter = PerUserRateLimiter(2)
    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-a") is False


@pytest.mark.asyncio
async def test_allow_isolated_per_user():
    limiter = PerUserRateLimiter(1)
    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-b") is True
    assert await limiter.allow("user-a") is False


@pytest.mark.asyncio
async def test_allow_disabled_when_max_zero():
    limiter = PerUserRateLimiter(0)
    for _ in range(5):
        assert await limiter.allow("user-a") is True


def test_agent_execute_returns_429_when_rate_limited(monkeypatch):
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "2")
    test_app = create_app()
    test_app.state.agent.execute_agent = AsyncMock(return_value={"ok": True})

    with TestClient(test_app) as client:
        payload = {
            "action": "generateStory",
            "parameters": {"storyId": "s1"},
            "user_id": "rate-test-user",
        }
        assert client.post("/agent/execute", json=payload).status_code == 200
        assert client.post("/agent/execute", json=payload).status_code == 200
        response = client.post("/agent/execute", json=payload)

    assert response.status_code == 429
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "RATE_LIMITED"


def test_agent_execute_rejects_missing_user_id(monkeypatch):
    """user_id is required so anonymous traffic cannot share one rate-limit bucket."""
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "1000")
    test_app = create_app()

    with TestClient(test_app) as client:
        response = client.post(
            "/agent/execute",
            json={"action": "generateStory", "parameters": {"storyId": "s1"}},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_agent_execute_rejects_empty_user_id(monkeypatch):
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "1000")
    test_app = create_app()

    with TestClient(test_app) as client:
        response = client.post(
            "/agent/execute",
            json={"action": "generateStory", "parameters": {"storyId": "s1"}, "user_id": ""},
        )

    assert response.status_code == 422
