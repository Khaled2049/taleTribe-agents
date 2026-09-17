"""HTTP trust boundary for POST /assistant/run."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agents.storyAgent.llm_provider import _byok_config, _firebase_token
from assistant.api import register_assistant
from assistant.events import validate_event_sequence
from config import Settings
from mcp_server.story_data import NotFound, StoryDataError
from rate_limit import PerUserRateLimiter

BODY = {
    "v": 1,
    "userId": "u1",
    "storyId": "s1",
    "clientMessageId": "c1",
    "message": {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
}


class TextProvider:
    def __init__(self):
        self.calls = 0
        self.context = None

    async def chat_stream(self, *_args, **_kwargs):
        self.calls += 1
        self.context = (_byok_config.get(), _firebase_token.get())
        yield {"type": "text_delta", "text": "Hello."}
        yield {
            "type": "usage",
            "provider": "mock",
            "model": "mock-1",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }
        yield {"type": "done", "finish_reason": "stop"}


def settings(**kwargs):
    return Settings(
        _env_file=None,
        google_cloud_project="test",
        environment="test",
        enable_mcp=False,
        enable_mcp_writes=False,
        story_data_url="http://story-data",
        **kwargs,
    )


def client(*, enabled=True, authorized=True, provider=None):
    app = FastAPI()
    app.state.rate_limiter = PerUserRateLimiter(20)
    app.state.agent = SimpleNamespace(
        llm_provider=provider or TextProvider(),
        postgres_context=None,
        embedding_provider=None,
    )

    async def verify():
        if not authorized:
            raise HTTPException(401)

    register_assistant(app, settings(assistant_api_enabled=enabled), verify)
    return TestClient(app), app.state.agent.llm_provider


def owned(factory):
    factory.return_value.get_story = AsyncMock(
        return_value={"id": "s1", "ownerId": "u1"}
    )
    factory.return_value.close = AsyncMock()


def test_defaults_and_ceiling_validation():
    config = settings()
    assert config.assistant_max_model_calls == 4
    assert config.assistant_max_tool_calls == 10
    assert config.assistant_max_output_tokens == 2048
    assert config.assistant_run_timeout_seconds == 120
    with pytest.raises(ValidationError):
        settings(assistant_max_model_calls=0)
    with pytest.raises(ValidationError):
        settings(assistant_max_output_tokens=8193)


@pytest.mark.parametrize(
    "enabled,authorized,status", [(False, True, 404), (True, False, 401)]
)
def test_gates(enabled, authorized, status):
    api, _ = client(enabled=enabled, authorized=authorized)
    if not enabled:
        with patch("assistant.api.StoryDataClient") as factory:
            owned(factory)
            assert api.post("/assistant/run", json=BODY).status_code == status
    else:
        assert api.post("/assistant/run", json=BODY).status_code == status


def test_unowned_story_is_refused_before_the_model():
    provider = TextProvider()
    api, _ = client(provider=provider)
    with patch("assistant.api.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(side_effect=NotFound())
        factory.return_value.close = AsyncMock()
        assert api.post("/assistant/run", json=BODY).status_code == 403
    assert provider.calls == 0


def test_published_story_owned_by_someone_else_is_refused_before_the_model():
    provider = TextProvider()
    api, _ = client(provider=provider)
    with patch("assistant.api.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(
            return_value={"id": "s1", "ownerId": "someone-else", "published": True}
        )
        factory.return_value.close = AsyncMock()
        assert api.post("/assistant/run", json=BODY).status_code == 403
    assert provider.calls == 0


def test_story_service_failure_is_not_reported_as_denial():
    api, _ = client()
    with patch("assistant.api.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(side_effect=StoryDataError("boom"))
        factory.return_value.close = AsyncMock()
        assert api.post("/assistant/run", json=BODY).status_code == 503


def test_owned_story_streams_a_valid_run_with_request_context():
    provider = TextProvider()
    api, _ = client(provider=provider)
    with patch("assistant.api.StoryDataClient") as factory:
        owned(factory)
        response = api.post(
            "/assistant/run",
            json=BODY,
            headers={"X-Firebase-Token": "firebase-token"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"

    frames = [frame for frame in response.text.split("\n\n") if frame.strip()]
    events = [json.loads(frame.removeprefix("data: ")) for frame in frames]
    parsed = validate_event_sequence(events)
    assert [event.type for event in parsed] == [
        "run.started",
        "text.delta",
        "usage",
        "text.done",
        "run.completed",
    ]
    assert provider.context == (
        {"user_id": "u1", "provider": "", "api_key": "", "model": ""},
        "firebase-token",
    )


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "apiKey": "fake"},
        {key: value for key, value in BODY.items() if key != "userId"},
        {**BODY, "message": {"role": "user", "parts": []}},
    ],
)
def test_malformed_requests_are_rejected(body):
    api, _ = client()
    assert api.post("/assistant/run", json=body).status_code == 422


def test_unsupported_protocol_version_is_a_stable_conflict():
    api, _ = client()
    response = api.post("/assistant/run", json={**BODY, "v": 2})
    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "unsupported_protocol_version",
            "message": "This version of the app is out of date. Reload to continue.",
        }
    }


def test_byok_credentials_reach_the_provider_context():
    """A run carrying the gateway's resolved key must bill that key, not the
    platform. The provider reads it from the ContextVar, so that is where it
    has to arrive."""
    provider = TextProvider()
    api, _ = client(provider=provider)
    with patch("assistant.api.StoryDataClient") as factory:
        owned(factory)
        response = api.post(
            "/assistant/run",
            json={
                **BODY,
                "providerConfig": {
                    "provider": "claude",
                    "apiKey": "user-supplied-key",
                    "model": "claude-sonnet-4-6",
                },
            },
            headers={"X-Firebase-Token": "firebase-token"},
        )

    assert response.status_code == 200
    assert provider.context == (
        {
            "user_id": "u1",
            "provider": "claude",
            "api_key": "user-supplied-key",
            "model": "claude-sonnet-4-6",
        },
        "firebase-token",
    )


def test_a_run_without_byok_sends_no_key():
    """The platform path must stay the default: absent settings mean absent
    credentials, not an empty-string provider the gateway might honour."""
    provider = TextProvider()
    api, _ = client(provider=provider)
    with patch("assistant.api.StoryDataClient") as factory:
        owned(factory)
        api.post("/assistant/run", json=BODY)

    config, _ = provider.context
    assert config["provider"] == ""
    assert config["api_key"] == ""


@pytest.mark.parametrize(
    "provider_config",
    [
        {"provider": "gemini"},
        {"provider": "gemini", "apiKey": ""},
        {"provider": "not-a-provider", "apiKey": "k"},
        {"provider": "gemini", "apiKey": "k", "extra": "field"},
    ],
)
def test_malformed_provider_config_is_rejected(provider_config):
    api, _ = client()
    response = api.post(
        "/assistant/run", json={**BODY, "providerConfig": provider_config}
    )
    assert response.status_code == 422
