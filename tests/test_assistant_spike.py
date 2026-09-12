import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from assistant.events import validate_event_sequence
from assistant_spike import mock_events, register_spike
from config import Settings
from mcp_server.story_data import NotFound, StoryDataError
from rate_limit import PerUserRateLimiter


def settings(**kwargs):
    return Settings(
        _env_file=None,
        google_cloud_project="test",
        environment="test",
        enable_mcp_writes=False,
        **kwargs,
    )


def client(enabled=True, authorized=True):
    app = FastAPI()
    app.state.rate_limiter = PerUserRateLimiter(20)

    async def verify():
        if not authorized:
            raise HTTPException(401)

    register_spike(
        app,
        settings(
            assistant_api_enabled=enabled,
            assistant_stream_spike_enabled=enabled,
            story_data_url="http://story-data",
        ),
        verify,
    )
    return TestClient(app)


BODY = {
    "v": 1,
    "userId": "u1",
    "storyId": "s1",
    "clientMessageId": "c1",
    "message": {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
}


def test_defaults_and_production_guard():
    config = settings()
    assert not config.assistant_api_enabled
    assert not config.assistant_edit_proposals_enabled
    assert not config.assistant_research_enabled
    assert config.assistant_legacy_fallback_enabled
    with pytest.raises(ValidationError, match="development/test"):
        Settings(
            google_cloud_project="test",
            environment="production",
            assistant_api_enabled=True,
            assistant_stream_spike_enabled=True,
        )


@pytest.mark.parametrize(
    "enabled,authorized,status", [(False, True, 404), (True, False, 401)]
)
def test_gates(enabled, authorized, status):
    assert (
        client(enabled, authorized).post("/assistant/spike", json=BODY).status_code
        == status
    )


def test_missing_story_is_denied():
    with patch("assistant_spike.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(side_effect=NotFound())
        factory.return_value.close = AsyncMock()
        assert client().post("/assistant/spike", json=BODY).status_code == 403
        factory.return_value.get_story.assert_awaited_once_with("u1", "s1")


def test_published_story_owned_by_someone_else_is_denied():
    """story-data serves a published story to any caller; the spike must not.

    Without the ownerId re-check this returns 200, which would widen the
    assistant from "your stories" to every published story on the platform.
    """
    with patch("assistant_spike.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(
            return_value={"id": "s1", "ownerId": "someone-else", "published": True}
        )
        factory.return_value.close = AsyncMock()
        assert client().post("/assistant/spike", json=BODY).status_code == 403


def test_story_service_failure_is_not_reported_as_denial():
    with patch("assistant_spike.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(side_effect=StoryDataError("boom"))
        factory.return_value.close = AsyncMock()
        assert client().post("/assistant/spike", json=BODY).status_code == 503


def test_owned_story_streams_a_valid_v1_run():
    with patch("assistant_spike.StoryDataClient") as factory:
        factory.return_value.get_story = AsyncMock(
            return_value={"id": "s1", "ownerId": "u1"}
        )
        factory.return_value.close = AsyncMock()
        response = client().post("/assistant/spike", json=BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = [f for f in response.text.split("\n\n") if f.strip()]
    events = [json.loads(f.removeprefix("data: ")) for f in frames]
    parsed = validate_event_sequence(events)
    assert [e.type for e in parsed] == [
        "run.started",
        "text.delta",
        "text.delta",
        "text.delta",
        "text.done",
        "usage",
        "run.completed",
    ]
    assert "".join(e.text for e in parsed if e.type == "text.delta") == (
        "Mock streaming works."
    )


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "apiKey": "fake"},
        {k: v for k, v in BODY.items() if k != "userId"},
        {**BODY, "message": {"role": "user", "parts": []}},
    ],
    ids=["unknown-field", "missing-uid", "empty-parts"],
)
def test_malformed_requests_rejected(body):
    assert client().post("/assistant/spike", json=body).status_code == 422


def test_unsupported_protocol_version_is_a_stable_conflict():
    response = client().post("/assistant/spike", json={**BODY, "v": 2})
    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "unsupported_protocol_version",
            "message": "This version of the app is out of date. Reload to continue.",
        }
    }


@pytest.mark.asyncio
async def test_disconnect_logs_cancellation_without_emitting_a_frame(caplog):
    """A dropped consumer is logged, not signalled.

    There is deliberately no run.cancelled frame here: yielding during
    GeneratorExit is illegal, and the peer that would read it has already gone.
    The agent-side terminal outcome in the log is what Phase 0 verified
    cancellation against, and it stays the signal.
    """
    with caplog.at_level(logging.INFO):
        stream = mock_events("test-run")
        assert '"type":"run.started"' in await anext(stream)
        assert '"text":"Mock"' in await anext(stream)
        await stream.aclose()
    assert "outcome=cancelled" in caplog.text
    assert "outcome=completed" not in caplog.text
