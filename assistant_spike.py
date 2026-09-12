"""Transport proof for the v1 assistant protocol. No inference or persistence.

Phase 0 shipped this with a hand-rolled ``{"type":"text-delta"}`` frame format
that the ADR explicitly said not to ship as the assistant API. Phase 1 keeps the
transport -- which is the part that was actually proven, including cancellation
reaching the agent -- and swaps the payload for real ``assistant.events`` frames.
That makes the spike the protocol's first end-to-end test: the TypeScript
validator checks itself against bytes off a socket rather than against a JSON
file, and it still costs no model call, no credits and no writes.

The mock deltas remain fixed. Phase 3 replaces this endpoint with the real run
loop; until then the only thing it proves is that the wire format survives the
gateway, which is exactly what it proved before.
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from assistant.errors import ErrorCode, safe_message
from assistant.events import (
    RunCompleted,
    RunEvents,
    RunStarted,
    TextDelta,
    TextDone,
    Usage,
    encode_sse,
)
from assistant.protocol import AgentRunRequest, TextPart
from assistant.version import ASSISTANT_PROTOCOL_VERSION
from mcp_server.data import StoryNotFoundError, get_owned_story
from mcp_server.story_data import StoryDataClient, StoryDataError

logger = logging.getLogger(__name__)

MOCK_DELTAS = ["Mock", " streaming", " works."]


async def mock_events(run_id: str):
    """Fixed deltas as v1 frames.

    Note what is deliberately *not* here: a ``run.cancelled`` frame in the
    ``finally``. When the consumer disconnects, the generator is torn down with
    ``GeneratorExit``, and yielding during that is illegal -- Python raises
    "async generator ignored GeneratorExit". It would also be pointless, since
    the peer that would read the frame is the one that just left.

    ``run.cancelled`` is therefore a *server-initiated* terminal state, for when
    the run loop stops itself (a budget ceiling, an internal abort) and the
    socket is still open. A peer disconnect needs no frame: the browser knows it
    disconnected. The Phase 0 gap this protocol actually closes is the other
    one -- a mid-stream failure with the socket intact, which is ``run.failed``.
    """
    events = RunEvents(run_id)
    completed = False
    try:
        yield encode_sse(events.emit(RunStarted, provider="mock", model="mock-1"))
        for text in MOCK_DELTAS:
            yield encode_sse(events.emit(TextDelta, text=text))
            await asyncio.sleep(0.75)
        yield encode_sse(
            events.emit(
                TextDone,
                part=TextPart(type="text", text="".join(MOCK_DELTAS)),
            )
        )
        yield encode_sse(
            events.emit(
                Usage,
                provider="mock",
                model="mock-1",
                prompt_tokens=0,
                completion_tokens=0,
                credits=0,
                billing="mock",
            )
        )
        yield encode_sse(events.emit(RunCompleted, finish_reason="stop"))
        completed = True
    finally:
        logger.info(
            "assistant_spike_finished run_id=%s outcome=%s",
            run_id,
            "completed" if completed else "cancelled",
        )


def register_spike(app, settings, verify_internal_token):
    @app.post("/assistant/spike")
    async def stream_spike(
        raw_body: dict[str, Any],
        request: Request,
        _: None = Depends(verify_internal_token),
    ):
        if raw_body.get("v") != ASSISTANT_PROTOCOL_VERSION:
            code = ErrorCode.UNSUPPORTED_PROTOCOL_VERSION
            raise HTTPException(
                409,
                detail={"code": code.value, "message": safe_message(code)},
            )
        try:
            body = AgentRunRequest.model_validate(raw_body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(), body=raw_body) from exc
        if (
            not settings.assistant_api_enabled
            or not settings.assistant_stream_spike_enabled
        ):
            raise HTTPException(404, "Assistant transport spike is disabled")
        if not await request.app.state.rate_limiter.allow(body.user_id):
            raise HTTPException(429, "Too many requests")
        if not settings.story_data_url:
            raise HTTPException(503, "Story service unavailable")
        client = StoryDataClient(
            settings.story_data_url, settings.story_data_service_token
        )
        try:
            # story-data serves a published story to any caller, so a 200 is not
            # proof of ownership. get_owned_story re-checks ownerId == uid.
            await get_owned_story(body.story_id, body.user_id, story_client=client)
        except StoryNotFoundError:
            raise HTTPException(403, "Story access denied") from None
        except StoryDataError:
            raise HTTPException(503, "Story service unavailable") from None
        finally:
            await client.close()
        run_id = uuid.uuid4().hex
        logger.info("assistant_spike_started run_id=%s", run_id)
        return StreamingResponse(
            mock_events(run_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
