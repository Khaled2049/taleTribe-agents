"""Authenticated HTTP surface for the streaming assistant run loop."""

from __future__ import annotations

import uuid
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from agents.storyAgent.llm_provider import _byok_config, _firebase_token
from assistant.errors import ErrorCode, safe_message
from assistant.events import encode_sse
from assistant.history import HistoryLimits
from assistant.protocol import AgentRunRequest
from assistant.run import RunLimits, run_assistant
from assistant.version import ASSISTANT_PROTOCOL_VERSION
from mcp_server.data import StoryNotFoundError, get_owned_story
from mcp_server.story_data import StoryDataClient, StoryDataError


def register_assistant(
    app: Any, settings: Any, verify_internal_token: Callable[..., Any]
) -> None:
    @app.post("/assistant/run")
    async def assistant_run(
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

        if not settings.assistant_api_enabled:
            raise HTTPException(404, "Assistant API is disabled")
        if not await request.app.state.rate_limiter.allow(body.user_id):
            raise HTTPException(429, "Too many requests")
        if not settings.story_data_url.strip():
            raise HTTPException(503, "Story service unavailable")

        # The Functions gateway checks ownership first, and agents checks again.
        # story-data permits public reads of published stories, so a 200 alone is
        # not authorization: get_owned_story compares ownerId with the asserted uid.
        client = StoryDataClient(
            settings.story_data_url.strip(), settings.story_data_service_token.strip()
        )
        try:
            await get_owned_story(body.story_id, body.user_id, story_client=client)
        except StoryNotFoundError:
            raise HTTPException(403, "Story access denied") from None
        except StoryDataError:
            raise HTTPException(503, "Story service unavailable") from None
        finally:
            await client.close()

        run_id = uuid.uuid4().hex
        firebase_token = request.headers.get("X-Firebase-Token", "").strip() or None
        limits = RunLimits.from_settings(settings)
        history_limits = HistoryLimits.from_settings(settings)

        async def frames():
            # ContextVars are installed inside the streaming task, not the route
            # task. Starlette may consume StreamingResponse in a child task whose
            # context was copied before the route returns.
            byok = body.provider_config
            byok_context = _byok_config.set(
                {
                    "user_id": body.user_id,
                    "provider": byok.provider if byok else "",
                    "api_key": byok.api_key if byok else "",
                    "model": (byok.model or "") if byok else "",
                }
            )
            firebase_context = _firebase_token.set(firebase_token)
            try:
                agent = request.app.state.agent
                async for event in run_assistant(
                    body,
                    run_id=run_id,
                    provider=agent.llm_provider,
                    postgres=agent.postgres_context,
                    embedder=agent.embedding_provider,
                    limits=limits,
                    edits_enabled=settings.assistant_edit_proposals_enabled,
                    history_limits=history_limits,
                ):
                    yield encode_sse(event)
            finally:
                _firebase_token.reset(firebase_context)
                _byok_config.reset(byok_context)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
