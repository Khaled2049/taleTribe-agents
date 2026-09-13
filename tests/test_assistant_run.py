"""The Phase 3 run loop: translation, tool rounds, and hard ceilings."""

import asyncio
import json

import pytest

from agents.storyAgent.llm_provider import InsufficientCreditsError
from assistant.events import validate_event_sequence
from assistant.protocol import AgentRunRequest
from assistant.run import RunLimits, run_assistant
from mcp_server import story_data
from tests.mcp_fakes import FakeStoryData

BODY = {
    "v": 1,
    "userId": "uid-1",
    "storyId": "story-1",
    "clientMessageId": "message-1",
    "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "How many chapters?"}],
    },
}


class FakeProvider:
    def __init__(self, *calls):
        self.scripts = list(calls)
        self.requests = []

    async def chat_stream(self, messages, tools, *, max_output_tokens, idempotency_key):
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "max_output_tokens": max_output_tokens,
                "idempotency_key": idempotency_key,
            }
        )
        script = self.scripts[min(len(self.requests) - 1, len(self.scripts) - 1)]
        for event in script:
            yield event


class FakePostgres:
    pool = object()

    async def slim_context(self, story_id):
        assert story_id == "story-1"
        return "Story: Saltmarsh\nCharacters: Mina\nChapters: The Lamp Room"


@pytest.fixture
def owned_story():
    fake = FakeStoryData()
    fake.seed_story("story-1", "uid-1", title="Saltmarsh")
    fake.seed_chapter(
        "story-1", "chapter-1", title="The Lamp Room", content="Brass polish."
    )
    fake.seed_entity(
        "story-1", "characters", "char-1", name="Mina", personality="Guarded."
    )
    story_data.configure(fake)
    yield fake
    story_data.configure(None)


def tool_round(name="get_story_overview", arguments="{}", call_id="call-1"):
    return [
        {
            "type": "tool_call_delta",
            "provider": "mock",
            "model": "mock-1",
            "tool_call": {
                "index": 0,
                "tool_call_id": call_id,
                "name": name,
            },
        },
        {
            "type": "tool_call_delta",
            "tool_call": {"index": 0, "arguments_delta": arguments},
        },
        {
            "type": "usage",
            "provider": "mock",
            "model": "mock-1",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
            "credits": 0,
        },
        {"type": "done", "finish_reason": "tool_calls"},
    ]


FINAL_ROUND = [
    {"type": "text_delta", "text": "Saltmarsh has one chapter."},
    {
        "type": "usage",
        "provider": "mock",
        "model": "mock-1",
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 7,
            "total_tokens": 157,
        },
        "credits": 0,
    },
    {"type": "done", "finish_reason": "stop"},
]


async def collect(provider, *, limits=None):
    request = AgentRunRequest.model_validate({**BODY, "userId": "uid-1"})
    return [
        event
        async for event in run_assistant(
            request,
            run_id="run-1",
            provider=provider,
            postgres=FakePostgres(),
            embedder=None,
            limits=limits or RunLimits(),
        )
    ]


async def test_tool_answer_runs_end_to_end_and_reenters_as_data(owned_story):
    provider = FakeProvider(tool_round(), FINAL_ROUND)
    events = await collect(provider)
    parsed = validate_event_sequence(
        [event.model_dump(by_alias=True, mode="json") for event in events]
    )

    assert [event.type for event in parsed] == [
        "run.started",
        "tool.started",
        "tool.args.delta",
        "usage",
        "tool.completed",
        "text.delta",
        "usage",
        "text.done",
        "run.completed",
    ]
    assert parsed[-1].finish_reason == "stop"
    assert len(provider.requests) == 2
    assert provider.requests[0]["idempotency_key"] == "run-1:0"
    assert provider.requests[1]["idempotency_key"] == "run-1:1"
    assert (
        "structured entity tools"
        in provider.requests[0]["messages"][0]["parts"][0]["text"]
    )
    assert "Story: Saltmarsh" in provider.requests[0]["messages"][0]["parts"][0]["text"]

    second_messages = provider.requests[1]["messages"]
    assert second_messages[-1]["role"] == "tool"
    tool_result = json.loads(second_messages[-1]["parts"][0]["text"])
    assert tool_result["title"] == "Saltmarsh"
    assert "user_id" not in second_messages[-2]["parts"][0]["arguments"]
    assert "story_id" not in second_messages[-2]["parts"][0]["arguments"]


async def test_entity_tool_emits_a_story_reference(owned_story):
    provider = FakeProvider(
        tool_round(
            "get_story_entity",
            '{"kind":"character","entityId":"char-1"}',
        ),
        FINAL_ROUND,
    )
    events = await collect(provider)
    references = [event for event in events if event.type == "reference.emitted"]
    assert len(references) == 1
    assert references[0].part.source_id == "char-1"


async def test_model_call_ceiling_is_a_hard_guard(owned_story):
    provider = FakeProvider(tool_round())
    events = await collect(provider, limits=RunLimits(max_model_calls=2))
    assert len(provider.requests) == 2
    assert events[-1].type == "run.completed"
    assert events[-1].finish_reason == "max_steps"


async def test_tool_call_ceiling_stops_before_execution(owned_story):
    two_calls = [
        {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 0,
                "tool_call_id": "call-1",
                "name": "get_story_overview",
                "arguments_delta": "{}",
            },
        },
        {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 1,
                "tool_call_id": "call-2",
                "name": "get_story_overview",
                "arguments_delta": "{}",
            },
        },
        {"type": "done", "finish_reason": "tool_calls"},
    ]
    provider = FakeProvider(two_calls)
    events = await collect(provider, limits=RunLimits(max_tool_calls=1))
    assert len(provider.requests) == 1
    assert not any(event.type == "tool.completed" for event in events)
    assert events[-1].finish_reason == "max_steps"


async def test_output_ceiling_is_forwarded_on_every_model_call(owned_story):
    provider = FakeProvider(tool_round(), FINAL_ROUND)
    await collect(provider, limits=RunLimits(max_output_tokens=321))
    assert [request["max_output_tokens"] for request in provider.requests] == [321, 321]


async def test_wall_clock_ceiling_cancels_a_hanging_model_call(owned_story):
    class HangingProvider:
        calls = 0

        async def chat_stream(self, *_args, **_kwargs):
            self.calls += 1
            await asyncio.sleep(60)
            if False:
                yield {}

    provider = HangingProvider()
    events = await collect(provider, limits=RunLimits(timeout_seconds=0.01))
    assert provider.calls == 1
    assert events[-1].type == "run.completed"
    assert events[-1].finish_reason == "max_steps"


async def test_provider_error_body_is_replaced_with_safe_message(owned_story):
    provider = FakeProvider(
        [
            {
                "type": "error",
                "error": {
                    "code": "provider_error",
                    "message": "secret provider body",
                },
            }
        ]
    )
    events = await collect(provider)
    assert events[-1].type == "run.failed"
    assert events[-1].code.value == "provider_error"
    assert "secret" not in events[-1].message


async def test_platform_credit_refusal_is_a_safe_quota_failure(owned_story):
    class QuotaProvider:
        async def chat_stream(self, *_args, **_kwargs):
            raise InsufficientCreditsError("sensitive gateway response")
            if False:
                yield {}

    events = await collect(QuotaProvider())
    assert events[-1].type == "run.failed"
    assert events[-1].code.value == "quota_exceeded"
    assert "sensitive" not in events[-1].message
