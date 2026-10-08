"""Specialist consults: context isolation, the shared budget, failure handling."""

import asyncio
import contextvars
import json

import pytest

from agents.storyAgent.llm_provider import RateLimitedError
from assistant.events import validate_event_sequence
from assistant.protocol import AgentRunRequest
from assistant.run import RunLimits, _model_tools, run_assistant
from assistant.specialists.base import SpecialistId
from assistant.specialists.budget import RunBudget
from assistant.specialists.context_builder import ConsultRejected, build_context
from assistant.specialists.registry import SPECIALISTS
from assistant.tools import (
    ConsultSpecialistArgs,
    FocusRef,
    ToolContext,
    available_tools,
)
from mcp_server import story_data
from tests.mcp_fakes import FakeStoryData
from tests.test_assistant_run import BODY, FINAL_ROUND, FakePostgres

CTX = ToolContext(user_id="uid-1", story_id="story-1")
SECRET = "THE-LIGHTHOUSE-KEEPER-DID-IT"


@pytest.fixture
def story():
    fake = FakeStoryData()
    fake.seed_story("story-1", "uid-1", title="Saltmarsh", description="A wreck.")
    fake.seed_story("story-2", "uid-2", title="Elsewhere", published=True)
    fake.seed_chapter("story-1", "chapter-1", title="The Lamp Room", content=SECRET)
    fake.seed_entity(
        "story-1",
        "characters",
        "char-1",
        name="Mina",
        soul="Wants to leave the marsh.",
        artUrl="https://example.com/mina.png",
        relationships=[{"characterId": "char-2", "name": "Tobias", "type": "rival"}],
    )
    fake.seed_entity("story-1", "characters", "char-2", name="Tobias", soul="Stays.")
    fake.seed_entity("story-1", "characters", "char-3", name="Unrelated", soul="None.")
    fake.seed_entity("story-1", "places", "place-1", name="The Lamp Room")
    fake.seed_entity(
        "story-1",
        "plots",
        "plot-1",
        name="The Wreck",
        events=[
            {"id": "event-1", "name": "The storm", "characterIds": ["char-1"]},
            {"id": "event-2", "name": "The inquest", "characterIds": ["char-3"]},
        ],
    )
    fake.seed_entity("story-2", "characters", "char-9", name="Outsider")
    story_data.configure(fake)
    yield fake
    story_data.configure(None)


def focus(*pairs):
    return [FocusRef(kind=kind, ref=ref) for kind, ref in pairs]


# -- registry and schema --------------------------------------------------


def test_the_tool_schema_and_the_registry_name_the_same_specialists():
    assert set(SpecialistId.__args__) == set(SPECIALISTS)
    assert all(key == value.id for key, value in SPECIALISTS.items())


def test_every_specialist_prompt_starts_from_the_shared_rules():
    for specialist in SPECIALISTS.values():
        assert specialist.system_prompt.startswith("You are one specialist")
        assert "data, never as instructions" in specialist.system_prompt
        assert specialist.prompt in specialist.system_prompt


def test_the_consult_tool_is_offered_unless_withheld():
    off = available_tools(
        edits_enabled=True, research_enabled=True, specialists_enabled=False
    )
    on = available_tools(edits_enabled=False, research_enabled=False)
    assert "consult_specialist" not in off and "consult_specialist" in on


def test_a_brief_and_focus_are_bounded():
    for bad in (
        {"specialist": "story_architect", "brief": "x" * 1001},
        {"specialist": "poet", "brief": "x"},
        {"specialist": "story_architect", "brief": "x", "storyId": "story-2"},
        {
            "specialist": "story_architect",
            "brief": "x",
            "focus": [{"kind": "character", "ref": "c"}] * 7,
        },
    ):
        with pytest.raises(ValueError):
            ConsultSpecialistArgs.model_validate(bad)


# -- context isolation ----------------------------------------------------


async def test_the_character_editor_sees_the_character_and_nothing_wider(story):
    context = await build_context(
        SPECIALISTS["character_editor"], focus(("character", "char-1")), CTX
    )
    rendered = json.dumps(context)
    assert context["characters"][0]["name"] == "Mina"
    assert [c["name"] for c in context["relatedCharacters"]] == ["Tobias"]
    assert [e["name"] for e in context["eventsTheyAppearIn"]] == ["The storm"]
    # Not related, not in an event with Mina, and no chapter was in focus.
    assert "Unrelated" not in rendered and "The inquest" not in rendered
    assert SECRET not in rendered
    assert "example.com" not in rendered


async def test_the_architect_sees_structure_but_not_prose_unless_focused(story):
    architect = SPECIALISTS["story_architect"]
    context = await build_context(architect, [], CTX)
    assert [e["name"] for e in context["plotLines"][0]["events"]] == [
        "The storm",
        "The inquest",
    ]
    assert {c["name"] for c in context["characters"]} == {"Mina", "Tobias", "Unrelated"}
    assert SECRET not in json.dumps(context)

    with_chapter = await build_context(architect, focus(("chapter", "chapter-1")), CTX)
    assert with_chapter["chapters"][0]["text"] == SECRET


async def test_context_is_clipped_to_the_specialists_ceiling(story):
    story.entities[("story-1", "plots")][0]["events"] = [
        {"id": f"event-{n}", "name": "Beat", "content": "x" * 4000} for n in range(40)
    ]
    architect = SPECIALISTS["story_architect"]
    context = await build_context(architect, [], CTX)
    assert len(json.dumps(context, separators=(",", ":"))) <= architect.context_chars
    assert context["truncated"] is True


async def test_large_context_keeps_focused_event_chapter_and_review(story):
    events = [
        {
            "id": f"event-{n}",
            "name": f"Beat {n}",
            "content": "Background action. " * 200,
            "characterIds": ["char-1"],
            "orderIndex": n,
            "tensionLevel": 5,
            "pacing": "moderate",
            "storyBeat": "rising_action",
        }
        for n in range(200)
    ]
    events[-1]["content"] = "Mina discovers the wreck's cause. " * 40
    story.entities[("story-1", "plots")][0]["events"] = events
    chapter = story.chapters["story-1"][0]
    chapter["content"] = "The lamp flickered. " * 300
    prior = [
        {
            "specialist": "Character Editor",
            "findings": {"analysis": "Mina needs a motive. " * 70},
        }
    ]
    critic = SPECIALISTS["critic"]
    refs = focus(("event", "event-199"), ("chapter", "chapter-1"))
    context = await build_context(critic, refs, CTX, prior)
    assert len(json.dumps(context)) <= critic.context_chars
    assert context["truncated"] is True
    assert context["focusEvents"][0]["id"] == "event-199"
    assert context["focusEvents"][0]["name"] == "Beat 199"
    assert context["focusEvents"][0]["content"] == events[-1]["content"]
    assert context["chapters"][0]["text"] == chapter["content"]
    assert context["priorFindings"] == prior

    # When focus itself overflows, shorten prose while retaining its identity.
    events[-1]["content"] *= 20
    context = await build_context(critic, refs, CTX, prior)
    assert len(json.dumps(context)) <= critic.context_chars
    assert context["focusEvents"][0]["id"] == "event-199"
    assert context["focusEvents"][0]["name"] == "Beat 199"
    assert context["focusEvents"][0]["content"].startswith("Mina discovers")
    assert context["chapters"][0]["id"] == "chapter-1"
    assert context["priorFindings"][0]["specialist"] == "Character Editor"


@pytest.mark.parametrize("ref", ["char-1", "Mina", "  mina "])
async def test_focus_resolves_an_id_or_an_exact_name(story, ref):
    context = await build_context(
        SPECIALISTS["character_editor"], focus(("character", ref)), CTX
    )
    assert [c["id"] for c in context["characters"]] == ["char-1"]


@pytest.mark.parametrize("ref", ["chapter-1", "the lamp room", "1"])
async def test_a_chapter_resolves_by_id_title_or_number(story, ref):
    context = await build_context(
        SPECIALISTS["story_architect"], focus(("chapter", ref)), CTX
    )
    assert context["chapters"][0]["text"] == SECRET


@pytest.mark.parametrize(
    "refs",
    [
        [],
        [("character", "char-404")],
        # Another owner's character is indistinguishable from a missing one.
        [("character", "char-9")],
        [("character", "Outsider")],
    ],
)
async def test_without_a_named_character_the_editor_gets_the_whole_cast(story, refs):
    context = await build_context(
        SPECIALISTS["character_editor"],
        focus(*refs),
        CTX,
        brief="How could I introduce a new character?",
    )
    assert {c["name"] for c in context["characters"]} == {"Mina", "Tobias", "Unrelated"}
    assert "No single character was named" in context["castNote"]
    # Only the caller's own echoed reference may mention it, never a record.
    assert "char-9" not in json.dumps({**context, "focusNotFound": []})
    assert ("focusNotFound" in context) == bool(refs)


async def test_an_ambiguous_name_asks_for_an_id(story):
    story.seed_entity("story-1", "characters", "char-7", name="MINA")
    with pytest.raises(ConsultRejected, match="More than one character"):
        await build_context(
            SPECIALISTS["character_editor"], focus(("character", "Mina")), CTX
        )


async def test_a_story_with_no_characters_still_gets_an_answer(story):
    story.entities[("story-1", "characters")] = []
    context = await build_context(SPECIALISTS["character_editor"], [], CTX)
    assert context["characters"] == []
    assert "no characters recorded yet" in context["castNote"]


async def test_a_wrong_optional_focus_narrows_the_consult_instead_of_stopping_it(
    story,
):
    context = await build_context(
        SPECIALISTS["story_architect"],
        focus(("plot", "central-plot"), ("chapter", "chapter-404")),
        CTX,
    )
    assert context["focusNotFound"] == [
        {"kind": "plot", "ref": "central-plot"},
        {"kind": "chapter", "ref": "chapter-404"},
    ]
    assert context["plotLines"] and "chapters" not in context


async def test_context_is_refused_for_a_story_the_caller_does_not_own(story):
    from mcp_server import data

    with pytest.raises(data.StoryNotFoundError):
        await build_context(
            SPECIALISTS["story_architect"],
            [],
            ToolContext(user_id="uid-1", story_id="story-2"),
        )


# -- budget ---------------------------------------------------------------


def test_the_budget_always_holds_one_call_back_for_the_director():
    budget = RunBudget(max_model_calls=3, max_consults=5)
    assert budget.take_director_call()  # the step that asks for consults
    assert budget.take_consult()
    assert not budget.take_consult()  # would leave nothing to answer with
    assert budget.take_director_call()
    assert not budget.take_director_call()


def test_a_refunded_consult_can_be_used_again():
    budget = RunBudget(max_model_calls=8, max_consults=1)
    assert budget.take_consult()
    budget.refund_consult()
    assert budget.take_consult() and not budget.take_consult()


# -- run loop -------------------------------------------------------------

FINDINGS = {
    "analysis": "Three events sit at the same tension, so nothing escalates.",
    "recommendations": [{"title": "Raise the inquest", "detail": "Make it cost Mina."}],
    "suggestedChanges": [
        {
            "operation": "event.update",
            "target": "The inquest",
            "change": "Raise tension to 8.",
        }
    ],
    "risks": ["May crowd the ending."],
    "confidence": 0.7,
}

USAGE = {
    "type": "usage",
    "provider": "mock",
    "model": "mock-1",
    "usage": {"prompt_tokens": 900, "completion_tokens": 120},
    "credits": 11,
}


def consult_step(*consults):
    """A director step requesting one or more consults at once."""
    events = []
    for index, (call_id, arguments) in enumerate(consults):
        events.append(
            {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "tool_call_id": call_id,
                    "name": "consult_specialist",
                    "arguments_delta": json.dumps(arguments),
                },
            }
        )
    return [*events, USAGE, {"type": "done", "finish_reason": "tool_calls"}]


def findings_reply(payload=FINDINGS, text=""):
    reply = []
    if text:
        reply.append({"type": "text_delta", "text": text})
    if payload is not None:
        reply.append(
            {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "tool_call_id": "sub-1",
                    "name": "submit_findings",
                    "arguments_delta": (
                        payload if isinstance(payload, str) else json.dumps(payload)
                    ),
                },
            }
        )
    return [*reply, USAGE, {"type": "done", "finish_reason": "tool_calls"}]


ARCHITECT = {"specialist": "story_architect", "brief": "Why does the middle sag?"}
EDITOR = {
    "specialist": "character_editor",
    "brief": "Is Mina consistent?",
    "focus": [{"kind": "character", "ref": "Mina"}],
}


class RoutingProvider:
    """Answers the director from a script and each specialist by its role."""

    def __init__(self, director, specialists=None):
        self.director = list(director)
        self.specialists = specialists or {}
        self.requests = []
        self.active = 0
        self.peak = 0

    async def chat_stream(
        self, messages, tools, *, max_output_tokens, idempotency_key, required_tool=None
    ):
        request = {
            "messages": messages,
            "tools": [tool["name"] for tool in tools],
            "idempotency_key": idempotency_key,
            "required_tool": required_tool,
            "max_output_tokens": max_output_tokens,
        }
        self.requests.append(request)
        system = messages[0]["parts"][0]["text"]
        if required_tool != "submit_findings":
            script = (
                self.director.pop(0) if len(self.director) > 1 else self.director[0]
            )
        else:
            key = next(
                name
                for name in ("Story Architect", "Character Editor")
                if name in system
            )
            script = self.specialists.get(key, findings_reply())
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
        if isinstance(script, BaseException):
            raise script
        for event in script:
            yield event

    def specialist_requests(self):
        return [r for r in self.requests if r["required_tool"] == "submit_findings"]


async def collect(provider, *, limits=None, enabled=True, **kwargs):
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
            specialists_enabled=enabled,
            **kwargs,
        )
    ]


def completed(events, call_id):
    return next(
        event.part
        for event in events
        if event.type == "tool.completed" and event.part.tool_call_id == call_id
    )


async def test_a_consult_returns_findings_and_the_director_answers(story):
    provider = RoutingProvider([consult_step(("call-1", ARCHITECT)), FINAL_ROUND])
    events = await collect(provider)
    validate_event_sequence(events)

    part = completed(events, "call-1")
    assert part.result["specialist"] == "story_architect"
    assert part.result["degraded"] is False
    assert part.result["findings"]["suggestedChanges"][0]["target"] == "The inquest"
    assert events[-1].type == "run.completed" and events[-1].finish_reason == "stop"
    # Director step, the consult, the director's answer: each billed once.
    assert [e.type for e in events].count("usage") == 3

    (sub,) = provider.specialist_requests()
    assert sub["tools"] == ["submit_findings"]
    assert sub["idempotency_key"].startswith("run-1:consult:")
    # The findings re-enter the director's prompt as a tool result, i.e. data.
    tool_message = provider.requests[-1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "same tension" in tool_message["parts"][0]["text"]


async def test_a_specialist_gets_no_history_tools_or_director_prompt(story):
    provider = RoutingProvider([consult_step(("call-1", EDITOR)), FINAL_ROUND])
    await collect(provider)
    (sub,) = provider.specialist_requests()
    assert [m["role"] for m in sub["messages"]] == ["system", "user"]
    rendered = json.dumps(sub["messages"])
    assert "consult_specialist" not in rendered
    assert "How many chapters?" not in rendered  # the writer's message
    assert "Unrelated" not in rendered and SECRET not in rendered
    assert "Is Mina consistent?" in rendered


async def test_consults_requested_together_run_in_parallel(story):
    provider = RoutingProvider(
        [consult_step(("call-1", ARCHITECT), ("call-2", EDITOR)), FINAL_ROUND]
    )
    events = await collect(provider)
    assert provider.peak == 2
    assert completed(events, "call-1").result["specialist"] == "story_architect"
    assert completed(events, "call-2").result["specialist"] == "character_editor"
    keys = [r["idempotency_key"] for r in provider.requests]
    assert len(set(keys)) == len(keys)


async def test_consults_beyond_the_cap_are_declined_not_run(story):
    three = consult_step(
        ("call-1", ARCHITECT),
        ("call-2", EDITOR),
        ("call-3", {"specialist": "critic", "brief": "And an editorial read?"}),
    )
    provider = RoutingProvider([three, FINAL_ROUND])
    events = await collect(provider)
    assert len(provider.specialist_requests()) == 2
    declined = completed(events, "call-3").result
    assert (
        declined["accepted"] is False and "No consults are left" in declined["reason"]
    )
    assert events[-1].finish_reason == "stop"


async def test_a_consult_cannot_take_the_directors_last_call(story):
    provider = RoutingProvider([consult_step(("call-1", ARCHITECT)), FINAL_ROUND])
    events = await collect(provider, limits=RunLimits(max_model_calls=2))
    assert provider.specialist_requests() == []
    assert completed(events, "call-1").result["accepted"] is False
    assert events[-1].type == "run.completed" and events[-1].finish_reason == "stop"


async def test_a_declined_consult_does_not_cost_one(story):
    bad = {
        "specialist": "drafter",
        "brief": "Write the confrontation.",
        "focus": [{"kind": "event", "ref": "The confrontation"}],
    }
    provider = CastProvider(
        [
            consult_step(("call-1", bad)),
            consult_step(("call-2", EDITOR), ("call-3", ARCHITECT)),
            FINAL_ROUND,
        ]
    )
    events = await collect(provider)
    reason = completed(events, "call-1").result["reason"]
    # The refusal names what does exist, so the retry needs no lookup.
    assert "is called 'The confrontation'" in reason and "The storm" in reason
    assert provider.calls_to("Drafting Agent") == []
    # Both consults in the next step still fit under the cap of two.
    assert len(provider.calls_to("Character Editor")) == 1
    assert len(provider.calls_to("Story Architect")) == 1


async def test_one_specialist_failing_leaves_the_other_and_the_answer(story):
    provider = RoutingProvider(
        [consult_step(("call-1", ARCHITECT), ("call-2", EDITOR)), FINAL_ROUND],
        specialists={"Story Architect": RateLimitedError("slow down")},
    )
    events = await collect(provider)
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.tool_call_id == "call-1" and failed.code.value == "rate_limited"
    assert completed(events, "call-2").result["accepted"] is True
    assert events[-1].type == "run.completed" and events[-1].finish_reason == "stop"
    assert "run.failed" not in [event.type for event in events]


async def test_malformed_specialist_ids_fail_only_the_tool(story):
    specialist = ["critic"]
    provider = RoutingProvider(
        [
            consult_step(
                ("call-bad", {"specialist": specialist, "brief": "Assess this."}),
                ("call-good", EDITOR),
            ),
            FINAL_ROUND,
        ]
    )
    events = await collect(provider)
    validate_event_sequence(events)
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.tool_call_id == "call-bad"
    assert failed.code.value == "provider_error"
    assert completed(events, "call-good").result["accepted"] is True
    assert len(provider.specialist_requests()) == 1
    assert events[-1].type == "run.completed"


async def test_malformed_findings_lists_fail_only_the_consult(story):
    provider = RoutingProvider(
        [consult_step(("call-bad", ARCHITECT), ("call-good", EDITOR)), FINAL_ROUND],
        specialists={
            "Story Architect": findings_reply(
                {"analysis": "", "recommendations": 42, "risks": True}
            )
        },
    )
    events = await collect(provider)
    validate_event_sequence(events)
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.tool_call_id == "call-bad"
    assert failed.code.value == "provider_error"
    assert completed(events, "call-good").result["accepted"] is True
    # Malformed output still incurred a model call: preserve all four usage events.
    assert sum(event.type == "usage" for event in events) == 4
    assert len(provider.specialist_requests()) == 2
    assert events[-1].type == "run.completed"


async def test_unexpected_specialist_errors_preserve_siblings_usage_and_privacy(
    story, caplog
):
    class Broken(RoutingProvider):
        async def chat_stream(self, messages, tools, **kwargs):
            if (
                kwargs.get("required_tool") == "submit_findings"
                and "Story Architect" in messages[0]["parts"][0]["text"]
            ):
                yield USAGE
                raise RuntimeError(SECRET)
            async for event in super().chat_stream(messages, tools, **kwargs):
                yield event

    provider = Broken(
        [consult_step(("call-bad", ARCHITECT), ("call-good", EDITOR)), FINAL_ROUND]
    )
    events = await collect(provider)
    validate_event_sequence(events)
    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.tool_call_id == "call-bad" and failed.code.value == "internal_error"
    assert completed(events, "call-good").result["accepted"] is True
    assert sum(event.type == "usage" for event in events) == 4
    assert events[-1].type == "run.completed"
    assert "error_type=RuntimeError" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in json.dumps([event.model_dump(mode="json") for event in events])


async def test_a_slow_specialist_times_out_without_failing_the_run(story):
    class Slow(RoutingProvider):
        async def chat_stream(self, messages, tools, **kwargs):
            if kwargs.get("required_tool") == "submit_findings":
                await asyncio.sleep(5)
            async for event in super().chat_stream(messages, tools, **kwargs):
                yield event

    provider = Slow([consult_step(("call-1", ARCHITECT)), FINAL_ROUND])
    events = await collect(provider, limits=RunLimits(specialist_timeout_seconds=0.05))
    assert any(event.type == "tool.failed" for event in events)
    assert events[-1].finish_reason == "stop"


@pytest.mark.parametrize(
    "reply, analysis",
    [
        (
            findings_reply('{"analysis": 5, "extra": true}', text="Plain words."),
            "Plain words.",
        ),
        (findings_reply('{"analysis": "Kept.", "confidence": 9}'), "Kept."),
        (
            findings_reply({"analysis": "Kept.", "recommendations": 42, "risks": True}),
            "Kept.",
        ),
        (findings_reply("{not json", text="Still useful."), "Still useful."),
    ],
)
async def test_an_answer_in_the_wrong_shape_degrades_to_plain_analysis(
    story, reply, analysis
):
    provider = RoutingProvider(
        [consult_step(("call-1", ARCHITECT)), FINAL_ROUND],
        specialists={"Story Architect": reply},
    )
    events = await collect(provider)
    result = completed(events, "call-1").result
    assert result["degraded"] is True
    assert result["findings"] == {"analysis": analysis}
    assert len(provider.specialist_requests()) == 1  # no repair round


@pytest.mark.parametrize(
    "reply",
    [
        findings_reply(None),
        [
            {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "name": "apply_story_changes",
                    "arguments_delta": json.dumps(FINDINGS),
                },
            },
            USAGE,
            {"type": "done", "finish_reason": "tool_calls"},
        ],
    ],
    ids=["empty", "forbidden-tool"],
)
async def test_an_unusable_answer_is_a_failed_consult(story, reply):
    provider = RoutingProvider(
        [consult_step(("call-1", ARCHITECT)), FINAL_ROUND],
        specialists={"Story Architect": reply},
    )
    events = await collect(provider)
    assert any(event.type == "tool.failed" for event in events)
    # The call was made, so it is still billed.
    assert [e.type for e in events].count("usage") == 3


async def test_the_tool_is_unknown_and_unadvertised_when_withheld(story):
    provider = RoutingProvider([consult_step(("call-1", ARCHITECT)), FINAL_ROUND])
    events = await collect(provider, enabled=False)
    assert any(event.type == "tool.failed" for event in events)
    assert provider.specialist_requests() == []
    assert "consult_specialist" not in provider.requests[0]["tools"]
    system = provider.requests[0]["messages"][0]["parts"][0]["text"]
    assert "writers' room" not in system


async def test_the_director_is_told_who_it_can_consult(story):
    provider = RoutingProvider([FINAL_ROUND])
    events = await collect(provider)
    system = provider.requests[0]["messages"][0]["parts"][0]["text"]
    assert "- story_architect:" in system and "- character_editor:" in system
    assert "at most 2 consults" in system
    # A plain question still makes no consult.
    assert provider.specialist_requests() == []
    assert not any(event.type == "tool.started" for event in events)


async def test_request_scoped_context_reaches_parallel_consults(story):
    """BYOK and the Firebase token ride ContextVars; gather must not drop them."""
    caller = contextvars.ContextVar("caller", default="unset")
    seen = []

    class Recording(RoutingProvider):
        async def chat_stream(self, messages, tools, **kwargs):
            seen.append(caller.get())
            async for event in super().chat_stream(messages, tools, **kwargs):
                yield event

    provider = Recording(
        [consult_step(("call-1", ARCHITECT), ("call-2", EDITOR)), FINAL_ROUND]
    )
    caller.set("uid-1-byok")
    await collect(provider)
    assert seen == ["uid-1-byok"] * 4


def test_the_submit_schema_reaches_the_provider_without_null_types():
    from assistant.run import _without_null_branches
    from assistant.specialists.runner import _submit_tool

    (tool,) = _submit_tool(_without_null_branches)
    assert tool["name"] == "submit_findings"
    rendered = json.dumps(tool)
    assert '"type": "null"' not in rendered
    # A forced tool call is compiled into a decoding constraint, and a provider
    # rejected the bounded, nested version as having too many states.
    for keyword in ("maxLength", "minLength", "maxItems", "maximum", "minimum"):
        assert keyword not in rendered
    assert len(rendered) < 2500
    properties = tool["parameters"]["properties"]
    assert set(properties) == {
        "analysis",
        "recommendations",
        "suggestedChanges",
        "risks",
        "confidence",
    }
    assert tool["parameters"]["required"] == ["analysis"]
    names = {tool["name"] for tool in _model_tools(edits_enabled=False)}
    assert "submit_findings" not in names


# -- critic, review round and drafting ------------------------------------

CRITIC = {"specialist": "critic", "brief": "Would that actually fix the sag?"}
DRAFT = {
    "specialist": "drafter",
    "brief": "Write the storm from Mina's point of view, tense and spare.",
    "focus": [{"kind": "event", "ref": "The storm"}],
}
PROSE = ["The lamp guttered. ", "Mina counted the seconds between the waves."]


def draft_reply(chunks=PROSE, finish="stop"):
    return [
        *({"type": "text_delta", "text": chunk} for chunk in chunks),
        USAGE,
        {"type": "done", "finish_reason": finish},
    ]


class CastProvider(RoutingProvider):
    """Routes each sub-call by the specialist named in its system prompt."""

    NAMES = ("Story Architect", "Character Editor", "Critic", "Drafting Agent")

    async def chat_stream(
        self, messages, tools, *, max_output_tokens, idempotency_key, required_tool=None
    ):
        system = messages[0]["parts"][0]["text"]
        name = next((n for n in self.NAMES if f"You are the {n}" in system), None)
        request = {
            "messages": messages,
            "tools": [tool["name"] for tool in tools],
            "idempotency_key": idempotency_key,
            "required_tool": required_tool,
            "max_output_tokens": max_output_tokens,
            "specialist": name,
        }
        self.requests.append(request)
        if name is None:
            script = (
                self.director.pop(0) if len(self.director) > 1 else self.director[0]
            )
        else:
            default = draft_reply() if name == "Drafting Agent" else findings_reply()
            script = self.specialists.get(name, default)
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
        if isinstance(script, BaseException):
            raise script
        for event in script:
            yield event

    def calls_to(self, name):
        return [r for r in self.requests if r["specialist"] == name]


def material(request):
    return request["messages"][1]["parts"][0]["text"]


def test_the_roster_covers_four_specialists_with_two_modes():
    assert set(SPECIALISTS) == {
        "story_architect",
        "character_editor",
        "critic",
        "drafter",
    }
    assert [s.id for s in SPECIALISTS.values() if s.mode == "draft"] == ["drafter"]
    # A drafter writes for the writer, so it is not told to submit findings.
    assert "submit_findings" not in SPECIALISTS["drafter"].system_prompt
    assert "submit_findings" in SPECIALISTS["critic"].system_prompt


async def test_a_review_is_shown_what_colleagues_found_in_the_same_step(story):
    step = consult_step(("call-1", ARCHITECT), ("call-2", {**CRITIC, "review": True}))
    provider = CastProvider(
        [step, FINAL_ROUND],
        specialists={
            "Critic": findings_reply({"analysis": "Raising tension alone is cosmetic."})
        },
    )
    events = await collect(provider)

    (critic_call,) = provider.calls_to("Critic")
    # Injected by the server from the architect's validated answer, verbatim.
    assert "priorFindings" in material(critic_call)
    assert "same tension" in material(critic_call)
    assert "priorFindings" not in material(provider.calls_to("Story Architect")[0])
    # The review waited for the first wave rather than racing it.
    assert provider.peak == 1
    assert completed(events, "call-2").result["reviewed"] is True
    assert completed(events, "call-1").result["reviewed"] is False
    # Both views reach the director, disagreement intact.
    final = json.dumps(provider.requests[-1]["messages"])
    assert "same tension" in final and "cosmetic" in final


async def test_a_review_in_a_later_step_sees_earlier_findings(story):
    provider = CastProvider(
        [
            consult_step(("call-1", ARCHITECT)),
            consult_step(("call-2", {**CRITIC, "review": True})),
            FINAL_ROUND,
        ]
    )
    await collect(provider)
    assert "same tension" in material(provider.calls_to("Critic")[0])


async def test_only_one_review_round_is_allowed(story):
    provider = CastProvider(
        [
            consult_step(("call-1", ARCHITECT)),
            consult_step(("call-2", {**CRITIC, "review": True})),
            consult_step(("call-3", {**ARCHITECT, "review": True})),
            FINAL_ROUND,
        ]
    )
    events = await collect(provider, limits=RunLimits(max_consults=4))
    assert len(provider.calls_to("Critic")) == 1
    assert len(provider.calls_to("Story Architect")) == 1
    declined = completed(events, "call-3").result
    assert declined["accepted"] is False and "one review round" in declined["reason"]


async def test_a_review_with_nothing_to_review_is_a_plain_consult(story):
    provider = CastProvider(
        [consult_step(("call-1", {**CRITIC, "review": True})), FINAL_ROUND]
    )
    events = await collect(provider)
    assert "priorFindings" not in material(provider.calls_to("Critic")[0])
    assert completed(events, "call-1").result["reviewed"] is False
    # The round was not spent, so a real review is still possible later.
    budget = RunBudget(max_model_calls=8, max_consults=2)
    assert budget.take_critique() and not budget.take_critique()


async def test_a_draft_streams_to_the_writer_and_ends_the_run(story):
    provider = CastProvider([consult_step(("call-1", DRAFT)), FINAL_ROUND])
    events = await collect(provider)
    validate_event_sequence(events)

    deltas = [event.text for event in events if event.type == "text.delta"]
    assert deltas == PROSE
    done = next(event for event in events if event.type == "text.done")
    assert done.part.text == "".join(PROSE)
    # The director gets a receipt, never the prose, and no further turn.
    receipt = completed(events, "call-1").result
    assert receipt == {
        "accepted": True,
        "specialist": "drafter",
        "name": "Drafting Agent",
        "delivered": True,
        "words": 10,
        "truncated": False,
    }
    assert events[-1].type == "run.completed" and events[-1].finish_reason == "stop"
    assert [r["specialist"] for r in provider.requests] == [None, "Drafting Agent"]
    (draft_call,) = provider.calls_to("Drafting Agent")
    assert draft_call["tools"] == [] and draft_call["required_tool"] is None
    assert draft_call["max_output_tokens"] == SPECIALISTS["drafter"].max_output_tokens


async def test_a_draft_sees_the_scene_and_not_the_rest_of_the_plot(story):
    story.entities[("story-1", "plots")][0]["events"] = [
        {"id": "event-0", "name": "The warning", "orderIndex": 0, "characterIds": []},
        {
            "id": "event-1",
            "name": "The storm",
            "orderIndex": 1,
            "characterIds": ["char-1"],
            "locationId": "place-1",
        },
        {"id": "event-2", "name": "The inquest", "orderIndex": 2, "characterIds": []},
    ]
    context = await build_context(
        SPECIALISTS["drafter"], focus(("event", "the storm")), CTX
    )
    assert [e["name"] for e in context["events"]] == ["The storm"]
    assert [c["name"] for c in context["charactersInScene"]] == ["Mina"]
    assert [p["name"] for p in context["setting"]] == ["The Lamp Room"]
    assert [e["name"] for e in context["whatCameBefore"]] == ["The warning"]
    # What happens next is exactly what a drafter must not be handed.
    assert "The inquest" not in json.dumps(context)
    assert "plotLines" not in context


@pytest.mark.parametrize(
    "refs, fragment",
    [
        ([], "needs an event or a chapter in focus"),
        ([("event", "The confrontation")], "is called 'The confrontation'"),
        ([("character", "Mina")], "needs an event or a chapter in focus"),
    ],
)
async def test_a_draft_without_an_anchor_is_refused_with_what_exists(
    story, refs, fragment
):
    with pytest.raises(ConsultRejected, match=fragment) as excinfo:
        await build_context(SPECIALISTS["drafter"], focus(*refs), CTX)
    assert "events: The storm, The inquest" in str(excinfo.value)
    assert "chapters: The Lamp Room" in str(excinfo.value)


async def test_a_chapter_alone_can_anchor_a_rewrite(story):
    context = await build_context(
        SPECIALISTS["drafter"], focus(("chapter", "The Lamp Room")), CTX
    )
    assert context["chapters"][0]["text"] == SECRET


async def test_a_refused_draft_returns_to_the_director_without_spending(story):
    bad = {**DRAFT, "focus": [{"kind": "event", "ref": "The confrontation"}]}
    provider = CastProvider([consult_step(("call-1", bad)), FINAL_ROUND])
    events = await collect(provider)
    assert provider.calls_to("Drafting Agent") == []
    assert completed(events, "call-1").result["accepted"] is False
    assert events[-1].finish_reason == "stop"
    assert len(provider.requests) == 2  # the director explains instead


async def test_a_draft_alongside_another_call_is_declined(story):
    provider = CastProvider(
        [consult_step(("call-1", DRAFT), ("call-2", ARCHITECT)), FINAL_ROUND]
    )
    events = await collect(provider)
    assert provider.calls_to("Drafting Agent") == []
    assert "alone" in completed(events, "call-1").result["reason"]
    assert completed(events, "call-2").result["accepted"] is True


async def test_a_truncated_draft_is_reported_as_cut_short(story):
    provider = CastProvider(
        [consult_step(("call-1", DRAFT)), FINAL_ROUND],
        specialists={"Drafting Agent": draft_reply(finish="length")},
    )
    events = await collect(provider)
    assert completed(events, "call-1").result["truncated"] is True
    assert events[-1].finish_reason == "length"


async def test_a_draft_follows_the_directors_words_on_a_new_paragraph(story):
    step = [
        {"type": "text_delta", "text": "Here is the storm."},
        *consult_step(("call-1", DRAFT)),
    ]
    events = await collect(CastProvider([step, FINAL_ROUND]))
    settled = [event.part.text for event in events if event.type == "text.done"]
    assert settled == ["Here is the storm.", "\n\n" + "".join(PROSE)]
    streamed = "".join(e.text for e in events if e.type == "text.delta")
    assert streamed == "Here is the storm.\n\n" + "".join(PROSE)


async def test_a_draft_that_fails_midway_keeps_what_was_shown(story):
    broken = [
        {"type": "text_delta", "text": "The lamp guttered. "},
        {"type": "error", "error": {"code": "provider_error"}},
    ]
    provider = CastProvider(
        [consult_step(("call-1", DRAFT)), FINAL_ROUND],
        specialists={"Drafting Agent": broken},
    )
    events = await collect(provider)
    types = [event.type for event in events]
    assert "tool.failed" in types and "run.failed" not in types
    assert any(
        event.type == "text.done" and event.part.text == "The lamp guttered. "
        for event in events
    )
    # The director still gets a turn to say what happened.
    assert events[-1].type == "run.completed" and len(provider.requests) == 3


async def test_an_empty_draft_is_a_failed_consult(story):
    provider = CastProvider(
        [consult_step(("call-1", DRAFT)), FINAL_ROUND],
        specialists={"Drafting Agent": draft_reply(chunks=[])},
    )
    events = await collect(provider)
    assert any(event.type == "tool.failed" for event in events)
    assert events[-1].finish_reason == "stop"


async def test_the_same_specialist_twice_in_one_step_is_asked_once(story):
    twice = consult_step(("call-1", CRITIC), ("call-2", {**CRITIC, "brief": "Again?"}))
    provider = CastProvider([twice, FINAL_ROUND])
    events = await collect(provider)
    assert len(provider.calls_to("Critic")) == 1
    assert completed(events, "call-1").result["accepted"] is True
    repeat = completed(events, "call-2").result
    assert repeat["accepted"] is False and "already asked" in repeat["reason"]


# -- writers' room mode ---------------------------------------------------

ROOM_STEP = consult_step(
    ("call-1", ARCHITECT),
    ("call-2", EDITOR),
    ("call-3", {**CRITIC, "review": True}),
)


async def collect_room(provider, *, room_enabled=True, mode="room", **kwargs):
    request = AgentRunRequest.model_validate({**BODY, "userId": "uid-1", "mode": mode})
    return [
        event
        async for event in run_assistant(
            request,
            run_id="run-1",
            provider=provider,
            postgres=FakePostgres(),
            embedder=None,
            limits=kwargs.pop("limits", RunLimits()),
            specialists_enabled=kwargs.pop("specialists_enabled", True),
            room_enabled=room_enabled,
            **kwargs,
        )
    ]


async def test_the_room_convenes_three_views_and_one_recommendation(story):
    provider = CastProvider([ROOM_STEP, FINAL_ROUND])
    events = await collect_room(provider)
    validate_event_sequence(events)

    # Past the normal cap of two, and each view is marked for its own card.
    for call_id in ("call-1", "call-2", "call-3"):
        result = completed(events, call_id).result
        assert result["accepted"] is True and result["room"] is True
    assert completed(events, "call-3").result["reviewed"] is True
    assert events[-1].type == "run.completed" and events[-1].finish_reason == "stop"
    # Director, three specialists, director: five model calls for the whole room.
    assert len(provider.requests) == 5

    first = provider.requests[0]
    assert first["required_tool"] == "consult_specialist"
    assert first["tools"] == ["consult_specialist"]
    system = first["messages"][0]["parts"][0]["text"]
    assert "convened the writers' room" in system and "at most 4 consults" in system
    # Only the opening step is forced; the synthesis is free to answer.
    assert provider.requests[-1]["required_tool"] is None


async def test_each_view_is_emitted_as_it_finishes_not_when_all_are_done(story):
    class Staggered(CastProvider):
        async def chat_stream(self, messages, tools, **kwargs):
            system = messages[0]["parts"][0]["text"]
            if "You are the Story Architect" in system:
                await asyncio.sleep(0.08)
            async for event in super().chat_stream(messages, tools, **kwargs):
                yield event

    step = consult_step(("call-1", ARCHITECT), ("call-2", EDITOR))
    events = await collect_room(Staggered([step, FINAL_ROUND]))
    order = [e.part.tool_call_id for e in events if e.type == "tool.completed"]
    # Requested first, finished last.
    assert order == ["call-2", "call-1"]


async def test_the_room_does_not_draft(story):
    step = consult_step(("call-1", ARCHITECT), ("call-2", DRAFT))
    provider = CastProvider([step, FINAL_ROUND])
    events = await collect_room(provider)
    assert provider.calls_to("Drafting Agent") == []
    refused = completed(events, "call-2").result
    assert refused["accepted"] is False and "does not draft" in refused["reason"]
    assert events[-1].finish_reason == "stop"


async def test_room_mode_comes_only_from_the_request_field(story):
    for kwargs in ({"room_enabled": False}, {"mode": None}):
        provider = CastProvider([ROOM_STEP, FINAL_ROUND])
        events = await collect_room(provider, **kwargs)
        # An ordinary run: nothing forced, the normal cap, no room cards.
        assert provider.requests[0]["required_tool"] is None
        assert "convened the writers' room" not in (
            provider.requests[0]["messages"][0]["parts"][0]["text"]
        )
        assert len(provider.specialist_requests()) == 2
        results = [e.part.result for e in events if e.type == "tool.completed"]
        assert not any(result.get("room") for result in results)


async def test_room_mode_is_inert_without_specialists(story):
    provider = CastProvider([FINAL_ROUND])
    events = await collect_room(provider, specialists_enabled=False)
    assert provider.requests[0]["required_tool"] is None
    assert events[-1].type == "run.completed"


async def test_the_room_still_answers_when_the_budget_runs_short(story):
    provider = CastProvider([ROOM_STEP, FINAL_ROUND])
    events = await collect_room(provider, limits=RunLimits(max_model_calls=3))
    # One consult fits beside the director's two calls; the rest are declined.
    assert len(provider.specialist_requests()) == 1
    assert events[-1].type == "run.completed" and events[-1].finish_reason == "stop"


async def test_a_cancelled_room_leaves_no_specialist_running(story):
    started = asyncio.Event()
    finished = []

    class Hanging(CastProvider):
        async def chat_stream(self, messages, tools, **kwargs):
            if kwargs.get("required_tool") == "submit_findings":
                started.set()
                try:
                    await asyncio.sleep(30)
                finally:
                    finished.append("cancelled")
            async for event in super().chat_stream(messages, tools, **kwargs):
                yield event

    provider = Hanging([consult_step(("call-1", ARCHITECT), ("call-2", EDITOR))])
    events = await collect_room(provider, limits=RunLimits(timeout_seconds=0.2))
    assert started.is_set()
    await asyncio.sleep(0)
    assert finished == ["cancelled", "cancelled"]
    assert events[-1].type == "run.completed"


def test_only_a_known_mode_is_accepted():
    from pydantic import ValidationError

    assert AgentRunRequest.model_validate({**BODY, "userId": "u", "mode": "room"})
    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate({**BODY, "userId": "u", "mode": "debate"})


@pytest.mark.parametrize(
    "brief, expected",
    [
        ("Is Mina consistent in the storm?", ["char-1"]),
        ("Why is TOBIAS so passive?", ["char-2"]),
        ("Is the protagonist consistent?", ["char-1", "char-2", "char-3"]),
    ],
)
async def test_a_character_named_only_in_the_brief_is_put_in_focus(
    story, brief, expected
):
    context = await build_context(SPECIALISTS["character_editor"], [], CTX, brief=brief)
    assert sorted(c["id"] for c in context["characters"]) == expected
    assert ("castNote" in context) == (len(expected) > 1)


async def test_a_shared_first_name_is_not_guessed(story):
    story.entities[("story-1", "characters")][0]["name"] = "Mina Stone"
    story.seed_entity("story-1", "characters", "char-7", name="Mina Vale")
    context = await build_context(
        SPECIALISTS["character_editor"], [], CTX, brief="What does Mina want?"
    )
    # Two Minas: neither is picked, so the editor sees the cast and says so.
    assert len(context["characters"]) == 4 and "castNote" in context


async def test_the_rooms_critic_reviews_even_when_not_asked_to(story):
    step = consult_step(("call-1", ARCHITECT), ("call-2", CRITIC))
    provider = CastProvider([step, FINAL_ROUND])
    events = await collect_room(provider)
    assert completed(events, "call-2").result["reviewed"] is True
    assert "priorFindings" in material(provider.calls_to("Critic")[0])
    # Outside the room the model's own choice stands.
    ordinary = CastProvider([step, FINAL_ROUND])
    plain = await collect(ordinary)
    assert completed(plain, "call-2").result["reviewed"] is False


# -- answering well, and always answering ---------------------------------


def test_specialists_are_told_to_advise_on_what_does_not_exist_yet():
    for specialist in SPECIALISTS.values():
        prompt = specialist.system_prompt
        assert "does not exist yet" in prompt
        assert "Never answer that the material" in prompt


async def test_the_room_withholds_the_proposal_tools(story):
    provider = CastProvider([ROOM_STEP, FINAL_ROUND])
    await collect_room(provider, edits_enabled=True)
    synthesis_tools = provider.requests[-1]["tools"]
    assert "consult_specialist" in synthesis_tools
    assert "propose_story_changes" not in synthesis_tools
    # An ordinary run still offers them.
    ordinary = CastProvider([FINAL_ROUND])
    await collect(ordinary)
    assert "propose_story_changes" in ordinary.requests[0]["tools"]


EMPTY_ROUND = [USAGE, {"type": "done", "finish_reason": "stop"}]


async def test_an_empty_reply_after_tool_use_is_asked_for_once_more(story):
    provider = CastProvider(
        [consult_step(("call-1", ARCHITECT)), EMPTY_ROUND, FINAL_ROUND]
    )
    events = await collect(provider)
    director = [r for r in provider.requests if r["specialist"] is None]
    assert len(director) == 3
    nudge = director[-1]["messages"][-1]
    assert nudge["role"] == "user" and "Write your reply" in nudge["parts"][0]["text"]
    texts = [e.part.text for e in events if e.type == "text.done"]
    assert texts == ["Saltmarsh has one chapter."]
    assert events[-1].finish_reason == "stop"


async def test_a_reply_that_stays_empty_says_so_instead_of_nothing(story):
    provider = CastProvider([consult_step(("call-1", ARCHITECT)), EMPTY_ROUND])
    events = await collect(provider)
    texts = [e.part.text for e in events if e.type == "text.done"]
    assert len(texts) == 1 and "could not put a summary together" in texts[0]
    # Asked once, not in a loop.
    assert len([r for r in provider.requests if r["specialist"] is None]) == 3
    assert events[-1].type == "run.completed"


async def test_an_empty_reply_with_no_tool_use_is_left_alone(story):
    provider = CastProvider([EMPTY_ROUND])
    events = await collect(provider)
    assert len(provider.requests) == 1
    assert not any(e.type == "text.done" for e in events)


async def test_findings_without_an_analysis_are_salvaged_from_the_rest(story):
    partial = {
        "analysis": "",
        "recommendations": [{"title": "Raise the inquest", "detail": "Cost Mina."}],
        "risks": ["May crowd the ending."],
    }
    provider = CastProvider(
        [consult_step(("call-1", ARCHITECT)), FINAL_ROUND],
        specialists={"Story Architect": findings_reply(partial)},
    )
    events = await collect(provider)
    result = completed(events, "call-1").result
    assert result["degraded"] is True
    assert "Raise the inquest — Cost Mina." in result["findings"]["analysis"]
    assert "May crowd the ending." in result["findings"]["analysis"]
