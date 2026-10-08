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


def test_the_consult_tool_is_offered_only_behind_its_flag():
    off = available_tools(edits_enabled=True, research_enabled=True)
    on = available_tools(
        edits_enabled=False, research_enabled=False, specialists_enabled=True
    )
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
    "refs, fragment",
    [
        ([], "needs a character in focus. This story's characters: Mina"),
        ([("character", "char-404")], "No character is called 'char-404'"),
        # Another owner's character is indistinguishable from a missing one.
        ([("character", "char-9")], "No character is called 'char-9'"),
        ([("character", "Outsider")], "No character is called 'Outsider'"),
    ],
)
async def test_a_missing_required_focus_is_rejected_with_who_exists(
    story, refs, fragment
):
    with pytest.raises(ConsultRejected, match=fragment):
        await build_context(SPECIALISTS["character_editor"], focus(*refs), CTX)


async def test_an_ambiguous_name_asks_for_an_id(story):
    story.seed_entity("story-1", "characters", "char-7", name="MINA")
    with pytest.raises(ConsultRejected, match="More than one character"):
        await build_context(
            SPECIALISTS["character_editor"], focus(("character", "Mina")), CTX
        )


async def test_a_story_with_no_characters_says_so(story):
    story.entities[("story-1", "characters")] = []
    with pytest.raises(ConsultRejected, match="no characters recorded yet"):
        await build_context(SPECIALISTS["character_editor"], [], CTX)


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
        ("call-1", ARCHITECT), ("call-2", EDITOR), ("call-3", ARCHITECT)
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


async def test_a_bad_focus_is_declined_and_does_not_cost_a_consult(story):
    bad = {**EDITOR, "focus": [{"kind": "character", "ref": "the protagonist"}]}
    provider = RoutingProvider(
        [
            consult_step(("call-1", bad)),
            consult_step(("call-2", EDITOR), ("call-3", ARCHITECT)),
            FINAL_ROUND,
        ]
    )
    events = await collect(provider)
    reason = completed(events, "call-1").result["reason"]
    # The refusal names who does exist, so the retry needs no lookup.
    assert "No character is called 'the protagonist'" in reason
    assert "Mina, Tobias, Unrelated" in reason
    assert len(provider.specialist_requests()) == 2


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


async def test_an_empty_answer_is_a_failed_consult(story):
    provider = RoutingProvider(
        [consult_step(("call-1", ARCHITECT)), FINAL_ROUND],
        specialists={"Story Architect": findings_reply(None)},
    )
    events = await collect(provider)
    assert any(event.type == "tool.failed" for event in events)
    # The call was made, so it is still billed.
    assert [e.type for e in events].count("usage") == 3


async def test_the_tool_is_unknown_and_unadvertised_when_the_flag_is_off(story):
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
