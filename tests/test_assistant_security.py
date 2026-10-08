"""Execution boundaries hold even when a model follows injected instructions.

These deterministic fixtures simulate compromised model decisions; they do
not measure a real model's ability to resist prompt injection.
"""

import json
from pathlib import Path

import pytest

from assistant.events import validate_event_sequence
from assistant.protocol import AgentRunRequest
from assistant.run import RunLimits, run_assistant
from mcp_server import story_data
from tests.mcp_fakes import FakeStoryData
from tests.test_assistant_run import (
    FINAL_ROUND,
    FakePostgres,
    FakeProvider,
    editor_body,
    tool_round,
)
from tests.test_assistant_specialists import (
    ARCHITECT,
    CRITIC,
    CastProvider,
    consult_step,
    findings_reply,
)

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "assistant_adversarial.json").read_text()
)
SECRET = "PRIVATE-OTHER-STORY-CONTENT"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
async def test_injected_tool_calls_cannot_widen_scope_or_authority(case, monkeypatch):
    from assistant import run

    class ScopedStoryData(FakeStoryData):
        async def get_entity(self, uid, story_id, kind, entity_id):
            assert (uid, story_id) == ("uid-1", "story-1")
            return await super().get_entity(uid, story_id, kind, entity_id)

    story = ScopedStoryData()
    story.seed_story("story-1", "uid-1", title="Saltmarsh")
    story.seed_story("story-2", "uid-2", published=True, description=SECRET)
    story.seed_entity(
        "story-1", "characters", "char-1", name="Mina", notes=case["injection"]
    )
    story.seed_entity("story-2", "characters", "char-9", name="Outsider", notes=SECRET)
    story.seed_chapter(
        "story-1", "chapter-1", title="The Lamp Room", content="Brass polish."
    )
    story_data.configure(story)
    before = repr((story.stories, story.entities, story.chapters))
    dispatched = []
    original_execute = run.execute_tool

    async def execute(name, args, runtime):
        dispatched.append(name)
        return await original_execute(name, args, runtime)

    async def forbidden_binding(*args, **kwargs):
        pytest.fail("A forbidden story proposal reached binding")

    monkeypatch.setattr(run, "execute_tool", execute)
    monkeypatch.setattr(run, "bind_story_changes", forbidden_binding)
    attack = [
        {
            "type": "tool_call_delta",
            "tool_call": {
                "index": index,
                "tool_call_id": f"attack-{index}",
                "name": call["name"],
                "arguments_delta": json.dumps(call["arguments"]),
            },
        }
        for index, call in enumerate(case["calls"])
    ] + [{"type": "done", "finish_reason": "tool_calls"}]
    if case["source"] == "prior_findings":
        provider = CastProvider(
            [
                consult_step(
                    ("architect", ARCHITECT), ("critic", {**CRITIC, "review": True})
                ),
                attack,
                FINAL_ROUND,
            ],
            specialists={
                name: findings_reply({"analysis": case["injection"]})
                for name in ("Story Architect", "Critic")
            },
        )
    else:
        prelude = (
            [
                tool_round(
                    "get_story_entity",
                    json.dumps({"kind": "character", "entityId": "char-1"}),
                )
            ]
            if case["source"] == "character_notes"
            else []
        )
        provider = FakeProvider(*prelude, attack, FINAL_ROUND)
    body = editor_body()
    body["mode"] = case.get("mode")
    if case["source"] == "user_message":
        body["message"]["parts"][0]["text"] = case["injection"]
    try:
        events = [
            event
            async for event in run_assistant(
                AgentRunRequest.model_validate(body),
                run_id="security-run",
                provider=provider,
                postgres=FakePostgres(),
                embedder=None,
                limits=RunLimits(),
                edits_enabled=True,
            )
        ]
        validate_event_sequence(events)
        assert {e.tool_call_id for e in events if e.type == "tool.failed"} == {
            f"attack-{index}"
            for index in case.get("failed_calls", range(len(case["calls"])))
        }
        assert events[-1].type == "run.completed"
        assert not any(e.type == "approval.requested" for e in events)
        assert repr((story.stories, story.entities, story.chapters)) == before
        assert SECRET not in json.dumps([e.model_dump(mode="json") for e in events])
        assert SECRET not in json.dumps(provider.requests)
        assert case["injection"] in json.dumps(provider.requests)
        assert set(dispatched) <= {"get_story_entity"}
        if case["source"] == "prior_findings":
            critic = provider.calls_to("Critic")[0]
            assert "priorFindings" in critic["messages"][1]["parts"][0]["text"]
    finally:
        story_data.configure(None)
