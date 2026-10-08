"""Story-change proposals: schema bounds, server binding, approval, continuation."""

import json

import pytest
from pydantic import ValidationError

from assistant.entity_changes import ProposalRejected, bind_story_changes
from assistant.events import validate_event_sequence
from assistant.protocol import AgentRunRequest, RunRequest, StoryChangeDraft
from assistant.run import RunLimits, _is_edit_request, run_assistant
from assistant.tools import ProposeStoryChangesDraft, ToolContext, available_tools
from mcp_server import story_data
from tests.mcp_fakes import FakeStoryData
from tests.test_assistant_run import (
    BODY,
    FINAL_ROUND,
    FakePostgres,
    FakeProvider,
    editor_body,
    tool_round,
)

CTX = ToolContext(user_id="uid-1", story_id="story-1")


@pytest.fixture
def story():
    fake = FakeStoryData()
    fake.seed_story("story-1", "uid-1", title="Saltmarsh")
    fake.seed_story("story-2", "uid-2", title="Elsewhere", published=True)
    fake.seed_entity("story-1", "characters", "char-1", name="Mina", revision=4)
    fake.seed_entity("story-1", "places", "place-1", name="The Lamp Room")
    fake.seed_entity(
        "story-1",
        "plots",
        "plot-1",
        name="The Wreck",
        events=[{"id": "event-1", "name": "The storm", "revision": 7}],
    )
    fake.seed_entity("story-2", "characters", "char-9", name="Outsider")
    story_data.configure(fake)
    yield fake
    story_data.configure(None)


def draft(*changes, summary="Sharpen the cast."):
    return {"summary": summary, "changes": list(changes)}


CREATE_PLACE = {
    "operation": "place.create",
    "fields": {"name": "Abandoned Hospital", "atmosphere": "Damp and silent."},
}
UPDATE_MINA = {
    "operation": "character.update",
    "entityId": "char-1",
    "fields": {"personality": "Guarded, then reckless."},
}


async def collect(provider, *, body=None, enabled=True, run_id="run-1", **kwargs):
    request = AgentRunRequest.model_validate({**(body or BODY), "userId": "uid-1"})
    return [
        event
        async for event in run_assistant(
            request,
            run_id=run_id,
            provider=provider,
            postgres=FakePostgres(),
            embedder=None,
            limits=RunLimits(),
            entity_proposals_enabled=enabled,
            **kwargs,
        )
    ]


def proposal_round(*changes, call_id="call-1"):
    return tool_round("propose_story_changes", json.dumps(draft(*changes)), call_id)


# -- schema ---------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "character.update", "fields": {"voice": "Low."}},
        {"operation": "character.create", "fields": {"voice": "Low."}},
        {"operation": "character.create", "entityId": "c", "fields": {"name": "A"}},
        {"operation": "place.update", "entityId": "p", "fields": {}},
        {"operation": "place.update", "entityId": "p", "fields": {"voice": "x"}},
        {"operation": "event.create", "fields": {"name": "A"}},
        {"operation": "plot.create", "plotLineId": "p", "fields": {"name": "A"}},
        {"operation": "character.delete", "entityId": "c", "fields": {"name": "A"}},
        {"operation": "character.update", "entityId": "c", "fields": {"artUrl": "x"}},
        {
            "operation": "character.update",
            "entityId": "c",
            "baseRevision": 1,
            "fields": {"name": "A"},
        },
        {
            "operation": "event.update",
            "entityId": "e",
            "plotLineId": "p",
            "fields": {"tensionLevel": 11},
        },
        {
            "operation": "event.update",
            "entityId": "e",
            "plotLineId": "p",
            "fields": {"pacing": "glacial"},
        },
    ],
)
def test_malformed_changes_are_rejected_by_the_schema(change):
    with pytest.raises(ValidationError):
        StoryChangeDraft.model_validate(change)


def test_explicit_nulls_are_treated_as_unset():
    change = StoryChangeDraft.model_validate(
        {
            "operation": "place.update",
            "entityId": "place-1",
            "fields": {"atmosphere": "Cold.", "voice": None, "age": None},
        }
    )
    assert change.fields.set_names() == {"atmosphere"}


def test_a_proposal_is_bounded_to_five_changes():
    many = [
        {"operation": "place.create", "fields": {"name": f"Place {n}"}}
        for n in range(6)
    ]
    with pytest.raises(ValidationError):
        ProposeStoryChangesDraft.model_validate(draft(*many))


def test_the_tool_can_be_withheld_and_never_exposes_its_apply_half():
    off = available_tools(
        edits_enabled=True, research_enabled=True, entity_proposals_enabled=False
    )
    on = available_tools(edits_enabled=False, research_enabled=False)
    assert "propose_story_changes" not in off
    assert "propose_story_changes" in on
    assert "apply_story_changes" not in on


def test_provider_facing_schemas_carry_no_null_type():
    """Optional fields reach the provider as plain, not-required fields."""
    from assistant.run import _model_tools
    from assistant.tools import TOOL_SCHEMAS

    tools = _model_tools(edits_enabled=True, entity_proposals_enabled=True)
    rendered = json.dumps(tools)
    assert '"type": "null"' not in rendered and '"default": null' not in rendered
    proposal = next(t for t in tools if t["name"] == "propose_story_changes")
    fields = proposal["parameters"]["$defs"]["StoryChangeFields"]["properties"]
    assert fields["pacing"]["enum"] == ["slow", "moderate", "fast"]
    assert fields["name"]["maxLength"] == 200
    # Tools without optional fields are passed through untouched.
    search = next(t for t in tools if t["name"] == "search_story")
    expected = TOOL_SCHEMAS["search_story"].model_json_schema(by_alias=True)
    expected.pop("description", None)
    assert search["parameters"] == expected


# -- binding --------------------------------------------------------------


async def bind(*changes):
    return await bind_story_changes(
        ProposeStoryChangesDraft.model_validate(draft(*changes)), CTX
    )


async def test_binding_reads_revision_and_label_from_story_data(story):
    bound = await bind(UPDATE_MINA, CREATE_PLACE)
    update, create = bound.changes
    assert (update.base_revision, update.label) == (4, "Mina")
    assert (create.base_revision, create.label) == (None, "Abandoned Hospital")


async def test_binding_an_event_update_uses_the_events_own_revision(story):
    bound = await bind(
        {
            "operation": "event.update",
            "entityId": "event-1",
            "plotLineId": "plot-1",
            "fields": {"tensionLevel": 9, "characterIds": ["char-1"]},
        }
    )
    assert (bound.changes[0].base_revision, bound.changes[0].label) == (7, "The storm")


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({**UPDATE_MINA, "entityId": "char-404"}, "No character"),
        # Another owner's entity is indistinguishable from a missing one.
        ({**UPDATE_MINA, "entityId": "char-9"}, "No character"),
        (
            {"operation": "place.create", "fields": {"name": "the lamp room"}},
            "already exists",
        ),
        (
            {
                "operation": "event.create",
                "plotLineId": "plot-404",
                "fields": {"name": "Landfall"},
            },
            "No plot line",
        ),
        (
            {
                "operation": "event.update",
                "entityId": "event-404",
                "plotLineId": "plot-1",
                "fields": {"name": "Landfall"},
            },
            "No event",
        ),
        (
            {
                "operation": "event.create",
                "plotLineId": "plot-1",
                "fields": {"name": "Landfall", "characterIds": ["char-404"]},
            },
            "Unknown characters",
        ),
        (
            {
                "operation": "event.create",
                "plotLineId": "plot-1",
                "fields": {"name": "Landfall", "locationId": "place-404"},
            },
            "Unknown places",
        ),
    ],
)
async def test_unbindable_changes_are_rejected_with_a_reason(story, change, fragment):
    with pytest.raises(ProposalRejected, match=fragment):
        await bind(change)


async def test_two_updates_to_one_entity_are_rejected(story):
    with pytest.raises(ProposalRejected, match="twice"):
        await bind(UPDATE_MINA, {**UPDATE_MINA, "fields": {"voice": "Low."}})


async def test_an_event_cannot_reference_a_character_the_proposal_creates(story):
    with pytest.raises(ProposalRejected, match="Unknown characters"):
        await bind(
            {"operation": "character.create", "fields": {"name": "Elena"}},
            {
                "operation": "event.create",
                "plotLineId": "plot-1",
                "fields": {"name": "Arrival", "characterIds": ["Elena"]},
            },
        )


async def test_binding_refuses_a_story_the_caller_does_not_own(story):
    from mcp_server import data

    with pytest.raises(data.StoryNotFoundError):
        await bind_story_changes(
            ProposeStoryChangesDraft.model_validate(draft(CREATE_PLACE)),
            ToolContext(user_id="uid-1", story_id="story-2"),
        )


# -- run loop -------------------------------------------------------------


async def test_a_proposal_pauses_for_approval_and_writes_nothing(story):
    provider = FakeProvider(proposal_round(UPDATE_MINA, CREATE_PLACE))
    before = repr(story.entities)
    events = await collect(provider)
    validate_event_sequence(events)

    assert [event.type for event in events][-5:] == [
        "tool.completed",
        "tool.started",
        "tool.args.delta",
        "approval.requested",
        "run.completed",
    ]
    assert events[-1].finish_reason == "tool_calls"
    proposal = next(event for event in events if event.type == "tool.completed")
    assert proposal.part.arguments["changes"][0]["baseRevision"] == 4
    proposal_id = proposal.part.result["proposalId"]
    approval = next(event for event in events if event.type == "approval.requested")
    assert approval.approval_id == f"approval-{proposal_id.removeprefix('proposal-')}"
    assert approval.summary == "Save these 2 changes to the story?"
    assert repr(story.entities) == before
    assert len(provider.requests) == 1


async def test_a_rejected_draft_goes_back_to_the_model_as_data(story):
    provider = FakeProvider(
        proposal_round({**UPDATE_MINA, "entityId": "char-404"}), FINAL_ROUND
    )
    events = await collect(provider)
    types = [event.type for event in events]
    assert "approval.requested" not in types
    assert types[-1] == "run.completed" and events[-1].finish_reason == "stop"
    tool_message = provider.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["parts"][0]["text"])["accepted"] is False


async def test_a_proposal_alongside_another_tool_is_declined_not_fatal(story):
    mixed = [
        {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 0,
                "tool_call_id": "call-1",
                "name": "get_story_overview",
            },
        },
        {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 1,
                "tool_call_id": "call-2",
                "name": "propose_story_changes",
                "arguments_delta": json.dumps(draft(CREATE_PLACE)),
            },
        },
        {"type": "done", "finish_reason": "tool_calls"},
    ]
    events = await collect(FakeProvider(mixed, FINAL_ROUND))
    types = [event.type for event in events]
    assert "approval.requested" not in types and "run.failed" not in types


async def test_the_tool_is_unknown_when_it_is_withheld(story):
    provider = FakeProvider(proposal_round(CREATE_PLACE), FINAL_ROUND)
    events = await collect(provider, enabled=False, specialists_enabled=False)
    types = [event.type for event in events]
    assert "tool.failed" in types and "approval.requested" not in types
    assert all(
        tool["name"] != "propose_story_changes"
        for tool in provider.requests[0]["tools"]
    )
    system = provider.requests[0]["messages"][0]["parts"][0]["text"]
    assert "propose_story_changes" not in system


async def test_a_plain_question_still_proposes_nothing(story):
    provider = FakeProvider(FINAL_ROUND)
    events = await collect(provider)
    types = [event.type for event in events]
    assert not {"tool.started", "approval.requested"} & set(types)
    assert types[-1] == "run.completed"
    system = provider.requests[0]["messages"][0]["parts"][0]["text"]
    assert "propose_story_changes" in system


# -- continuation ---------------------------------------------------------


async def first_proposal(story, *changes):
    events = await collect(FakeProvider(proposal_round(*changes)), run_id="run-1")
    part = next(event.part for event in events if event.type == "tool.completed")
    proposal_id = part.result["proposalId"]
    suffix = proposal_id.removeprefix("proposal-")
    return {
        "kind": "entity_approval",
        "previousRunId": "run-1",
        "approvalId": f"approval-{suffix}",
        "toolCallId": f"apply-{suffix}",
        "proposalId": proposal_id,
        "proposal": part.arguments,
    }


async def continue_with(continuation, **kwargs):
    provider = FakeProvider(FINAL_ROUND)
    events = await collect(
        provider,
        body={**BODY, "continuation": continuation},
        run_id="run-2",
        **kwargs,
    )
    return provider, events


async def test_an_applied_continuation_records_what_was_saved_unbilled(story):
    base = await first_proposal(story, UPDATE_MINA, CREATE_PLACE)
    provider, events = await continue_with(
        {
            **base,
            "decision": "applied",
            "results": [
                {"index": 0, "status": "applied", "entityId": "char-1"},
                {"index": 1, "status": "applied", "entityId": "place-2"},
            ],
        }
    )
    assert [event.type for event in events] == [
        "run.started",
        "approval.resolved",
        "text.done",
        "run.completed",
    ]
    assert events[1].approved is True
    assert events[2].part.text == (
        "Saved: updated character Mina; created place Abandoned Hospital."
    )
    assert provider.requests == []


async def test_a_partial_apply_says_what_landed_and_what_did_not(story):
    base = await first_proposal(story, UPDATE_MINA, CREATE_PLACE)
    provider, events = await continue_with(
        {
            **base,
            "decision": "apply_failed",
            "results": [
                {"index": 0, "status": "stale"},
                {"index": 1, "status": "skipped"},
            ],
        }
    )
    text = events[2].part.text
    assert events[1].approved is False
    assert text.startswith("Not saved: character Mina (it changed since")
    assert "place Abandoned Hospital (an earlier change did not save)" in text
    assert provider.requests == []


async def test_a_rejection_is_recorded_without_a_model_call(story):
    base = await first_proposal(story, CREATE_PLACE)
    provider, events = await continue_with({**base, "decision": "rejected"})
    assert events[2].part.text == "Left the story unchanged."
    assert provider.requests == []


async def test_a_revision_request_reaches_the_model_with_the_prior_proposal(story):
    base = await first_proposal(story, CREATE_PLACE)
    provider, events = await continue_with(
        {**base, "decision": "revision_requested", "feedback": "Make it a morgue."}
    )
    assert events[-1].type == "run.completed"
    user_text = provider.requests[0]["messages"][-1]["parts"][0]["text"]
    assert "Abandoned Hospital" in user_text and "Make it a morgue." in user_text


@pytest.mark.parametrize(
    "patch",
    [
        {"decision": "applied"},
        {"decision": "applied", "results": [{"index": 0, "status": "failed"}]},
        {"decision": "apply_failed", "results": [{"index": 0, "status": "applied"}]},
        {"decision": "rejected", "results": [{"index": 0, "status": "applied"}]},
        {"decision": "revision_requested"},
        {"decision": "rejected", "proposalId": "proposal-forged"},
        {"decision": "rejected", "previousRunId": "run-other"},
    ],
)
async def test_inconsistent_or_forged_continuations_fail_the_run(story, patch):
    base = await first_proposal(story, CREATE_PLACE)
    provider, events = await continue_with({**base, **patch})
    assert events[-1].type == "run.failed"
    assert provider.requests == []


async def test_a_tampered_proposal_breaks_the_linkage(story):
    base = await first_proposal(story, CREATE_PLACE)
    tampered = json.loads(json.dumps(base))
    tampered["proposal"]["changes"][0]["fields"]["name"] = "Something Else"
    _, events = await continue_with({**tampered, "decision": "rejected"})
    assert events[-1].type == "run.failed"


async def test_entity_continuations_are_refused_when_proposals_are_withheld(story):
    base = await first_proposal(story, CREATE_PLACE)
    _, events = await continue_with({**base, "decision": "rejected"}, enabled=False)
    assert events[-1].type == "run.failed"


def test_the_browser_request_accepts_both_continuation_kinds():
    with pytest.raises(ValidationError):
        RunRequest.model_validate(
            {**BODY, "userId": None, "continuation": {"kind": "something_else"}}
        )


# -- selection hijack -----------------------------------------------------


@pytest.mark.parametrize(
    "text, direct",
    [
        ("Make the villain more interesting", False),
        ("Improve the pacing of the second act", False),
        ("Make this tighter", True),
        ("Rewrite the selected paragraph", True),
        ("Shorten", True),
        ("Tighten it up please", True),
    ],
)
def test_an_edit_verb_alone_does_not_claim_the_selection(text, direct):
    assert _is_edit_request(text, require_selection_reference=True) is direct
    assert _is_edit_request(text) is True


async def test_a_story_request_with_a_selection_keeps_every_tool(story):
    body = {
        **editor_body(),
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Make the villain more interesting"}],
        },
    }
    provider = FakeProvider(FINAL_ROUND)
    await collect(provider, body=body, edits_enabled=True)
    request = provider.requests[0]
    assert request["required_tool"] is None
    names = {tool["name"] for tool in request["tools"]}
    assert {"propose_story_changes", "propose_editor_edit"} <= names


# -- targets by name ------------------------------------------------------


@pytest.mark.parametrize("ref", ["char-1", "Mina", " mina "])
async def test_an_update_can_name_its_target_by_id_or_exact_name(story, ref):
    bound = await bind({**UPDATE_MINA, "entityId": ref})
    change = bound.changes[0]
    # The browser always receives the real id and revision, whatever was sent.
    assert (change.entity_id, change.base_revision, change.label) == (
        "char-1",
        4,
        "Mina",
    )


async def test_an_event_update_can_name_the_plot_line_and_event(story):
    bound = await bind(
        {
            "operation": "event.update",
            "entityId": "the storm",
            "plotLineId": "The Wreck",
            "fields": {"tensionLevel": 9},
        }
    )
    change = bound.changes[0]
    assert (change.entity_id, change.plot_line_id) == ("event-1", "plot-1")
    assert change.base_revision == 7


async def test_an_event_finds_its_own_plot_line_when_the_one_given_is_wrong(story):
    # A search hit carries the event's id, and the model guesses the rest.
    bound = await bind(
        {
            "operation": "event.update",
            "entityId": "event-1",
            "plotLineId": "event-1",
            "fields": {"tensionLevel": 9},
        }
    )
    assert bound.changes[0].plot_line_id == "plot-1"


async def test_an_event_id_used_as_a_plot_line_is_explained_not_just_refused(story):
    with pytest.raises(ProposalRejected) as excinfo:
        await bind(
            {
                "operation": "plot.update",
                "entityId": "event-1",
                "fields": {"description": "The tally never resolves."},
            }
        )
    reason = str(excinfo.value)
    assert "That is the event 'The storm' in plot line 'The Wreck'" in reason
    assert "This story's plot lines: The Wreck" in reason


async def test_a_name_and_an_id_for_one_row_still_count_as_the_same_target(story):
    with pytest.raises(ProposalRejected, match="updates character 'Mina' twice"):
        await bind(
            UPDATE_MINA, {**UPDATE_MINA, "entityId": "Mina", "fields": {"voice": "x"}}
        )


async def test_an_ambiguous_name_asks_for_an_id(story):
    story.seed_entity("story-1", "characters", "char-7", name="MINA")
    with pytest.raises(ProposalRejected, match="More than one character"):
        await bind({**UPDATE_MINA, "entityId": "Mina"})
