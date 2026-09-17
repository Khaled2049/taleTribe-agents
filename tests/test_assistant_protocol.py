"""Protocol-level invariants: versioning, the trust boundary, and sequences."""

import json

import pytest
from pydantic import ValidationError

from assistant.errors import ErrorCode, safe_message
from assistant.events import (
    AssistantEventAdapter,
    RunCancelled,
    RunCompleted,
    RunEvents,
    RunStarted,
    TextDelta,
    encode_sse,
    validate_event_sequence,
)
from assistant.protocol import AgentRunRequest, RunRequest
from assistant.version import ASSISTANT_PROTOCOL_VERSION

BROWSER_BODY = {
    "v": 1,
    "storyId": "story-1",
    "clientMessageId": "client-1",
    "message": {"role": "user", "parts": [{"type": "text", "text": "Tighten this"}]},
}


def test_run_request_round_trips_camel_case():
    request = RunRequest.model_validate(BROWSER_BODY)
    assert request.story_id == "story-1"
    assert request.model_dump(by_alias=True, exclude_none=True) == BROWSER_BODY


def test_editor_context_from_alias_survives_round_trip():
    """`from` is a Python keyword; the alias is the part that can silently break."""
    body = {
        **BROWSER_BODY,
        "editorContext": {
            "chapterId": "chapter-2",
            "persistedRevision": 7,
            "documentVersion": 42,
            "selection": {"from": 840, "to": 1062, "text": "selected"},
            "buffer": {"text": "bounded editor text", "truncated": False},
            "dirty": True,
        },
    }
    request = RunRequest.model_validate(body)
    assert request.editor_context.selection.from_ == 840
    assert request.model_dump(by_alias=True, exclude_none=True) == body


def test_editor_continuation_is_strict_and_bounded():
    body = {
        **BROWSER_BODY,
        "continuation": {
            "kind": "editor_approval",
            "previousRunId": "run-1",
            "approvalId": "approval-1",
            "toolCallId": "apply-1",
            "proposalId": "proposal-1",
            "decision": "rejected",
            "proposal": {
                "chapterId": "chapter-2",
                "baseRevision": 7,
                "baseDocumentVersion": 42,
                "summary": "Tighten this.",
                "operations": [
                    {
                        "type": "replace",
                        "from": 5,
                        "to": 13,
                        "originalText": "selected",
                        "replacementText": "revised",
                    }
                ],
            },
        },
    }
    request = RunRequest.model_validate(body)
    assert request.continuation.proposal.operations[0].from_ == 5
    assert request.model_dump(by_alias=True, exclude_none=True) == body
    with pytest.raises(ValidationError):
        RunRequest.model_validate(
            {
                **body,
                "continuation": {**body["continuation"], "userId": "other"},
            }
        )


@pytest.mark.parametrize("version", [0, 2, 99])
def test_unsupported_protocol_version_rejected(version):
    with pytest.raises(ValidationError):
        RunRequest.model_validate({**BROWSER_BODY, "v": version})


def test_protocol_version_is_required():
    without_version = {key: value for key, value in BROWSER_BODY.items() if key != "v"}
    with pytest.raises(ValidationError):
        RunRequest.model_validate(without_version)


def test_browser_cannot_assert_an_identity():
    """The trust boundary: RunRequest has no user_id field to populate."""
    assert "user_id" not in RunRequest.model_fields
    with pytest.raises(ValidationError):
        RunRequest.model_validate({**BROWSER_BODY, "userId": "someone-else"})


def test_gateway_adds_the_verified_uid():
    request = AgentRunRequest.model_validate({**BROWSER_BODY, "userId": "uid-1"})
    assert request.user_id == "uid-1"
    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(BROWSER_BODY)


def test_message_text_is_bounded_tighter_than_assistant_output():
    from assistant.protocol import MAX_MESSAGE_CHARS

    oversized = {
        **BROWSER_BODY,
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "x" * (MAX_MESSAGE_CHARS + 1)}],
        },
    }
    with pytest.raises(ValidationError):
        RunRequest.model_validate(oversized)


def test_unknown_event_type_is_rejected():
    """Compatibility policy, half one: an unrecognized tag is not a stream we know."""
    with pytest.raises(ValidationError):
        AssistantEventAdapter.validate_python(
            {"v": 1, "runId": "r", "seq": 0, "type": "text.bogus"}
        )


def test_unknown_event_field_is_ignored():
    """Compatibility policy, half two: a newer optional field must not break us."""
    event = AssistantEventAdapter.validate_python(
        {"v": 1, "runId": "r", "seq": 0, "type": "run.cancelled", "addedLater": 7}
    )
    assert event.type == "run.cancelled"
    assert not hasattr(event, "addedLater")


def test_event_writers_reject_unknown_fields():
    """The tolerant read policy must not hide mistakes in server-owned writes."""
    with pytest.raises(ValidationError):
        RunCancelled(v=1, run_id="r", seq=0, added_later=7)


@pytest.mark.parametrize(
    "event",
    [
        lambda: RunCancelled(v=1, run_id="", seq=0),
        lambda: TextDelta(v=1, run_id="r", seq=0, text=""),
    ],
)
def test_required_event_strings_are_non_empty(event):
    with pytest.raises(ValidationError):
        event()


def test_run_events_stamps_version_and_dense_seq():
    events = RunEvents("run-1")
    emitted = [
        events.emit(RunStarted, provider="mock", model="mock-1"),
        events.emit(TextDelta, text="a"),
        events.emit(TextDelta, text="b"),
        events.emit(RunCompleted),
    ]
    assert [e.seq for e in emitted] == [0, 1, 2, 3]
    assert {e.v for e in emitted} == {ASSISTANT_PROTOCOL_VERSION}
    assert {e.run_id for e in emitted} == {"run-1"}


def test_second_terminal_event_is_refused():
    events = RunEvents("run-1")
    events.emit(RunCompleted)
    with pytest.raises(RuntimeError, match="terminal"):
        events.emit(RunCancelled)


def test_failure_carries_only_a_code_and_its_canned_message():
    """No provider body, no exception text -- the Phase 0 logging rule on the wire."""
    failure = RunEvents("run-1").failure(ErrorCode.PROVIDER_UNAVAILABLE)
    assert failure.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert failure.message == safe_message(ErrorCode.PROVIDER_UNAVAILABLE)


def test_every_error_code_has_a_safe_message():
    assert all(safe_message(code) for code in ErrorCode)


def test_encode_sse_emits_one_terminated_frame():
    frame = encode_sse(RunEvents("run-1").emit(TextDelta, text="hi"))
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[6:]) == {
        "v": 1,
        "runId": "run-1",
        "seq": 0,
        "type": "text.delta",
        "text": "hi",
    }


@pytest.mark.parametrize(
    "events,message",
    [
        ([], "at least one event"),
        (
            [
                {"v": 1, "runId": "r", "seq": 0, "type": "run.started"},
                {"v": 1, "runId": "r", "seq": 2, "type": "run.completed"},
            ],
            "dense",
        ),
        (
            [
                {"v": 1, "runId": "r", "seq": 0, "type": "run.completed"},
                {"v": 1, "runId": "r", "seq": 1, "type": "run.cancelled"},
            ],
            "exactly one terminal",
        ),
        (
            [
                {"v": 1, "runId": "r", "seq": 0, "type": "run.started"},
                {"v": 1, "runId": "r", "seq": 1, "type": "text.delta", "text": "x"},
            ],
            "exactly one terminal",
        ),
        (
            [
                {"v": 1, "runId": "a", "seq": 0, "type": "run.started"},
                {"v": 1, "runId": "b", "seq": 1, "type": "run.completed"},
            ],
            "one runId",
        ),
    ],
)
def test_malformed_sequences_are_rejected(events, message):
    with pytest.raises(ValueError, match=message):
        validate_event_sequence(events)
