"""Regenerate the canonical assistant protocol fixtures.

The fixtures are generated from the Pydantic models rather than hand-written,
because Python is the declared source of truth for this protocol: a hand-written
fixture that disagrees with the models would be a second, silently-wrong
specification. Generating them also makes the sequence invariants free -- every
run is built through ``RunEvents``, so ``seq`` is dense and there is exactly one
terminal event by construction.

Their job is to be the fixed point that the TypeScript and Go consumers check
themselves against. Regenerate deliberately, review the diff, and re-run the
round-trip tests in all three repositories:

    python -m scripts.generate_assistant_fixtures
    python scripts/sync_assistant_fixtures.py --check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assistant.errors import ErrorCode, safe_message
from assistant.events import (
    ApprovalRequested,
    ApprovalResolved,
    ReferenceEmitted,
    RunCancelled,
    RunCompleted,
    RunEvents,
    RunStarted,
    TextDelta,
    TextDone,
    ToolArgsDelta,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
    Usage,
    encode_sse,
    validate_event_sequence,
)
from assistant.protocol import SourcePart, TextPart, ToolCallPart

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "assistant" / "fixtures"


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [e.model_dump(by_alias=True, exclude_none=True, mode="json") for e in events]


def text_only() -> list[Any]:
    r = RunEvents("run-text-only")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(TextDelta, text="The lighthouse "),
        r.emit(TextDelta, text="had been dark for a year."),
        r.emit(
            TextDone,
            part=TextPart(type="text", text="The lighthouse had been dark for a year."),
        ),
        r.emit(
            Usage,
            provider="mock",
            model="mock-1",
            prompt_tokens=412,
            completion_tokens=9,
            credits=5,
            billing="mock",
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


def single_tool_round() -> list[Any]:
    r = RunEvents("run-single-tool")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ToolStarted, tool_call_id="call-1", name="get_story_overview"),
        r.emit(ToolArgsDelta, tool_call_id="call-1", delta="{}"),
        r.emit(
            ToolCompleted,
            part=ToolCallPart(
                type="tool_call",
                tool_call_id="call-1",
                name="get_story_overview",
                arguments={},
                result={"title": "Saltmarsh", "chapter_count": 3},
            ),
        ),
        r.emit(TextDelta, text="Saltmarsh has three chapters."),
        r.emit(
            TextDone,
            part=TextPart(type="text", text="Saltmarsh has three chapters."),
        ),
        r.emit(
            Usage,
            provider="mock",
            model="mock-1",
            prompt_tokens=530,
            completion_tokens=7,
            credits=6,
            billing="platform",
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


def multi_tool() -> list[Any]:
    """Two tools plus interleaved text, so ordering is pinned, not assumed."""
    r = RunEvents("run-multi-tool")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ToolStarted, tool_call_id="call-1", name="list_story_entities"),
        r.emit(ToolArgsDelta, tool_call_id="call-1", delta='{"kind":"character"}'),
        r.emit(
            ToolCompleted,
            part=ToolCallPart(
                type="tool_call",
                tool_call_id="call-1",
                name="list_story_entities",
                arguments={"kind": "character"},
                result=[{"id": "ent-1", "name": "Mina"}],
            ),
        ),
        r.emit(TextDelta, text="Mina is the only named character so far. "),
        r.emit(ToolStarted, tool_call_id="call-2", name="search_story"),
        r.emit(ToolArgsDelta, tool_call_id="call-2", delta='{"query":"Mina harbour"}'),
        r.emit(
            ToolCompleted,
            part=ToolCallPart(
                type="tool_call",
                tool_call_id="call-2",
                name="search_story",
                arguments={"query": "Mina harbour", "limit": 8},
                result=[{"chunk_id": "chunk-9", "score": 0.81}],
            ),
        ),
        r.emit(TextDelta, text="She appears once, at the harbour."),
        r.emit(
            TextDone,
            part=TextPart(
                type="text",
                text=(
                    "Mina is the only named character so far. "
                    "She appears once, at the harbour."
                ),
            ),
        ),
        r.emit(
            Usage,
            provider="mock",
            model="mock-1",
            prompt_tokens=880,
            completion_tokens=18,
            credits=9,
            billing="platform",
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


def approval_pause_resume() -> list[Any]:
    """A proposal stream ends with a real, resultless approval gate."""
    r = RunEvents("run-approval")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ToolStarted, tool_call_id="call-1", name="propose_editor_edit"),
        r.emit(
            ToolCompleted,
            part=ToolCallPart(
                type="tool_call",
                tool_call_id="call-1",
                name="propose_editor_edit",
                arguments={
                    "chapterId": "chapter-2",
                    "baseRevision": 7,
                    "baseDocumentVersion": 42,
                    "summary": "Tighten the exchange and make Mina sound guarded.",
                    "operations": [
                        {
                            "type": "replace",
                            "from": 840,
                            "to": 1062,
                            "originalText": "original selected text",
                            "replacementText": "proposed replacement",
                        }
                    ],
                },
                result={"proposalId": "proposal-1"},
            ),
        ),
        r.emit(ToolStarted, tool_call_id="call-2", name="apply_editor_edit"),
        r.emit(
            ToolArgsDelta,
            tool_call_id="call-2",
            delta='{"proposalId":"proposal-1"}',
        ),
        r.emit(
            ApprovalRequested,
            approval_id="approval-1",
            tool_call_id="call-2",
            summary="Apply 1 replacement to chapter 2?",
        ),
        r.emit(RunCompleted, finish_reason="tool_calls"),
    ]


def approval_applied() -> list[Any]:
    r = RunEvents("run-approval-applied")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ApprovalResolved, approval_id="approval-1", approved=True),
        r.emit(
            TextDone,
            part=TextPart(
                type="text", text="Applied and saved in the current chapter."
            ),
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


def approval_rejected() -> list[Any]:
    r = RunEvents("run-approval-rejected")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ApprovalResolved, approval_id="approval-1", approved=False),
        r.emit(
            TextDone,
            part=TextPart(
                type="text",
                text="Kept the suggestion without changing the chapter.",
            ),
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


def research_citations() -> list[Any]:
    r = RunEvents("run-research")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ToolStarted, tool_call_id="call-1", name="research_web"),
        r.emit(
            ToolCompleted,
            part=ToolCallPart(
                type="tool_call",
                tool_call_id="call-1",
                name="research_web",
                arguments={"query": "Fresnel lens maintenance", "maxResults": 3},
                result={"results": 2},
            ),
        ),
        r.emit(
            ReferenceEmitted,
            part=SourcePart(
                type="source",
                source_id="src-1",
                kind="web",
                title="Fresnel lens",
                url="https://example.org/fresnel",
                snippet="A Fresnel lens is a compact lens design.",
            ),
        ),
        r.emit(
            ReferenceEmitted,
            part=SourcePart(
                type="source",
                source_id="chunk-9",
                kind="story",
                title="Chapter 2",
                snippet="The lamp room smelled of brass polish.",
            ),
        ),
        r.emit(
            TextDone,
            part=TextPart(type="text", text="Keepers polished the lens nightly."),
        ),
        r.emit(
            Usage,
            provider="mock",
            model="mock-1",
            prompt_tokens=1220,
            completion_tokens=11,
            credits=13,
            billing="byok",
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


def max_steps() -> list[Any]:
    """The loop hit ASSISTANT_MAX_MODEL_CALLS with an answer still unfinished.

    Terminal but successful: the text.done part is a real partial answer, and
    the two usage events are the two model calls it cost. A client that treats
    the first usage event as the run total will under-report here, which is
    exactly why the case is a fixture rather than a comment.
    """
    r = RunEvents("run-max-steps")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ToolStarted, tool_call_id="call-1", name="search_story"),
        r.emit(ToolArgsDelta, tool_call_id="call-1", delta='{"query":"the keeper"}'),
        r.emit(
            ToolCompleted,
            part=ToolCallPart(
                type="tool_call",
                tool_call_id="call-1",
                name="search_story",
                arguments={"query": "the keeper", "limit": 8},
                result=[{"chunk_id": "chunk-3", "score": 0.74}],
            ),
        ),
        r.emit(
            Usage,
            provider="mock",
            model="mock-1",
            prompt_tokens=610,
            completion_tokens=12,
            credits=7,
            billing="platform",
        ),
        r.emit(TextDelta, text="The keeper appears in chapter two"),
        r.emit(
            TextDone,
            part=TextPart(type="text", text="The keeper appears in chapter two"),
        ),
        r.emit(
            Usage,
            provider="mock",
            model="mock-1",
            prompt_tokens=940,
            completion_tokens=8,
            credits=10,
            billing="platform",
        ),
        r.emit(RunCompleted, finish_reason="max_steps"),
    ]


def cancellation() -> list[Any]:
    """Stop after a partial delta. No text.done: nothing settled."""
    r = RunEvents("run-cancelled")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(TextDelta, text="The lighthouse "),
        r.emit(RunCancelled),
    ]


def provider_error() -> list[Any]:
    """A mid-stream failure, which the Phase 0 protocol could not express."""
    r = RunEvents("run-provider-error")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(TextDelta, text="The light"),
        r.failure(ErrorCode.PROVIDER_UNAVAILABLE),
    ]


def stale_edit() -> list[Any]:
    r = RunEvents("run-stale-edit")
    return [
        r.emit(RunStarted, provider="mock", model="mock-1"),
        r.emit(ToolStarted, tool_call_id="call-1", name="apply_editor_edit"),
        r.emit(
            ToolFailed,
            tool_call_id="call-1",
            code=ErrorCode.STALE_PROPOSAL,
            message=safe_message(ErrorCode.STALE_PROPOSAL),
        ),
        r.emit(
            TextDone,
            part=TextPart(
                type="text",
                text="The chapter changed. Ask me again for a fresh edit.",
            ),
        ),
        r.emit(RunCompleted, finish_reason="stop"),
    ]


FIXTURES = {
    "text-only": ("The minimum viable run.", text_only),
    "single-tool-round": ("One tool call, its result, and a reply.", single_tool_round),
    "multi-tool": ("Two tool rounds interleaved with text.", multi_tool),
    "approval-pause-resume": (
        "Proposal stream paused at a resultless editor approval.",
        approval_pause_resume,
    ),
    "approval-applied": (
        "A separate continuation confirms an applied editor proposal.",
        approval_applied,
    ),
    "approval-rejected": (
        "A separate continuation confirms a rejected editor proposal.",
        approval_rejected,
    ),
    "research-citations": (
        "Web and story references emitted as parts.",
        research_citations,
    ),
    "max-steps": (
        "The step ceiling ended the run with a partial answer, not a failure.",
        max_steps,
    ),
    "cancellation": ("Stopped mid-delta; run.cancelled is terminal.", cancellation),
    "provider-error": ("Mid-stream provider failure with a safe code.", provider_error),
    "stale-edit": ("An edit refused because the document moved on.", stale_edit),
}


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, (description, build) in FIXTURES.items():
        events = build()
        payload = _dump(events)
        validate_event_sequence(payload)
        # The SSE rendering is part of the contract too: a consumer that reads
        # frames off the wire should be able to check itself against bytes, not
        # only against parsed objects.
        document = {
            "name": name,
            "description": description,
            "protocolVersion": 1,
            "events": payload,
            "sse": "".join(encode_sse(event) for event in events),
        }
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
        print(f"wrote {path.name} ({len(payload)} events)")


if __name__ == "__main__":
    main()
