"""The normalized assistant event stream.

One discriminated union on ``type``. Every event carries ``v`` so a stray frame
is self-describing, ``runId`` so agent and gateway logs correlate, and ``seq``
so a later phase can add resume without a version bump.

Event models are strict when constructed for writing. The exported read adapter
overrides that setting with ``extra="ignore"``: a newer agent adding an optional
field must not break an older client, while an unrecognized ``type`` must fail
loudly. Keeping those paths separate makes the compatibility policy real rather
than documenting strict writes while silently discarding writer mistakes.

Two invariants hold over any well-formed run and are checked by
``validate_event_sequence``: ``seq`` is dense and ascending from 0, and there is
exactly one terminal event, last. The second closes a real Phase 0 hole -- the
spike protocol had no terminal error, so a mid-stream agent failure and a
dropped socket were indistinguishable to the browser.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Iterable, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.alias_generators import to_camel

from assistant.errors import ErrorCode, safe_message
from assistant.protocol import (
    MAX_CONTENT_CHARS,
    MAX_ID_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_TOOL_NAME_CHARS,
    SourcePart,
    TextPart,
    ToolCallPart,
)
from assistant.version import ASSISTANT_PROTOCOL_VERSION

TERMINAL_EVENT_TYPES = frozenset({"run.completed", "run.failed", "run.cancelled"})

BillingMode = Literal["platform", "byok", "local", "mock"]


class EventModel(BaseModel):
    """Read-tolerant, write-strict. See the module docstring."""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class BaseEvent(EventModel):
    v: Literal[ASSISTANT_PROTOCOL_VERSION]
    run_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    seq: int = Field(ge=0)


class RunStarted(BaseEvent):
    type: Literal["run.started"]
    provider: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_TOOL_NAME_CHARS
    )
    model: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_TOOL_NAME_CHARS
    )


class TextDelta(BaseEvent):
    """Incremental. The frontend accumulates; deltas are not cumulative."""

    type: Literal["text.delta"]
    text: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)


class TextDone(BaseEvent):
    """The settled text part, so a reload never depends on replaying deltas."""

    type: Literal["text.done"]
    part: TextPart


class ToolStarted(BaseEvent):
    type: Literal["tool.started"]
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    name: str = Field(min_length=1, max_length=MAX_TOOL_NAME_CHARS)


class ToolArgsDelta(BaseEvent):
    type: Literal["tool.args.delta"]
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    delta: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)


class ToolCompleted(BaseEvent):
    type: Literal["tool.completed"]
    part: ToolCallPart


class ToolFailed(BaseEvent):
    type: Literal["tool.failed"]
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    code: ErrorCode
    message: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)


class ApprovalRequested(BaseEvent):
    """Reserved. No v1 tool requires approval until Phase 5 adds editor writes."""

    type: Literal["approval.requested"]
    approval_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)


class ApprovalResolved(BaseEvent):
    type: Literal["approval.resolved"]
    approval_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    approved: bool


class ReferenceEmitted(BaseEvent):
    type: Literal["reference.emitted"]
    part: SourcePart


class Usage(BaseEvent):
    type: Literal["usage"]
    provider: str = Field(min_length=1, max_length=MAX_TOOL_NAME_CHARS)
    model: str = Field(min_length=1, max_length=MAX_TOOL_NAME_CHARS)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    credits: int = Field(ge=0)
    billing: BillingMode


class RunCompleted(BaseEvent):
    """Terminal success. ``finishReason`` says *why* the run stopped talking.

    ``max_steps`` is the orchestrator's own ceiling, and it is a success rather
    than a failure: the user has a real partial answer, so ``run.failed`` would
    both discard it and render a message about a daily allowance that was never
    reached. ``stop`` would be a lie in the other direction -- it claims the
    model was finished -- and Phase 4 needs to tell the two apart to offer
    "continue".
    """

    type: Literal["run.completed"]
    finish_reason: Literal["stop", "length", "tool_calls", "max_steps"] = "stop"


class RunFailed(BaseEvent):
    type: Literal["run.failed"]
    code: ErrorCode
    message: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)


class RunCancelled(BaseEvent):
    type: Literal["run.cancelled"]


AssistantEvent = Annotated[
    Union[
        RunStarted,
        TextDelta,
        TextDone,
        ToolStarted,
        ToolArgsDelta,
        ToolCompleted,
        ToolFailed,
        ApprovalRequested,
        ApprovalResolved,
        ReferenceEmitted,
        Usage,
        RunCompleted,
        RunFailed,
        RunCancelled,
    ],
    Field(discriminator="type"),
]

_StrictAssistantEventAdapter: TypeAdapter[Any] = TypeAdapter(AssistantEvent)


class _ReadTolerantAssistantEventAdapter:
    """Read known events tolerantly while leaving writer models strict."""

    def validate_python(self, value: Any) -> Any:
        return _StrictAssistantEventAdapter.validate_python(value, extra="ignore")

    def json_schema(self, **kwargs: Any) -> dict[str, Any]:
        return _StrictAssistantEventAdapter.json_schema(**kwargs)


AssistantEventAdapter = _ReadTolerantAssistantEventAdapter()


def encode_sse(event: BaseEvent) -> str:
    """Render one event as an SSE frame, camelCase, terminated by a blank line."""
    payload = event.model_dump(by_alias=True, exclude_none=True, mode="json")
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


class RunEvents:
    """Stamps ``v``, ``runId`` and a dense ``seq`` so callers cannot get it wrong.

    Emitting through a single object is what makes the sequence invariants
    structural: a caller that constructs events by hand can skip a ``seq`` or
    emit two terminals, and nothing would catch it until a fixture test ran.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._seq = 0
        self._terminated = False

    def emit(self, event_cls: type[BaseEvent], **fields: Any) -> BaseEvent:
        if self._terminated:
            raise RuntimeError("run already produced its terminal event")
        event = event_cls(
            v=ASSISTANT_PROTOCOL_VERSION,
            run_id=self.run_id,
            seq=self._seq,
            type=get_args(event_cls.model_fields["type"].annotation)[0],
            **fields,
        )
        self._seq += 1
        if event.type in TERMINAL_EVENT_TYPES:  # type: ignore[attr-defined]
            self._terminated = True
        return event

    def frame(self, event_cls: type[BaseEvent], **fields: Any) -> str:
        return encode_sse(self.emit(event_cls, **fields))

    def failure(self, code: ErrorCode) -> BaseEvent:
        """Terminal failure carrying only a code and its canned safe message."""
        return self.emit(RunFailed, code=code, message=safe_message(code))


def validate_event_sequence(events: Iterable[Any]) -> list[Any]:
    """Parse a whole run and assert the two sequence invariants.

    Used by the fixture round-trip tests and by anything that wants to treat a
    recorded run as trustworthy.
    """
    parsed = [AssistantEventAdapter.validate_python(event) for event in events]
    if not parsed:
        raise ValueError("a run has at least one event")

    run_ids = {event.run_id for event in parsed}
    if len(run_ids) != 1:
        raise ValueError(f"a run has one runId, got {sorted(run_ids)}")

    for expected, event in enumerate(parsed):
        if event.seq != expected:
            raise ValueError(f"seq must be dense from 0; got {event.seq} at {expected}")

    terminals = [event for event in parsed if event.type in TERMINAL_EVENT_TYPES]
    if len(terminals) != 1:
        raise ValueError(f"a run has exactly one terminal event, got {len(terminals)}")
    if parsed[-1].type not in TERMINAL_EVENT_TYPES:
        raise ValueError("the terminal event is last")
    return parsed
