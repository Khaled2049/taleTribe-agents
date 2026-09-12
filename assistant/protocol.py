"""Assistant run requests and the message parts that persist across a reload.

Two models carry the trust boundary in their types rather than in a comment.
``RunRequest`` is what the browser sends and has no identity field at all.
``AgentRunRequest`` adds ``user_id`` and is what the Functions gateway forwards
after it has verified the Firebase token and checked story ownership. A browser
body cannot become an ``AgentRunRequest`` by adding a field, because
``extra="forbid"`` rejects the attempt.

Deviation from the shape sketched in the integration plan, which says field
names are illustrative: ``storyId`` is top level on the run request rather than
nested inside ``editorContext``. The story is what the run is scoped to and what
ownership is checked against, so it exists whether or not an editor is open, and
having one copy means there is no way for the two to disagree.

Parts are the persisted vocabulary. The UI reloads a thread by re-rendering
parts, so a tool call stays a ``ToolCallPart`` rather than being flattened back
into a string -- flattening is a one-way door that Phase 7's durable threads
cannot reopen.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from agents.storyAgent.action_schemas import (
    MAX_CONTENT_CHARS,
    MAX_ID_CHARS,
    MAX_PROMPT_CHARS,
)
from assistant.version import ASSISTANT_PROTOCOL_VERSION

# Bounds. The shared ones come from action_schemas so the assistant and the
# existing actions cannot drift into two different ideas of "too long".
MAX_MESSAGE_CHARS = MAX_PROMPT_CHARS
MAX_SELECTION_CHARS = MAX_PROMPT_CHARS
MAX_PARTS_PER_MESSAGE = 16
MAX_TOOL_NAME_CHARS = 64
MAX_SUMMARY_CHARS = 500
MAX_URL_CHARS = 2048


class StrictModel(BaseModel):
    """Wire model for anything read at a trust boundary or written by us.

    camelCase on the wire, snake_case in Python. ``populate_by_name`` lets tests
    and fixtures construct with either spelling; ``model_dump(by_alias=True)``
    is what goes on the wire.
    """

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class TextPart(StrictModel):
    """Assistant-produced text, and the persisted form of a settled reply."""

    type: Literal["text"]
    text: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)


class UserTextPart(StrictModel):
    """The same wire shape, bounded far tighter.

    Input crosses a trust boundary and costs prompt tokens, so it is capped at
    ``MAX_MESSAGE_CHARS`` rather than the ``MAX_CONTENT_CHARS`` an assistant
    reply may reach. Same ``type`` tag, because to a renderer it is the same
    thing.
    """

    type: Literal["text"]
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ToolCallPart(StrictModel):
    """A completed tool round, in the shape the UI reloads it from."""

    type: Literal["tool_call"]
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    name: str = Field(min_length=1, max_length=MAX_TOOL_NAME_CHARS)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None


class SourcePart(StrictModel):
    """A story or web reference. ``url`` is absent for story-internal sources."""

    type: Literal["source"]
    source_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    kind: Literal["story", "web"]
    title: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    url: Optional[str] = Field(default=None, min_length=1, max_length=MAX_URL_CHARS)
    snippet: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_PROMPT_CHARS
    )


AssistantPart = Annotated[
    Union[TextPart, ToolCallPart, SourcePart],
    Field(discriminator="type"),
]


class Selection(StrictModel):
    from_: int = Field(ge=0, alias="from")
    to: int = Field(ge=0)
    text: str = Field(min_length=0, max_length=MAX_SELECTION_CHARS)


class EditorContext(StrictModel):
    """Freshness for the active buffer. Optional, and unused until Phase 5.

    Specified now so Phase 5 does not need a protocol change. Deliberately not a
    place to put the manuscript: send the selection and the minimum context
    needed for freshness, and let the server retrieve persisted story text.
    """

    chapter_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_ID_CHARS
    )
    persisted_revision: Optional[int] = Field(default=None, ge=0)
    document_version: Optional[int] = Field(default=None, ge=0)
    selection: Optional[Selection] = None
    dirty: bool = False


class UserMessage(StrictModel):
    """v1 user input is text-only; the list is for forward room, not features."""

    role: Literal["user"]
    parts: list[UserTextPart] = Field(min_length=1, max_length=MAX_PARTS_PER_MESSAGE)


class RunRequest(StrictModel):
    """What the browser sends. Carries no identity -- see the module docstring."""

    v: Literal[ASSISTANT_PROTOCOL_VERSION]
    story_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    thread_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_ID_CHARS
    )
    client_message_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    message: UserMessage
    editor_context: Optional[EditorContext] = None


class AgentRunRequest(RunRequest):
    """What the gateway forwards: the browser's request plus a verified uid.

    ``user_id`` is derived from the Firebase token by the Functions gateway. It
    is never read from a browser body and never exposed as a tool argument.
    """

    user_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
