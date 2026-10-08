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

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator
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
MAX_EDITOR_WINDOW_CHARS = 8_000
MAX_PARTS_PER_MESSAGE = 16
MAX_TOOL_NAME_CHARS = 64
MAX_SUMMARY_CHARS = 500
# Generous for every provider's key format; bounded so a body cannot grow on it.
MAX_API_KEY_CHARS = 512
MAX_URL_CHARS = 2048
MAX_EDIT_OPERATIONS = 20
# Entity proposals. Tighter than story-data's own ceilings (200 / 20 000): a
# proposal has to fit in one bounded tool call, and five changes at story-data's
# prose limit would not.
MAX_STORY_CHANGES = 5
MAX_ENTITY_NAME_CHARS = 200
MAX_ENTITY_SHORT_CHARS = 100
MAX_ENTITY_PROSE_CHARS = 4_000
MAX_EVENT_CHARACTERS = 20


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


class EditorTextWindow(StrictModel):
    """A bounded plain-text view of the live editor, never HTML or TipTap JSON."""

    text: str = Field(min_length=0, max_length=MAX_EDITOR_WINDOW_CHARS)
    truncated: bool = False


class ReplaceOperation(StrictModel):
    """The Phase 5 editor operation. An empty replacement is a deletion."""

    type: Literal["replace"] = "replace"
    from_: int = Field(ge=0, alias="from")
    to: int = Field(ge=0)
    original_text: str = Field(min_length=1, max_length=MAX_SELECTION_CHARS)
    replacement_text: str = Field(
        default="", min_length=0, max_length=MAX_SELECTION_CHARS
    )


class InsertOperation(StrictModel):
    """Reserved for a later editor phase; Phase 5 rejects it at execution."""

    type: Literal["insert"] = "insert"
    at: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=MAX_SELECTION_CHARS)


EditOperation = Annotated[
    Union[ReplaceOperation, InsertOperation],
    Field(discriminator="type"),
]


class ProposeEditorEditArgs(StrictModel):
    chapter_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    base_revision: int = Field(ge=0)
    base_document_version: int = Field(ge=0)
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    operations: list[EditOperation] = Field(
        min_length=1, max_length=MAX_EDIT_OPERATIONS
    )


EditorApplyStatus = Literal[
    "saved",
    "applied_local_save_failed",
    "applied_local_save_conflict",
    "stale",
    "invalid",
]


class EditorApplyResult(StrictModel):
    """Bounded browser report. It never authorizes or performs a server write."""

    status: EditorApplyStatus
    chapter_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    document_version: int = Field(ge=0)
    persisted_revision: Optional[int] = Field(default=None, ge=0)


class EditorContinuation(StrictModel):
    """Stateless second request after a browser-owned approval decision."""

    kind: Literal["editor_approval"] = "editor_approval"
    previous_run_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    approval_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    proposal_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    decision: Literal["applied", "rejected", "revision_requested", "apply_failed"]
    proposal: ProposeEditorEditArgs
    result: Optional[EditorApplyResult] = None
    feedback: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SUMMARY_CHARS
    )


StoryChangeOperation = Literal[
    "character.create",
    "character.update",
    "place.create",
    "place.update",
    "plot.create",
    "plot.update",
    "event.create",
    "event.update",
]

# What a proposal may set, per entity kind. Deliberately narrower than
# story-data's inputs: no image URLs, no relationships, no event ordering or
# dependencies. The browser merges these onto the full current record.
STORY_CHANGE_FIELDS: dict[str, frozenset[str]] = {
    "character": frozenset(
        {
            "name",
            "age",
            "soul",
            "personality",
            "voice",
            "backstory",
            "affiliations",
            "notes",
        }
    ),
    "place": frozenset(
        {
            "name",
            "description",
            "atmosphere",
            "geography",
            "history",
            "significance",
            "notes",
        }
    ),
    "plot": frozenset({"name", "description"}),
    "event": frozenset(
        {
            "name",
            "content",
            "tension_level",
            "pacing",
            "story_beat",
            "emotional_tone",
            "character_ids",
            "location_id",
            "notes",
        }
    ),
}


def _prose() -> Any:
    return Field(default=None, min_length=1, max_length=MAX_ENTITY_PROSE_CHARS)


def _short() -> Any:
    return Field(default=None, min_length=1, max_length=MAX_ENTITY_SHORT_CHARS)


class StoryChangeFields(StrictModel):
    """Only the fields being set. Which ones apply depends on the entity kind."""

    name: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_ENTITY_NAME_CHARS
    )
    age: Optional[int] = Field(default=None, ge=0, le=100_000)
    soul: Optional[str] = _prose()
    personality: Optional[str] = _prose()
    voice: Optional[str] = _prose()
    backstory: Optional[str] = _prose()
    affiliations: Optional[str] = _prose()
    description: Optional[str] = _prose()
    atmosphere: Optional[str] = _prose()
    geography: Optional[str] = _prose()
    history: Optional[str] = _prose()
    significance: Optional[str] = _prose()
    content: Optional[str] = _prose()
    tension_level: Optional[int] = Field(default=None, ge=1, le=10)
    # The plot board renders these as fixed choices, so free text would not show.
    pacing: Optional[Literal["slow", "moderate", "fast"]] = None
    story_beat: Optional[
        Literal[
            "exposition",
            "inciting_incident",
            "rising_action",
            "midpoint",
            "climax",
            "falling_action",
            "resolution",
        ]
    ] = None
    emotional_tone: Optional[str] = _short()
    character_ids: Optional[
        list[Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]]
    ] = Field(default=None, max_length=MAX_EVENT_CHARACTERS)
    location_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_ID_CHARS
    )
    notes: Optional[str] = _prose()

    def set_names(self) -> frozenset[str]:
        return frozenset(
            name for name in self.model_fields_set if getattr(self, name) is not None
        )


class StoryChangeDraft(StrictModel):
    """One create or update, as the model writes it. No revision: the server binds it."""

    operation: StoryChangeOperation
    entity_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=MAX_ID_CHARS,
        description=(
            "Required for an update: the entity's exact name, or its id. "
            "Omit for a create."
        ),
    )
    plot_line_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=MAX_ID_CHARS,
        description=(
            "Required for event.create and event.update: the plot line the "
            "event belongs to, by exact name or id."
        ),
    )
    fields: StoryChangeFields

    @property
    def kind(self) -> str:
        return self.operation.split(".", 1)[0]

    @property
    def is_create(self) -> bool:
        return self.operation.endswith(".create")

    @model_validator(mode="after")
    def _check_shape(self) -> "StoryChangeDraft":
        # Providers often emit explicit nulls for optional fields; treat as unset.
        set_fields = self.fields.set_names()
        if not set_fields:
            raise ValueError("a change must set at least one field")
        unexpected = set_fields - STORY_CHANGE_FIELDS[self.kind]
        if unexpected:
            raise ValueError(
                f"{self.kind} changes cannot set: {', '.join(sorted(unexpected))}"
            )
        if self.is_create:
            if self.entity_id is not None:
                raise ValueError("a create cannot name an existing entity")
            if self.fields.name is None:
                raise ValueError("a create requires a name")
        elif self.entity_id is None:
            raise ValueError("an update requires entityId")
        if (self.kind == "event") != (self.plot_line_id is not None):
            raise ValueError("plotLineId is required for events and only for events")
        return self


class StoryChange(StoryChangeDraft):
    """A draft plus what the server bound: the target's revision and a label."""

    base_revision: Optional[int] = Field(default=None, ge=0)
    label: str = Field(min_length=1, max_length=MAX_ENTITY_NAME_CHARS)

    @model_validator(mode="after")
    def _check_revision(self) -> "StoryChange":
        if self.is_create != (self.base_revision is None):
            raise ValueError(
                "baseRevision is required for updates and only for updates"
            )
        return self


class ProposeStoryChangesArgs(StrictModel):
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    reason: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SUMMARY_CHARS
    )
    changes: list[StoryChange] = Field(min_length=1, max_length=MAX_STORY_CHANGES)


class StoryChangeResult(StrictModel):
    """Bounded browser report for one change. It never authorizes a server write."""

    index: int = Field(ge=0, lt=MAX_STORY_CHANGES)
    status: Literal["applied", "stale", "failed", "skipped"]
    entity_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_ID_CHARS
    )


class EntityContinuation(StrictModel):
    """Stateless second request after the writer decides on a story-change proposal."""

    kind: Literal["entity_approval"]
    previous_run_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    approval_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    tool_call_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    proposal_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    decision: Literal["applied", "rejected", "revision_requested", "apply_failed"]
    proposal: ProposeStoryChangesArgs
    results: Optional[list[StoryChangeResult]] = Field(
        default=None, min_length=1, max_length=MAX_STORY_CHANGES
    )
    feedback: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SUMMARY_CHARS
    )


def _continuation_kind(value: Any) -> str:
    # Editor continuations predate the tag and may omit it.
    if isinstance(value, dict):
        return str(value.get("kind", "editor_approval"))
    return str(getattr(value, "kind", "editor_approval"))


Continuation = Annotated[
    Union[
        Annotated[EditorContinuation, Tag("editor_approval")],
        Annotated[EntityContinuation, Tag("entity_approval")],
    ],
    Discriminator(_continuation_kind),
]


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
    buffer: Optional[EditorTextWindow] = None
    dirty: bool = False


class UserMessage(StrictModel):
    """v1 user input is text-only; the list is for forward room, not features."""

    role: Literal["user"]
    parts: list[UserTextPart] = Field(min_length=1, max_length=MAX_PARTS_PER_MESSAGE)


class RunRequest(StrictModel):
    """What the browser sends. Carries no identity -- see the module docstring."""

    v: Literal[ASSISTANT_PROTOCOL_VERSION]  # type: ignore[valid-type]  # Runtime schema shares the version constant.
    story_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    thread_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_ID_CHARS
    )
    client_message_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    message: UserMessage
    editor_context: Optional[EditorContext] = None
    continuation: Optional[Continuation] = None


class ProviderConfig(StrictModel):
    """A caller's own provider credentials, resolved by the gateway.

    Like ``user_id`` this is gateway-owned: the browser sends its key to the
    Function once, over the encrypted settings path, and never on a run request.
    The gateway decrypts it per run and the agent forwards it to creditProxy,
    which bills the user's provider instead of platform credits.
    """

    provider: Literal["gemini", "claude", "openai"]
    api_key: str = Field(min_length=1, max_length=MAX_API_KEY_CHARS)
    model: Optional[str] = Field(default=None, min_length=1, max_length=MAX_ID_CHARS)


class AgentRunRequest(RunRequest):
    """What the gateway forwards: the browser's request plus a verified uid.

    ``user_id`` is derived from the Firebase token by the Functions gateway. It
    is never read from a browser body and never exposed as a tool argument.
    ``provider_config`` carries the same guarantee for BYOK credentials.
    """

    user_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    provider_config: Optional[ProviderConfig] = None
