"""Strict argument schemas for the assistant's tools. Schemas only -- no executors.

Phase 3 ports the read tools, Phase 5 the editor tools, Phase 6 research. What
Phase 1 owns is the shape, the bounds, and one structural invariant.

**No tool argument model declares an identity field.** Not "identity fields are
validated", not "identity fields are overwritten by the server" -- they do not
exist. ``user_id`` and ``story_id`` live on ``ToolContext``, which the
orchestrator constructs from the verified Firebase token and the ownership gate
in stories.py, and which no model output ever touches. Combined with
``extra="forbid"``, a model that tries to assert whose story to read produces a
validation error rather than a read. This is stronger than checking the value
afterwards, because it removes the field a check could be forgotten on, and
``test_assistant_tools.py`` asserts it over every registered schema so it cannot
erode as Phases 3-6 add tools.

Model output is untrusted even when it matches a schema. The bounds here are the
first of two checks; Phase 3 re-validates at the execution boundary, where the
result sizes these arguments imply are also capped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from assistant.protocol import (
    MAX_ID_CHARS,
    MAX_SELECTION_CHARS,
    MAX_SUMMARY_CHARS,
    ProposeEditorEditArgs,
    StrictModel,
)

MAX_QUERY_CHARS = 500
MAX_TOOL_RESULTS = 20
MAX_CHAPTER_WINDOW_CHARS = 20_000
MAX_RESEARCH_RESULTS = 5

EntityKind = Literal["character", "place", "plot"]

# Every spelling of "who is asking" and "which story", in both cases. A tool
# argument model that declares one of these has reopened the hole the
# ToolContext split exists to close.
IDENTITY_FIELD_NAMES = frozenset(
    {
        "uid",
        "user_id",
        "userId",
        "owner_id",
        "ownerId",
        "story_id",
        "storyId",
        "firebase_token",
        "firebaseToken",
    }
)


@dataclass(frozen=True)
class ToolContext:
    """Server-owned execution scope. Never serialized into a model prompt.

    Frozen, and a plain dataclass rather than a pydantic model on purpose: there
    is no ``model_validate`` on it, so there is no code path that builds one
    from parsed JSON.
    """

    user_id: str
    story_id: str


class GetStoryOverviewArgs(StrictModel):
    """Metadata and ordered chapter titles. Scope comes entirely from context."""


class SearchStoryArgs(StrictModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    limit: int = Field(default=8, ge=1, le=MAX_TOOL_RESULTS)


class ListStoryEntitiesArgs(StrictModel):
    kind: EntityKind
    limit: int = Field(default=20, ge=1, le=MAX_TOOL_RESULTS)


class GetStoryEntityArgs(StrictModel):
    kind: EntityKind
    entity_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)


class ReadChapterArgs(StrictModel):
    """A bounded window, so a read cannot pull a whole manuscript into a prompt."""

    chapter_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8_000, ge=1, le=MAX_CHAPTER_WINDOW_CHARS)


class ReadCurrentEditorArgs(StrictModel):
    """The live buffer the browser supplied on the run request, not a fresh read."""

    selection_only: bool = True


class ApplyEditorEditArgs(StrictModel):
    proposal_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)


class ProposeEditorEditDraft(StrictModel):
    """Model-authored content; the server supplies every editor coordinate."""

    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    replacement_text: str = Field(
        default="", min_length=0, max_length=MAX_SELECTION_CHARS
    )


class ResearchWebArgs(StrictModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    max_results: int = Field(default=3, ge=1, le=MAX_RESEARCH_RESULTS)


# Read tools are always available; the other two families are gated on the
# server flags declared in P0-T4, which today gate nothing else because no
# executor exists yet.
READ_TOOLS: dict[str, type[BaseModel]] = {
    "get_story_overview": GetStoryOverviewArgs,
    "search_story": SearchStoryArgs,
    "list_story_entities": ListStoryEntitiesArgs,
    "get_story_entity": GetStoryEntityArgs,
    "read_chapter": ReadChapterArgs,
    "read_current_editor": ReadCurrentEditorArgs,
}

EDIT_TOOLS: dict[str, type[BaseModel]] = {
    "propose_editor_edit": ProposeEditorEditArgs,
    "apply_editor_edit": ApplyEditorEditArgs,
}

# Only proposals are provider-facing in Phase 5. The apply schema is retained
# for the synthesized browser approval part and continuation validation.
MODEL_EDIT_TOOLS: dict[str, type[BaseModel]] = {
    "propose_editor_edit": ProposeEditorEditDraft,
}

RESEARCH_TOOLS: dict[str, type[BaseModel]] = {
    "research_web": ResearchWebArgs,
}

# Adding a tool here also requires a ``Capability`` in ``assistant/help.py``:
# the browser renders /help from that catalog without asking the model, and
# ``tests/test_assistant_help.py`` asserts the catalog partitions this map, so
# an undescribed tool fails rather than going quietly unmentioned.
TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    **READ_TOOLS,
    **EDIT_TOOLS,
    **RESEARCH_TOOLS,
}

# Tools whose effect requires explicit user approval before it is applied.
# propose_editor_edit is absent on purpose: producing a proposal changes nothing.
APPROVAL_REQUIRED = frozenset({"apply_editor_edit"})


class UnknownToolError(ValueError):
    """The model named a tool outside the server-owned allowlist."""


def available_tools(
    *, edits_enabled: bool, research_enabled: bool
) -> dict[str, type[BaseModel]]:
    """The allowlist for one run. Server-owned; a model cannot widen it."""
    tools = dict(READ_TOOLS)
    if edits_enabled:
        tools.update(MODEL_EDIT_TOOLS)
    if research_enabled:
        tools.update(RESEARCH_TOOLS)
    return tools


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate model-supplied arguments and return the normalized camelCase payload.

    Mirrors ``action_schemas.validate_action_parameters`` so the two validation
    entry points in this service behave the same way.
    """
    schema = TOOL_SCHEMAS.get(name)
    if schema is None:
        raise UnknownToolError(f"Unknown tool: {name}")
    validated = schema.model_validate(arguments)
    return validated.model_dump(by_alias=True, exclude_none=True)
