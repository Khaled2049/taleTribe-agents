"""Validation schemas for StoryAgent actions."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ActionName = Literal[
    "generateChapter",
    "brainstormIdeas",
    "brainstormCharacter",
    "brainstormPlot",
    "generateNextLines",
    "chatWithContext",
    "enhanceText",
    "enhanceWizardInput",
    "generateStoryChoices",
    "summarizeChapter",
    "clearMemory",
    "indexChapter",
    "deleteChapterChunks",
    "indexEntity",
    "deleteEntityChunks",
]

MAX_CONTENT_CHARS = 100_000
MAX_ID_CHARS = 128
MAX_PROMPT_CHARS = 10_000
# Bounds for the chapter-continuity payload. Neighbor bodies are truncated by
# the caller (Cloud Functions) before they get here; these are defensive
# ceilings so a misbehaving caller can't balloon the prompt (and the bill).
MAX_NEIGHBOR_CONTENT_CHARS = 8_000


def _story_id_field() -> Any:
    return Field(
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("storyId", "story_id"),
        serialization_alias="storyId",
    )


def _chapter_id_field() -> Any:
    return Field(
        default=None,
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("chapterId", "chapter_id"),
        serialization_alias="chapterId",
    )


def _user_id_field(default: Any = None) -> Any:
    return Field(
        default=default,
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
    )


class StrictModel(BaseModel):
    """Base model with strict unknown-field handling."""

    model_config = ConfigDict(extra="forbid")


class LenientModel(BaseModel):
    """Base for nested continuity payloads — ignores unknown fields (so adding a
    field on the caller side won't break validation) while still enforcing the
    size ceilings below."""

    model_config = ConfigDict(extra="ignore")


class NeighborChapterParam(LenientModel):
    chapter_number: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("chapterNumber", "chapter_number"),
        serialization_alias="chapterNumber",
    )
    order: Optional[float] = None
    title: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    content: Optional[str] = Field(default=None, max_length=MAX_NEIGHBOR_CONTENT_CHARS)


class GenerateChapterParams(StrictModel):
    story_id: str = _story_id_field()
    chapter_number: int = Field(
        validation_alias=AliasChoices("chapterNumber", "chapter_number"),
        serialization_alias="chapterNumber",
    )
    order: Optional[float] = Field(
        default=None,
        description="Float ordering key of the chapter being generated.",
    )
    previous_chapters: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        validation_alias=AliasChoices("previousChapters", "previous_chapters"),
        serialization_alias="previousChapters",
    )
    # Bounded continuity context (preferred over previous_chapters): truncated
    # text of the immediate neighbors only.
    prev_chapter: Optional[NeighborChapterParam] = Field(
        default=None,
        validation_alias=AliasChoices("prevChapter", "prev_chapter"),
        serialization_alias="prevChapter",
    )
    next_chapter: Optional[NeighborChapterParam] = Field(
        default=None,
        validation_alias=AliasChoices("nextChapter", "next_chapter"),
        serialization_alias="nextChapter",
    )
    plot_context: Optional[str] = Field(
        default=None,
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("plotContext", "plot_context"),
        serialization_alias="plotContext",
    )


class BrainstormIdeasParams(StrictModel):
    story_id: str = _story_id_field()
    idea_type: str = Field(
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("type", "idea_type"),
        serialization_alias="type",
    )
    prompt: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    count: int = Field(default=5, ge=1, le=20)


class BrainstormCharacterParams(StrictModel):
    story_id: str = _story_id_field()
    role: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    archetype: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)


class BrainstormPlotParams(StrictModel):
    story_id: str = _story_id_field()
    plot_type: str = Field(
        default="conflict",
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("plotType", "plot_type"),
        serialization_alias="plotType",
    )


class GenerateNextLinesParams(StrictModel):
    story_id: str = _story_id_field()
    content: str = Field(max_length=MAX_CONTENT_CHARS)
    cursor_position: int = Field(
        ge=0,
        validation_alias=AliasChoices("cursorPosition", "cursor_position"),
        serialization_alias="cursorPosition",
    )
    chapter_id: Optional[str] = _chapter_id_field()


class ChatWithContextParams(StrictModel):
    story_id: str = _story_id_field()
    message: str = Field(max_length=MAX_CONTENT_CHARS)
    context: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        validation_alias=AliasChoices("chatHistory", "chat_history"),
        serialization_alias="chatHistory",
    )
    user_id: Optional[str] = _user_id_field()


class EnhanceTextParams(StrictModel):
    story_id: str = _story_id_field()
    action: Literal["expand", "dialogue", "rewrite"]
    selected_text: str = Field(
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("selectedText", "selected_text"),
        serialization_alias="selectedText",
    )
    chapter_id: Optional[str] = _chapter_id_field()


class GenerateStoryChoicesParams(StrictModel):
    story_id: str = _story_id_field()
    mode: Literal["opening", "continuation", "ending"]
    current_content: str = Field(
        default="",
        max_length=MAX_CONTENT_CHARS,
        validation_alias=AliasChoices("currentContent", "current_content"),
        serialization_alias="currentContent",
    )
    chapter_id: Optional[str] = _chapter_id_field()
    turn_count: int = Field(
        default=0,
        validation_alias=AliasChoices("turnCount", "turn_count"),
        serialization_alias="turnCount",
    )
    user_id: Optional[str] = _user_id_field()


class SummarizeChapterParams(StrictModel):
    story_id: str = _story_id_field()
    content: str = Field(max_length=MAX_CONTENT_CHARS)
    chapter_id: Optional[str] = _chapter_id_field()


class ClearMemoryParams(StrictModel):
    story_id: str = _story_id_field()
    user_id: str = _user_id_field(default="anonymous")


class IndexChapterParams(StrictModel):
    story_id: str = _story_id_field()
    chapter_id: str = Field(
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("chapterId", "chapter_id"),
        serialization_alias="chapterId",
    )
    title: Optional[str] = Field(default="", max_length=MAX_PROMPT_CHARS)
    content: str = Field(default="", max_length=MAX_CONTENT_CHARS)
    chapter_number: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("chapterNumber", "chapter_number"),
        serialization_alias="chapterNumber",
    )


class DeleteChapterChunksParams(StrictModel):
    story_id: str = _story_id_field()
    chapter_id: str = Field(
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("chapterId", "chapter_id"),
        serialization_alias="chapterId",
    )


class IndexEntityParams(StrictModel):
    story_id: str = _story_id_field()
    kind: Literal["character", "place", "plot"]
    entity_id: str = Field(
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("entityId", "entity_id"),
        serialization_alias="entityId",
    )
    data: Dict[str, Any] = Field(default_factory=dict)


class DeleteEntityChunksParams(StrictModel):
    story_id: str = _story_id_field()
    entity_id: str = Field(
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("entityId", "entity_id"),
        serialization_alias="entityId",
    )


class EnhanceWizardInputParams(StrictModel):
    wizard_type: Literal["premise", "character", "place", "conflict", "blueprint"] = (
        Field(
            validation_alias=AliasChoices("type", "wizard_type"),
            serialization_alias="type",
        )
    )
    data: Dict[str, Any]
    user_id: str = _user_id_field(default=...)


_ACTION_SCHEMAS = {
    "generateChapter": GenerateChapterParams,
    "brainstormIdeas": BrainstormIdeasParams,
    "brainstormCharacter": BrainstormCharacterParams,
    "brainstormPlot": BrainstormPlotParams,
    "generateNextLines": GenerateNextLinesParams,
    "chatWithContext": ChatWithContextParams,
    "enhanceText": EnhanceTextParams,
    "enhanceWizardInput": EnhanceWizardInputParams,
    "generateStoryChoices": GenerateStoryChoicesParams,
    "summarizeChapter": SummarizeChapterParams,
    "clearMemory": ClearMemoryParams,
    "indexChapter": IndexChapterParams,
    "deleteChapterChunks": DeleteChapterChunksParams,
    "indexEntity": IndexEntityParams,
    "deleteEntityChunks": DeleteEntityChunksParams,
}


def validate_action_parameters(
    action: str, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate action parameters and return normalized camelCase payload."""
    schema = _ACTION_SCHEMAS.get(action)
    if not schema:
        raise ValueError(f"Unknown action: {action}")
    validated = schema.model_validate(parameters)
    return validated.model_dump(by_alias=True, exclude_none=True)
