"""Validation schemas for StoryAgent actions."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ActionName = Literal[
    "generateStory",
    "generateChapter",
    "brainstormIdeas",
    "brainstormCharacter",
    "brainstormPlot",
    "generateNextLines",
    "chatWithContext",
    "enhanceText",
    "enhanceWizardInput",
    "generateStoryChoices",
    "clearMemory",
]

MAX_CONTENT_CHARS = 100_000
MAX_ID_CHARS = 128
MAX_PROMPT_CHARS = 10_000

_STORY_ID = Field(
    max_length=MAX_ID_CHARS,
    validation_alias=AliasChoices("storyId", "story_id"),
    serialization_alias="storyId",
)
_CHAPTER_ID = Field(
    default=None,
    max_length=MAX_ID_CHARS,
    validation_alias=AliasChoices("chapterId", "chapter_id"),
    serialization_alias="chapterId",
)


class StrictModel(BaseModel):
    """Base model with strict unknown-field handling."""

    model_config = ConfigDict(extra="forbid")


class GenerateStoryParams(StrictModel):
    story_id: str = _STORY_ID
    genre: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    tone: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    length: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    generate_first_chapter_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("generateFirstChapterOnly", "generate_first_chapter_only"),
        serialization_alias="generateFirstChapterOnly",
    )
    plot_context: Optional[str] = Field(
        default=None,
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("plotContext", "plot_context"),
        serialization_alias="plotContext",
    )


class GenerateChapterParams(StrictModel):
    story_id: str = _STORY_ID
    chapter_number: int = Field(
        validation_alias=AliasChoices("chapterNumber", "chapter_number"),
        serialization_alias="chapterNumber",
    )
    previous_chapters: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        validation_alias=AliasChoices("previousChapters", "previous_chapters"),
        serialization_alias="previousChapters",
    )
    plot_context: Optional[str] = Field(
        default=None,
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("plotContext", "plot_context"),
        serialization_alias="plotContext",
    )


class BrainstormIdeasParams(StrictModel):
    story_id: str = _STORY_ID
    idea_type: str = Field(
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("type", "idea_type"),
        serialization_alias="type",
    )
    prompt: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    count: int = Field(default=5, ge=1, le=20)


class BrainstormCharacterParams(StrictModel):
    story_id: str = _STORY_ID
    role: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    archetype: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)


class BrainstormPlotParams(StrictModel):
    story_id: str = _STORY_ID
    plot_type: str = Field(
        default="conflict",
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("plotType", "plot_type"),
        serialization_alias="plotType",
    )


class GenerateNextLinesParams(StrictModel):
    story_id: str = _STORY_ID
    content: str = Field(max_length=MAX_CONTENT_CHARS)
    cursor_position: int = Field(
        ge=0,
        validation_alias=AliasChoices("cursorPosition", "cursor_position"),
        serialization_alias="cursorPosition",
    )
    chapter_id: Optional[str] = _CHAPTER_ID


class ChatWithContextParams(StrictModel):
    story_id: str = _STORY_ID
    message: str = Field(max_length=MAX_CONTENT_CHARS)
    context: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        validation_alias=AliasChoices("chatHistory", "chat_history"),
        serialization_alias="chatHistory",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
    )


class EnhanceTextParams(StrictModel):
    story_id: str = _STORY_ID
    action: Literal["expand", "dialogue", "rewrite"]
    selected_text: str = Field(
        max_length=MAX_CONTENT_CHARS,
        validation_alias=AliasChoices("selectedText", "selected_text"),
        serialization_alias="selectedText",
    )
    chapter_id: Optional[str] = _CHAPTER_ID


class GenerateStoryChoicesParams(StrictModel):
    story_id: str = _STORY_ID
    mode: Literal["opening", "continuation", "ending"]
    current_content: str = Field(
        default="",
        max_length=MAX_CONTENT_CHARS,
        validation_alias=AliasChoices("currentContent", "current_content"),
        serialization_alias="currentContent",
    )
    chapter_id: Optional[str] = _CHAPTER_ID
    turn_count: int = Field(
        default=0,
        validation_alias=AliasChoices("turnCount", "turn_count"),
        serialization_alias="turnCount",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
    )


class ClearMemoryParams(StrictModel):
    story_id: str = _STORY_ID
    user_id: str = Field(
        default="anonymous",
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
    )


class EnhanceWizardInputParams(StrictModel):
    wizard_type: Literal["premise", "character", "place", "conflict", "blueprint"] = Field(
        validation_alias=AliasChoices("type", "wizard_type"),
        serialization_alias="type",
    )
    data: Dict[str, Any]
    user_id: str = Field(
        max_length=MAX_ID_CHARS,
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
    )


_ACTION_SCHEMAS = {
    "generateStory": GenerateStoryParams,
    "generateChapter": GenerateChapterParams,
    "brainstormIdeas": BrainstormIdeasParams,
    "brainstormCharacter": BrainstormCharacterParams,
    "brainstormPlot": BrainstormPlotParams,
    "generateNextLines": GenerateNextLinesParams,
    "chatWithContext": ChatWithContextParams,
    "enhanceText": EnhanceTextParams,
    "enhanceWizardInput": EnhanceWizardInputParams,
    "generateStoryChoices": GenerateStoryChoicesParams,
    "clearMemory": ClearMemoryParams,
}


def validate_action_parameters(action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate action parameters and return normalized camelCase payload."""
    schema = _ACTION_SCHEMAS.get(action)
    if not schema:
        raise ValueError(f"Unknown action: {action}")
    validated = schema.model_validate(parameters)
    return validated.model_dump(by_alias=True, exclude_none=True)
