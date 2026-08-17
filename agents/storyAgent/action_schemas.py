"""Validation schemas for StoryAgent actions."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ActionName = Literal[
    "brainstormIdeas",
    "generateNextLines",
    "chatWithContext",
    "enhanceText",
    "enhanceWizardInput",
    "generateStoryChoices",
    "summarizeChapter",
]

MAX_CONTENT_CHARS = 100_000
MAX_ID_CHARS = 128
MAX_PROMPT_CHARS = 10_000


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


class BrainstormIdeasParams(StrictModel):
    story_id: str = _story_id_field()
    idea_type: str = Field(
        max_length=MAX_PROMPT_CHARS,
        validation_alias=AliasChoices("type", "idea_type"),
        serialization_alias="type",
    )
    prompt: Optional[str] = Field(default=None, max_length=MAX_PROMPT_CHARS)
    count: int = Field(default=5, ge=1, le=20)


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
    "brainstormIdeas": BrainstormIdeasParams,
    "generateNextLines": GenerateNextLinesParams,
    "chatWithContext": ChatWithContextParams,
    "enhanceText": EnhanceTextParams,
    "enhanceWizardInput": EnhanceWizardInputParams,
    "generateStoryChoices": GenerateStoryChoicesParams,
    "summarizeChapter": SummarizeChapterParams,
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
