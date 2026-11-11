"""Validation schemas for StoryAgent actions."""

from typing import Any, Dict, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ActionName = Literal["enhanceWizardInput"]

MAX_CONTENT_CHARS = 100_000
MAX_ID_CHARS = 128
MAX_PROMPT_CHARS = 10_000


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


class EnhanceWizardInputParams(StrictModel):
    wizard_type: Literal["premise", "character", "place", "conflict", "blueprint"] = (
        Field(
            validation_alias=AliasChoices("type", "wizard_type"),
            serialization_alias="type",
        )
    )
    data: Dict[str, Any]
    user_id: str = _user_id_field(default=...)


_ACTION_SCHEMAS: Dict[str, type[StrictModel]] = {
    "enhanceWizardInput": EnhanceWizardInputParams,
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
