"""The one shape every specialist answers in.

Structured so the director can weigh two specialists against each other, but
deliberately shallow. A specialist's answer is forced through one tool call,
and providers compile a forced tool's schema into a decoding constraint: a
nested copy of the story-change schema was rejected outright ("too many states
for serving"). So a suggested change is a sentence about a named target, and
the director -- the only one allowed to call ``propose_story_changes`` -- turns
the ones the writer wants into a real, server-bound proposal.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import Field

from assistant.protocol import (
    MAX_ENTITY_NAME_CHARS,
    StoryChangeOperation,
    StrictModel,
)

MAX_ANALYSIS_CHARS = 3_000
MAX_RECOMMENDATIONS = 4
MAX_RECOMMENDATION_CHARS = 1_000
MAX_SUGGESTED_CHANGES = 4
MAX_SUGGESTED_CHANGE_CHARS = 600
MAX_RISKS = 4
MAX_RISK_CHARS = 300

# Keywords that only add states to a provider's decoding constraint. They stay
# on the model the server validates against.
_CONSTRAINT_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "default",
        "title",
    }
)


class Recommendation(StrictModel):
    title: str = Field(min_length=1, max_length=MAX_ENTITY_NAME_CHARS)
    detail: str = Field(min_length=1, max_length=MAX_RECOMMENDATION_CHARS)


class SuggestedChange(StrictModel):
    operation: StoryChangeOperation
    target: str = Field(
        min_length=1,
        max_length=MAX_ENTITY_NAME_CHARS,
        description="Name of the entity to change, or the name a new one should have.",
    )
    change: str = Field(
        min_length=1,
        max_length=MAX_SUGGESTED_CHANGE_CHARS,
        description="Which fields to set and to what, in one or two sentences.",
    )


class SpecialistFindings(StrictModel):
    """Submit your findings. Call this exactly once."""

    analysis: str = Field(
        min_length=1,
        max_length=MAX_ANALYSIS_CHARS,
        description="What is happening and why, grounded in the material.",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list, max_length=MAX_RECOMMENDATIONS
    )
    suggested_changes: list[SuggestedChange] = Field(
        default_factory=list,
        max_length=MAX_SUGGESTED_CHANGES,
        description="Optional concrete creates or updates the writer could accept.",
    )
    risks: list[Annotated[str, Field(min_length=1, max_length=MAX_RISK_CHARS)]] = Field(
        default_factory=list, max_length=MAX_RISKS
    )
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


def provider_schema(node: Any) -> Any:
    """The schema without the bounds a forced tool call cannot afford."""
    if isinstance(node, list):
        return [provider_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: (
            # A property may itself be named "title" or "default".
            {name: provider_schema(value) for name, value in child.items()}
            if key == "properties" and isinstance(child, dict)
            else provider_schema(child)
        )
        for key, child in node.items()
        if key not in _CONSTRAINT_KEYWORDS
    }
