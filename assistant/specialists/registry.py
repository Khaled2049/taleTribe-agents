"""The specialists a run may consult. Server-owned; a model cannot add one."""

from __future__ import annotations

from assistant.specialists.architect import STORY_ARCHITECT
from assistant.specialists.base import Specialist
from assistant.specialists.character_editor import CHARACTER_EDITOR
from assistant.specialists.critic import CRITIC
from assistant.specialists.drafter import DRAFTER

SPECIALISTS: dict[str, Specialist] = {
    specialist.id: specialist
    for specialist in (STORY_ARCHITECT, CHARACTER_EDITOR, CRITIC, DRAFTER)
}


def roster() -> str:
    """What the director is told about whom it can consult."""
    return "\n".join(
        f"- {specialist.id}: {specialist.description}"
        for specialist in SPECIALISTS.values()
    )
