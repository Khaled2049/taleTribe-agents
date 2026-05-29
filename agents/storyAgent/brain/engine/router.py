"""Memory router — decides which layers to query per request."""

from dataclasses import dataclass

BRAINSTORM_HINTS = {
    "brainstorm",
    "brainstormcharacter",
    "brainstormplot",
    "brainstormideas",
}


@dataclass
class RouterDecision:
    fetch_working: bool = True
    fetch_procedural: bool = True
    fetch_semantic: bool = True
    fetch_episodic: bool = True
    semantic_query: str = ""
    episodic_query: str = ""


class MemoryRouter:
    def route(self, user_message: str, action_hint: str = "") -> RouterDecision:
        hint = action_hint.lower().replace("_", "")
        is_brainstorm = hint in BRAINSTORM_HINTS

        return RouterDecision(
            fetch_working=True,
            fetch_procedural=True,
            # Brainstorming doesn't need episodic history — avoid noise
            fetch_semantic=not is_brainstorm,
            fetch_episodic=not is_brainstorm,
            semantic_query=user_message,
            episodic_query=user_message[-200:] if user_message else "",
        )
