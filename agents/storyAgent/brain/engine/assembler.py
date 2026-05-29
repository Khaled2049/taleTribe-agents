"""Prompt assembler — builds layered context string from memory layers."""

from ..types import (
    AssembledPrompt,
    MemoryDocument,
    ProceduralMemoryState,
    WorkingMemoryState,
)


class PromptAssembler:
    def assemble(
        self,
        user_message: str,
        working: WorkingMemoryState | None,
        procedural: ProceduralMemoryState | None,
        semantic_docs: list[MemoryDocument],
        episodic_docs: list[MemoryDocument],
    ) -> AssembledPrompt:
        parts = []
        procedural_injected = False
        working_injected = False

        if procedural and _has_content(procedural):
            parts.append("=== WRITER STYLE & PREFERENCES ===")
            if procedural.tone:
                parts.append(f"Tone: {procedural.tone}")
            if procedural.style:
                parts.append(f"Style: {procedural.style}")
            if procedural.pov:
                parts.append(f"POV: {procedural.pov}")
            if procedural.genre:
                parts.append(f"Genre: {procedural.genre}")
            if procedural.narrative_rules:
                parts.append("Rules:")
                for rule in procedural.narrative_rules:
                    parts.append(f"- {rule}")
            procedural_injected = True

        if working and _has_working_content(working):
            parts.append("\n=== CURRENT SCENE ===")
            if working.current_scene:
                parts.append(f"Scene: {working.current_scene}")
            if working.active_characters:
                parts.append(
                    f"Active characters: {', '.join(working.active_characters)}"
                )
            if working.recent_events:
                parts.append("Recent events:")
                for event in working.recent_events:
                    parts.append(f"- {event}")
            if working.mood:
                parts.append(f"Mood: {working.mood}")
            working_injected = True

        if semantic_docs:
            parts.append("\n=== RELEVANT FACTS & LORE ===")
            for doc in semantic_docs:
                parts.append(f"- {doc.text}")

        if episodic_docs:
            parts.append("\n=== RELEVANT PAST EVENTS ===")
            for doc in episodic_docs:
                entry = doc.summary if doc.summary else doc.text
                parts.append(f"- {entry}")

        parts.append(f"\n=== CURRENT REQUEST ===\n{user_message}")

        return AssembledPrompt(
            text="\n".join(parts),
            working_injected=working_injected,
            procedural_injected=procedural_injected,
            semantic_count=len(semantic_docs),
            episodic_count=len(episodic_docs),
        )


def _has_content(p: ProceduralMemoryState) -> bool:
    return bool(p.tone or p.style or p.pov or p.genre or p.narrative_rules)


def _has_working_content(w: WorkingMemoryState) -> bool:
    return bool(w.current_scene or w.active_characters or w.recent_events or w.mood)
