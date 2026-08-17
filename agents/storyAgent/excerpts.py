"""Render retrieved vector chunks for a chat prompt.

Extracted from chapter_rag.py when the Firestore vector path was deleted: the
chunks now come from pgvector via postgres_context.retrieve(), but the metadata
shape they carry — kind, title, chapterNumber, name — is unchanged, so the
formatter moved across intact.
"""

from typing import List


def format_excerpts(excerpts: List[dict]) -> str:
    """Render retrieved chunks for the chat prompt. Empty string when none.

    Handles every chunk kind: chapter excerpts, plus character/place/plot details."""
    if not excerpts:
        return ""
    lines = ["RELEVANT STORY DETAILS (retrieved for this question):"]
    for e in excerpts:
        kind = e.get("kind") or "chapter"
        if kind == "chapter":
            num = e.get("chapterNumber")
            label = f"Ch{num}" if num is not None else "Chapter"
            title = e.get("title") or ""
            head = f"{label}: {title}".strip().rstrip(":").strip()
        else:
            name = e.get("name") or e.get("title") or "Untitled"
            head = f"{kind.capitalize()}: {name}"
        lines.append(f"[{head}] {e.get('text', '')}")
    return "\n".join(lines)
