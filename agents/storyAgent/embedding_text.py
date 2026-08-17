"""Text preparation for embedding: chunking prose, composing entity text.

Extracted from chapter_rag.py when the Firestore vector path was deleted. Both
helpers are backend-agnostic — the outbox worker in postgres_context.py chunks
canonical PostgreSQL text through them — so they outlived the RAG class they used
to sit beside.
"""

from typing import Any, List

from .entity_schema import ENTITY_FIELD_SCHEMA

# Chunks stay well under any embedding model's input limit while staying large
# enough to carry a coherent scene beat; the overlap avoids splitting a thought
# across a chunk boundary so retrieval doesn't miss it.
CHUNK_WORDS = 250
CHUNK_OVERLAP_WORDS = 40


def _chunk_text(text: str) -> List[str]:
    """Split prose into overlapping word windows. Returns [] for empty text."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP_WORDS)
    for start in range(0, len(words), step):
        window = words[start : start + CHUNK_WORDS]
        if window:
            chunks.append(" ".join(window))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def _entity_name(data: dict) -> str:
    return data.get("name") or data.get("title") or "Untitled"


def _stringify(value: Any) -> str:
    """Flatten list/dict field values (e.g. traits) into readable text."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def compose_entity_text(kind: str, data: dict) -> str:
    """Build the text we embed for a metadata entity from its known fields.

    Field names + order come from the shared ENTITY_FIELD_SCHEMA (entity_schema.py),
    which the full-context prompt builder also iterates — so the two can't drift. That
    schema mirrors the real Firestore schema (src/types/ICharacter.ts, IPlace.ts,
    IPlot.ts) and the frontend's SIGNATURE_FIELDS (entityIndexTrigger.ts), which decides
    when a re-embed fires. Unknown kinds fall back to name + description so nothing
    silently indexes as empty."""
    name = _entity_name(data)
    schema = ENTITY_FIELD_SCHEMA.get(kind)

    if schema is None:
        # Unknown kind: name + description fallback.
        parts = [f"{kind}: {name}"]
        if data.get("description"):
            parts.append(_stringify(data["description"]))
        return "\n".join(p for p in parts if p)

    parts: List[str] = [f"{kind.capitalize()}: {name}"]
    for field, label, _cap in schema:
        value = data.get(field)
        if value:
            parts.append(f"{label}: {_stringify(value)}")

    # Array-of-object fields (formatted bespoke, see ENTITY_ARRAY_FIELDS).
    if kind == "character":
        for rel in data.get("relationships") or []:
            if isinstance(rel, dict):
                rn = rel.get("name", "")
                rt = rel.get("type", "")
                rd = rel.get("description", "")
                if rn or rt or rd:
                    parts.append(f"Relationship - {rn} ({rt}): {rd}".strip())
    elif kind == "plot":
        for ev in data.get("events") or []:
            if isinstance(ev, dict):
                en = ev.get("name", "")
                ec = ev.get("content", "")
                if en or ec:
                    parts.append(f"Event - {en}: {ec}".strip())
            elif ev:
                parts.append(f"Event: {ev}")

    return "\n".join(p for p in parts if p)
