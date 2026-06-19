"""Chapter RAG — embed chapter content on write, retrieve relevant excerpts on read.

This is the production fix for the chat-context scaling problem: instead of
re-sending whole chapters per message (cost grows with book length) or sending
only chapter titles (no real recall), we index each chapter's body once at write
time and retrieve only the top-k relevant chunks per user message.

Storage: stories/{storyId}/chapter_chunks/{chunkId}
  { kind, chapterId/entityId, chapterNumber/name, chunkIndex, text, embedding(Vector), createdAt }

The embed → store → top-k mechanics live in the shared ``VectorStore`` (native
``find_nearest`` only — no brute-force fallback; an unavailable/!READY index or the
emulator yields no excerpts). This module owns only the chapter/entity-specific
concerns: chunking, doc-id schemes, delete-before-reindex, and how a retrieved chunk
is rendered into the chat prompt.

See wiki/chat-scaling-design.md.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from google.cloud import firestore

from .brain.vector_store import VectorStore
from .entity_schema import ENTITY_FIELD_SCHEMA

logger = logging.getLogger(__name__)

# Words per chunk and overlap between consecutive chunks. ~250 words keeps each
# chunk well under any embedding model's input limit while staying large enough
# to carry a coherent scene beat; overlap avoids splitting a thought across the
# chunk boundary so retrieval doesn't miss it.
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


class ChapterRAG:
    """Indexes chapter bodies and retrieves relevant excerpts via vector search.

    `embedder` is the shared EmbeddingProvider from StoryAgent. `db` is the shared
    Firestore client. Both are injected so we reuse the process-wide singletons
    rather than reconnecting per request. Retrieval/storage is delegated to a shared
    ``VectorStore`` over the chapter_chunks collection (native KNN, which is
    provisioned for this collection in firestore.indexes.json).
    """

    def __init__(self, db: firestore.Client, embedder):
        self._db = db
        self._embedder = embedder
        self._store = VectorStore(embedder)

    def _collection(self, story_id: str):
        return (
            self._db.collection("stories")
            .document(story_id)
            .collection("chapter_chunks")
        )

    # ---- write path -------------------------------------------------------

    async def index_chapter(
        self,
        story_id: str,
        chapter_id: str,
        title: str,
        content: str,
        chapter_number: Optional[int] = None,
    ) -> int:
        """(Re)index one chapter. Deletes any prior chunks for it, then writes new
        ones. Returns the number of chunks written. Embedding cost is paid here,
        once per edit — never per chat message."""
        await self.delete_chapter(story_id, chapter_id)

        chunks = _chunk_text(content or "")
        if not chunks:
            return 0

        col = self._collection(story_id)
        now = datetime.now(timezone.utc)
        for idx, chunk in enumerate(chunks):
            metadata = {
                "kind": "chapter",
                "chapterId": chapter_id,
                "chapterNumber": chapter_number,
                "title": title,
                "chunkIndex": idx,
                "createdAt": now,
            }
            await self._store.upsert(col, f"{chapter_id}_{idx}", chunk, metadata)

        logger.info(
            "chapter_rag_indexed story_id=%s chapter_id=%s chunks=%d",
            story_id,
            chapter_id,
            len(chunks),
        )
        return len(chunks)

    async def delete_chapter(self, story_id: str, chapter_id: str) -> int:
        """Remove all chunks for a chapter (on chapter edit before re-index, or
        on chapter delete). Returns count removed."""
        removed = await self._store.delete_where(
            self._collection(story_id), "chapterId", chapter_id
        )
        if removed:
            logger.info(
                "chapter_rag_deleted story_id=%s chapter_id=%s chunks=%d",
                story_id,
                chapter_id,
                removed,
            )
        return removed

    async def index_entity(
        self,
        story_id: str,
        kind: str,
        entity_id: str,
        data: dict,
    ) -> int:
        """(Re)index one metadata entity (character/place/plot) into the same vector
        collection as chapters. Lets chat retrieve a character's backstory, a place
        description, or a plot's events on demand — not just their names."""
        await self.delete_entity(story_id, entity_id)

        text = compose_entity_text(kind, data or {})
        chunks = _chunk_text(text)
        if not chunks:
            return 0

        name = _entity_name(data or {})
        col = self._collection(story_id)
        now = datetime.now(timezone.utc)
        for idx, chunk in enumerate(chunks):
            metadata = {
                "kind": kind,
                "entityId": entity_id,
                "name": name,
                "chunkIndex": idx,
                "createdAt": now,
            }
            await self._store.upsert(col, f"{kind}_{entity_id}_{idx}", chunk, metadata)

        logger.info(
            "entity_rag_indexed story_id=%s kind=%s entity_id=%s chunks=%d",
            story_id,
            kind,
            entity_id,
            len(chunks),
        )
        return len(chunks)

    async def delete_entity(self, story_id: str, entity_id: str) -> int:
        """Remove all chunks for an entity (on edit before re-index, or on delete)."""
        removed = await self._store.delete_where(
            self._collection(story_id), "entityId", entity_id
        )
        if removed:
            logger.info(
                "entity_rag_deleted story_id=%s entity_id=%s chunks=%d",
                story_id,
                entity_id,
                removed,
            )
        return removed

    # ---- read path --------------------------------------------------------

    async def retrieve(
        self,
        story_id: str,
        query: str,
        top_k: int = 4,
        query_embedding: Optional[List[float]] = None,
    ) -> List[dict]:
        """Return the top-k most relevant chunks (chapters + entities) for `query`.

        Pass ``query_embedding`` to reuse a vector already computed for this message
        (shared with semantic memory, which queries on the same full message).

        Native vector search only (flat cost regardless of book length), via the shared
        VectorStore. There is no brute-force fallback: if this returns nothing, the
        cause is the index (not built / !READY) or the emulator — not a silent fallback.
        """
        results = await self._store.query(
            self._collection(story_id), query, top_k, query_embedding=query_embedding
        )
        return [self._to_excerpt(data) for data in results]

    @staticmethod
    def _to_excerpt(data: dict) -> dict:
        return {
            "kind": data.get("kind", "chapter"),
            "chapterNumber": data.get("chapterNumber"),
            "title": data.get("title", ""),
            "name": data.get("name", ""),
            "text": data.get("text", ""),
        }


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
