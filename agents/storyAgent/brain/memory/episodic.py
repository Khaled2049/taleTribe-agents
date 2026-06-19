"""Episodic memory layer — past events and session summaries, retrieved via embedding search.

Shares the common ``VectorStore`` mechanism with SemanticMemoryLayer and ChapterRAG
(one retrieval implementation, one dimension contract). Native KNN only (no
brute-force fallback). See wiki/chat-scaling-design.md (#2).
"""

import logging
import uuid
from datetime import datetime, timezone

from ..types import MemoryDocument
from ..vector_store import VectorStore, _to_list

logger = logging.getLogger(__name__)


class EpisodicMemoryLayer:
    def __init__(self, db, context_id: str, embedder):
        self._db = db
        self._context_id = context_id
        self._embedder = embedder
        self._store = VectorStore(embedder)

    def _collection(self):
        return (
            self._db.collection("stories")
            .document(self._context_id)
            .collection("episodic_memory")
        )

    async def retrieve(self, query: str, top_k: int = 3) -> list[MemoryDocument]:
        if not query:
            return []
        rows = await self._store.query(self._collection(), query, top_k)
        return [
            MemoryDocument(
                id=row.get("id", ""),
                text=row.get("text", ""),
                embedding=_to_list(row.get("embedding")),
                created_at=row.get("created_at", datetime.now(timezone.utc)),
                summary=row.get("summary", ""),
            )
            for row in rows
        ]

    async def store(self, text: str, summary: str) -> str:
        doc_id = str(uuid.uuid4())
        await self._store.upsert(
            self._collection(),
            doc_id,
            text,
            {"summary": summary, "created_at": datetime.now(timezone.utc)},
        )
        return doc_id

    async def clear(self) -> None:
        await self._store.delete_all(self._collection())
