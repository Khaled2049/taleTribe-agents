"""Semantic memory layer — facts, characters, lore, retrieved via embedding search."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from google.cloud import firestore
import anyio
import numpy as np

from ..types import MemoryDocument
from .constants import MEMORY_FETCH_LIMIT

logger = logging.getLogger(__name__)


class SemanticMemoryLayer:
    def __init__(self, db: firestore.Client, context_id: str, embedder):
        self._db = db
        self._context_id = context_id
        self._embedder = embedder

    def _collection(self):
        return self._db.collection("stories").document(self._context_id).collection("semantic_memory")

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryDocument]:
        if not query:
            return []

        embedding_list = await self._embedder.embed(query)
        query_vec = np.array(embedding_list, dtype=np.float32)

        def _fetch_recent():
            return [
                doc
                for doc in self._collection()
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(MEMORY_FETCH_LIMIT)
                .stream()
            ]

        docs = await anyio.to_thread.run_sync(_fetch_recent)
        if not docs:
            return []

        scored = []
        for doc in docs:
            data = doc.to_dict()
            emb = data.get("embedding")
            if not emb:
                continue
            score = _cosine_similarity(query_vec, np.array(emb, dtype=np.float32))
            scored.append((score, doc.id, data))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc_id, data in scored[:top_k]:
            results.append(MemoryDocument(
                id=doc_id,
                text=data.get("text", ""),
                embedding=data.get("embedding", []),
                created_at=data.get("created_at", datetime.now(timezone.utc)),
                data=data.get("data", {}),
                type=data.get("type", ""),
            ))
        return results

    async def store(self, text: str, type: str = "", data: dict[str, Any] | None = None) -> str:
        embedding = await self._embedder.embed(text)
        doc_id = str(uuid.uuid4())
        doc = {
            "text": text,
            "type": type,
            "data": data or {},
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc),
        }
        def _set():
            self._collection().document(doc_id).set(doc)
        await anyio.to_thread.run_sync(_set)
        return doc_id


    async def clear(self) -> None:
        def _delete_all():
            for doc in self._collection().stream():
                doc.reference.delete()
        await anyio.to_thread.run_sync(_delete_all)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:  # mismatched dims (e.g. old 384-dim vs new 768-dim) → skip
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
