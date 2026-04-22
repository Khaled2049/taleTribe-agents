"""Episodic memory layer — past events and session summaries, retrieved via embedding search."""
import logging
import uuid
from datetime import datetime, timezone
from google.cloud import firestore
import anyio
import numpy as np

from ..types import MemoryDocument
from .semantic import _cosine_similarity

logger = logging.getLogger(__name__)


class EpisodicMemoryLayer:
    def __init__(self, db: firestore.Client, context_id: str, embedder):
        self._db = db
        self._context_id = context_id
        self._embedder = embedder

    def _collection(self):
        return self._db.collection("stories").document(self._context_id).collection("episodic_memory")

    async def retrieve(self, query: str, top_k: int = 3) -> list[MemoryDocument]:
        if not query:
            return []

        query_vec = await anyio.to_thread.run_sync(
            lambda: self._embedder.encode(query, convert_to_numpy=True)
        )

        def _fetch_all():
            return [doc for doc in self._collection().stream()]

        docs = await anyio.to_thread.run_sync(_fetch_all)
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
                summary=data.get("summary", ""),
            ))
        return results

    async def store(self, text: str, summary: str) -> str:
        embedding = await anyio.to_thread.run_sync(
            lambda: self._embedder.encode(text, convert_to_numpy=True).tolist()
        )
        doc_id = str(uuid.uuid4())
        doc = {
            "text": text,
            "summary": summary,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc),
        }
        def _set():
            self._collection().document(doc_id).set(doc)
        await anyio.to_thread.run_sync(_set)
        return doc_id
