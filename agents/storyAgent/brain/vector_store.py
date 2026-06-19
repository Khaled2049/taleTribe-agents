"""Shared embed → store → top-k primitive for every vector collection.

Before this existed, ChapterRAG (native ``find_nearest``) and the brain's
SemanticMemoryLayer (brute-force cosine) implemented the same job two different
ways, with two failure modes and two places that could drift on embedding
dimension. This is the single mechanism both delegate to, so there is exactly one
retrieval implementation and one dimension contract (asserted once at startup —
see embedding_provider.verify_embedding_dimension).

Retrieval is native Firestore KNN (``find_nearest``) ONLY. There is deliberately no
brute-force scan fallback: an un-indexed query would read up to the whole collection
per call, which is the cost/latency cliff we refuse to ship. When native search is
unavailable (no/!READY index, emulator), ``query`` returns ``[]`` and the caller
degrades gracefully (chat still answers from the slim roster). Provision and build
the vector index (firestore.indexes.json) before relying on retrieval.

It is deliberately collection-agnostic: callers pass the target CollectionReference
per call, because ChapterRAG's collection varies by story while the brain's is
fixed per context. Doc-id schemes and delete-before-reindex semantics stay with the
callers, since those encode each store's lifecycle (e.g. clearMemory must wipe
semantic memory but never chapters).

See wiki/chat-scaling-design.md (#2).
"""

import logging
from typing import Any, List, Optional

import anyio

logger = logging.getLogger(__name__)

# Firestore caps a WriteBatch at 500 operations per commit.
_BATCH_LIMIT = 500


class VectorStore:
    """Embed text, store it with its embedding, and retrieve top-k via native
    Firestore KNN (``find_nearest``).

    Retrieval is native-only by design — see the module docstring. When native
    search is unavailable, ``query`` returns ``[]`` rather than scanning the
    collection.
    """

    def __init__(self, embedder):
        self._embedder = embedder

    # ---- write path -------------------------------------------------------

    async def upsert(self, collection, doc_id: str, text: str, metadata: dict) -> None:
        """Embed ``text`` and write ``metadata`` + the embedding under ``doc_id``.
        Storing the embedding as a Firestore Vector is what makes find_nearest
        indexable; on older clients it falls back to a plain list (read path
        tolerates both)."""
        embedding = await self._embedder.embed(text)
        doc = {**metadata, "text": text, "embedding": _make_vector(embedding)}
        await anyio.to_thread.run_sync(lambda: collection.document(doc_id).set(doc))

    # ---- read path --------------------------------------------------------

    async def query(
        self,
        collection,
        text: str,
        top_k: int,
        query_embedding: Optional[List[float]] = None,
    ) -> List[dict]:
        """Top-k most similar docs to ``text`` as plain dicts, each with its
        Firestore document id injected under ``"id"``.

        Pass ``query_embedding`` to reuse a vector already computed for this message
        (e.g. chapter retrieval and semantic memory both query on the full message —
        embedding it once and sharing avoids a redundant embedding API call).

        Native KNN only: returns ``[]`` (never a brute-force scan) when there's no
        embedding to query with or native search is unavailable.
        """
        if query_embedding is None:
            if not text or self._embedder is None:
                return []
            query_embedding = await self._embedder.embed(text)
        native = await self._query_native(collection, query_embedding, top_k)
        return native if native is not None else []

    async def _query_native(
        self, collection, query_embedding: List[float], top_k: int
    ) -> Optional[List[dict]]:
        """Firestore KNN via find_nearest. Returns None (not []) when native search
        is unavailable, so ``query`` yields no excerpts (no scan fallback)."""
        try:
            from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
            from google.cloud.firestore_v1.vector import Vector
        except ImportError:
            return None

        def _run() -> Optional[List[dict]]:
            try:
                vq = collection.find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_embedding),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=top_k,
                )
                return [_with_id(d) for d in vq.get()]
            except Exception as exc:  # no/!READY index, emulator, etc.
                logger.warning(
                    "vector_store_native_unavailable err=%s "
                    "(returning no excerpts; provision/build the vector index)",
                    exc,
                )
                return None

        return await anyio.to_thread.run_sync(_run)

    # ---- delete helpers ---------------------------------------------------

    async def delete_where(self, collection, field: str, value: Any) -> int:
        """Delete every doc where ``field == value``. Returns the count removed."""

        def _delete() -> int:
            docs = list(collection.where(field, "==", value).stream())
            return _batched_delete(collection, docs)

        return await anyio.to_thread.run_sync(_delete)

    async def delete_all(self, collection) -> int:
        """Delete every doc in the collection. Returns the count removed."""

        def _delete() -> int:
            docs = list(collection.stream())
            return _batched_delete(collection, docs)

        return await anyio.to_thread.run_sync(_delete)


def _batched_delete(collection, docs) -> int:
    """Delete ``docs`` using Firestore WriteBatch (≤500 ops/commit) instead of one
    RPC per doc. Re-indexing a long chapter deletes ~dozens of chunks; batching turns
    that into a couple of commits. Returns the count removed."""
    client = collection._client  # the Client the collection belongs to
    for start in range(0, len(docs), _BATCH_LIMIT):
        batch = client.batch()
        for d in docs[start : start + _BATCH_LIMIT]:
            batch.delete(d.reference)
        batch.commit()
    return len(docs)


def _with_id(snapshot) -> dict:
    data = snapshot.to_dict() or {}
    data["id"] = snapshot.id
    return data


def _make_vector(embedding: List[float]):
    """Wrap an embedding as a Firestore Vector when available, else a plain list.

    Storing as Vector is what makes find_nearest indexable. On older clients without
    the Vector type, we store a plain list — but such docs are NOT indexed by
    find_nearest and (since there is no brute-force fallback) are effectively
    invisible to retrieval. Keep the firestore client current in any real deployment."""
    try:
        from google.cloud.firestore_v1.vector import Vector

        return Vector(embedding)
    except ImportError:
        return embedding


def _to_list(embedding: Any) -> List[float]:
    """Normalize an embedding read back from Firestore (Vector or list) to a list."""
    if embedding is None:
        return []
    value = getattr(embedding, "value", None)
    if value is not None:
        return list(value)
    return list(embedding)
