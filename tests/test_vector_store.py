"""Unit tests for the shared VectorStore (native find_nearest via in-memory fake).

Exercises upsert → query top-k ordering + id injection (through the native KNN path),
the no-fallback empty result when native search is unavailable, and the delete
helpers — without a real Firestore or vector index.
"""

import pytest

from agents.storyAgent.brain.embedding_provider import MockEmbeddingProvider
from agents.storyAgent.brain.vector_store import VectorStore, _to_list

pytestmark = pytest.mark.unit


# ---- minimal in-memory Firestore fake ------------------------------------


class _FakeDoc:
    def __init__(self, store, doc_id):
        self._store = store
        self.id = doc_id

    def set(self, data):
        self._store[self.id] = dict(data)


class _FakeSnapshot:
    def __init__(self, store, doc_id, data):
        self._store = store
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return dict(self._data)

    @property
    def reference(self):
        outer = self

        class _Ref:
            def delete(self_inner):
                outer._store.pop(outer.id, None)

        return _Ref()


class _FakeQuery:
    def __init__(self, store, items):
        self._store = store
        self._items = items

    def where(self, field, _op, value):
        return _FakeQuery(
            self._store, [(i, d) for i, d in self._items if d.get(field) == value]
        )

    def order_by(self, field, direction=None):
        return _FakeQuery(
            self._store,
            sorted(self._items, key=lambda kv: kv[1].get(field), reverse=True),
        )

    def limit(self, n):
        return _FakeQuery(self._store, self._items[:n])

    def stream(self):
        return [_FakeSnapshot(self._store, i, d) for i, d in self._items]


def _cos(a, b):
    """Pure-python cosine so the fake's KNN ranking has no numpy dependency."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class _FakeVectorQuery:
    def __init__(self, snapshots):
        self._snapshots = snapshots

    def get(self):
        return self._snapshots


class _FakeBatch:
    """Mimics a Firestore WriteBatch: queue deletes, apply on commit."""

    def __init__(self):
        self._ops = []

    def delete(self, ref):
        self._ops.append(ref)

    def commit(self):
        for ref in self._ops:
            ref.delete()
        self._ops = []


class _FakeClient:
    def batch(self):
        return _FakeBatch()


class _FakeCollection:
    def __init__(self):
        self._store = {}
        self._client = _FakeClient()

    def document(self, doc_id):
        return _FakeDoc(self._store, doc_id)

    def _all(self):
        return _FakeQuery(self._store, list(self._store.items()))

    def where(self, *a):
        return self._all().where(*a)

    def order_by(self, *a, **k):
        return self._all().order_by(*a, **k)

    def limit(self, n):
        return self._all().limit(n)

    def stream(self):
        return self._all().stream()

    def find_nearest(self, vector_field, query_vector, distance_measure, limit):
        """Stand in for Firestore native KNN: cosine-rank in memory, top-`limit`."""
        q = _to_list(query_vector)
        scored = [
            (_cos(q, _to_list(data.get(vector_field))), doc_id, data)
            for doc_id, data in self._store.items()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return _FakeVectorQuery(
            [
                _FakeSnapshot(self._store, doc_id, data)
                for _s, doc_id, data in scored[:limit]
            ]
        )


class _NoNativeCollection(_FakeCollection):
    """Emulator-like: find_nearest raises, so retrieval has no native path."""

    def find_nearest(self, *a, **k):
        raise RuntimeError("find_nearest unsupported (emulator / no index)")


@pytest.fixture
def store():
    return VectorStore(MockEmbeddingProvider())


@pytest.mark.asyncio
async def test_upsert_then_query_returns_most_similar_first(store):
    col = _FakeCollection()
    await store.upsert(col, "a", "dragons and castles", {"kind": "chapter"})
    await store.upsert(col, "b", "spreadsheets and taxes", {"kind": "chapter"})

    rows = await store.query(col, "dragons and castles", top_k=2)
    assert [r["id"] for r in rows][0] == "a"  # exact match ranks first
    assert rows[0]["text"] == "dragons and castles"
    assert rows[0]["kind"] == "chapter"


@pytest.mark.asyncio
async def test_query_respects_top_k(store):
    col = _FakeCollection()
    for i in range(5):
        await store.upsert(col, f"d{i}", f"text number {i}", {})
    rows = await store.query(col, "text number 2", top_k=3)
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_upsert_stores_embedding_as_readable_vector(store):
    col = _FakeCollection()
    await store.upsert(col, "a", "hello world", {})
    stored = col._store["a"]
    # embedding round-trips to a 768-dim list regardless of Vector vs list storage.
    assert len(_to_list(stored["embedding"])) == 768


@pytest.mark.asyncio
async def test_empty_query_returns_nothing(store):
    col = _FakeCollection()
    await store.upsert(col, "a", "something", {})
    assert await store.query(col, "", top_k=4) == []


@pytest.mark.asyncio
async def test_query_returns_empty_when_native_unavailable(store):
    # No brute-force fallback: when find_nearest can't run (no/!READY index, emulator)
    # query yields no excerpts instead of scanning the collection.
    col = _NoNativeCollection()
    await store.upsert(col, "a", "dragons and castles", {})
    assert await store.query(col, "dragons and castles", top_k=2) == []


@pytest.mark.asyncio
async def test_delete_where_removes_only_matching(store):
    col = _FakeCollection()
    await store.upsert(col, "c1", "x", {"chapterId": "ch1"})
    await store.upsert(col, "c2", "y", {"chapterId": "ch1"})
    await store.upsert(col, "c3", "z", {"chapterId": "ch2"})

    removed = await store.delete_where(col, "chapterId", "ch1")
    assert removed == 2
    assert set(col._store.keys()) == {"c3"}


@pytest.mark.asyncio
async def test_delete_all_clears_collection(store):
    col = _FakeCollection()
    await store.upsert(col, "a", "x", {})
    await store.upsert(col, "b", "y", {})
    removed = await store.delete_all(col)
    assert removed == 2
    assert col._store == {}
