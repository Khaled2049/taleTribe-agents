"""In-memory Firestore fake for mcp_server tests.

Not a test module (no test_ prefix). Supports the subset the MCP code uses:
document get/set/create/update/delete with last_update_time preconditions,
auto-id document refs, subcollections, and where/select/limit/stream queries.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Optional

from google.api_core import exceptions as gcp_exceptions

_versions = itertools.count(1)
_auto_ids = itertools.count(1)


class FakeWriteOption:
    def __init__(self, last_update_time):
        self.last_update_time = last_update_time


class FakeWriteResult:
    """What the real client returns from update(); writes.py reads update_time
    off it to report the revision an edit produced."""

    def __init__(self, update_time):
        self.update_time = update_time


class FakeSnapshot:
    def __init__(self, doc_id: str, data: Optional[dict], update_time):
        self.id = doc_id
        self._data = data
        self.update_time = update_time

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> Optional[dict]:
        return copy.deepcopy(self._data) if self._data is not None else None

    def get(self, field: str):
        return (self._data or {}).get(field)


class FakeDocRef:
    def __init__(self, db: "FakeFirestoreClient", path: str):
        self._db = db
        self._path = path

    @property
    def id(self) -> str:
        return self._path.rsplit("/", 1)[-1]

    def collection(self, name: str) -> "FakeCollection":
        return FakeCollection(self._db, f"{self._path}/{name}")

    def get(self, transaction=None) -> FakeSnapshot:
        entry = self._db.docs.get(self._path)
        if entry is None:
            return FakeSnapshot(self.id, None, None)
        data, version = entry
        return FakeSnapshot(self.id, data, version)

    def set(self, data: dict, merge: bool = False) -> None:
        entry = self._db.docs.get(self._path)
        if merge and entry is not None:
            merged = dict(entry[0])
            merged.update(copy.deepcopy(data))
            self._db.docs[self._path] = (merged, next(_versions))
            return
        self._db.docs[self._path] = (copy.deepcopy(data), next(_versions))

    def create(self, data: dict) -> None:
        """Fail if the document already exists — the real create() semantics.

        This is what makes the write-idempotency reservation in writes.py a
        genuine claim rather than a read-then-write race.
        """
        if self._path in self._db.docs:
            raise gcp_exceptions.AlreadyExists(f"document {self._path} exists")
        self._db.docs[self._path] = (copy.deepcopy(data), next(_versions))

    def update(
        self, patch: dict, option: Optional[FakeWriteOption] = None
    ) -> FakeWriteResult:
        if self._db.before_update is not None:
            # Lets a test act as a concurrent writer in the window between the
            # caller's read and its precondition-guarded update. Without this
            # the retry loop in writes.create_chapter is untestable.
            self._db.before_update(self._path)
        entry = self._db.docs.get(self._path)
        if entry is None:
            raise gcp_exceptions.NotFound(f"no document {self._path}")
        data, version = entry
        if option is not None and option.last_update_time != version:
            raise gcp_exceptions.FailedPrecondition("stale update_time")
        merged = dict(data)
        merged.update(copy.deepcopy(patch))
        version = next(_versions)
        self._db.docs[self._path] = (merged, version)
        return FakeWriteResult(version)

    def delete(self, option: Optional[FakeWriteOption] = None) -> None:
        entry = self._db.docs.get(self._path)
        if entry is None:
            if option is not None:
                raise gcp_exceptions.NotFound(f"no document {self._path}")
            return
        _, version = entry
        if option is not None and option.last_update_time != version:
            raise gcp_exceptions.FailedPrecondition("stale update_time")
        del self._db.docs[self._path]


class FakeQuery:
    def __init__(self, db: "FakeFirestoreClient", prefix: str):
        self._db = db
        self._prefix = prefix
        self._filters: list[tuple[str, str, Any]] = []
        self._select: Optional[list[str]] = None
        self._limit: Optional[int] = None
        self._order: Optional[tuple[str, str]] = None

    def _clone(self) -> "FakeQuery":
        q = FakeQuery(self._db, self._prefix)
        q._filters = list(self._filters)
        q._select = self._select
        q._limit = self._limit
        q._order = self._order
        return q

    def where(self, *args, filter=None) -> "FakeQuery":
        q = self._clone()
        if filter is not None:
            q._filters.append((filter.field_path, filter.op_string, filter.value))
        else:
            field, op, value = args
            q._filters.append((field, op, value))
        return q

    def select(self, field_paths: list[str]) -> "FakeQuery":
        q = self._clone()
        q._select = list(field_paths)
        return q

    def limit(self, count: int) -> "FakeQuery":
        q = self._clone()
        q._limit = count
        return q

    def order_by(self, field_path: str, direction: str = "ASCENDING") -> "FakeQuery":
        q = self._clone()
        q._order = (field_path, direction)
        return q

    def stream(self):
        # (sort key, snapshot) pairs — the key is read from the unprojected doc
        # so select() and order_by() can name different fields.
        results: list[tuple[Any, FakeSnapshot]] = []
        for path, (data, version) in self._db.docs.items():
            head, _, doc_id = path.rpartition("/")
            if head != self._prefix:
                continue
            if not all(self._matches(data, f) for f in self._filters):
                continue
            sort_key = None
            if self._order is not None:
                sort_key = data.get(self._order[0])
                # Firestore omits documents missing the order_by field.
                if sort_key is None:
                    continue
            out = data
            if self._select is not None:
                out = {k: v for k, v in data.items() if k in self._select}
            results.append((sort_key, FakeSnapshot(doc_id, out, version)))
        if self._order is not None:
            results.sort(key=lambda r: r[0], reverse=self._order[1] == "DESCENDING")
        else:
            # Firestore orders an unordered query by __name__ ascending. Modelled
            # here so that limit()-without-order_by returns a document-id slice
            # rather than a convenient insertion-order one — otherwise the fake
            # would hide exactly the truncate-before-sort bug it should catch.
            results.sort(key=lambda r: r[1].id)
        snapshots = [snap for _key, snap in results]
        if self._limit is not None:
            snapshots = snapshots[: self._limit]
        return iter(snapshots)

    @staticmethod
    def _matches(data: dict, f: tuple[str, str, Any]) -> bool:
        field, op, value = f
        if op == "==":
            return data.get(field) == value
        raise NotImplementedError(f"fake query op {op}")


class FakeCollection(FakeQuery):
    def __init__(self, db: "FakeFirestoreClient", prefix: str):
        super().__init__(db, prefix)

    def document(self, doc_id: Optional[str] = None) -> FakeDocRef:
        if doc_id is None:
            # Fixed width so the __name__ sort in stream() stays deterministic
            # past ten documents. Real auto-ids are random, so id order is NOT
            # insertion order in production — a test that depends on the two
            # agreeing is testing the fake, not the code.
            doc_id = f"auto{next(_auto_ids):06d}"
        return FakeDocRef(self._db, f"{self._prefix}/{doc_id}")


class FakeFirestoreClient:
    def __init__(self, project: str = "test-project"):
        self.project = project
        # path -> (data, version)
        self.docs: dict[str, tuple[dict, int]] = {}
        # Test hook: called with the doc path at the top of every update().
        self.before_update: Optional[Callable[[str], None]] = None

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(self, name)

    @staticmethod
    def write_option(*, last_update_time) -> FakeWriteOption:
        return FakeWriteOption(last_update_time)

    # -- test helpers -------------------------------------------------

    def seed(self, path: str, data: dict) -> None:
        self.docs[path] = (copy.deepcopy(data), next(_versions))


# ---------------------------------------------------------------------------
# story-data fake, backing the MCP read tools
# ---------------------------------------------------------------------------


class FakeStoryData:
    """Stands in for StoryDataClient, seeded with story-data's JSON shapes.

    Reproduces story-data's *authorization* semantics, not just its payloads,
    because that is what the read path has to be tested against:

    - `GET /v1/stories` is scoped to the caller (WHERE owner_id).
    - `GET /v1/stories/{id}` and the chapter routes also serve a **published**
      story to a non-owner, which is why data.get_owned_story re-checks ownerId.
      A fake that simply refused non-owners everywhere would make that check
      look redundant and let a real widening of MCP's scope pass the suite.
    - The worldbuilding routes are owner-only (they go through store.owner).
    """

    def __init__(self):
        self.stories: dict[str, dict] = {}
        self.chapters: dict[str, list[dict]] = {}
        self.entities: dict[tuple[str, str], list[dict]] = {}
        self.threads: dict[str, dict] = {}
        self.thread_messages: dict[str, list[dict]] = {}
        # Every path requested, so a test can assert on content=false.
        self.requests: list[tuple[str, dict]] = []

    # -- seeding ------------------------------------------------------

    def seed_story(self, story_id: str, owner_id: str, **fields: Any) -> dict:
        record = {
            "id": story_id,
            "ownerId": owner_id,
            "title": "Untitled",
            "description": "",
            "authorName": "",
            "category": "",
            "tags": [],
            "published": False,
            "revision": 1,
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
        }
        record.update(fields)
        self.stories[story_id] = record
        self.chapters.setdefault(story_id, [])
        return record

    def seed_chapter(self, story_id: str, chapter_id: str, **fields: Any) -> dict:
        record = {
            "id": chapter_id,
            "storyId": story_id,
            "title": "Untitled",
            "content": "",
            "position": float(len(self.chapters.get(story_id, [])) + 1),
            "wordCount": 0,
            "revision": 1,
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
        }
        record.update(fields)
        self.chapters.setdefault(story_id, []).append(record)
        return record

    def seed_entity(
        self, story_id: str, kind: str, entity_id: str, **fields: Any
    ) -> dict:
        record = {
            "id": entity_id,
            "storyId": story_id,
            "name": "Unnamed",
            "revision": 1,
        }
        record.update(fields)
        self.entities.setdefault((story_id, kind), []).append(record)
        return record

    # -- internals ----------------------------------------------------

    def _visible_story(self, uid: str, story_id: str) -> dict:
        """story-data's GetStory: owner always, others only when published."""
        from mcp_server import story_data

        record = self.stories.get(story_id)
        if record is None:
            raise story_data.NotFound(story_id)
        if record["ownerId"] != uid and not record.get("published"):
            raise story_data.NotFound(story_id)
        return copy.deepcopy(record)

    def _owned_story(self, uid: str, story_id: str) -> dict:
        """store.owner: owner-only, as the worldbuilding routes are."""
        from mcp_server import story_data

        record = self.stories.get(story_id)
        if record is None or record["ownerId"] != uid:
            raise story_data.NotFound(story_id)
        return record

    def seed_thread(self, thread_id: str, story_id: str) -> dict:
        record = {"id": thread_id, "storyId": story_id, "messageCount": 0}
        self.threads[thread_id] = record
        self.thread_messages.setdefault(thread_id, [])
        return record

    def seed_message(
        self, thread_id: str, role: str, text: str, status: str = "complete"
    ) -> dict:
        rows = self.thread_messages.setdefault(thread_id, [])
        record = {
            "id": f"msg-{len(rows) + 1}",
            "threadId": thread_id,
            "sequence": len(rows) + 1,
            "role": role,
            "parts": [{"type": "text", "text": text}],
            "status": status,
        }
        rows.append(record)
        self.threads[thread_id]["messageCount"] = len(rows)
        return record

    # -- client interface ---------------------------------------------

    async def list_stories(self, uid: str) -> list[dict]:
        self.requests.append(("/v1/stories", {}))
        mine = [s for s in self.stories.values() if s["ownerId"] == uid]
        mine.sort(key=lambda s: s.get("updatedAt") or "", reverse=True)
        return copy.deepcopy(mine)

    async def get_story(self, uid: str, story_id: str) -> dict:
        self.requests.append((f"/v1/stories/{story_id}", {}))
        return self._visible_story(uid, story_id)

    async def list_chapter_index(self, uid: str, story_id: str) -> list[dict]:
        self.requests.append((f"/v1/stories/{story_id}/chapters", {"content": "false"}))
        self._visible_story(uid, story_id)
        rows = []
        for chapter in sorted(
            self.chapters.get(story_id, []), key=lambda c: c["position"]
        ):
            lean = copy.deepcopy(chapter)
            lean["content"] = ""  # what ?content=false returns
            rows.append(lean)
        return rows

    async def get_chapter(self, uid: str, story_id: str, chapter_id: str) -> dict:
        from mcp_server import story_data

        self.requests.append((f"/v1/stories/{story_id}/chapters/{chapter_id}", {}))
        self._visible_story(uid, story_id)
        for chapter in self.chapters.get(story_id, []):
            if chapter["id"] == chapter_id:
                return copy.deepcopy(chapter)
        raise story_data.NotFound(chapter_id)

    async def list_entities(self, uid: str, story_id: str, kind: str) -> list[dict]:
        self.requests.append((f"/v1/stories/{story_id}/{kind}", {}))
        self._owned_story(uid, story_id)
        rows = list(self.entities.get((story_id, kind), []))
        if kind in ("characters", "places"):
            rows.sort(key=lambda e: str(e.get("name") or "").lower())
        return copy.deepcopy(rows)

    async def get_entity(
        self, uid: str, story_id: str, kind: str, entity_id: str
    ) -> dict:
        from mcp_server import story_data

        self.requests.append((f"/v1/stories/{story_id}/{kind}/{entity_id}", {}))
        self._owned_story(uid, story_id)
        for entity in self.entities.get((story_id, kind), []):
            if entity["id"] == entity_id:
                return copy.deepcopy(entity)
        raise story_data.NotFound(entity_id)

    async def get_assistant_thread(
        self, uid: str, story_id: str, thread_id: str
    ) -> dict:
        from mcp_server import story_data

        self.requests.append(
            (f"/v1/stories/{story_id}/assistant-threads/{thread_id}", {})
        )
        self._owned_story(uid, story_id)
        thread = self.threads.get(thread_id)
        if thread is None or thread["storyId"] != story_id:
            raise story_data.NotFound(thread_id)
        return copy.deepcopy(thread)

    async def list_assistant_messages(
        self,
        uid: str,
        story_id: str,
        thread_id: str,
        *,
        cursor: int = 0,
        limit: int = 20,
    ) -> dict:
        params = {"limit": str(limit)}
        if cursor > 0:
            params["cursor"] = str(cursor)
        self.requests.append(
            (f"/v1/stories/{story_id}/assistant-threads/{thread_id}/messages", params)
        )
        from mcp_server import story_data

        self._owned_story(uid, story_id)
        thread = self.threads.get(thread_id)
        if thread is None or thread["storyId"] != story_id:
            raise story_data.NotFound(thread_id)
        rows = [
            row
            for row in self.thread_messages.get(thread_id, [])
            if row["sequence"] > cursor
        ]
        return {"messages": copy.deepcopy(rows[:limit])}

    async def close(self) -> None:
        return None
