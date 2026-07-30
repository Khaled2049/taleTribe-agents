"""Owner-enforced Firestore reads backing the MCP tools.

Every function takes the caller's Firebase uid and refuses to return anything
the caller does not own: `stories/{id}.userId == uid`. Missing and non-owned
stories are indistinguishable to the caller ("not found") so story IDs cannot
be probed for existence.

Deliberately does NOT reuse StoryContextBuilder.build_story_context — that
path fetches all four subcollections with full chapter bodies and TTL-caches
the result, which is wasteful for targeted reads and would cache across
callers. The projection/limit/sort patterns are mirrored from it instead.

All functions are synchronous (google-cloud-firestore sync client); tools.py
bridges them with anyio.to_thread.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.query import Query

from agents.storyAgent.entity_schema import embedded_field_names

# Mirrors StoryContextBuilder.COLLECTION_FETCH_LIMIT: bounded subcollection reads.
COLLECTION_FETCH_LIMIT = 200
MAX_STORY_LIST_LIMIT = 100
SHORT_TEXT_LIMIT = 300

# MCP tools address entities by Firestore collection name (plural); the shared
# field vocabulary in entity_schema.py is keyed by entity kind (singular). One
# explicit mapping instead of the two lists happening to line up.
ENTITY_KIND_BY_COLLECTION = {
    "characters": "character",
    "places": "place",
    "plots": "plot",
}
ENTITY_COLLECTIONS = tuple(ENTITY_KIND_BY_COLLECTION)

# First non-empty of these becomes the one-line descriptor in list_entities.
#
# This is editorial priority, not a field set, which is why it can't simply be
# derived from entity_schema.py: that module's order exists to fix prompt and
# embedding layout, and for characters it opens with "age" — accurate, useless
# as a one-line descriptor. So the names stay explicit here, and
# test_descriptor_fields_exist_in_entity_schema keeps them honest by checking
# every one against embedded_field_names() for its kind. A renamed field fails
# the suite instead of silently producing blank descriptors forever.
_DESCRIPTOR_FIELDS = {
    "characters": ("personality", "soul", "backstory"),
    "places": ("description", "atmosphere"),
    "plots": ("description",),
}

# User-authored fields that belong in get_entity's output but aren't part of the
# embedding vocabulary in entity_schema.py (they're presentation, not prose).
_ENTITY_EXTRA_FIELDS = {
    "characters": ("artUrl",),
    "places": ("imageUrl",),
    "plots": (),
}


class StoryNotFoundError(Exception):
    """Story missing OR not owned by the caller — deliberately the same error."""


class EntityNotFoundError(Exception):
    """Chapter/character/place/plot not found within an owned story."""


class Page(NamedTuple):
    """A capped subcollection read: the rows, plus whether more were left behind.

    `truncated` exists because silence is the wrong answer here. A caller that
    receives 200 chapters cannot tell "that is the whole book" from "that is
    where we stopped counting", and the LLM on the other end will happily
    reason about a manuscript with a hole in it.
    """

    items: list[dict]
    truncated: bool


def _page(query: Any, limit: int = COLLECTION_FETCH_LIMIT) -> tuple[list, bool]:
    """Stream at most `limit` documents, reading one extra to detect truncation.

    The probe row is fetched and discarded: one surplus document is far cheaper
    than a count query, and it turns "we got exactly the cap" (which might mean
    either) into a definite answer.
    """
    docs = list(query.limit(limit + 1).stream())
    return docs[:limit], len(docs) > limit


def _truncate(value: Any, limit: int = SHORT_TEXT_LIMIT) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _display_name(data: dict) -> str:
    """Always a non-empty str, so callers can sort on it.

    Firestore holds whatever the writer's client put there: a character named
    "7" arrives as an int and would break a `.lower()` sort.
    """
    for field in ("name", "title"):
        value = data.get(field)
        if value is None or isinstance(value, bool):
            continue
        text = str(value).strip()
        if text:
            return text
    return "Unnamed"


def _chapter_sort_key(chapter: dict) -> float:
    """Float `order` first, then `chapterNumber`, then 0.0 — same as context_builder."""
    for field in ("order", "chapterNumber"):
        value = chapter.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else None


def get_owned_story(db: Any, story_id: str, uid: str) -> dict:
    """The single ownership gate every story-scoped read goes through."""
    snap = db.collection("stories").document(story_id).get()
    data = snap.to_dict() if snap.exists else None
    if not data or data.get("userId") != uid:
        raise StoryNotFoundError(story_id)
    data["id"] = snap.id
    return data


def list_stories_for_user(db: Any, uid: str, limit: int) -> list[dict]:
    limit = max(1, min(int(limit), MAX_STORY_LIST_LIMIT))
    # Order in Firestore, not in Python: sorting a client-side page would rank
    # only an arbitrary slice of a prolific author's stories, so the "most
    # recently updated" promise would break above the page size. Backed by the
    # existing (userId ASC, updatedAt DESC) composite index. Documents without
    # `updatedAt` are excluded by the order_by — every story-creation path
    # writes it.
    docs = (
        db.collection("stories")
        .where(filter=FieldFilter("userId", "==", uid))
        .order_by("updatedAt", direction=Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    stories = []
    for doc in docs:
        data = doc.to_dict() or {}
        stories.append(
            {
                "story_id": doc.id,
                "title": data.get("title", "Untitled"),
                "description": _truncate(data.get("description")),
                "is_published": bool(data.get("isPublished", False)),
                "chapter_count": data.get("chapterCount"),
                "updated_at": _iso(data.get("updatedAt")),
            }
        )
    return stories


def get_story_overview(db: Any, story_id: str, uid: str) -> dict:
    story = get_owned_story(db, story_id, uid)

    # The denormalized index is rebuilt from an uncapped query
    # (chapterIndexTrigger.ts), so when it exists this path is always complete.
    chapter_index = story.get("chapterIndex")
    truncated = False
    if not isinstance(chapter_index, list):
        # Same backfill fallback as _fetch_slim_chat_context: titles only.
        docs, truncated = _page(_chapters_in_reading_order(db, story_id))
        chapter_index = [doc.to_dict() or {} for doc in docs]
    chapters = sorted(chapter_index, key=_chapter_sort_key)

    return {
        "story_id": story["id"],
        "title": story.get("title", "Untitled"),
        "description": story.get("description"),
        "author": story.get("author"),
        "tags": story.get("tags") or [],
        "category": story.get("category"),
        "is_published": bool(story.get("isPublished", False)),
        "chapter_count": story.get("chapterCount"),
        "updated_at": _iso(story.get("updatedAt")),
        "chapters": [
            {
                "title": chapter.get("title", "Untitled"),
                "order": chapter.get("order"),
                "chapter_number": chapter.get("chapterNumber"),
            }
            for chapter in chapters
        ],
        "chapters_truncated": truncated,
    }


def _chapters_in_reading_order(db: Any, story_id: str) -> Any:
    """Chapters ordered by `order` in Firestore, not in Python.

    Ordering has to happen server-side: `limit()` on an unordered query returns
    documents by document id, so sorting the result would rank an arbitrary
    slice and a long book would come back missing chapters from the middle with
    nothing to show for it. Same reasoning as list_stories_for_user.

    Firestore omits documents that lack the order field, which makes `order` the
    one safe choice: every write path sets it (`StoriesRepo.addChapter` and
    `generateChapterTask` in the frontend repo), whereas `chapterNumber` is only
    written by the generation path. This matches what
    `StoryContextBuilder.build_story_context` already does for the same
    collection, so the two read paths agree on which chapters exist.
    """
    return (
        db.collection("stories")
        .document(story_id)
        .collection("chapters")
        .select(["title", "order", "chapterNumber", "wordCount"])
        .order_by("order", direction=Query.ASCENDING)
    )


def list_chapters(db: Any, story_id: str, uid: str) -> Page:
    get_owned_story(db, story_id, uid)
    docs, truncated = _page(_chapters_in_reading_order(db, story_id))
    chapters = []
    for doc in docs:
        data = doc.to_dict() or {}
        chapters.append(
            {
                "chapter_id": doc.id,
                "title": data.get("title", "Untitled"),
                "order": data.get("order"),
                "chapter_number": data.get("chapterNumber"),
                "word_count": data.get("wordCount"),
            }
        )
    # Already ordered by the query; re-sorted so the `order`-then-chapterNumber
    # contract in _chapter_sort_key holds for ties within the page too.
    chapters.sort(key=_chapter_sort_key)
    return Page(chapters, truncated)


def get_chapter(
    db: Any,
    story_id: str,
    chapter_id: str,
    uid: str,
    offset: int,
    max_chars: int,
) -> dict:
    get_owned_story(db, story_id, uid)
    snap = (
        db.collection("stories")
        .document(story_id)
        .collection("chapters")
        .document(chapter_id)
        .get()
    )
    if not snap.exists:
        raise EntityNotFoundError(chapter_id)
    data = snap.to_dict() or {}
    content = data.get("content") or ""

    offset = max(0, int(offset))
    max_chars = max(1, min(int(max_chars), 50_000))
    window = content[offset : offset + max_chars]
    next_offset = offset + max_chars if offset + max_chars < len(content) else None

    return {
        "chapter_id": snap.id,
        "title": data.get("title", "Untitled"),
        "order": data.get("order"),
        "chapter_number": data.get("chapterNumber"),
        "word_count": data.get("wordCount"),
        "total_chars": len(content),
        "offset": offset,
        "content": window,
        "next_offset": next_offset,
    }


def _require_entity_type(entity_type: str) -> str:
    if entity_type not in ENTITY_COLLECTIONS:
        raise ValueError(f"entity_type must be one of {', '.join(ENTITY_COLLECTIONS)}")
    return entity_type


def list_entities(db: Any, story_id: str, uid: str, entity_type: str) -> Page:
    """Entities for one story, alphabetical, capped at COLLECTION_FETCH_LIMIT.

    Unlike chapters this cannot order in Firestore, so the page really is an
    arbitrary slice (document id order) when `truncated` is set. There is no
    field to order on: the display name is whichever of `name`/`title` the
    writer's client happened to populate, and Firestore holds it as whatever
    type was written — `_display_name` exists precisely because a character
    called "7" arrives as an int. Ordering on a field that is sometimes absent
    would silently drop entities, which is worse than an honest flag.
    """
    _require_entity_type(entity_type)
    get_owned_story(db, story_id, uid)
    docs, truncated = _page(
        db.collection("stories").document(story_id).collection(entity_type)
    )
    entities = []
    for doc in docs:
        data = doc.to_dict() or {}
        descriptor = None
        for field in _DESCRIPTOR_FIELDS[entity_type]:
            descriptor = _truncate(data.get(field))
            if descriptor:
                break
        entities.append(
            {
                "entity_id": doc.id,
                "name": _display_name(data),
                "descriptor": descriptor,
            }
        )
    entities.sort(key=lambda e: e["name"].lower())
    return Page(entities, truncated)


def entity_content_fields(entity_type: str) -> tuple[str, ...]:
    """The fields get_entity will return, minus the computed ones."""
    kind = ENTITY_KIND_BY_COLLECTION[entity_type]
    return tuple(embedded_field_names(kind)) + _ENTITY_EXTRA_FIELDS[entity_type]


def get_entity(
    db: Any, story_id: str, uid: str, entity_type: str, entity_id: str
) -> dict:
    """One entity, projected onto an explicit field set.

    Projected rather than passed through — the same discipline get_chapter uses.
    A denylist ("pop embedding") was both leaky and unsafe: it shipped userId,
    storyId, signature and embeddingUpdatedAt as noise in every response, and
    any *other* raw Firestore value would have broken JSON serialization
    outright, because embeddings are firestore_v1.vector.Vector and nothing
    guarantees `embedding` is the only field ever to hold one.

    Array-of-object fields (relationships, events) are passed through whole:
    they carry the story structure a reader actually wants, and only the
    frontend writes them, so their contents are plain JSON by construction.
    Empty and absent values are omitted rather than sent as nulls.
    """
    _require_entity_type(entity_type)
    get_owned_story(db, story_id, uid)
    snap = (
        db.collection("stories")
        .document(story_id)
        .collection(entity_type)
        .document(entity_id)
        .get()
    )
    if not snap.exists:
        raise EntityNotFoundError(entity_id)
    raw = snap.to_dict() or {}

    entity: dict[str, Any] = {
        "entity_id": snap.id,
        "name": _display_name(raw),
        "updated_at": _iso(raw.get("updatedAt") or raw.get("createdAt")),
    }
    for field in entity_content_fields(entity_type):
        if field == "name":
            continue  # already set, and coerced to a str
        value = raw.get(field)
        if value is None or value == "":
            continue
        entity[field] = value
    return entity
