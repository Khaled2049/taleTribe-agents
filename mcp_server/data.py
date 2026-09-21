"""Owner-enforced story-data reads backing the MCP tools.

Every function takes the caller's Firebase uid and refuses to return anything
the caller does not own. Missing and non-owned stories are indistinguishable to
the caller ("not found") so story IDs cannot be probed for existence.

Reads only — nothing here mutates. The write tools live in writes.py and reuse
get_owned_story below as their own gate.

Ownership is asserted twice on purpose. story-data enforces it (the asserted uid
scopes `GET /v1/stories`, and worldbuilding goes through its owner check), but
`GetStory` and `ListChapters` deliberately serve a *published* story to any
caller so the public reader can use them. MCP is owner-only on every tool, so
get_owned_story re-checks `ownerId == uid` rather than inheriting that allowance.

All functions are async: the transport is HTTP now, so tools.py awaits them
directly rather than bridging with anyio.to_thread.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from agents.storyAgent.entity_schema import embedded_field_names
from mcp_server import blocks, story_data

# Bounded reads, unchanged from the Firestore path so page sizes and the
# `truncated` contract stay the same for callers.
COLLECTION_FETCH_LIMIT = 200
MAX_STORY_LIST_LIMIT = 100
SHORT_TEXT_LIMIT = 300
MAX_BLOCKS_PER_PAGE = 1000
DEFAULT_BLOCKS_PER_PAGE = 500

# MCP tools address entities by collection name (plural), which is also the
# story-data path segment; the shared field vocabulary in entity_schema.py is
# keyed by entity kind (singular). One explicit mapping instead of the two lists
# happening to line up.
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
    """A capped read: the rows, plus whether more were left behind.

    `truncated` exists because silence is the wrong answer here. A caller that
    receives 200 chapters cannot tell "that is the whole book" from "that is
    where we stopped counting", and the LLM on the other end will happily
    reason about a manuscript with a hole in it.
    """

    items: list[dict]
    truncated: bool


def _cap(
    rows: list[dict], limit: int = COLLECTION_FETCH_LIMIT
) -> tuple[list[dict], bool]:
    """Trim to `limit`, reporting whether anything was dropped."""
    return rows[:limit], len(rows) > limit


def _truncate(value: Any, limit: int = SHORT_TEXT_LIMIT) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _display_name(data: dict) -> str:
    """Always a non-empty str, so callers can sort on it."""
    for field in ("name", "title"):
        value = data.get(field)
        if value is None or isinstance(value, bool):
            continue
        text = str(value).strip()
        if text:
            return text
    return "Unnamed"


def _order(chapter: dict) -> float:
    """story-data's `position` is the running-order key."""
    value = chapter.get("position")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _numbered(chapters: list[dict]) -> list[tuple[int, dict]]:
    """Pair each chapter with its 1-based place in reading order.

    story-data has no `chapterNumber` column, and `position` cannot stand in for
    one: it is a sort key that keeps gaps after a delete and takes fractional
    values on an insert between neighbours. The ordinal callers want is the index
    within the ordered list, so it is derived here rather than read.
    """
    return list(enumerate(sorted(chapters, key=_order), start=1))


def _revision(record: dict) -> str:
    """story-data carries an integer `revision` per row; callers want a token."""
    return str(record.get("revision", ""))


async def get_owned_story(
    story_id: str,
    uid: str,
    *,
    story_client: Optional[story_data.StoryDataClient] = None,
) -> dict:
    """The single ownership gate every story-scoped read goes through."""
    client = story_client or story_data.client()
    try:
        story = await client.get_story(uid, story_id)
    except story_data.NotFound as exc:
        raise StoryNotFoundError(story_id) from exc
    # story-data serves a published story to any caller; MCP is owner-only.
    if not isinstance(story, dict) or story.get("ownerId") != uid:
        raise StoryNotFoundError(story_id)
    return story


async def list_stories_for_user(uid: str, limit: int) -> list[dict]:
    limit = max(1, min(int(limit), MAX_STORY_LIST_LIMIT))
    # story-data already returns the caller's own stories, updated_at DESC
    # (stories_owner_updated_idx), so the "most recently updated" promise holds
    # above the page size and the cap can be applied here.
    rows = await story_data.client().list_stories(uid)
    stories = []
    for record in rows[:limit]:
        stories.append(
            {
                "story_id": record.get("id"),
                "title": record.get("title") or "Untitled",
                "description": _truncate(record.get("description")),
                "is_published": bool(record.get("published", False)),
                # No count column, and counting would mean a request per story.
                # get_story_overview reports the real number.
                "chapter_count": None,
                "updated_at": record.get("updatedAt"),
            }
        )
    return stories


async def get_story_overview(story_id: str, uid: str) -> dict:
    story = await get_owned_story(story_id, uid)
    rows = await story_data.client().list_chapter_index(uid, story_id)
    chapters, truncated = _cap(rows if isinstance(rows, list) else [])

    return {
        "story_id": story.get("id"),
        "title": story.get("title") or "Untitled",
        "description": story.get("description"),
        "author": story.get("authorName"),
        "tags": story.get("tags") or [],
        "category": story.get("category"),
        "is_published": bool(story.get("published", False)),
        "chapter_count": len(chapters),
        "updated_at": story.get("updatedAt"),
        "chapters": [
            {
                "title": chapter.get("title") or "Untitled",
                "order": chapter.get("position"),
                "chapter_number": number,
            }
            for number, chapter in _numbered(chapters)
        ],
        "chapters_truncated": truncated,
    }


async def list_chapters(story_id: str, uid: str) -> Page:
    await get_owned_story(story_id, uid)
    # content=false: the default listing carries every chapter body, and this
    # tool only reports the running order.
    rows = await story_data.client().list_chapter_index(uid, story_id)
    capped, truncated = _cap(rows if isinstance(rows, list) else [])
    chapters = [
        {
            "chapter_id": chapter.get("id"),
            "title": chapter.get("title") or "Untitled",
            "order": chapter.get("position"),
            "chapter_number": number,
            "word_count": chapter.get("wordCount"),
        }
        for number, chapter in _numbered(capped)
    ]
    return Page(chapters, truncated)


async def _owned_chapter(story_id: str, chapter_id: str, uid: str) -> dict:
    """One chapter with its body, behind the ownership gate."""
    await get_owned_story(story_id, uid)
    try:
        return await story_data.client().get_chapter(uid, story_id, chapter_id)
    except story_data.NotFound as exc:
        raise EntityNotFoundError(chapter_id) from exc


async def _chapter_number(story_id: str, chapter_id: str, uid: str) -> Optional[int]:
    """The chapter's place in reading order, or None if it is no longer listed.

    Costs one extra metadata request (no bodies), which is what buys the same
    ordinal the list tools report. Deriving it from `position` instead would be
    wrong the moment a chapter is deleted or inserted between two others.
    """
    rows = await story_data.client().list_chapter_index(uid, story_id)
    for number, chapter in _numbered(rows if isinstance(rows, list) else []):
        if chapter.get("id") == chapter_id:
            return number
    return None


async def get_chapter(
    story_id: str,
    chapter_id: str,
    uid: str,
    offset: int,
    max_chars: int,
) -> dict:
    record = await _owned_chapter(story_id, chapter_id, uid)
    content = record.get("content") or ""

    offset = max(0, int(offset))
    max_chars = max(1, min(int(max_chars), 50_000))
    window = content[offset : offset + max_chars]
    next_offset = offset + max_chars if offset + max_chars < len(content) else None

    return {
        "chapter_id": record.get("id"),
        "title": record.get("title") or "Untitled",
        "order": record.get("position"),
        "chapter_number": await _chapter_number(story_id, chapter_id, uid),
        "word_count": record.get("wordCount"),
        "total_chars": len(content),
        "offset": offset,
        "content": window,
        "next_offset": next_offset,
        "revision": _revision(record),
    }


async def get_chapter_blocks(
    story_id: str,
    chapter_id: str,
    uid: str,
    start_index: int,
    max_blocks: int,
) -> dict:
    """The chapter's top-level blocks, as addresses for the edit tools.

    A separate read from get_chapter because that one windows by *character*
    offset, and a window can begin in the middle of a block — block indices
    attached to a partial character window would be meaningless. This returns
    structure and short previews instead of prose, which is both what the
    editing workflow needs and far cheaper than re-shipping a 100k-char
    chapter to locate one paragraph.
    """
    record = await _owned_chapter(story_id, chapter_id, uid)
    all_blocks = blocks.split_blocks(record.get("content") or "")

    start_index = max(0, int(start_index))
    max_blocks = max(1, min(int(max_blocks), MAX_BLOCKS_PER_PAGE))
    window = all_blocks[start_index : start_index + max_blocks]
    end = start_index + max_blocks
    next_index = end if end < len(all_blocks) else None

    return {
        "chapter_id": record.get("id"),
        "title": record.get("title") or "Untitled",
        "revision": _revision(record),
        "block_count": len(all_blocks),
        "start_index": start_index,
        "blocks": [
            {
                "index": start_index + position,
                "tag": block.tag,
                "preview": blocks.block_preview(block.html),
                "chars": len(block.html),
            }
            for position, block in enumerate(window)
        ],
        "next_index": next_index,
    }


def _require_entity_type(entity_type: str) -> str:
    if entity_type not in ENTITY_COLLECTIONS:
        raise ValueError(f"entity_type must be one of {', '.join(ENTITY_COLLECTIONS)}")
    return entity_type


async def list_entities(story_id: str, uid: str, entity_type: str) -> Page:
    """Entities for one story, alphabetical, capped at COLLECTION_FETCH_LIMIT.

    story-data orders characters and places by name and plot lines by creation
    time, so unlike the Firestore path a truncated page is a real prefix rather
    than an arbitrary slice. Sorted again here so all three kinds agree.
    """
    _require_entity_type(entity_type)
    await get_owned_story(story_id, uid)
    rows = await story_data.client().list_entities(uid, story_id, entity_type)
    capped, truncated = _cap(rows if isinstance(rows, list) else [])
    entities = []
    for record in capped:
        descriptor = None
        for field in _DESCRIPTOR_FIELDS[entity_type]:
            descriptor = _truncate(record.get(field))
            if descriptor:
                break
        entities.append(
            {
                "entity_id": record.get("id"),
                "name": _display_name(record),
                "descriptor": descriptor,
            }
        )
    entities.sort(key=lambda e: e["name"].lower())
    return Page(entities, truncated)


def entity_content_fields(entity_type: str) -> tuple[str, ...]:
    """The fields get_entity will return, minus the computed ones."""
    kind = ENTITY_KIND_BY_COLLECTION[entity_type]
    return tuple(embedded_field_names(kind)) + _ENTITY_EXTRA_FIELDS[entity_type]


async def get_entity(story_id: str, uid: str, entity_type: str, entity_id: str) -> dict:
    """One entity, projected onto an explicit field set.

    Projected rather than passed through, so a column added to story-data cannot
    start appearing in tool output unreviewed. Array-of-object fields
    (relationships, events) are passed through whole: they carry the story
    structure a reader actually wants. Empty and absent values are omitted
    rather than sent as nulls.
    """
    _require_entity_type(entity_type)
    await get_owned_story(story_id, uid)
    try:
        raw = await story_data.client().get_entity(
            uid, story_id, entity_type, entity_id
        )
    except story_data.NotFound as exc:
        raise EntityNotFoundError(entity_id) from exc

    entity: dict[str, Any] = {
        "entity_id": raw.get("id"),
        "name": _display_name(raw),
        "updated_at": raw.get("updatedAt") or raw.get("createdAt"),
    }
    for field in entity_content_fields(entity_type):
        if field == "name":
            continue  # already set, and coerced to a str
        value = raw.get(field)
        if value is None or value == "":
            continue
        entity[field] = value
    return entity
