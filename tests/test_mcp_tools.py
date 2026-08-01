"""Unit tests for mcp_server.data (owner-enforced reads) and mcp_server.tools."""

import itertools
import json
import os
import pathlib
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("USE_MOCK", "true")

from mcp.server.auth.provider import AccessToken  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from mcp_server import app as mcp_app  # noqa: E402
from mcp_server import data, writes  # noqa: E402
from mcp_server.tools import UNTRUSTED_NOTICE, register_tools  # noqa: E402
from rate_limit import PerUserRateLimiter  # noqa: E402
from tests.mcp_fakes import FakeFirestoreClient  # noqa: E402

UID_A = "user-a"
UID_B = "user-b"


def _seeded_db() -> FakeFirestoreClient:
    db = FakeFirestoreClient()
    db.seed(
        "stories/story-a",
        {
            "userId": UID_A,
            "title": "Story A",
            "description": "d" * 400,
            "isPublished": False,
            "chapterCount": 2,
            "author": "Author A",
            "updatedAt": datetime(2026, 7, 1, tzinfo=timezone.utc),
            "chapterIndex": [
                {"title": "Chapter Two", "order": 2},
                {"title": "Chapter One", "order": 1},
            ],
        },
    )
    db.seed(
        "stories/story-a/chapters/ch1",
        {
            "title": "Chapter One",
            "order": 1,
            "wordCount": 5,
            "content": "0123456789" * 2500,  # 25_000 chars
        },
    )
    db.seed(
        "stories/story-a/chapters/ch2",
        {"title": "Chapter Two", "order": 2, "wordCount": 3, "content": "short"},
    )
    db.seed(
        "stories/story-a/characters/char1",
        {
            "name": "Mira",
            "personality": "p" * 400,
            "soul": "steadfast",
            "embedding": [0.1, 0.2],
            "relationships": [{"name": "Bran", "relation": "brother"}],
        },
    )
    # A second user-a story WITHOUT chapterIndex (exercises the fallback).
    db.seed(
        "stories/story-c",
        {
            "userId": UID_A,
            "title": "Story C",
            "updatedAt": datetime(2026, 7, 20, tzinfo=timezone.utc),
        },
    )
    db.seed(
        "stories/story-c/chapters/cc1",
        {"title": "Only Chapter", "order": 1, "content": "hello"},
    )
    # Another user's story: must be invisible to user-a.
    db.seed(
        "stories/story-b",
        {"userId": UID_B, "title": "Story B", "chapterIndex": []},
    )
    db.seed(
        "stories/story-b/chapters/chb",
        {"title": "Secret", "order": 1, "content": "secret text"},
    )
    return db


# ---------------------------------------------------------------------------
# data.py — ownership
# ---------------------------------------------------------------------------


def test_list_stories_scoped_to_owner_and_sorted():
    db = _seeded_db()
    stories = data.list_stories_for_user(db, UID_A, limit=20)
    assert [s["story_id"] for s in stories] == ["story-c", "story-a"]  # newest first
    assert all("Story B" != s["title"] for s in stories)
    # Long description is truncated with an ellipsis.
    story_a = next(s for s in stories if s["story_id"] == "story-a")
    assert len(story_a["description"]) <= data.SHORT_TEXT_LIMIT
    assert story_a["description"].endswith("…")


def test_list_stories_limit_clamped():
    db = _seeded_db()
    assert len(data.list_stories_for_user(db, UID_A, limit=1)) == 1
    assert len(data.list_stories_for_user(db, UID_A, limit=99999)) == 2


def test_list_stories_ranks_across_the_whole_collection():
    """Regression: ordering must happen in Firestore, not over a client-side page.

    With more stories than the page size, a sort-after-fetch would rank only an
    arbitrary slice and could miss the genuinely newest story entirely.
    """
    db = FakeFirestoreClient()
    for i in range(data.MAX_STORY_LIST_LIMIT + 20):
        db.seed(
            f"stories/story-{i:04d}",
            {
                "userId": UID_A,
                "title": f"Story {i}",
                # Newest story sorts LAST by document id, so it only surfaces
                # if the ordering is applied before the limit.
                "updatedAt": datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=i),
            },
        )
    stories = data.list_stories_for_user(db, UID_A, limit=3)
    assert [s["title"] for s in stories] == ["Story 119", "Story 118", "Story 117"]


def test_idor_story_of_other_user_is_not_found():
    db = _seeded_db()
    with pytest.raises(data.StoryNotFoundError):
        data.get_owned_story(db, "story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        data.get_story_overview(db, "story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        data.list_chapters(db, "story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        data.get_chapter(db, "story-b", "chb", UID_A, 0, 1000)
    with pytest.raises(data.StoryNotFoundError):
        data.list_entities(db, "story-b", UID_A, "characters")


def test_missing_story_indistinguishable_from_not_owned():
    db = _seeded_db()
    with pytest.raises(data.StoryNotFoundError):
        data.get_owned_story(db, "no-such-story", UID_A)


# ---------------------------------------------------------------------------
# data.py — reads
# ---------------------------------------------------------------------------


def test_story_overview_uses_sorted_chapter_index():
    db = _seeded_db()
    overview = data.get_story_overview(db, "story-a", UID_A)
    assert overview["title"] == "Story A"
    assert [c["title"] for c in overview["chapters"]] == [
        "Chapter One",
        "Chapter Two",
    ]


def test_story_overview_falls_back_to_projected_chapters():
    db = _seeded_db()
    overview = data.get_story_overview(db, "story-c", UID_A)
    assert [c["title"] for c in overview["chapters"]] == ["Only Chapter"]
    assert overview["chapters_truncated"] is False


def test_story_overview_from_chapter_index_is_never_truncated():
    """The denormalized index is rebuilt uncapped, so that path is complete."""
    db = _seeded_db()
    assert data.get_story_overview(db, "story-a", UID_A)["chapters_truncated"] is False


def test_list_chapters_sorted_with_ids():
    db = _seeded_db()
    page = data.list_chapters(db, "story-a", UID_A)
    assert [c["chapter_id"] for c in page.items] == ["ch1", "ch2"]
    assert page.items[0]["word_count"] == 5
    assert "content" not in page.items[0]
    assert page.truncated is False


def test_get_chapter_pagination_arithmetic():
    db = _seeded_db()
    first = data.get_chapter(db, "story-a", "ch1", UID_A, 0, 10_000)
    assert first["total_chars"] == 25_000
    assert len(first["content"]) == 10_000
    assert first["next_offset"] == 10_000

    last = data.get_chapter(db, "story-a", "ch1", UID_A, 20_000, 10_000)
    assert len(last["content"]) == 5_000
    assert last["next_offset"] is None

    # Windows tile the content exactly.
    middle = data.get_chapter(db, "story-a", "ch1", UID_A, 10_000, 10_000)
    full = first["content"] + middle["content"] + last["content"]
    assert full == "0123456789" * 2500


def test_get_chapter_clamps_inputs():
    db = _seeded_db()
    clamped = data.get_chapter(db, "story-a", "ch1", UID_A, -5, 999_999)
    assert clamped["offset"] == 0
    assert len(clamped["content"]) == 25_000  # max_chars clamped to 50k > total
    missing = pytest.raises(
        data.EntityNotFoundError, data.get_chapter, db, "story-a", "nope", UID_A, 0, 10
    )
    assert missing


def _seed_long_book(db, story_id: str = "story-long") -> int:
    """A story with more chapters than the cap, and NO chapterIndex.

    Document ids run opposite to `order`, which is what makes the ordering
    testable: a truncate-before-sort read returns the END of the book by
    document id and drops the beginning entirely.
    """
    total = data.COLLECTION_FETCH_LIMIT + 50
    db.seed(f"stories/{story_id}", {"userId": UID_A, "title": "Long Book"})
    for i in range(total):
        order = total - i
        db.seed(
            f"stories/{story_id}/chapters/ch-{i:04d}",
            {"title": f"Chapter {order}", "order": order, "content": "x"},
        )
    return total


def test_list_chapters_ranks_across_the_whole_collection():
    """The cap must slice reading order, not document-id order."""
    db = _seeded_db()
    _seed_long_book(db)

    page = data.list_chapters(db, "story-long", UID_A)

    assert len(page.items) == data.COLLECTION_FETCH_LIMIT
    assert page.truncated is True
    # Chapter 1 is present and first, rather than the book opening at chapter 51.
    assert [c["order"] for c in page.items] == list(
        range(1, data.COLLECTION_FETCH_LIMIT + 1)
    )


def test_story_overview_fallback_reports_truncation():
    db = _seeded_db()
    _seed_long_book(db)
    overview = data.get_story_overview(db, "story-long", UID_A)
    assert overview["chapters_truncated"] is True
    assert len(overview["chapters"]) == data.COLLECTION_FETCH_LIMIT
    assert overview["chapters"][0]["order"] == 1


def test_list_entities_reports_truncation():
    """Entities can't be ordered server-side, so the flag is the whole guarantee."""
    db = _seeded_db()
    for i in range(data.COLLECTION_FETCH_LIMIT + 5):
        db.seed(f"stories/story-a/plots/pl-{i:04d}", {"title": f"Plot {i}"})

    page = data.list_entities(db, "story-a", UID_A, "plots")

    assert len(page.items) == data.COLLECTION_FETCH_LIMIT
    assert page.truncated is True


def test_list_entities_descriptor_and_sorting():
    db = _seeded_db()
    page = data.list_entities(db, "story-a", UID_A, "characters")
    assert page.items == [
        {
            "entity_id": "char1",
            "name": "Mira",
            "descriptor": page.items[0]["descriptor"],
        }
    ]
    assert page.items[0]["descriptor"].startswith("p")
    assert len(page.items[0]["descriptor"]) <= data.SHORT_TEXT_LIMIT
    assert page.truncated is False


def test_list_entities_survives_non_string_names():
    """A name Firestore holds as a non-string must not break the sort."""
    db = _seeded_db()
    db.seed("stories/story-a/places/p1", {"name": 7, "description": "a tower"})
    db.seed("stories/story-a/places/p2", {"title": True, "description": "a moor"})
    db.seed("stories/story-a/places/p3", {"name": "   ", "description": "a fen"})
    db.seed("stories/story-a/places/p4", {"name": "Harbor"})

    entities = data.list_entities(db, "story-a", UID_A, "places").items

    by_id = {e["entity_id"]: e["name"] for e in entities}
    assert by_id["p1"] == "7"  # coerced, not crashed
    assert by_id["p2"] == "Unnamed"  # bool is never a name
    assert by_id["p3"] == "Unnamed"  # whitespace-only falls through
    assert by_id["p4"] == "Harbor"
    assert [e["name"] for e in entities] == ["7", "Harbor", "Unnamed", "Unnamed"]


def test_list_entities_falls_back_to_title():
    db = _seeded_db()
    db.seed("stories/story-a/plots/pl1", {"title": "The Reckoning"})
    page = data.list_entities(db, "story-a", UID_A, "plots")
    assert page.items == [
        {"entity_id": "pl1", "name": "The Reckoning", "descriptor": None}
    ]


def test_descriptor_fields_exist_in_entity_schema():
    """Guard the one copy of field names data.py still keeps.

    _DESCRIPTOR_FIELDS encodes editorial priority, so it can't be derived from
    entity_schema.py — but every name in it must exist there for its kind, or
    list_entities silently returns blank descriptors after a rename.
    """
    from agents.storyAgent.entity_schema import embedded_field_names

    for collection, fields in data._DESCRIPTOR_FIELDS.items():
        kind = data.ENTITY_KIND_BY_COLLECTION[collection]
        known = embedded_field_names(kind)
        assert known, f"entity_schema has no fields for kind {kind!r}"
        for field in fields:
            assert field in known, (
                f"{collection}: descriptor field {field!r} is not in "
                f"entity_schema.py for kind {kind!r} (known: {known})"
            )


def test_entity_collections_cover_every_schema_kind():
    """A new entity kind in entity_schema.py should surface through MCP too."""
    from agents.storyAgent.entity_schema import ENTITY_FIELD_SCHEMA

    assert set(data.ENTITY_KIND_BY_COLLECTION.values()) == set(ENTITY_FIELD_SCHEMA)
    assert set(data._DESCRIPTOR_FIELDS) == set(data.ENTITY_COLLECTIONS)


def test_tool_entity_type_literal_matches_collections():
    """The tool signature's Literal is the one copy the type system forces.

    `Literal[...]` members must be spelled out at the call site — they can't be
    unpacked from ENTITY_COLLECTIONS — so this is where drift would show up as a
    tool that advertises an entity type data.py rejects.
    """
    from typing import get_args

    from mcp_server.tools import EntityType

    assert set(get_args(EntityType)) == set(data.ENTITY_COLLECTIONS)


def test_list_entities_rejects_unknown_type():
    db = _seeded_db()
    with pytest.raises(ValueError):
        data.list_entities(db, "story-a", UID_A, "chapters")


def test_get_entity_returns_story_content():
    db = _seeded_db()
    entity = data.get_entity(db, "story-a", UID_A, "characters", "char1")
    assert entity["name"] == "Mira"
    assert entity["soul"] == "steadfast"
    assert entity["relationships"] == [{"name": "Bran", "relation": "brother"}]
    assert "embedding" not in entity
    with pytest.raises(data.EntityNotFoundError):
        data.get_entity(db, "story-a", UID_A, "characters", "nope")


def test_get_entity_projects_instead_of_passing_through():
    """Internal bookkeeping must not reach the client, named or not.

    The old denylist popped only `embedding`, so it shipped these as noise —
    and would have shipped any future internal field automatically.
    """
    db = _seeded_db()
    db.seed(
        "stories/story-a/characters/char2",
        {
            "name": "Bran",
            "personality": "wry",
            "artUrl": "https://example.test/bran.png",
            # Internal bookkeeping the client has no use for:
            "userId": UID_A,
            "storyId": "story-a",
            "signature": "abc123",
            "embeddingUpdatedAt": "2026-07-01T00:00:00Z",
            "embedding": [0.1, 0.2],
        },
    )
    entity = data.get_entity(db, "story-a", UID_A, "characters", "char2")

    assert entity["personality"] == "wry"
    assert entity["artUrl"] == "https://example.test/bran.png"
    for leaked in ("userId", "storyId", "signature", "embeddingUpdatedAt", "embedding"):
        assert leaked not in entity, f"{leaked} leaked into the tool result"


def test_get_entity_survives_a_firestore_vector_in_any_field():
    """A raw Firestore type outside `embedding` used to break serialization.

    Embeddings are stored as firestore_v1.vector.Vector, which is not JSON
    serializable; the projection is what makes that structurally impossible
    rather than dependent on remembering the field's name.
    """
    from google.cloud.firestore_v1.vector import Vector

    db = _seeded_db()
    db.seed(
        "stories/story-a/places/p1",
        {
            "name": "The Tower",
            "description": "tall",
            "embedding": Vector([0.1, 0.2]),
            "styleVector": Vector([0.3, 0.4]),  # a field no denylist would name
        },
    )
    entity = data.get_entity(db, "story-a", UID_A, "places", "p1")

    assert entity["description"] == "tall"
    assert "styleVector" not in entity
    json.dumps(entity)  # would raise TypeError on a Vector


def test_get_entity_omits_empty_values():
    db = _seeded_db()
    db.seed(
        "stories/story-a/places/p2",
        {"name": "Bare", "description": "", "atmosphere": None, "history": "damp"},
    )
    entity = data.get_entity(db, "story-a", UID_A, "places", "p2")
    assert entity["history"] == "damp"
    assert "description" not in entity
    assert "atmosphere" not in entity


def test_entity_content_fields_track_the_shared_schema():
    """get_entity's field set is the shared vocabulary plus named extras."""
    from agents.storyAgent.entity_schema import embedded_field_names

    for collection, kind in data.ENTITY_KIND_BY_COLLECTION.items():
        fields = data.entity_content_fields(collection)
        assert set(embedded_field_names(kind)).issubset(fields)
        assert set(data._ENTITY_EXTRA_FIELDS[collection]).issubset(fields)


# ---------------------------------------------------------------------------
# tools.py — wiring (uid from token, error mapping, rate limit, notice)
# ---------------------------------------------------------------------------


def _tool_server(
    db,
    max_rpm: int = 1000,
    *,
    enable_writes: bool = False,
    write_rpm: int = 1000,
) -> FastMCP:
    mcp = FastMCP("test")
    register_tools(
        mcp,
        db=db,
        rate_limiter=PerUserRateLimiter(max_rpm),
        write_rate_limiter=PerUserRateLimiter(write_rpm),
        enable_writes=enable_writes,
    )
    return mcp


def _token(uid: str = UID_A, scopes=("stories:read",)) -> AccessToken:
    return AccessToken(
        token="mcp_at_test", client_id="c1", scopes=list(scopes), subject=uid
    )


_RW = ("stories:read", "stories:write")


async def _call(mcp: FastMCP, name: str, arguments: dict) -> dict:
    """call_tool returns content blocks; the tool's dict rides as JSON text."""
    result = await mcp.call_tool(name, arguments)
    if isinstance(result, dict):
        return result
    return json.loads(result[0].text)


async def test_tools_use_token_subject_and_attach_notice():
    mcp = _tool_server(_seeded_db())
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        result = await _call(mcp, "list_my_stories", {})
    assert result["count"] == 2
    assert result["notice"] == UNTRUSTED_NOTICE
    assert {s["story_id"] for s in result["stories"]} == {"story-a", "story-c"}


async def test_tool_idor_returns_not_found():
    mcp = _tool_server(_seeded_db())
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A)):
        with pytest.raises(ToolError, match="Story not found"):
            await mcp.call_tool("get_story_overview", {"story_id": "story-b"})


async def test_tool_unauthenticated_rejected():
    mcp = _tool_server(_seeded_db())
    with patch("mcp_server.tools.get_access_token", return_value=None):
        with pytest.raises(ToolError, match="Not authenticated"):
            await mcp.call_tool("list_my_stories", {})


async def test_tool_rate_limit():
    mcp = _tool_server(_seeded_db(), max_rpm=1)
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        await mcp.call_tool("list_my_stories", {})
        with pytest.raises(ToolError, match="Rate limit"):
            await mcp.call_tool("list_my_stories", {})


async def test_tool_get_chapter_roundtrip():
    mcp = _tool_server(_seeded_db())
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        result = await _call(
            mcp,
            "get_chapter",
            {"story_id": "story-a", "chapter_id": "ch2"},
        )
    assert result["content"] == "short"
    assert result["next_offset"] is None
    assert result["notice"] == UNTRUSTED_NOTICE


async def test_tool_entities_roundtrip():
    mcp = _tool_server(_seeded_db())
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        listed = await _call(
            mcp, "list_entities", {"story_id": "story-a", "entity_type": "characters"}
        )
        entity = await _call(
            mcp,
            "get_entity",
            {
                "story_id": "story-a",
                "entity_type": "characters",
                "entity_id": listed["entities"][0]["entity_id"],
            },
        )
    assert entity["name"] == "Mira"
    assert "embedding" not in entity


# ---------------------------------------------------------------------------
# writes.py — ownership (the highest-value tests in this file)
#
# Every denial case asserts the document count is unchanged. The bug worth
# catching is not "denied", it is "denied but wrote anyway".
# ---------------------------------------------------------------------------


def _doc_count(db) -> int:
    return len(db.docs)


_fake_versions = itertools.count(1_000_000)


def _bump() -> int:
    """A fresh document version, for simulating a concurrent writer."""
    return next(_fake_versions)


async def test_create_chapter_idor_raises_story_not_found():
    db = _seeded_db()
    before = _doc_count(db)
    with pytest.raises(data.StoryNotFoundError):
        writes.create_chapter(db, UID_A, "story-b", "Sneaky", "text")
    assert _doc_count(db) == before


async def test_create_chapter_missing_story_indistinguishable_from_not_owned():
    db = _seeded_db()
    with pytest.raises(data.StoryNotFoundError):
        writes.create_chapter(db, UID_A, "no-such-story", "T", "x")
    with pytest.raises(data.StoryNotFoundError):
        writes.create_chapter(db, UID_A, "story-b", "T", "x")


async def test_chapter_userid_matches_story_owner():
    db = _seeded_db()
    result = writes.create_chapter(db, UID_A, "story-a", "New", "body")
    stored, _ = db.docs[f"stories/story-a/chapters/{result['chapter_id']}"]
    assert stored["userId"] == UID_A


async def test_no_write_argument_can_target_another_users_story():
    """Guards against a later `user_id`-style parameter reopening the IDOR."""
    db = _seeded_db()
    story = writes.create_story(db, UID_B, "B's book")
    with pytest.raises(data.StoryNotFoundError):
        writes.create_chapter(db, UID_A, story["story_id"], "T", "x")


# ---------------------------------------------------------------------------
# writes.py — derived fields and cross-layer round-trip
# ---------------------------------------------------------------------------


async def test_created_story_is_visible_to_list_stories_for_user():
    """The strongest test here: Firestore's order_by silently DROPS documents
    missing `updatedAt`, so a story written without it exists but is invisible.
    An existence assertion would not catch that; this does."""
    db = _seeded_db()
    created = writes.create_story(db, UID_A, "Fresh")
    listed = data.list_stories_for_user(db, UID_A, 20)
    assert created["story_id"] in {s["story_id"] for s in listed}


async def test_create_story_writes_every_field_mapstorydoc_reads():
    # Field list mirrors StoriesRepo.createStory / mapStoryDoc in the frontend
    # repo (src/services/StoriesRepo.ts) — mapStoryDoc calls .toDate() on the
    # timestamps without guarding, so an absent one throws in story lists.
    db = FakeFirestoreClient()
    result = writes.create_story(db, UID_A, "T", "D", "Fantasy", ["a", "b"])
    stored, _ = db.docs[f"stories/{result['story_id']}"]
    for field in (
        "id",
        "title",
        "description",
        "userId",
        "author",
        "isPublished",
        "createdAt",
        "updatedAt",
        "chapterCount",
        "views",
        "likes",
        "category",
        "tags",
        "targetAudience",
        "language",
        "copyright",
        "coverImageUrl",
    ):
        assert field in stored, f"missing {field}"
    assert stored["id"] == result["story_id"]
    assert stored["userId"] == UID_A
    assert stored["isPublished"] is False
    assert stored["chapterCount"] == 0
    assert stored["createdAt"].tzinfo is not None
    assert stored["updatedAt"].tzinfo is not None


async def test_create_story_author_from_public_profile():
    db = FakeFirestoreClient()
    db.seed(f"publicProfiles/{UID_A}", {"username": "mira"})
    result = writes.create_story(db, UID_A, "T")
    assert db.docs[f"stories/{result['story_id']}"][0]["author"] == "mira"


async def test_create_story_author_blank_when_profile_missing():
    db = FakeFirestoreClient()
    result = writes.create_story(db, UID_A, "T")
    assert db.docs[f"stories/{result['story_id']}"][0]["author"] == ""


async def test_create_chapter_bumps_chapter_count_and_story_updated_at():
    db = _seeded_db()
    before = db.docs["stories/story-a"][0]["updatedAt"]
    writes.create_chapter(db, UID_A, "story-a", "Three", "text")
    story, _ = db.docs["stories/story-a"]
    assert story["chapterCount"] == 3
    assert story["updatedAt"] > before


async def test_created_chapter_is_visible_to_read_tools():
    db = _seeded_db()
    created = writes.create_chapter(db, UID_A, "story-a", "Three", "one two three")
    page = data.list_chapters(db, "story-a", UID_A)
    assert created["chapter_id"] in {c["chapter_id"] for c in page.items}
    fetched = data.get_chapter(db, "story-a", created["chapter_id"], UID_A, 0, 1000)
    assert fetched["title"] == "Three"
    assert fetched["word_count"] == 3


# ---------------------------------------------------------------------------
# writes.py — ordering and concurrency
# ---------------------------------------------------------------------------


async def test_order_derives_from_max_order_not_chapter_count():
    """Post-delete drift: chapterCount lags max(order) because the frontend's
    deleteChapter decrements without renumbering. Deriving order from the
    counter (as StoriesRepo.addChapter does) would collide at 4."""
    db = FakeFirestoreClient()
    db.seed("stories/s", {"userId": UID_A, "chapterCount": 4, "title": "S"})
    for idx, order in enumerate((0, 1, 3, 4)):
        db.seed(f"stories/s/chapters/c{idx}", {"title": f"C{idx}", "order": order})
    result = writes.create_chapter(db, UID_A, "s", "Next", "x")
    assert result["order"] == 5


async def test_create_chapter_retries_on_lost_precondition():
    db = _seeded_db()
    bumped = {"done": False}

    def concurrent_writer(path):
        # Act as a second caller landing between our read and our update.
        if path == "stories/story-a" and not bumped["done"]:
            bumped["done"] = True
            db.docs[path] = (dict(db.docs[path][0]), 999_999)

    db.before_update = concurrent_writer
    result = writes.create_chapter(db, UID_A, "story-a", "Three", "x")
    assert result["attempts"] == 2
    assert result["order"] == 3


async def test_orders_are_unique_under_repeated_contention():
    db = _seeded_db()
    orders = []
    for n in range(4):
        state = {"done": False}

        def concurrent_writer(path, state=state):
            if path == "stories/story-a" and not state["done"]:
                state["done"] = True
                db.docs[path] = (dict(db.docs[path][0]), _bump())

        db.before_update = concurrent_writer
        # Distinct titles: identical arguments would hit the idempotency
        # replay and return the same chapter rather than exercising ordering.
        orders.append(
            writes.create_chapter(db, UID_A, "story-a", f"C{n}", "x")["order"]
        )
    assert len(set(orders)) == len(orders)


async def test_no_order_collision_with_a_claimed_but_unwritten_chapter():
    """The interleaving the retry loop alone cannot save: caller A claims its
    slot (the story doc moves), but its chapter document is not written yet, so
    a max(order) read by caller B still reports the old ceiling. B must take
    its order from the nextChapterOrder counter A's claim bumped — deriving it
    from the subcollection gave both callers the same value."""
    db = FakeFirestoreClient()
    db.seed("stories/s", {"userId": UID_A, "title": "S", "chapterCount": 3})
    for i in range(3):
        db.seed(f"stories/s/chapters/c{i}", {"title": f"C{i}", "order": i})

    # Caller A: claim the counters exactly as _append_chapter does, then stall
    # before the chapter write.
    story_ref = db.collection("stories").document("s")
    snap_a = data.get_owned_story_snapshot(db, "s", UID_A)
    story_a = snap_a.to_dict()
    order_a = max(
        int(story_a.get("nextChapterOrder") or 0), writes._next_order(db, "s")
    )
    story_ref.update(
        {
            "chapterCount": 4,
            "nextChapterOrder": order_a + 1,
            "updatedAt": writes._now(),
        },
        option=db.write_option(last_update_time=snap_a.update_time),
    )

    # Caller B: full create_chapter while A's chapter is still unwritten.
    result_b = writes.create_chapter(db, UID_A, "s", "From B", "text")

    # Caller A finally writes its chapter.
    chapter_a = story_ref.collection("chapters").document()
    chapter_a.set({"id": chapter_a.id, "title": "From A", "order": order_a})

    assert result_b["order"] != order_a
    orders = [
        (doc.to_dict() or {}).get("order")
        for doc in story_ref.collection("chapters").stream()
    ]
    assert len(set(orders)) == len(orders)


async def test_order_counter_self_heals_below_existing_chapters():
    """A frontend addChapter can write an order the counter never saw; the
    max(counter, subcollection) floor must step over it rather than collide."""
    db = FakeFirestoreClient()
    # Counter says 2, but a chapter with order 5 already exists.
    db.seed(
        "stories/s",
        {"userId": UID_A, "title": "S", "chapterCount": 1, "nextChapterOrder": 2},
    )
    db.seed("stories/s/chapters/c0", {"title": "C0", "order": 5})
    result = writes.create_chapter(db, UID_A, "s", "Next", "x")
    assert result["order"] == 6


async def test_write_conflict_after_exhausting_attempts():
    db = _seeded_db()

    def always_stale(path):
        # A NEW version every time, so no retry can ever match what it read.
        if path == "stories/story-a":
            db.docs[path] = (dict(db.docs[path][0]), _bump())

    db.before_update = always_stale
    with pytest.raises(writes.WriteConflictError):
        writes.create_chapter(db, UID_A, "story-a", "C", "x")


# ---------------------------------------------------------------------------
# writes.py — limits the Admin SDK bypasses
# ---------------------------------------------------------------------------


async def test_story_cap_enforced_from_denormalized_counter():
    db = FakeFirestoreClient()
    db.seed(f"users/{UID_A}", {"storyCount": writes.MAX_STORIES_PER_USER})
    before = _doc_count(db)
    with pytest.raises(writes.LimitExceededError) as exc:
        writes.create_story(db, UID_A, "One too many")
    assert exc.value.limit_name == "stories_per_user"
    assert _doc_count(db) == before

    db.seed(f"users/{UID_A}", {"storyCount": writes.MAX_STORIES_PER_USER - 1})
    assert writes.create_story(db, UID_A, "Just fits")["story_id"]


async def test_missing_user_doc_counts_as_zero_stories():
    # Matches the userStoryCount() helper in firestore.rules.
    db = FakeFirestoreClient()
    assert writes.create_story(db, UID_A, "First")["story_id"]


async def test_chapter_cap_enforced():
    db = FakeFirestoreClient()
    db.seed(
        "stories/s",
        {"userId": UID_A, "chapterCount": writes.MAX_CHAPTERS_PER_STORY},
    )
    before = _doc_count(db)
    with pytest.raises(writes.LimitExceededError) as exc:
        writes.create_chapter(db, UID_A, "s", "Too many", "x")
    assert exc.value.limit_name == "chapters_per_story"
    assert _doc_count(db) == before


async def test_content_char_cap_measures_the_stored_string():
    db = _seeded_db()
    before = _doc_count(db)
    # One word, long enough that the stored <p>-wrapped form exceeds the cap.
    with pytest.raises(writes.LimitExceededError) as exc:
        writes.create_chapter(
            db, UID_A, "story-a", "T", "x" * (writes.MAX_CHAPTER_CONTENT_CHARS + 1)
        )
    assert exc.value.limit_name == "chapter_content_chars"
    assert _doc_count(db) == before


async def test_content_char_cap_boundary_is_inclusive():
    db = _seeded_db()
    # "<p>" + body + "</p>" == exactly the cap.
    body = "x" * (writes.MAX_CHAPTER_CONTENT_CHARS - len("<p></p>"))
    result = writes.create_chapter(db, UID_A, "story-a", "T", body)
    stored, _ = db.docs[f"stories/story-a/chapters/{result['chapter_id']}"]
    assert len(stored["content"]) == writes.MAX_CHAPTER_CONTENT_CHARS


async def test_word_cap_enforced():
    db = _seeded_db()
    before = _doc_count(db)
    with pytest.raises(writes.LimitExceededError) as exc:
        writes.create_chapter(
            db, UID_A, "story-a", "T", "word " * (writes.MAX_CHAPTER_WORDS + 1)
        )
    assert exc.value.limit_name == "chapter_words"
    assert _doc_count(db) == before


@pytest.mark.parametrize("bad_title", ["", "   ", "t" * 201])
async def test_blank_and_oversize_titles_rejected(bad_title):
    db = _seeded_db()
    with pytest.raises(ValueError):
        writes.create_story(db, UID_A, bad_title)
    with pytest.raises(ValueError):
        writes.create_chapter(db, UID_A, "story-a", bad_title, "x")


async def test_tag_count_and_length_capped():
    db = FakeFirestoreClient()
    with pytest.raises(ValueError):
        writes.create_story(db, UID_A, "T", tags=[f"t{i}" for i in range(11)])
    with pytest.raises(ValueError):
        writes.create_story(db, UID_A, "T", tags=["x" * 41])


async def test_description_cap_enforced():
    db = FakeFirestoreClient()
    with pytest.raises(ValueError):
        writes.create_story(db, UID_A, "T", "d" * 2001)


# ---------------------------------------------------------------------------
# writes.py — content transform
# ---------------------------------------------------------------------------


async def test_content_is_escaped_and_paragraph_wrapped():
    db = _seeded_db()
    result = writes.create_chapter(
        db, UID_A, "story-a", "T", "a & b <script>x</script>\n\nsecond"
    )
    stored, _ = db.docs[f"stories/story-a/chapters/{result['chapter_id']}"]
    assert stored["content"] == (
        "<p>a &amp; b &lt;script&gt;x&lt;/script&gt;</p>\n<p>second</p>"
    )
    assert "<script" not in stored["content"]


async def test_word_count_matches_the_frontend_formula():
    # StoriesRepo.countWords is content.trim().split(/\s+/) over the raw HTML,
    # and the editor recomputes it on the next save — so counting the stored
    # string here keeps the number from jumping under the user.
    db = _seeded_db()
    plain = "one two three\n\nfour five"
    result = writes.create_chapter(db, UID_A, "story-a", "T", plain)
    stored, _ = db.docs[f"stories/story-a/chapters/{result['chapter_id']}"]
    assert result["word_count"] == len(stored["content"].split())
    assert result["word_count"] == len(plain.split())


async def test_empty_content_stores_empty_string():
    db = _seeded_db()
    result = writes.create_chapter(db, UID_A, "story-a", "T", "")
    stored, _ = db.docs[f"stories/story-a/chapters/{result['chapter_id']}"]
    assert stored["content"] == ""
    assert stored["wordCount"] == 0


# ---------------------------------------------------------------------------
# writes.py — idempotency
# ---------------------------------------------------------------------------


async def test_identical_create_story_returns_same_id_and_writes_once():
    db = FakeFirestoreClient()
    first = writes.create_story(db, UID_A, "Twice", "same")
    second = writes.create_story(db, UID_A, "Twice", "same")
    assert second["story_id"] == first["story_id"]
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert len([p for p in db.docs if p.startswith("stories/")]) == 1


async def test_identical_create_chapter_returns_same_id_and_writes_once():
    db = _seeded_db()
    first = writes.create_chapter(db, UID_A, "story-a", "Dup", "body")
    second = writes.create_chapter(db, UID_A, "story-a", "Dup", "body")
    assert second["chapter_id"] == first["chapter_id"]
    assert db.docs["stories/story-a"][0]["chapterCount"] == 3  # bumped once


async def test_idempotency_is_scoped_to_the_caller():
    """Cross-user dedup would be both an information leak and a denial of
    service — B could pre-claim A's key and block the write."""
    db = FakeFirestoreClient()
    a = writes.create_story(db, UID_A, "Same title", "same")
    b = writes.create_story(db, UID_B, "Same title", "same")
    assert a["story_id"] != b["story_id"]


async def test_idempotency_window_expires():
    db = FakeFirestoreClient()
    first = writes.create_story(db, UID_A, "Later", "again")
    later = datetime.now(timezone.utc) + timedelta(
        seconds=writes.IDEMPOTENCY_TTL_SECONDS + 1
    )
    with patch("mcp_server.writes._now", return_value=later):
        second = writes.create_story(db, UID_A, "Later", "again")
    assert second["story_id"] != first["story_id"]


async def test_failed_write_releases_the_reservation():
    db = FakeFirestoreClient()
    db.seed(f"users/{UID_A}", {"storyCount": writes.MAX_STORIES_PER_USER})
    with pytest.raises(writes.LimitExceededError):
        writes.create_story(db, UID_A, "Blocked")
    assert not [p for p in db.docs if p.startswith(f"{writes.WRITES_COLLECTION}/")]
    # And the same call succeeds once the blocker is gone.
    db.seed(f"users/{UID_A}", {"storyCount": 0})
    assert writes.create_story(db, UID_A, "Blocked")["story_id"]


async def test_concurrent_identical_call_is_reported_not_silently_dropped():
    db = FakeFirestoreClient()
    key = writes.idempotency_key(
        UID_A,
        "create_story",
        {"title": "Inflight", "description": "", "category": "", "tags": []},
    )
    db.seed(
        f"{writes.WRITES_COLLECTION}/{key}",
        {
            "uid": UID_A,
            "tool": "create_story",
            "expiresAt": datetime.now(timezone.utc) + timedelta(seconds=60),
        },
    )
    with pytest.raises(writes.DuplicateInFlightError):
        writes.create_story(db, UID_A, "Inflight")


# ---------------------------------------------------------------------------
# tools.py — write wiring and scope enforcement
# ---------------------------------------------------------------------------


async def test_read_only_token_cannot_call_write_tools():
    db = _seeded_db()
    mcp = _tool_server(db, enable_writes=True)
    before = _doc_count(db)
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        with pytest.raises(ToolError, match="read-only"):
            await mcp.call_tool("create_story", {"title": "Nope"})
        with pytest.raises(ToolError, match="read-only"):
            await mcp.call_tool(
                "create_chapter", {"story_id": "story-a", "title": "Nope"}
            )
    assert _doc_count(db) == before


async def test_write_scoped_token_can_call_write_tools():
    db = _seeded_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        story = await _call(mcp, "create_story", {"title": "Made by MCP"})
        chapter = await _call(
            mcp,
            "create_chapter",
            {"story_id": story["story_id"], "title": "One", "content": "hello"},
        )
    assert story["notice"] == UNTRUSTED_NOTICE
    assert chapter["notice"] == UNTRUSTED_NOTICE
    assert chapter["order"] == 0


async def test_write_scoped_token_can_still_call_read_tools():
    mcp = _tool_server(_seeded_db(), enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        result = await _call(mcp, "list_my_stories", {})
    assert result["count"] == 2


async def test_write_tools_absent_when_disabled():
    mcp = _tool_server(_seeded_db(), enable_writes=False)
    names = {t.name for t in await mcp.list_tools()}
    assert "create_story" not in names
    assert "create_chapter" not in names
    assert len(names) == 6


async def test_write_tools_present_when_enabled():
    mcp = _tool_server(_seeded_db(), enable_writes=True)
    names = {t.name for t in await mcp.list_tools()}
    assert {"create_story", "create_chapter"} <= names
    assert len(names) == 8


async def test_write_tool_idor_returns_story_not_found():
    db = _seeded_db()
    mcp = _tool_server(db, enable_writes=True)
    before = _doc_count(db)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="Story not found"):
            await mcp.call_tool(
                "create_chapter", {"story_id": "story-b", "title": "Sneaky"}
            )
    assert _doc_count(db) == before


async def test_write_tool_unauthenticated_rejected():
    mcp = _tool_server(_seeded_db(), enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=None):
        with pytest.raises(ToolError, match="Not authenticated"):
            await mcp.call_tool("create_story", {"title": "x"})


async def test_write_rate_limit_is_separate_and_tighter():
    mcp = _tool_server(_seeded_db(), enable_writes=True, write_rpm=1)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        await mcp.call_tool("create_story", {"title": "First"})
        with pytest.raises(ToolError, match="Write rate limit"):
            await mcp.call_tool("create_story", {"title": "Second"})
        # The shared bucket is untouched: reads still work.
        result = await _call(mcp, "list_my_stories", {})
    assert result["count"] >= 2


async def test_write_roundtrip_through_read_tools():
    db = _seeded_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        story = await _call(mcp, "create_story", {"title": "Round Trip"})
        await _call(
            mcp,
            "create_chapter",
            {"story_id": story["story_id"], "title": "Ch1", "content": "a b c"},
        )
        listed = await _call(mcp, "list_my_stories", {})
        overview = await _call(
            mcp, "get_story_overview", {"story_id": story["story_id"]}
        )
        chapters = await _call(mcp, "list_chapters", {"story_id": story["story_id"]})
    assert story["story_id"] in {s["story_id"] for s in listed["stories"]}
    assert overview["title"] == "Round Trip"
    assert [c["title"] for c in chapters["chapters"]] == ["Ch1"]


# ---------------------------------------------------------------------------
# Drift guards
# ---------------------------------------------------------------------------


def test_write_scope_is_not_in_required_scopes():
    """The single most important regression guard in this change.

    required_scopes is enforced CONJUNCTIVELY by RequireAuthMiddleware over the
    whole /mcp mount, so it is a FLOOR: every scope in it is demanded of every
    caller, at the transport layer, before any tool runs. Moving the write
    scope there would not enable writes — it would make write access mandatory
    and read-only connections impossible, collapsing the read/write
    distinction this server is built around.

    It would also invalidate every issued token, but that is the lesser reason
    and it stops being true once tokens are cheap to re-mint. The one above
    never stops being true. If you are here because consolidating the two
    lists looked like tidying: it isn't.
    """
    assert mcp_app.MCP_WRITE_SCOPE not in mcp_app.MCP_REQUIRED_SCOPES
    assert mcp_app.MCP_WRITE_SCOPE in mcp_app.MCP_VALID_SCOPES
    assert mcp_app.MCP_REQUIRED_SCOPES == [mcp_app.MCP_READ_SCOPE]


def test_default_scopes_stay_read_only():
    """A client that registers without an explicit `scope` must not be handed
    write access by omission."""
    assert mcp_app.MCP_DEFAULT_SCOPES == [mcp_app.MCP_READ_SCOPE]


def test_tools_write_scope_constant_matches_app():
    from mcp_server.tools import WRITE_SCOPE

    assert WRITE_SCOPE == mcp_app.MCP_WRITE_SCOPE


def test_write_limits_match_the_client_limits():
    """The Admin SDK bypasses firestore.rules, so writes.py re-declares the
    ceilings. Parse the frontend's own sources and fail when they drift.

    Skipped when the sibling repo isn't checked out — this has none of the
    CI-credential cost that retired the previous cross-repo test.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "novelsync-frontend"
    repo = root / "src" / "services" / "StoriesRepo.ts"
    rules = root / "firestore.rules"
    if not repo.exists() or not rules.exists():
        pytest.skip("novelsync-frontend not checked out beside this repo")

    repo_src = repo.read_text()
    word_limit = int(re.search(r"WORD_LIMIT\s*=\s*(\d+)", repo_src).group(1))
    chapter_limit = int(re.search(r"CHAPTER_LIMIT\s*=\s*(\d+)", repo_src).group(1))
    assert writes.MAX_CHAPTER_WORDS == word_limit
    assert writes.MAX_CHAPTERS_PER_STORY == chapter_limit

    rules_src = rules.read_text()
    assert f"content.size() <= {writes.MAX_CHAPTER_CONTENT_CHARS}" in rules_src
    # firestore.rules spells the cap inline: "... < 100; // MAX_STORIES_PER_USER"
    stories_cap = int(
        re.search(r"userStoryCount\([^)]*\)\s*<\s*(\d+)", rules_src).group(1)
    )
    assert writes.MAX_STORIES_PER_USER == stories_cap
