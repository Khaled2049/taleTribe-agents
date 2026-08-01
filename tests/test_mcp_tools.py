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
from structlog.testing import capture_logs  # noqa: E402

from mcp_server import app as mcp_app  # noqa: E402
from mcp_server import blocks as blocks_module  # noqa: E402
from mcp_server import data  # noqa: E402
from mcp_server import writes  # noqa: E402
from mcp_server import tools as tools_module  # noqa: E402
from mcp_server.tools import UNTRUSTED_NOTICE, register_tools  # noqa: E402
from rate_limit import PerUserRateLimiter  # noqa: E402
from tests import mcp_fakes  # noqa: E402
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


async def test_expired_reservation_is_claimed_by_exactly_one_racer():
    """Taking over an expired reservation has to keep the exactly-one-winner
    property that `create` gives the common path for free.

    The loser of the takeover is told the call is already in flight rather than
    being allowed to write a second story.
    """
    db = FakeFirestoreClient()
    writes.create_story(db, UID_A, "Later", "again")
    stories_before = len([p for p in db.docs if p.startswith("stories/")])

    later = datetime.now(timezone.utc) + timedelta(
        seconds=writes.IDEMPOTENCY_TTL_SECONDS + 1
    )
    key = writes.idempotency_key(
        UID_A,
        "create_story",
        {"title": "Later", "description": "again", "category": "", "tags": []},
    )
    path = f"{writes.WRITES_COLLECTION}/{key}"

    def steal(doc_path: str) -> None:
        # Act as the other racer: land a write on the reservation in the window
        # between this caller's read of it and its takeover.
        if doc_path != path:
            return
        db.before_update = None
        data, _version = db.docs[path]
        db.docs[path] = (data, next(mcp_fakes._versions))

    db.before_update = steal
    with patch("mcp_server.writes._now", return_value=later):
        with pytest.raises(writes.DuplicateInFlightError):
            writes.create_story(db, UID_A, "Later", "again")
    db.before_update = None
    assert len([p for p in db.docs if p.startswith("stories/")]) == stories_before


async def test_a_vanished_reservation_is_reclaimed():
    """TTL collection or a concurrent _release can remove the document between
    the create that lost and the read that follows it. The key is free then."""
    db = FakeFirestoreClient()
    first = writes.create_story(db, UID_A, "Gone", "poof")
    for path in [p for p in db.docs if p.startswith(f"{writes.WRITES_COLLECTION}/")]:
        del db.docs[path]
    second = writes.create_story(db, UID_A, "Gone", "poof")
    assert second["story_id"] != first["story_id"]
    assert second["idempotent_replay"] is False


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
    assert names.isdisjoint(
        {"create_story", "create_chapter", "append_to_chapter", "edit_chapter_blocks"}
    )
    # get_chapter_blocks is a READ tool: it stays available with writes off.
    assert "get_chapter_blocks" in names
    assert len(names) == 7


async def test_write_tools_present_when_enabled():
    mcp = _tool_server(_seeded_db(), enable_writes=True)
    names = {t.name for t in await mcp.list_tools()}
    assert {
        "create_story",
        "create_chapter",
        "append_to_chapter",
        "edit_chapter_blocks",
    } <= names
    assert len(names) == 11


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
# Chapter editing — reads that make blocks addressable
# ---------------------------------------------------------------------------

# A chapter shaped like real editor output: heading, paragraph, a list whose
# items contain nested <p>, a blockquote, and an image. Byte-identity of the
# blocks nobody touched is asserted against this.
RICH = (
    "<h2>The Descent</h2>"
    "<p>She counted the steps.</p>"
    '<ul class="list-disc"><li><p>rope</p></li><li><p>lamp</p></li></ul>'
    "<blockquote><p>Nothing waits below.</p></blockquote>"
    '<img src="cave.png" data-display-mode="wrap">'
)
RICH_BLOCKS = ["h2", "p", "ul", "blockquote", "img"]


def _rich_db() -> FakeFirestoreClient:
    db = _seeded_db()
    db.seed(
        "stories/story-a/chapters/rich",
        {"title": "Rich", "order": 3, "wordCount": 9, "content": RICH},
    )
    return db


def _revision_of(db, path: str = "stories/story-a/chapters/rich") -> str:
    return str(db.docs[path][1])


def _content_of(db, path: str = "stories/story-a/chapters/rich") -> str:
    return db.docs[path][0]["content"]


def test_get_chapter_exposes_a_revision():
    db = _rich_db()
    chapter = data.get_chapter(db, "story-a", "rich", UID_A, 0, 100)
    assert chapter["revision"] == _revision_of(db)


def test_get_chapter_blocks_lists_indices_tags_and_previews():
    db = _rich_db()
    listing = data.get_chapter_blocks(db, "story-a", "rich", UID_A, 0, 500)
    assert listing["block_count"] == 5
    assert [b["tag"] for b in listing["blocks"]] == RICH_BLOCKS
    assert [b["index"] for b in listing["blocks"]] == [0, 1, 2, 3, 4]
    assert listing["blocks"][1]["preview"] == "She counted the steps."
    # An image has no text of its own; the tag is what identifies it.
    assert listing["blocks"][4]["preview"] == ""
    assert listing["next_index"] is None
    assert listing["revision"] == _revision_of(db)


def test_get_chapter_blocks_paginates():
    db = _rich_db()
    first = data.get_chapter_blocks(db, "story-a", "rich", UID_A, 0, 2)
    assert [b["index"] for b in first["blocks"]] == [0, 1]
    assert first["next_index"] == 2
    second = data.get_chapter_blocks(
        db, "story-a", "rich", UID_A, first["next_index"], 2
    )
    assert [b["index"] for b in second["blocks"]] == [2, 3]
    last = data.get_chapter_blocks(db, "story-a", "rich", UID_A, 4, 2)
    assert [b["index"] for b in last["blocks"]] == [4]
    assert last["next_index"] is None


def test_get_chapter_blocks_clamps_inputs():
    db = _rich_db()
    listing = data.get_chapter_blocks(db, "story-a", "rich", UID_A, -5, 999_999)
    assert listing["start_index"] == 0
    assert len(listing["blocks"]) == 5


def test_get_chapter_blocks_is_owner_scoped():
    db = _rich_db()
    with pytest.raises(data.StoryNotFoundError):
        data.get_chapter_blocks(db, "story-b", "chb", UID_A, 0, 10)
    with pytest.raises(data.EntityNotFoundError):
        data.get_chapter_blocks(db, "story-a", "no-such-chapter", UID_A, 0, 10)


def test_get_chapter_blocks_on_an_empty_chapter():
    db = _seeded_db()
    db.seed("stories/story-a/chapters/blank", {"title": "Blank", "content": ""})
    listing = data.get_chapter_blocks(db, "story-a", "blank", UID_A, 0, 10)
    assert listing["block_count"] == 0
    assert listing["blocks"] == []


# ---------------------------------------------------------------------------
# Chapter editing — the revision guard
#
# This is the safety property of the whole feature: an edit must never land on
# a version the caller did not read.
# ---------------------------------------------------------------------------


async def test_stale_revision_is_refused_and_writes_nothing():
    db = _rich_db()
    before = _content_of(db)
    with pytest.raises(writes.StaleRevisionError):
        writes.append_to_chapter(
            db, UID_A, "story-a", "rich", "more", "not-the-version"
        )
    assert _content_of(db) == before


async def test_missing_revision_is_refused():
    db = _rich_db()
    with pytest.raises(ValueError, match="revision is required"):
        writes.append_to_chapter(db, UID_A, "story-a", "rich", "more", "")


async def test_fresh_revision_succeeds_and_returns_the_new_one():
    db = _rich_db()
    old = _revision_of(db)
    result = writes.append_to_chapter(db, UID_A, "story-a", "rich", "more", old)
    assert result["revision"] == _revision_of(db)
    assert result["revision"] != old


async def test_concurrent_writer_between_check_and_update_is_caught():
    """The window the precondition exists to close.

    The pre-check passes, then a concurrent editor save lands, then our update
    runs. Without the last_update_time option this would silently clobber it.
    """
    db = _rich_db()
    revision = _revision_of(db)
    path = "stories/story-a/chapters/rich"

    def concurrent_editor(updated_path: str) -> None:
        if updated_path != path:
            return
        db.before_update = None  # fire once
        record, _ = db.docs[path]
        db.docs[path] = ({**record, "content": "<p>the human typed this</p>"}, _bump())

    db.before_update = concurrent_editor
    with pytest.raises(writes.StaleRevisionError):
        writes.append_to_chapter(db, UID_A, "story-a", "rich", "appended", revision)
    assert _content_of(db) == "<p>the human typed this</p>"


async def test_revision_from_get_chapter_blocks_is_accepted_by_an_edit():
    """The two sides of the contract must agree on how a revision is spelled."""
    db = _rich_db()
    listing = data.get_chapter_blocks(db, "story-a", "rich", UID_A, 0, 10)
    writes.append_to_chapter(db, UID_A, "story-a", "rich", "ok", listing["revision"])


# ---------------------------------------------------------------------------
# append_to_chapter
# ---------------------------------------------------------------------------


async def test_append_preserves_existing_content_byte_for_byte():
    db = _rich_db()
    writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "New line.", _revision_of(db)
    )
    stored = _content_of(db)
    assert stored.startswith(RICH)
    assert stored == RICH + "\n<p>New line.</p>"


async def test_append_to_an_empty_chapter_adds_no_leading_separator():
    db = _seeded_db()
    db.seed("stories/story-a/chapters/blank", {"title": "Blank", "content": ""})
    path = "stories/story-a/chapters/blank"
    writes.append_to_chapter(
        db, UID_A, "story-a", "blank", "First.", _revision_of(db, path)
    )
    assert _content_of(db, path) == "<p>First.</p>"


async def test_append_escapes_markup_and_splits_paragraphs():
    db = _rich_db()
    writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "<b>bold</b>\n\nsecond", _revision_of(db)
    )
    stored = _content_of(db)
    assert "<p>&lt;b&gt;bold&lt;/b&gt;</p>" in stored
    assert stored.endswith("<p>second</p>")


async def test_append_block_count_survives_malformed_existing_content():
    """block_count must be what split_blocks will say next time, not the sum of
    the two halves — an unclosed tag makes those differ, and the number is the
    index range the caller's next edit addresses."""
    db = _rich_db()
    db.seed(
        "stories/story-a/chapters/broken",
        {"title": "Broken", "order": 9, "wordCount": 1, "content": "<p>unclosed"},
    )
    result = writes.append_to_chapter(
        db,
        UID_A,
        "story-a",
        "broken",
        "New paragraph.",
        _revision_of(db, "stories/story-a/chapters/broken"),
    )
    content = _content_of(db, "stories/story-a/chapters/broken")
    assert result["block_count"] == len(blocks_module.split_blocks(content)) == 1
    assert result["appended_blocks"] == 1  # the sum would have said 2


async def test_append_recomputes_word_count_and_touches_the_story():
    db = _rich_db()
    story_before = db.docs["stories/story-a"][0]["updatedAt"]
    result = writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "one two three", _revision_of(db)
    )
    record = db.docs["stories/story-a/chapters/rich"][0]
    assert record["wordCount"] == result["word_count"]
    assert record["wordCount"] == len(_content_of(db).split())
    assert db.docs["stories/story-a"][0]["updatedAt"] > story_before


async def test_append_does_not_touch_chapter_count():
    """No chapter is created by an edit, so the counter must not move."""
    db = _rich_db()
    before = db.docs["stories/story-a"][0]["chapterCount"]
    writes.append_to_chapter(db, UID_A, "story-a", "rich", "x", _revision_of(db))
    assert db.docs["stories/story-a"][0]["chapterCount"] == before


async def test_append_rejects_empty_content():
    db = _rich_db()
    with pytest.raises(ValueError, match="must not be empty"):
        writes.append_to_chapter(db, UID_A, "story-a", "rich", "   ", _revision_of(db))


async def test_append_enforces_the_word_cap_on_the_rebuilt_string():
    db = _seeded_db()
    body = "<p>" + " ".join(["w"] * (writes.MAX_CHAPTER_WORDS - 1)) + "</p>"
    db.seed("stories/story-a/chapters/big", {"title": "Big", "content": body})
    path = "stories/story-a/chapters/big"
    with pytest.raises(writes.LimitExceededError) as exc:
        writes.append_to_chapter(
            db, UID_A, "story-a", "big", "two more words", _revision_of(db, path)
        )
    assert exc.value.limit_name == "chapter_words"
    assert _content_of(db, path) == body  # denied, and nothing written


async def test_append_enforces_the_char_cap_on_the_rebuilt_string():
    db = _seeded_db()
    body = "<p>" + "x" * (writes.MAX_CHAPTER_CONTENT_CHARS - 10) + "</p>"
    db.seed("stories/story-a/chapters/big", {"title": "Big", "content": body})
    path = "stories/story-a/chapters/big"
    with pytest.raises(writes.LimitExceededError) as exc:
        writes.append_to_chapter(
            db, UID_A, "story-a", "big", "y" * 100, _revision_of(db, path)
        )
    assert exc.value.limit_name == "chapter_content_chars"
    assert _content_of(db, path) == body


async def test_append_is_owner_scoped():
    db = _rich_db()
    path = "stories/story-b/chapters/chb"
    with pytest.raises(data.StoryNotFoundError):
        writes.append_to_chapter(
            db, UID_A, "story-b", "chb", "x", _revision_of(db, path)
        )


# ---------------------------------------------------------------------------
# edit_chapter_blocks
# ---------------------------------------------------------------------------


def _edit(db, ops, chapter="rich", uid=UID_A):
    path = f"stories/story-a/chapters/{chapter}"
    return writes.edit_chapter_blocks(
        db, uid, "story-a", chapter, ops, _revision_of(db, path)
    )


async def test_replace_leaves_every_other_block_byte_identical():
    """The central promise of block editing."""
    db = _rich_db()
    original = blocks_module.split_blocks(RICH)
    _edit(db, [{"action": "replace", "index": 1, "text": "She counted twice."}])
    after = blocks_module.split_blocks(_content_of(db))
    assert after[1].html == "<p>She counted twice.</p>"
    for position in (0, 2, 3, 4):
        assert after[position].html == original[position].html


async def test_replace_with_empty_text_removes_the_block():
    db = _rich_db()
    result = _edit(db, [{"action": "replace", "index": 4, "text": ""}])
    assert result["block_count"] == 4
    assert "cave.png" not in _content_of(db)


async def test_removing_every_block_leaves_an_empty_chapter():
    db = _rich_db()
    ops = [{"action": "replace", "index": i, "text": ""} for i in range(5)]
    result = _edit(db, ops)
    assert _content_of(db) == ""
    assert result["block_count"] == 0
    assert result["word_count"] == 0


async def test_insert_after_places_the_block_in_the_right_seam():
    db = _rich_db()
    _edit(db, [{"action": "insert_after", "index": 1, "text": "Then she stopped."}])
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(db))]
    assert tags == ["h2", "p", "p", "ul", "blockquote", "img"]
    assert blocks_module.split_blocks(_content_of(db))[2].html == (
        "<p>Then she stopped.</p>"
    )


async def test_insert_after_minus_one_prepends():
    db = _rich_db()
    _edit(db, [{"action": "insert_after", "index": -1, "text": "Prologue."}])
    assert _content_of(db).startswith("<p>Prologue.</p>\n<h2>")


async def test_insert_after_last_index_appends():
    db = _rich_db()
    _edit(db, [{"action": "insert_after", "index": 4, "text": "The end."}])
    assert _content_of(db).endswith("<p>The end.</p>")


async def test_one_op_may_introduce_several_blocks():
    db = _rich_db()
    result = _edit(
        db, [{"action": "replace", "index": 1, "text": "First para.\n\nSecond para."}]
    )
    assert result["block_count"] == 6
    after = blocks_module.split_blocks(_content_of(db))
    assert [b.html for b in after[1:3]] == ["<p>First para.</p>", "<p>Second para.</p>"]


async def test_replacing_a_heading_keeps_its_level():
    """A heading is rebuilt as itself, not flattened into a paragraph.

    RICH[0] is an <h2>. Routing it through _to_paragraph_html like a paragraph
    would silently demote it, and nothing in the product could restore it.
    """
    db = _rich_db()
    _edit(db, [{"action": "replace", "index": 0, "text": "The Ascent"}])
    after = blocks_module.split_blocks(_content_of(db))
    assert after[0] == blocks_module.Block(tag="h2", html="<h2>The Ascent</h2>")


async def test_heading_replacement_escapes_markup():
    db = _rich_db()
    _edit(db, [{"action": "replace", "index": 0, "text": "A <b>bold</b> title"}])
    assert "<h2>A &lt;b&gt;bold&lt;/b&gt; title</h2>" in _content_of(db)


async def test_multi_paragraph_text_is_refused_for_a_heading():
    db = _rich_db()
    with pytest.raises(ValueError, match="single line"):
        _edit(db, [{"action": "replace", "index": 0, "text": "One.\n\nTwo."}])
    assert _content_of(db) == RICH


@pytest.mark.parametrize("index", [2, 4])  # <ul>, <img>
async def test_replacing_a_structural_block_is_refused(index):
    """Refused rather than flattened: _to_paragraph_html would turn the list
    into one soft-wrapped <p> and drop every item boundary."""
    db = _rich_db()
    with pytest.raises(ValueError, match="cannot rewrite"):
        _edit(db, [{"action": "replace", "index": index, "text": "rope and lamp"}])
    assert _content_of(db) == RICH


async def test_a_structural_block_can_still_be_deleted():
    """Deletion stays open for every tag — it is what the caller asked for,
    where a rewrite would be a downgrade they did not ask for."""
    db = _rich_db()
    _edit(db, [{"action": "replace", "index": 2, "text": ""}])
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(db))]
    assert tags == ["h2", "p", "blockquote", "img"]


async def test_a_refused_op_abandons_the_whole_call():
    """One bad op must not leave the other ops half-applied."""
    db = _rich_db()
    ops = [
        {"action": "replace", "index": 1, "text": "Rewritten."},
        {"action": "replace", "index": 2, "text": "flattened list"},
    ]
    with pytest.raises(ValueError, match="cannot rewrite"):
        _edit(db, ops)
    assert _content_of(db) == RICH


@pytest.mark.parametrize("bad_text", [0, False, [], None, 12])
async def test_a_non_string_text_is_refused_not_treated_as_a_deletion(bad_text):
    """`op.get("text") or ""` used to coerce every falsy non-string to "",
    which is the deletion sentinel — a malformed op silently destroyed a
    paragraph. Pydantic catches these at the tool layer; this is the boundary
    behind it holding on its own."""
    db = _rich_db()
    with pytest.raises(ValueError, match="text must be a string"):
        _edit(db, [{"action": "replace", "index": 1, "text": bad_text}])
    assert _content_of(db) == RICH


async def test_an_absent_text_key_still_means_deletion():
    """ "" is the documented schema default, so an omitted key keeps deleting."""
    db = _rich_db()
    _edit(db, [{"action": "replace", "index": 1}])
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(db))]
    assert tags == ["h2", "ul", "blockquote", "img"]


async def test_indices_refer_to_the_original_listing_regardless_of_op_order():
    """Ops are applied back-to-front, so a caller never compensates for shift.

    Submitting the same edits in either argument order must also produce
    byte-identical content — the normalisation sorts before applying.
    """
    ops = [
        {"action": "replace", "index": 0, "text": "New heading."},
        {"action": "insert_after", "index": 2, "text": "After the list."},
    ]
    first = _rich_db()
    _edit(first, ops)
    second = _rich_db()
    _edit(second, list(reversed(ops)))
    assert _content_of(first) == _content_of(second)

    after = blocks_module.split_blocks(_content_of(first))
    # Block 0 is the <h2>, so the replacement is rebuilt at that level.
    assert after[0].html == "<h2>New heading.</h2>"
    # Original block 2 was the list; the insert landed straight after it.
    assert after[2].tag == "ul"
    assert after[3].html == "<p>After the list.</p>"


async def test_removal_and_insert_in_one_call_use_original_indices():
    db = _rich_db()
    _edit(
        db,
        [
            {"action": "replace", "index": 3, "text": ""},  # drop the blockquote
            {"action": "insert_after", "index": 0, "text": "Subtitle."},
        ],
    )
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(db))]
    assert tags == ["h2", "p", "p", "ul", "img"]


async def test_duplicate_index_is_rejected():
    db = _rich_db()
    with pytest.raises(ValueError, match="two operations target block 1"):
        _edit(
            db,
            [
                {"action": "replace", "index": 1, "text": "a"},
                {"action": "insert_after", "index": 1, "text": "b"},
            ],
        )
    assert _content_of(db) == RICH


@pytest.mark.parametrize(
    "op",
    [
        {"action": "replace", "index": 5, "text": "x"},
        {"action": "replace", "index": -1, "text": "x"},
        {"action": "insert_after", "index": 5, "text": "x"},
        {"action": "insert_after", "index": -2, "text": "x"},
    ],
)
async def test_out_of_range_indices_are_rejected(op):
    db = _rich_db()
    with pytest.raises(ValueError, match="out of range"):
        _edit(db, [op])
    assert _content_of(db) == RICH


async def test_empty_chapter_guides_the_caller_to_insert_after_minus_one():
    db = _seeded_db()
    db.seed("stories/story-a/chapters/blank", {"title": "Blank", "content": ""})
    with pytest.raises(ValueError, match="no blocks yet"):
        _edit(db, [{"action": "replace", "index": 0, "text": "x"}], chapter="blank")
    _edit(
        db, [{"action": "insert_after", "index": -1, "text": "First."}], chapter="blank"
    )
    assert _content_of(db, "stories/story-a/chapters/blank") == "<p>First.</p>"


async def test_op_list_validation():
    db = _rich_db()
    for bad, match in [
        ([], "at least one"),
        ([{"action": "delete", "index": 0}], "action must be one of"),
        (
            [{"action": "replace", "index": "1", "text": "x"}],
            "index must be an integer",
        ),
        (
            [{"action": "replace", "index": True, "text": "x"}],
            "index must be an integer",
        ),
        ([{"action": "insert_after", "index": 0, "text": "  "}], "non-empty text"),
        (["not-an-object"], "each op must be an object"),
        ("not-a-list", "ops must be a list"),
    ]:
        with pytest.raises(ValueError, match=match):
            _edit(db, bad)
    assert _content_of(db) == RICH


async def test_too_many_ops_rejected():
    db = _rich_db()
    ops = [
        {"action": "insert_after", "index": i, "text": "x"}
        for i in range(writes.MAX_OPS_PER_CALL + 1)
    ]
    with pytest.raises(ValueError, match="at most 20 operations"):
        _edit(db, ops)


async def test_edit_escapes_markup():
    db = _rich_db()
    _edit(db, [{"action": "replace", "index": 1, "text": "<script>x</script>"}])
    assert "<script>" not in _content_of(db)
    assert "&lt;script&gt;" in _content_of(db)


async def test_edit_enforces_caps_on_the_rebuilt_string():
    db = _rich_db()
    with pytest.raises(writes.LimitExceededError) as exc:
        _edit(
            db,
            [
                {
                    "action": "replace",
                    "index": 1,
                    "text": " ".join(["w"] * (writes.MAX_CHAPTER_WORDS + 1)),
                }
            ],
        )
    assert exc.value.limit_name == "chapter_words"
    assert _content_of(db) == RICH


async def test_edit_is_owner_scoped():
    db = _rich_db()
    path = "stories/story-b/chapters/chb"
    with pytest.raises(data.StoryNotFoundError):
        writes.edit_chapter_blocks(
            db,
            UID_A,
            "story-b",
            "chb",
            [{"action": "replace", "index": 0, "text": "mine now"}],
            _revision_of(db, path),
        )


async def test_edit_on_a_missing_chapter_raises_entity_not_found():
    db = _rich_db()
    with pytest.raises(data.EntityNotFoundError):
        writes.edit_chapter_blocks(
            db,
            UID_A,
            "story-a",
            "no-such-chapter",
            [{"action": "replace", "index": 0, "text": "x"}],
            "1",
        )


# ---------------------------------------------------------------------------
# Editing — idempotency
# ---------------------------------------------------------------------------


async def test_identical_append_replays_instead_of_appending_twice():
    db = _rich_db()
    revision = _revision_of(db)
    first = writes.append_to_chapter(db, UID_A, "story-a", "rich", "Once.", revision)
    after_first = _content_of(db)
    second = writes.append_to_chapter(db, UID_A, "story-a", "rich", "Once.", revision)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert second["revision"] == first["revision"]
    assert _content_of(db) == after_first


async def test_identical_edit_replays_instead_of_editing_twice():
    db = _rich_db()
    revision = _revision_of(db)
    ops = [{"action": "insert_after", "index": 0, "text": "Inserted."}]
    first = writes.edit_chapter_blocks(db, UID_A, "story-a", "rich", ops, revision)
    after_first = _content_of(db)
    second = writes.edit_chapter_blocks(db, UID_A, "story-a", "rich", ops, revision)
    assert second["idempotent_replay"] is True
    assert _content_of(db) == after_first
    assert _content_of(db).count("Inserted.") == 1
    assert first["block_count"] == second["block_count"]


async def test_same_ops_against_the_new_revision_is_a_fresh_edit():
    """A replay is anchored to the base version, not to the text of the call."""
    db = _rich_db()
    ops = [{"action": "insert_after", "index": 0, "text": "Again."}]
    writes.edit_chapter_blocks(db, UID_A, "story-a", "rich", ops, _revision_of(db))
    writes.edit_chapter_blocks(db, UID_A, "story-a", "rich", ops, _revision_of(db))
    assert _content_of(db).count("Again.") == 2


async def test_op_order_does_not_change_the_idempotency_key():
    db = _rich_db()
    revision = _revision_of(db)
    ops = [
        {"action": "replace", "index": 0, "text": "A."},
        {"action": "insert_after", "index": 2, "text": "B."},
    ]
    writes.edit_chapter_blocks(db, UID_A, "story-a", "rich", ops, revision)
    replay = writes.edit_chapter_blocks(
        db, UID_A, "story-a", "rich", list(reversed(ops)), revision
    )
    assert replay["idempotent_replay"] is True


async def test_failed_edit_releases_the_reservation():
    """A corrected retry must not be blocked by the failed attempt's claim."""
    db = _rich_db()
    revision = _revision_of(db)
    with pytest.raises(writes.LimitExceededError):
        writes.edit_chapter_blocks(
            db,
            UID_A,
            "story-a",
            "rich",
            [
                {
                    "action": "replace",
                    "index": 1,
                    "text": " ".join(["w"] * (writes.MAX_CHAPTER_WORDS + 1)),
                }
            ],
            revision,
        )
    ok = writes.edit_chapter_blocks(
        db,
        UID_A,
        "story-a",
        "rich",
        [{"action": "replace", "index": 1, "text": "short"}],
        revision,
    )
    assert ok["idempotent_replay"] is False


async def test_another_users_completed_edit_cannot_be_replayed():
    """Ownership is checked before the reservation is consulted, so an attacker
    who somehow knew the exact arguments of someone else's successful edit gets
    "not found" rather than a replay of its result."""
    db = _rich_db()
    story = writes.create_story(db, UID_B, "B's book")
    chapter = writes.create_chapter(db, UID_B, story["story_id"], "Ch", "text")
    path = f"stories/{story['story_id']}/chapters/{chapter['chapter_id']}"
    revision = _revision_of(db, path)
    writes.append_to_chapter(
        db, UID_B, story["story_id"], chapter["chapter_id"], "hi", revision
    )
    with pytest.raises(data.StoryNotFoundError):
        writes.append_to_chapter(
            db, UID_A, story["story_id"], chapter["chapter_id"], "hi", revision
        )


# ---------------------------------------------------------------------------
# Editing — through the MCP tool layer
# ---------------------------------------------------------------------------


async def test_edit_tools_require_the_write_scope():
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A)):
        for name, args in [
            ("append_to_chapter", {"content": "x"}),
            (
                "edit_chapter_blocks",
                {"ops": [{"action": "replace", "index": 0, "text": "x"}]},
            ),
        ]:
            with pytest.raises(ToolError, match="read-only"):
                await mcp.call_tool(
                    name,
                    {
                        "story_id": "story-a",
                        "chapter_id": "rich",
                        "revision": _revision_of(db),
                        **args,
                    },
                )
    assert _content_of(db) == RICH


async def test_edit_tool_idor_returns_story_not_found():
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="Story not found"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-b",
                    "chapter_id": "chb",
                    "content": "mine now",
                    "revision": _revision_of(db, "stories/story-b/chapters/chb"),
                },
            )
    assert _content_of(db, "stories/story-b/chapters/chb") == "secret text"


async def test_edit_tool_missing_chapter_says_chapter_not_found():
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="Chapter not found"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-a",
                    "chapter_id": "nope",
                    "content": "x",
                    "revision": "1",
                },
            )


async def test_stale_revision_through_the_tool_layer_tells_the_model_to_reread():
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="changed since you read it"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-a",
                    "chapter_id": "rich",
                    "content": "x",
                    "revision": "stale",
                },
            )
    assert _content_of(db) == RICH


async def test_edit_tools_share_the_write_rate_limiter():
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True, write_rpm=1)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        await _call(
            mcp,
            "append_to_chapter",
            {
                "story_id": "story-a",
                "chapter_id": "rich",
                "content": "one",
                "revision": _revision_of(db),
            },
        )
        with pytest.raises(ToolError, match="Write rate limit"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-a",
                    "chapter_id": "rich",
                    "content": "two",
                    "revision": _revision_of(db),
                },
            )


@pytest.mark.parametrize(
    "tool,arguments,expected_extra",
    [
        (
            "create_story",
            {"title": "Audited", "description": "blurb"},
            {"title_chars": 7, "description_chars": 5, "tag_count": 0},
        ),
        (
            "create_chapter",
            {"story_id": "story-a", "title": "Audited", "content": "One two."},
            {"order": 4, "chapter_count": 3, "attempts": 1},
        ),
    ],
)
async def test_every_write_is_audited_with_the_caller_and_target(
    tool, arguments, expected_extra
):
    """The audit line is the only record of who changed what, so its spine —
    uid, connector, story, replay flag — has to survive refactoring."""
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with capture_logs() as logs:
            await _call(mcp, tool, arguments)

    line = next(entry for entry in logs if entry["event"].startswith("mcp_write_"))
    assert line["uid"] == UID_A
    assert line["client_id"] == "c1"
    assert line["story_id"]
    assert line["idempotent_replay"] is False
    for field, value in expected_extra.items():
        assert line[field] == value


async def test_audit_lines_carry_no_user_prose():
    """Lengths and counts only. A result field holding the author's words must
    never reach the logs, which is why _audit forwards nothing implicitly."""
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    secret = "Zephyrine unmistakable prose"
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with capture_logs() as logs:
            await _call(
                mcp,
                "create_story",
                {"title": secret, "description": secret, "tags": [secret[:20]]},
            )
    assert not any(
        secret[:20] in str(value) for entry in logs for value in entry.values()
    )


async def test_chapter_edits_are_audited_with_the_chapter_id():
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with capture_logs() as logs:
            await _call(
                mcp,
                "append_to_chapter",
                {
                    "story_id": "story-a",
                    "chapter_id": "rich",
                    "content": "Postscript.",
                    "revision": _revision_of(db),
                },
            )
    line = next(e for e in logs if e["event"] == "mcp_write_chapter_appended")
    assert line["chapter_id"] == "rich"
    assert line["appended_chars"] == len("Postscript.")
    assert line["block_count"] == 6


async def test_full_edit_cycle_through_the_tools():
    """Read blocks, edit by index with the revision it gave, read back."""
    db = _rich_db()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        listing = await _call(
            mcp,
            "get_chapter_blocks",
            {"story_id": "story-a", "chapter_id": "rich"},
        )
        assert listing["notice"] == UNTRUSTED_NOTICE
        target = next(b for b in listing["blocks"] if b["preview"].startswith("She"))

        edited = await _call(
            mcp,
            "edit_chapter_blocks",
            {
                "story_id": "story-a",
                "chapter_id": "rich",
                "revision": listing["revision"],
                "ops": [
                    {
                        "action": "replace",
                        "index": target["index"],
                        "text": "She counted the steps twice.",
                    }
                ],
            },
        )
        # Chain a second edit on the revision the first returned: no re-read.
        await _call(
            mcp,
            "append_to_chapter",
            {
                "story_id": "story-a",
                "chapter_id": "rich",
                "content": "Postscript.",
                "revision": edited["revision"],
            },
        )
        final = await _call(
            mcp, "get_chapter", {"story_id": "story-a", "chapter_id": "rich"}
        )

    assert "She counted the steps twice." in final["content"]
    assert final["content"].endswith("<p>Postscript.</p>")
    # Untouched blocks kept their exact markup through two edits.
    assert '<ul class="list-disc"><li><p>rope</p></li><li><p>lamp</p></li></ul>' in (
        final["content"]
    )
    assert final["revision"] == _revision_of(db)


# ---------------------------------------------------------------------------
# Drift guards
# ---------------------------------------------------------------------------


def test_revision_token_agrees_across_the_two_real_firestore_types():
    """The fake cannot catch this, and it breaks every chained edit if wrong.

    A read gives DocumentSnapshot.update_time (DatetimeWithNanoseconds); a
    write gives WriteResult.update_time (protobuf Timestamp). str() spells the
    same instant completely differently for the two, so a token taken straight
    from str() would make an edit's returned revision never match the next
    read's — refusing every chained edit as stale with nothing else going on.
    """
    from google.api_core.datetime_helpers import DatetimeWithNanoseconds
    from google.protobuf import timestamp_pb2

    from_write = timestamp_pb2.Timestamp(seconds=1_785_000_000, nanos=745_934_000)
    # The library's own conversion, so both objects are the same instant by
    # construction rather than by my arithmetic.
    from_read = DatetimeWithNanoseconds.from_timestamp_pb(from_write)

    assert str(from_write) != str(from_read)  # the trap this guards
    assert data.revision_token(from_write) == data.revision_token(from_read)


def test_revision_token_distinguishes_adjacent_versions():
    from google.protobuf import timestamp_pb2

    a = timestamp_pb2.Timestamp(seconds=1_785_000_000, nanos=1)
    b = timestamp_pb2.Timestamp(seconds=1_785_000_000, nanos=2)
    assert data.revision_token(a) != data.revision_token(b)


def test_block_op_actions_match_the_writes_layer():
    """The tool schema tells the model what to send; writes.py decides what is
    accepted. If they drift, valid-looking ops start being refused."""
    from typing import get_args

    field = tools_module.BlockOp.model_fields["action"]
    assert set(get_args(field.annotation)) == set(writes.OP_ACTIONS)


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
