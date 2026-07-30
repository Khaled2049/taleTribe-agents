"""Unit tests for mcp_server.data (owner-enforced reads) and mcp_server.tools."""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("USE_MOCK", "true")

from mcp.server.auth.provider import AccessToken  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from mcp_server import data  # noqa: E402
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


def _tool_server(db, max_rpm: int = 1000) -> FastMCP:
    mcp = FastMCP("test")
    register_tools(mcp, db=db, rate_limiter=PerUserRateLimiter(max_rpm))
    return mcp


def _token(uid: str = UID_A) -> AccessToken:
    return AccessToken(
        token="mcp_at_test", client_id="c1", scopes=["stories:read"], subject=uid
    )


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
