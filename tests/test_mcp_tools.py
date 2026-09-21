"""Unit tests for mcp_server.data (owner-enforced reads) and mcp_server.tools."""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("USE_MOCK", "true")

from mcp.server.auth.provider import AccessToken  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402
from structlog.testing import capture_logs  # noqa: E402

from capability_catalog import MCP_READ_TOOL_NAMES, MCP_WRITE_TOOL_NAMES  # noqa: E402
from mcp_server import app as mcp_app  # noqa: E402
from mcp_server import blocks as blocks_module  # noqa: E402
from mcp_server import data  # noqa: E402
from mcp_server import story_data  # noqa: E402
from mcp_server import writes  # noqa: E402
from mcp_server import tools as tools_module  # noqa: E402
from mcp_server.tools import UNTRUSTED_NOTICE, register_tools  # noqa: E402
from rate_limit import PerUserRateLimiter  # noqa: E402
from tests import mcp_fakes  # noqa: E402
from tests.mcp_fakes import FakeFirestoreClient, FakeStoryData  # noqa: E402

UID_A = "user-a"
UID_B = "user-b"


@pytest.fixture(autouse=True)
def _no_ambient_story_data():
    """Leave no client installed between tests.

    data.py reads through a process-wide client, so a fake left behind by one
    test would silently serve the next. Tests that need one call
    _use_story_data() explicitly.
    """
    story_data.configure(None)
    yield
    story_data.configure(None)


def _use_story_data(fake: FakeStoryData) -> FakeStoryData:
    story_data.configure(fake)
    return fake


def _seeded_story_data() -> FakeStoryData:
    """The read fixture, in story-data's shapes.

    Mirrors the Firestore fixture it replaces: two stories owned by user-a (one
    with chapters and a character), and one owned by user-b that must stay
    invisible.
    """
    fake = FakeStoryData()
    fake.seed_story(
        "story-a",
        UID_A,
        title="Story A",
        description="d" * 400,
        authorName="Author A",
        published=False,
        updatedAt="2026-07-01T00:00:00Z",
    )
    fake.seed_chapter(
        "story-a",
        "ch1",
        title="Chapter One",
        position=1.0,
        wordCount=5,
        content="0123456789" * 2500,  # 25_000 chars
    )
    fake.seed_chapter(
        "story-a",
        "ch2",
        title="Chapter Two",
        position=2.0,
        wordCount=3,
        content="short",
    )
    fake.seed_entity(
        "story-a",
        "characters",
        "char1",
        name="Mira",
        personality="p" * 400,
        soul="steadfast",
        relationships=[{"name": "Bran", "relation": "brother"}],
    )
    fake.seed_story("story-c", UID_A, title="Story C", updatedAt="2026-07-20T00:00:00Z")
    fake.seed_chapter(
        "story-c", "cc1", title="Only Chapter", position=1.0, content="hello"
    )
    fake.seed_story("story-b", UID_B, title="Story B")
    fake.seed_chapter(
        "story-b", "chb", title="Secret", position=1.0, content="secret text"
    )
    return _use_story_data(fake)


# ---------------------------------------------------------------------------
# data.py — ownership
# ---------------------------------------------------------------------------


async def test_list_stories_scoped_to_owner_and_sorted():
    _seeded_story_data()
    stories = await data.list_stories_for_user(UID_A, limit=20)
    assert [s["story_id"] for s in stories] == ["story-c", "story-a"]  # newest first
    assert all("Story B" != s["title"] for s in stories)
    # Long description is truncated with an ellipsis.
    story_a = next(s for s in stories if s["story_id"] == "story-a")
    assert len(story_a["description"]) <= data.SHORT_TEXT_LIMIT
    assert story_a["description"].endswith("…")


async def test_list_stories_limit_clamped():
    _seeded_story_data()
    assert len(await data.list_stories_for_user(UID_A, limit=1)) == 1
    assert len(await data.list_stories_for_user(UID_A, limit=99999)) == 2


async def test_list_stories_ranks_across_the_whole_collection():
    """Ordering must be applied before the limit, not over an arbitrary slice.

    story-data returns the caller's stories already ordered by updated_at DESC,
    so the cap here can only ever trim the tail. With more stories than the cap,
    a sort-after-trim would miss the genuinely newest one.
    """
    fake = _use_story_data(FakeStoryData())
    for i in range(data.MAX_STORY_LIST_LIMIT + 20):
        fake.seed_story(
            f"story-{i:04d}",
            UID_A,
            title=f"Story {i}",
            updatedAt=(
                f"2026-01-01T00:00:{i:02d}Z"
                if i < 60
                else f"2026-01-02T{i - 60:02d}:00:00Z"
            ),
        )
    stories = await data.list_stories_for_user(UID_A, limit=3)
    assert [s["title"] for s in stories] == ["Story 119", "Story 118", "Story 117"]


async def test_idor_story_of_other_user_is_not_found():
    _seeded_story_data()
    with pytest.raises(data.StoryNotFoundError):
        await data.get_owned_story("story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        await data.get_story_overview("story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        await data.list_chapters("story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        await data.get_chapter("story-b", "chb", UID_A, 0, 1000)
    with pytest.raises(data.StoryNotFoundError):
        await data.list_entities("story-b", UID_A, "characters")


async def test_missing_story_indistinguishable_from_not_owned():
    _seeded_story_data()
    with pytest.raises(data.StoryNotFoundError):
        await data.get_owned_story("no-such-story", UID_A)


async def test_published_story_of_another_user_is_still_not_found():
    """The check that stops the port from widening MCP's scope.

    story-data serves a *published* story to any caller so the public reader can
    use the same endpoints. MCP is owner-only on every tool, so get_owned_story
    re-checks ownerId. Without that check this test returns a story, and MCP
    quietly becomes "every published story on the platform".
    """
    fake = _seeded_story_data()
    fake.stories["story-b"]["published"] = True
    # Confirm the fake really does expose it, so this is testing our check and
    # not an artifact of the fake refusing non-owners anyway.
    assert await fake.get_story(UID_A, "story-b")

    with pytest.raises(data.StoryNotFoundError):
        await data.get_owned_story("story-b", UID_A)
    with pytest.raises(data.StoryNotFoundError):
        await data.get_chapter("story-b", "chb", UID_A, 0, 1000)
    with pytest.raises(data.StoryNotFoundError):
        await data.list_chapters("story-b", UID_A)


async def test_unreachable_story_data_is_an_error_not_an_empty_story():
    """A transport failure must not read as "you have no stories"."""

    class Broken(FakeStoryData):
        async def list_stories(self, uid):
            raise story_data.StoryDataError("connection refused")

    _use_story_data(Broken())
    with pytest.raises(story_data.StoryDataError):
        await data.list_stories_for_user(UID_A, limit=20)


# ---------------------------------------------------------------------------
# data.py — reads
# ---------------------------------------------------------------------------


async def test_story_overview_lists_chapters_in_reading_order():
    _seeded_story_data()
    overview = await data.get_story_overview("story-a", UID_A)
    assert overview["title"] == "Story A"
    assert [c["title"] for c in overview["chapters"]] == [
        "Chapter One",
        "Chapter Two",
    ]
    # chapter_count is derived from the index; story-data has no count column.
    assert overview["chapter_count"] == 2
    assert overview["author"] == "Author A"
    assert overview["chapters_truncated"] is False


async def test_story_overview_numbers_chapters_by_position():
    """chapter_number is the ordinal, not the position key.

    story-data has no chapterNumber column, and `position` cannot stand in for
    one: it keeps gaps after a delete and takes fractional values on an insert.
    """
    fake = _use_story_data(FakeStoryData())
    fake.seed_story("s", UID_A, title="Gappy")
    fake.seed_chapter("s", "a", title="First", position=1.0)
    fake.seed_chapter("s", "b", title="Second", position=7.5)
    fake.seed_chapter("s", "c", title="Third", position=99.0)

    overview = await data.get_story_overview("s", UID_A)
    assert [c["chapter_number"] for c in overview["chapters"]] == [1, 2, 3]
    assert [c["order"] for c in overview["chapters"]] == [1.0, 7.5, 99.0]


async def test_story_overview_on_a_story_with_one_chapter():
    _seeded_story_data()
    overview = await data.get_story_overview("story-c", UID_A)
    assert [c["title"] for c in overview["chapters"]] == ["Only Chapter"]
    assert overview["chapters_truncated"] is False


async def test_list_chapters_sorted_with_ids():
    _seeded_story_data()
    page = await data.list_chapters("story-a", UID_A)
    assert [c["chapter_id"] for c in page.items] == ["ch1", "ch2"]
    assert page.items[0]["word_count"] == 5
    assert page.items[0]["chapter_number"] == 1
    assert "content" not in page.items[0]
    assert page.truncated is False


async def test_list_chapters_does_not_fetch_chapter_bodies():
    """Regression: listing a book must not transfer every word of it.

    story-data's default chapter listing includes full content, so the index
    request has to ask for content=false. Without it a table of contents costs
    the whole manuscript.
    """
    fake = _seeded_story_data()
    await data.list_chapters("story-a", UID_A)
    index_calls = [
        params
        for path, params in fake.requests
        if path == "/v1/stories/story-a/chapters"
    ]
    assert index_calls, "expected the chapter index to be requested"
    assert all(params.get("content") == "false" for params in index_calls)


async def test_get_chapter_pagination_arithmetic():
    _seeded_story_data()
    first = await data.get_chapter("story-a", "ch1", UID_A, 0, 10_000)
    assert first["total_chars"] == 25_000
    assert len(first["content"]) == 10_000
    assert first["next_offset"] == 10_000

    last = await data.get_chapter("story-a", "ch1", UID_A, 20_000, 10_000)
    assert len(last["content"]) == 5_000
    assert last["next_offset"] is None

    # Windows tile the content exactly.
    middle = await data.get_chapter("story-a", "ch1", UID_A, 10_000, 10_000)
    full = first["content"] + middle["content"] + last["content"]
    assert full == "0123456789" * 2500


async def test_get_chapter_clamps_inputs():
    _seeded_story_data()
    clamped = await data.get_chapter("story-a", "ch1", UID_A, -5, 999_999)
    assert clamped["offset"] == 0
    assert len(clamped["content"]) == 25_000  # max_chars clamped to 50k > total
    with pytest.raises(data.EntityNotFoundError):
        await data.get_chapter("story-a", "nope", UID_A, 0, 10)


async def test_get_chapter_reports_its_reading_order_number():
    _seeded_story_data()
    second = await data.get_chapter("story-a", "ch2", UID_A, 0, 10)
    assert second["chapter_number"] == 2
    assert second["order"] == 2.0


def _seed_long_book(fake: FakeStoryData, story_id: str = "story-long") -> int:
    """A story with more chapters than the cap.

    Ids run opposite to position, so a truncate-before-sort read would return the
    END of the book and drop the beginning entirely.
    """
    total = data.COLLECTION_FETCH_LIMIT + 50
    fake.seed_story(story_id, UID_A, title="Long Book")
    for i in range(total):
        position = total - i
        fake.seed_chapter(
            story_id,
            f"ch-{i:04d}",
            title=f"Chapter {position}",
            position=float(position),
            content="x",
        )
    return total


async def test_list_chapters_ranks_across_the_whole_collection():
    """The cap must slice reading order, not insertion order."""
    fake = _seeded_story_data()
    _seed_long_book(fake)

    page = await data.list_chapters("story-long", UID_A)

    assert len(page.items) == data.COLLECTION_FETCH_LIMIT
    assert page.truncated is True
    # Chapter 1 is present and first, rather than the book opening at chapter 51.
    assert [c["order"] for c in page.items] == [
        float(n) for n in range(1, data.COLLECTION_FETCH_LIMIT + 1)
    ]


async def test_story_overview_reports_truncation():
    fake = _seeded_story_data()
    _seed_long_book(fake)
    overview = await data.get_story_overview("story-long", UID_A)
    assert overview["chapters_truncated"] is True
    assert len(overview["chapters"]) == data.COLLECTION_FETCH_LIMIT
    assert overview["chapters"][0]["order"] == 1.0


async def test_list_entities_reports_truncation():
    _seeded_story_data()
    fake = story_data.client()
    for i in range(data.COLLECTION_FETCH_LIMIT + 5):
        fake.seed_entity("story-a", "plots", f"pl-{i:04d}", name=f"Plot {i}")

    page = await data.list_entities("story-a", UID_A, "plots")

    assert len(page.items) == data.COLLECTION_FETCH_LIMIT
    assert page.truncated is True


async def test_list_entities_descriptor_and_sorting():
    _seeded_story_data()
    page = await data.list_entities("story-a", UID_A, "characters")
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


async def test_list_entities_survives_non_string_names():
    """A name that is not a string must not break the sort."""
    fake = _seeded_story_data()
    fake.seed_entity("story-a", "places", "p1", name=7, description="a tower")
    fake.seed_entity(
        "story-a", "places", "p2", name=None, title=True, description="a moor"
    )
    fake.seed_entity("story-a", "places", "p3", name="   ", description="a fen")
    fake.seed_entity("story-a", "places", "p4", name="Harbor")

    entities = (await data.list_entities("story-a", UID_A, "places")).items

    by_id = {e["entity_id"]: e["name"] for e in entities}
    assert by_id["p1"] == "7"  # coerced, not crashed
    assert by_id["p2"] == "Unnamed"  # bool is never a name
    assert by_id["p3"] == "Unnamed"  # whitespace-only falls through
    assert by_id["p4"] == "Harbor"
    assert [e["name"] for e in entities] == ["7", "Harbor", "Unnamed", "Unnamed"]


async def test_list_entities_falls_back_to_title():
    fake = _seeded_story_data()
    fake.seed_entity("story-a", "plots", "pl1", name=None, title="The Reckoning")
    page = await data.list_entities("story-a", UID_A, "plots")
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


async def test_list_entities_rejects_unknown_type():
    _seeded_story_data()
    with pytest.raises(ValueError):
        await data.list_entities("story-a", UID_A, "chapters")


async def test_get_entity_returns_story_content():
    _seeded_story_data()
    entity = await data.get_entity("story-a", UID_A, "characters", "char1")
    assert entity["name"] == "Mira"
    assert entity["soul"] == "steadfast"
    assert entity["relationships"] == [{"name": "Bran", "relation": "brother"}]
    with pytest.raises(data.EntityNotFoundError):
        await data.get_entity("story-a", UID_A, "characters", "nope")


async def test_get_entity_projects_instead_of_passing_through():
    """Internal bookkeeping must not reach the client, named or not.

    The projection is what makes that structural: a column added to story-data
    cannot start appearing in tool output without being listed here.
    """
    fake = _seeded_story_data()
    fake.seed_entity(
        "story-a",
        "characters",
        "char2",
        name="Bran",
        personality="wry",
        artUrl="https://example.test/bran.png",
        # Internal bookkeeping the client has no use for:
        storyId="story-a",
        signature="abc123",
        embeddingUpdatedAt="2026-07-01T00:00:00Z",
    )
    entity = await data.get_entity("story-a", UID_A, "characters", "char2")

    assert entity["name"] == "Bran"
    assert entity["artUrl"] == "https://example.test/bran.png"
    for internal in ("storyId", "signature", "embeddingUpdatedAt", "revision"):
        assert internal not in entity, f"{internal} leaked into tool output"


async def test_get_entity_omits_empty_values():
    fake = _seeded_story_data()
    fake.seed_entity(
        "story-a",
        "places",
        "p2",
        name="Bare",
        description="",
        atmosphere=None,
        history="damp",
    )
    entity = await data.get_entity("story-a", UID_A, "places", "p2")
    assert entity["history"] == "damp"
    assert "description" not in entity
    assert "atmosphere" not in entity


async def test_get_entity_is_owner_only_even_when_published():
    """Worldbuilding is owner-only in story-data, and must stay so through MCP."""
    fake = _seeded_story_data()
    fake.stories["story-b"]["published"] = True
    fake.seed_entity("story-b", "characters", "secret", name="Hidden")
    with pytest.raises(data.StoryNotFoundError):
        await data.get_entity("story-b", UID_A, "characters", "secret")
    with pytest.raises(data.StoryNotFoundError):
        await data.list_entities("story-b", UID_A, "characters")


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


def _read_tool_server(max_rpm: int = 1000, **kwargs) -> FastMCP:
    """A tool server whose reads come from story-data.

    The Firestore client is left empty deliberately: if a read tool still
    reached for it, these tests would fail rather than quietly pass on seeded
    Firestore data that production no longer has.
    """
    _seeded_story_data()
    return _tool_server(FakeFirestoreClient(), max_rpm, **kwargs)


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
    mcp = _read_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        result = await _call(mcp, "list_my_stories", {})
    assert result["count"] == 2
    assert result["notice"] == UNTRUSTED_NOTICE
    assert {s["story_id"] for s in result["stories"]} == {"story-a", "story-c"}


async def test_tool_idor_returns_not_found():
    mcp = _read_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A)):
        with pytest.raises(ToolError, match="Story not found"):
            await mcp.call_tool("get_story_overview", {"story_id": "story-b"})


async def test_tool_unauthenticated_rejected():
    mcp = _tool_server(FakeFirestoreClient())
    with patch("mcp_server.tools.get_access_token", return_value=None):
        with pytest.raises(ToolError, match="Not authenticated"):
            await mcp.call_tool("list_my_stories", {})


async def test_tool_rate_limit():
    mcp = _read_tool_server(max_rpm=1)
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        await mcp.call_tool("list_my_stories", {})
        with pytest.raises(ToolError, match="Rate limit"):
            await mcp.call_tool("list_my_stories", {})


async def test_tool_get_chapter_roundtrip():
    mcp = _read_tool_server()
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
    mcp = _read_tool_server()
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
# Every denial case asserts that story-data saw no write at all. The bug worth
# catching is not "denied", it is "denied but wrote anyway".
# ---------------------------------------------------------------------------


def _doc_count(db) -> int:
    """Firestore documents — now only the mcpWrites reservations."""
    return len(db.docs)


def _write_env() -> tuple[FakeFirestoreClient, FakeStoryData]:
    """The two backends a write touches, seeded as the read fixture is.

    Content goes to story-data; Firestore holds only the idempotency
    reservation, so the fakes stay separate and an assertion says which one it
    means.
    """
    return FakeFirestoreClient(), _seeded_story_data()


def _blank_env() -> tuple[FakeFirestoreClient, FakeStoryData]:
    return FakeFirestoreClient(), _use_story_data(FakeStoryData())


async def test_create_chapter_idor_raises_story_not_found():
    db, fake = _write_env()
    with pytest.raises(data.StoryNotFoundError):
        await writes.create_chapter(db, UID_A, "story-b", "Sneaky", "text")
    assert fake.writes == []
    assert _doc_count(db) == 0


async def test_create_chapter_missing_story_indistinguishable_from_not_owned():
    db, _ = _write_env()
    with pytest.raises(data.StoryNotFoundError):
        await writes.create_chapter(db, UID_A, "no-such-story", "T", "x")
    with pytest.raises(data.StoryNotFoundError):
        await writes.create_chapter(db, UID_A, "story-b", "T", "x")


async def test_no_write_argument_can_target_another_users_story():
    """Guards against a later `user_id`-style parameter reopening the IDOR."""
    db, fake = _write_env()
    story = await writes.create_story(db, UID_B, "B's book")
    with pytest.raises(data.StoryNotFoundError):
        await writes.create_chapter(db, UID_A, story["story_id"], "T", "x")


async def test_the_asserted_uid_is_the_one_story_data_is_told():
    """story-data derives ownership from X-User-ID, so the uid writes.py passes
    IS the authorization decision. A caller's own id must reach it unchanged."""
    db, fake = _blank_env()
    await writes.create_story(db, UID_B, "B's book")
    assert [s["ownerId"] for s in fake.stories.values()] == [UID_B]


# ---------------------------------------------------------------------------
# writes.py — derived fields and cross-layer round-trip
# ---------------------------------------------------------------------------


async def test_create_story_sends_the_fields_story_data_stores():
    """story-data decodes with DisallowUnknownFields, so an unrecognised key is
    a 400 for the whole request rather than a silently dropped field."""
    db, fake = _blank_env()
    fake.seed_profile(UID_A, username="mira")
    result = await writes.create_story(db, UID_A, "T", "D", "Fantasy", ["a", "b"])

    path, payload = fake.writes[-1]
    assert path == "POST /v1/stories"
    assert payload == {
        "title": "T",
        "description": "D",
        "authorName": "mira",
        "category": "Fantasy",
        "tags": ["a", "b"],
        "published": False,
    }
    assert result["story_id"] in fake.stories
    assert result["is_published"] is False


async def test_create_story_author_from_public_profile():
    db, fake = _blank_env()
    fake.seed_profile(UID_A, username="mira")
    await writes.create_story(db, UID_A, "T")
    assert fake.writes[-1][1]["authorName"] == "mira"


async def test_create_story_author_blank_when_profile_missing():
    """A story must still be creatable before the writer has a public profile —
    the byline is cosmetic, and story-data serves the username on read anyway."""
    db, fake = _blank_env()
    await writes.create_story(db, UID_A, "T")
    assert fake.writes[-1][1]["authorName"] == ""


async def test_create_chapter_reports_the_chapter_count_it_observed():
    db, fake = _write_env()
    result = await writes.create_chapter(db, UID_A, "story-a", "Three", "text")
    assert result["chapter_count"] == 3
    assert len(fake.chapters["story-a"]) == 3


async def test_created_chapter_carries_the_fields_a_listing_needs():
    db, fake = _write_env()
    created = await writes.create_chapter(
        db, UID_A, "story-a", "Three", "one two three"
    )
    stored = next(
        c for c in fake.chapters["story-a"] if c["id"] == created["chapter_id"]
    )
    assert stored["title"] == "Three"
    assert stored["position"] == created["order"]
    # Counted by story-data, reported back rather than recomputed here.
    assert created["word_count"] == stored["wordCount"] == 3


# ---------------------------------------------------------------------------
# writes.py — ordering and concurrency
#
# `position` carries a UNIQUE (story_id, position) constraint in story-data, so
# a lost race is a 409 on insert rather than a duplicate row. These tests drive
# that path through the fake's before_create_chapter hook.
# ---------------------------------------------------------------------------


async def test_position_derives_from_the_highest_position_not_the_count():
    """Positions keep gaps after a mid-book delete, so deriving the next one
    from the number of chapters would collide at 4."""
    db, fake = _blank_env()
    fake.seed_story("s", UID_A, title="S")
    for idx, position in enumerate((0, 1, 3, 4)):
        fake.seed_chapter("s", f"c{idx}", title=f"C{idx}", position=float(position))
    result = await writes.create_chapter(db, UID_A, "s", "Next", "x")
    assert result["position" if "position" in result else "order"] == 5.0


async def test_create_chapter_retries_when_the_position_is_taken():
    db, fake = _write_env()
    stolen = {"done": False}

    def concurrent_writer():
        # A second caller lands between our index read and our insert.
        if not stolen["done"]:
            stolen["done"] = True
            fake.seed_chapter("story-a", "rival", title="Rival", position=3.0)

    fake.before_create_chapter = concurrent_writer
    result = await writes.create_chapter(db, UID_A, "story-a", "Three", "x")
    assert result["attempts"] == 2
    assert result["order"] == 4.0


async def test_positions_are_unique_under_repeated_contention():
    db, fake = _write_env()
    positions = []
    for n in range(4):
        state = {"done": False}

        def concurrent_writer(state=state, n=n):
            if not state["done"]:
                state["done"] = True
                fake.seed_chapter(
                    "story-a",
                    f"rival{n}",
                    title="Rival",
                    position=_next_free_position(fake),
                )

        fake.before_create_chapter = concurrent_writer
        # Distinct titles: identical arguments would hit the idempotency
        # replay and return the same chapter rather than exercising ordering.
        positions.append(
            (await writes.create_chapter(db, UID_A, "story-a", f"C{n}", "x"))["order"]
        )
    all_positions = [c["position"] for c in fake.chapters["story-a"]]
    assert len(set(positions)) == len(positions)
    assert len(set(all_positions)) == len(all_positions)


def _next_free_position(fake: FakeStoryData, story_id: str = "story-a") -> float:
    return max(c["position"] for c in fake.chapters[story_id]) + 1.0


async def test_write_conflict_after_exhausting_attempts():
    db, fake = _write_env()

    def always_steal():
        # Take the slot every time, so no retry can ever find a free one.
        fake.seed_chapter(
            "story-a",
            f"rival{len(fake.chapters['story-a'])}",
            title="Rival",
            position=_next_free_position(fake),
        )

    fake.before_create_chapter = always_steal
    with pytest.raises(writes.WriteConflictError):
        await writes.create_chapter(db, UID_A, "story-a", "C", "x")


# ---------------------------------------------------------------------------
# writes.py — limits
#
# The per-user, per-story and per-chapter ceilings belong to story-data, which
# counts them transactionally; writes.py keeps only the stored-size bound that
# has no counterpart there. These tests pin that split rather than the numbers.
# ---------------------------------------------------------------------------


async def test_story_cap_is_reported_from_story_data():
    db, fake = _blank_env()
    for i in range(FakeStoryData.STORY_LIMIT):
        fake.seed_story(f"s{i}", UID_A)
    with pytest.raises(story_data.Rejected, match="limit of 100 stories"):
        await writes.create_story(db, UID_A, "One too many")


async def test_chapter_cap_is_reported_from_story_data():
    db, fake = _blank_env()
    fake.seed_story("s", UID_A)
    for i in range(FakeStoryData.CHAPTER_LIMIT):
        fake.seed_chapter("s", f"c{i}", position=float(i))
    with pytest.raises(story_data.Rejected, match="limit of 50 chapters"):
        await writes.create_chapter(db, UID_A, "s", "Too many", "x")


async def test_word_cap_is_reported_from_story_data():
    db, fake = _write_env()
    with pytest.raises(story_data.Rejected, match="5000 words"):
        await writes.create_chapter(
            db, UID_A, "story-a", "T", "word " * (FakeStoryData.WORD_LIMIT + 1)
        )


async def test_content_char_cap_measures_the_stored_string():
    """The one ceiling still applied here: story-data bounds words, not bytes,
    so nothing downstream would refuse megabytes of markup."""
    db, fake = _write_env()
    # One word, long enough that the stored <p>-wrapped form exceeds the cap.
    with pytest.raises(writes.LimitExceededError) as exc:
        await writes.create_chapter(
            db, UID_A, "story-a", "T", "x" * (writes.MAX_CHAPTER_CONTENT_CHARS + 1)
        )
    assert exc.value.limit_name == "chapter_content_chars"
    assert fake.writes == []
    assert _doc_count(db) == 0


async def test_content_char_cap_boundary_is_inclusive():
    db, fake = _write_env()
    # "<p>" + body + "</p>" == exactly the cap.
    body = "x" * (writes.MAX_CHAPTER_CONTENT_CHARS - len("<p></p>"))
    result = await writes.create_chapter(db, UID_A, "story-a", "T", body)
    stored = next(
        c for c in fake.chapters["story-a"] if c["id"] == result["chapter_id"]
    )
    assert len(stored["content"]) == writes.MAX_CHAPTER_CONTENT_CHARS


@pytest.mark.parametrize("bad_title", ["", "   ", "t" * 201])
async def test_blank_and_oversize_titles_rejected(bad_title):
    db, fake = _write_env()
    with pytest.raises(ValueError):
        await writes.create_story(db, UID_A, bad_title)
    with pytest.raises(ValueError):
        await writes.create_chapter(db, UID_A, "story-a", bad_title, "x")
    assert fake.writes == []


async def test_tag_count_and_length_capped():
    db, _ = _blank_env()
    with pytest.raises(ValueError):
        await writes.create_story(db, UID_A, "T", tags=[f"t{i}" for i in range(11)])
    with pytest.raises(ValueError):
        await writes.create_story(db, UID_A, "T", tags=["x" * 41])


async def test_description_cap_enforced():
    db, _ = _blank_env()
    with pytest.raises(ValueError):
        await writes.create_story(db, UID_A, "T", "d" * 2001)


# ---------------------------------------------------------------------------
# writes.py — content transform
# ---------------------------------------------------------------------------


async def test_content_is_escaped_and_paragraph_wrapped():
    db, fake = _write_env()
    result = await writes.create_chapter(
        db, UID_A, "story-a", "T", "a & b <script>x</script>\n\nsecond"
    )
    stored = next(
        c for c in fake.chapters["story-a"] if c["id"] == result["chapter_id"]
    )
    assert stored["content"] == (
        "<p>a &amp; b &lt;script&gt;x&lt;/script&gt;</p>\n<p>second</p>"
    )
    assert "<script" not in stored["content"]


async def test_word_count_comes_back_from_story_data():
    """Counted once, by the service that stores it — so the number cannot drift
    from what the next read reports."""
    db, fake = _write_env()
    plain = "one two three\n\nfour five"
    result = await writes.create_chapter(db, UID_A, "story-a", "T", plain)
    stored = next(
        c for c in fake.chapters["story-a"] if c["id"] == result["chapter_id"]
    )
    assert result["word_count"] == len(stored["content"].split())
    assert result["word_count"] == len(plain.split())


async def test_empty_content_stores_empty_string():
    db, fake = _write_env()
    result = await writes.create_chapter(db, UID_A, "story-a", "T", "")
    stored = next(
        c for c in fake.chapters["story-a"] if c["id"] == result["chapter_id"]
    )
    assert stored["content"] == ""
    assert stored["wordCount"] == 0


# ---------------------------------------------------------------------------
# writes.py — idempotency
#
# The reservation is the one part of the write path still in Firestore: it is
# the connector's own bookkeeping, not story content. These tests therefore
# still assert against the Firestore fake, and against story-data for the
# "wrote once" half.
# ---------------------------------------------------------------------------


async def test_identical_create_story_returns_same_id_and_writes_once():
    db, fake = _blank_env()
    first = await writes.create_story(db, UID_A, "Twice", "same")
    second = await writes.create_story(db, UID_A, "Twice", "same")
    assert second["story_id"] == first["story_id"]
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert len(fake.stories) == 1


async def test_identical_create_chapter_returns_same_id_and_writes_once():
    db, fake = _write_env()
    first = await writes.create_chapter(db, UID_A, "story-a", "Dup", "body")
    second = await writes.create_chapter(db, UID_A, "story-a", "Dup", "body")
    assert second["chapter_id"] == first["chapter_id"]
    assert len(fake.chapters["story-a"]) == 3  # created once


async def test_idempotency_is_scoped_to_the_caller():
    """Cross-user dedup would be both an information leak and a denial of
    service — B could pre-claim A's key and block the write."""
    db, _ = _blank_env()
    a = await writes.create_story(db, UID_A, "Same title", "same")
    b = await writes.create_story(db, UID_B, "Same title", "same")
    assert a["story_id"] != b["story_id"]


async def test_idempotency_window_expires():
    db, _ = _blank_env()
    first = await writes.create_story(db, UID_A, "Later", "again")
    later = datetime.now(timezone.utc) + timedelta(
        seconds=writes.IDEMPOTENCY_TTL_SECONDS + 1
    )
    with patch("mcp_server.writes._now", return_value=later):
        second = await writes.create_story(db, UID_A, "Later", "again")
    assert second["story_id"] != first["story_id"]


async def test_expired_reservation_is_claimed_by_exactly_one_racer():
    """Taking over an expired reservation has to keep the exactly-one-winner
    property that `create` gives the common path for free.

    The loser of the takeover is told the call is already in flight rather than
    being allowed to write a second story.
    """
    db, fake = _blank_env()
    await writes.create_story(db, UID_A, "Later", "again")
    stories_before = len(fake.stories)

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
        record, _version = db.docs[path]
        db.docs[path] = (record, next(mcp_fakes._versions))

    db.before_update = steal
    with patch("mcp_server.writes._now", return_value=later):
        with pytest.raises(writes.DuplicateInFlightError):
            await writes.create_story(db, UID_A, "Later", "again")
    db.before_update = None
    assert len(fake.stories) == stories_before


async def test_a_vanished_reservation_is_reclaimed():
    """TTL collection or a concurrent _release can remove the document between
    the create that lost and the read that follows it. The key is free then."""
    db, _ = _blank_env()
    first = await writes.create_story(db, UID_A, "Gone", "poof")
    for path in [p for p in db.docs if p.startswith(f"{writes.WRITES_COLLECTION}/")]:
        del db.docs[path]
    second = await writes.create_story(db, UID_A, "Gone", "poof")
    assert second["story_id"] != first["story_id"]
    assert second["idempotent_replay"] is False


async def test_failed_write_releases_the_reservation():
    """A rejection from story-data must free the key, or the caller's corrected
    retry would be refused as a duplicate for two minutes."""
    db, fake = _blank_env()
    for i in range(FakeStoryData.STORY_LIMIT):
        fake.seed_story(f"s{i}", UID_A)
    with pytest.raises(story_data.Rejected):
        await writes.create_story(db, UID_A, "Blocked")
    assert not [p for p in db.docs if p.startswith(f"{writes.WRITES_COLLECTION}/")]
    # And the same call succeeds once the blocker is gone.
    fake.stories.pop("s0")
    assert (await writes.create_story(db, UID_A, "Blocked"))["story_id"]


async def test_concurrent_identical_call_is_reported_not_silently_dropped():
    db, _ = _blank_env()
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
        await writes.create_story(db, UID_A, "Inflight")


# ---------------------------------------------------------------------------
# tools.py — write wiring and scope enforcement
# ---------------------------------------------------------------------------


def _write_tool_server(**kwargs) -> tuple[FastMCP, FakeFirestoreClient, FakeStoryData]:
    """A tool server with writes on, wired to both fakes."""
    db, fake = _write_env()
    return _tool_server(db, enable_writes=True, **kwargs), db, fake


async def test_read_only_token_cannot_call_write_tools():
    mcp, db, fake = _write_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        with pytest.raises(ToolError, match="read-only"):
            await mcp.call_tool("create_story", {"title": "Nope"})
        with pytest.raises(ToolError, match="read-only"):
            await mcp.call_tool(
                "create_chapter", {"story_id": "story-a", "title": "Nope"}
            )
    assert fake.writes == []
    assert _doc_count(db) == 0


async def test_write_scoped_token_can_call_write_tools():
    mcp, _db, _fake = _write_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        story = await _call(mcp, "create_story", {"title": "Made by MCP"})
        chapter = await _call(
            mcp,
            "create_chapter",
            {"story_id": story["story_id"], "title": "One", "content": "hello"},
        )
    assert story["notice"] == UNTRUSTED_NOTICE
    assert chapter["notice"] == UNTRUSTED_NOTICE
    # story-data seeds "Chapter 1" at position 0, so the first added chapter
    # lands after it.
    assert chapter["order"] == 1.0


async def test_write_scoped_token_can_still_call_read_tools():
    mcp = _read_tool_server(enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        result = await _call(mcp, "list_my_stories", {})
    assert result["count"] == 2


async def test_write_tools_absent_when_disabled():
    mcp = _read_tool_server(enable_writes=False)
    names = {t.name for t in await mcp.list_tools()}
    assert names == MCP_READ_TOOL_NAMES
    assert names.isdisjoint(MCP_WRITE_TOOL_NAMES)
    # get_chapter_blocks is a READ tool: it stays available with writes off.
    assert "get_chapter_blocks" in names
    assert len(names) == 7


async def test_write_tools_present_when_enabled():
    mcp, _db, _fake = _write_tool_server()
    names = {t.name for t in await mcp.list_tools()}
    assert names == MCP_READ_TOOL_NAMES | MCP_WRITE_TOOL_NAMES


async def test_write_tool_idor_returns_story_not_found():
    mcp, db, fake = _write_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="Story not found"):
            await mcp.call_tool(
                "create_chapter", {"story_id": "story-b", "title": "Sneaky"}
            )
    assert fake.writes == []
    assert _doc_count(db) == 0


async def test_write_tool_unauthenticated_rejected():
    mcp, _db, _fake = _write_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=None):
        with pytest.raises(ToolError, match="Not authenticated"):
            await mcp.call_tool("create_story", {"title": "x"})


async def test_write_rate_limit_is_separate_and_tighter():
    mcp, _db, _fake = _write_tool_server(write_rpm=1)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        await mcp.call_tool("create_story", {"title": "First"})
        with pytest.raises(ToolError, match="Write rate limit"):
            await mcp.call_tool("create_story", {"title": "Second"})
        # The shared bucket is untouched: reads still work.
        result = await _call(mcp, "list_my_stories", {})
    # The seeded two, plus the one just written.
    assert result["count"] == 3


async def test_a_story_data_outage_is_reported_not_swallowed():
    """A write that never reached the service must not look like a refusal the
    model can fix by rewording."""
    mcp, _db, fake = _write_tool_server()

    async def boom(*args, **kwargs):
        raise story_data.StoryDataError("connection reset")

    fake.create_story = boom
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="story service is unavailable"):
            await mcp.call_tool("create_story", {"title": "x"})


async def test_story_data_limit_message_reaches_the_model():
    """The ceilings live in story-data now, so its wording is what the caller
    must see — a generic "rejected" would leave the model guessing."""
    mcp, _db, fake = _write_tool_server()
    for i in range(FakeStoryData.STORY_LIMIT):
        fake.seed_story(f"cap{i}", UID_A)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="limit of 100 stories"):
            await mcp.call_tool("create_story", {"title": "One too many"})


async def test_write_tools_round_trip_through_the_read_tools():
    """The point of the port: one backend, so a tool can read back what it just
    wrote. Under the Firestore write path this could not pass — and the config
    it needed was rejected outright."""
    mcp, _db, _fake = _write_tool_server()
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        story = await _call(mcp, "create_story", {"title": "Round Trip"})
        chapter = await _call(
            mcp,
            "create_chapter",
            {"story_id": story["story_id"], "title": "Ch1", "content": "a b c"},
        )
        overview = await _call(
            mcp, "get_story_overview", {"story_id": story["story_id"]}
        )
        read_back = await _call(
            mcp,
            "get_chapter",
            {"story_id": story["story_id"], "chapter_id": chapter["chapter_id"]},
        )

    assert overview["title"] == "Round Trip"
    assert "Ch1" in [c["title"] for c in overview["chapters"]]
    assert read_back["content"] == "<p>a b c</p>"
    assert read_back["word_count"] == 3


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


def _rich_story_data() -> FakeStoryData:
    """The rich-content chapter, served from story-data."""
    fake = _seeded_story_data()
    fake.seed_chapter(
        "story-a",
        "rich",
        title="Rich",
        position=3.0,
        wordCount=9,
        content=RICH,
        revision=4,
    )
    return fake


def _rich_env() -> tuple[FakeFirestoreClient, FakeStoryData]:
    return FakeFirestoreClient(), _rich_story_data()


def _chapter_row(fake: FakeStoryData, chapter: str, story: str = "story-a") -> dict:
    return next(c for c in fake.chapters[story] if c["id"] == chapter)


def _revision_of(fake, chapter: str = "rich", story: str = "story-a") -> str:
    """The token a read hands the caller: story-data's integer revision."""
    return str(_chapter_row(fake, chapter, story)["revision"])


def _content_of(fake, chapter: str = "rich", story: str = "story-a") -> str:
    return _chapter_row(fake, chapter, story)["content"]


async def test_get_chapter_exposes_a_revision():
    """story-data carries an integer revision per row; the token is that value."""
    _rich_story_data()
    chapter = await data.get_chapter("story-a", "rich", UID_A, 0, 100)
    assert chapter["revision"] == "4"


async def test_get_chapter_blocks_lists_indices_tags_and_previews():
    _rich_story_data()
    listing = await data.get_chapter_blocks("story-a", "rich", UID_A, 0, 500)
    assert listing["block_count"] == 5
    assert [b["tag"] for b in listing["blocks"]] == RICH_BLOCKS
    assert [b["index"] for b in listing["blocks"]] == [0, 1, 2, 3, 4]
    assert listing["blocks"][1]["preview"] == "She counted the steps."
    # An image has no text of its own; the tag is what identifies it.
    assert listing["blocks"][4]["preview"] == ""
    assert listing["next_index"] is None
    assert listing["revision"] == "4"


async def test_get_chapter_blocks_paginates():
    _rich_story_data()
    first = await data.get_chapter_blocks("story-a", "rich", UID_A, 0, 2)
    assert [b["index"] for b in first["blocks"]] == [0, 1]
    assert first["next_index"] == 2
    second = await data.get_chapter_blocks(
        "story-a", "rich", UID_A, first["next_index"], 2
    )
    assert [b["index"] for b in second["blocks"]] == [2, 3]
    last = await data.get_chapter_blocks("story-a", "rich", UID_A, 4, 2)
    assert [b["index"] for b in last["blocks"]] == [4]
    assert last["next_index"] is None


async def test_get_chapter_blocks_clamps_inputs():
    _rich_story_data()
    listing = await data.get_chapter_blocks("story-a", "rich", UID_A, -5, 999_999)
    assert listing["start_index"] == 0
    assert listing["block_count"] == 5


async def test_get_chapter_blocks_is_owner_scoped():
    _rich_story_data()
    with pytest.raises(data.StoryNotFoundError):
        await data.get_chapter_blocks("story-b", "chb", UID_A, 0, 10)
    with pytest.raises(data.EntityNotFoundError):
        await data.get_chapter_blocks("story-a", "no-such-chapter", UID_A, 0, 10)


async def test_get_chapter_blocks_on_an_empty_chapter():
    fake = _seeded_story_data()
    fake.seed_chapter("story-a", "blank", title="Blank", content="")
    listing = await data.get_chapter_blocks("story-a", "blank", UID_A, 0, 10)
    assert listing["block_count"] == 0
    assert listing["blocks"] == []


# ---------------------------------------------------------------------------
# Chapter editing — the revision guard
#
# This is the safety property of the whole feature: an edit must never land on
# a version the caller did not read. story-data carries an integer `revision`
# per row and takes it back as If-Match, so the guard is that header plus a
# cheap pre-check on the read.
# ---------------------------------------------------------------------------


async def test_stale_revision_is_refused_and_writes_nothing():
    db, fake = _rich_env()
    before = _content_of(fake)
    with pytest.raises(writes.StaleRevisionError):
        await writes.append_to_chapter(db, UID_A, "story-a", "rich", "more", "999")
    assert _content_of(fake) == before
    assert not any(path.startswith("PATCH") for path, _ in fake.writes)


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "0", "-3"])
async def test_unusable_revisions_are_refused_before_any_request(bad):
    """Parsed here rather than forwarded: a caller that invents a revision gets
    a sentence it can act on, not a 428 from a service it has never heard of."""
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="revision"):
        await writes.append_to_chapter(db, UID_A, "story-a", "rich", "more", bad)
    assert fake.writes == []


async def test_fresh_revision_succeeds_and_returns_the_new_one():
    db, fake = _rich_env()
    old = _revision_of(fake)
    result = await writes.append_to_chapter(db, UID_A, "story-a", "rich", "more", old)
    assert result["revision"] == _revision_of(fake)
    assert result["revision"] != old


async def test_concurrent_writer_between_check_and_update_is_caught():
    """The window If-Match exists to close.

    The pre-check passes, then a concurrent editor save lands, then our PATCH
    runs. Without the header this would silently clobber it.
    """
    db, fake = _rich_env()
    revision = _revision_of(fake)

    def concurrent_editor() -> None:
        fake.before_update_chapter = None  # fire once
        row = _chapter_row(fake, "rich")
        row["content"] = "<p>the human typed this</p>"
        row["revision"] += 1

    fake.before_update_chapter = concurrent_editor
    with pytest.raises(writes.StaleRevisionError):
        await writes.append_to_chapter(
            db, UID_A, "story-a", "rich", "appended", revision
        )
    assert _content_of(fake) == "<p>the human typed this</p>"


async def test_the_revision_a_read_hands_out_is_the_one_a_write_accepts():
    """The handshake the port exists to restore.

    Under the Firestore write path a read returned story-data's integer
    revision while a write expected a timestamp token, so the two never met —
    which is why config refused that combination outright. Taking the token
    straight from the read tool and passing it to a write must now work, and
    the write's own returned revision must chain into a second edit.
    """
    db, fake = _rich_env()
    listing = await data.get_chapter_blocks("story-a", "rich", UID_A, 0, 500)
    first = await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "ok", listing["revision"]
    )
    second = await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "again", first["revision"]
    )
    assert second["revision"] == _revision_of(fake)
    assert _content_of(fake).endswith("<p>ok</p>\n<p>again</p>")


# ---------------------------------------------------------------------------
# append_to_chapter
# ---------------------------------------------------------------------------


async def test_append_preserves_existing_content_byte_for_byte():
    db, fake = _rich_env()
    await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "New line.", _revision_of(fake)
    )
    stored = _content_of(fake)
    assert stored.startswith(RICH)
    assert stored == RICH + "\n<p>New line.</p>"


async def test_append_to_an_empty_chapter_adds_no_leading_separator():
    db, fake = _rich_env()
    fake.seed_chapter("story-a", "blank", title="Blank", content="", position=9.0)
    await writes.append_to_chapter(
        db, UID_A, "story-a", "blank", "First.", _revision_of(fake, "blank")
    )
    assert _content_of(fake, "blank") == "<p>First.</p>"


async def test_append_escapes_markup_and_splits_paragraphs():
    db, fake = _rich_env()
    await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "<b>bold</b>\n\nsecond", _revision_of(fake)
    )
    stored = _content_of(fake)
    assert "<p>&lt;b&gt;bold&lt;/b&gt;</p>" in stored
    assert stored.endswith("<p>second</p>")


async def test_append_block_count_survives_malformed_existing_content():
    """block_count must be what split_blocks will say next time, not the sum of
    the two halves — an unclosed tag makes those differ, and the number is the
    index range the caller's next edit addresses."""
    db, fake = _rich_env()
    fake.seed_chapter(
        "story-a", "broken", title="Broken", position=9.0, content="<p>unclosed"
    )
    result = await writes.append_to_chapter(
        db, UID_A, "story-a", "broken", "New paragraph.", _revision_of(fake, "broken")
    )
    content = _content_of(fake, "broken")
    assert result["block_count"] == len(blocks_module.split_blocks(content)) == 1
    assert result["appended_blocks"] == 1  # the sum would have said 2


async def test_append_reports_the_word_count_story_data_stored():
    db, fake = _rich_env()
    result = await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "one two three", _revision_of(fake)
    )
    row = _chapter_row(fake, "rich")
    assert row["wordCount"] == result["word_count"]
    assert row["wordCount"] == len(_content_of(fake).split())


async def test_append_carries_the_title_and_position_through_the_patch():
    """story-data's chapter PATCH takes a whole ChapterInput, so a body-only
    edit that omitted these would rename the chapter and move it to the front
    of the book."""
    db, fake = _rich_env()
    await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "x", _revision_of(fake)
    )
    row = _chapter_row(fake, "rich")
    assert row["title"] == "Rich"
    assert row["position"] == 3.0


async def test_append_does_not_create_a_chapter():
    db, fake = _rich_env()
    before = len(fake.chapters["story-a"])
    await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "x", _revision_of(fake)
    )
    assert len(fake.chapters["story-a"]) == before


async def test_append_rejects_empty_content():
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="must not be empty"):
        await writes.append_to_chapter(
            db, UID_A, "story-a", "rich", "   ", _revision_of(fake)
        )


async def test_append_word_cap_is_reported_from_story_data():
    db, fake = _rich_env()
    body = "<p>" + " ".join(["w"] * (FakeStoryData.WORD_LIMIT - 1)) + "</p>"
    fake.seed_chapter("story-a", "big", title="Big", position=9.0, content=body)
    with pytest.raises(story_data.Rejected, match="5000 words"):
        await writes.append_to_chapter(
            db, UID_A, "story-a", "big", "two more words", _revision_of(fake, "big")
        )
    assert _content_of(fake, "big") == body  # denied, and nothing written


async def test_append_enforces_the_char_cap_on_the_rebuilt_string():
    db, fake = _rich_env()
    body = "<p>" + "x" * (writes.MAX_CHAPTER_CONTENT_CHARS - 10) + "</p>"
    fake.seed_chapter("story-a", "big", title="Big", position=9.0, content=body)
    with pytest.raises(writes.LimitExceededError) as exc:
        await writes.append_to_chapter(
            db, UID_A, "story-a", "big", "y" * 100, _revision_of(fake, "big")
        )
    assert exc.value.limit_name == "chapter_content_chars"
    assert _content_of(fake, "big") == body


async def test_append_is_owner_scoped():
    db, fake = _rich_env()
    with pytest.raises(data.StoryNotFoundError):
        await writes.append_to_chapter(db, UID_A, "story-b", "chb", "x", "1")
    assert fake.writes == []


# ---------------------------------------------------------------------------
# edit_chapter_blocks
# ---------------------------------------------------------------------------


async def _edit(db, fake, ops, chapter="rich", uid=UID_A):
    return await writes.edit_chapter_blocks(
        db, uid, "story-a", chapter, ops, _revision_of(fake, chapter)
    )


async def test_replace_leaves_every_other_block_byte_identical():
    """The central promise of block editing."""
    db, fake = _rich_env()
    original = blocks_module.split_blocks(RICH)
    await _edit(
        db, fake, [{"action": "replace", "index": 1, "text": "She counted twice."}]
    )
    after = blocks_module.split_blocks(_content_of(fake))
    assert after[1].html == "<p>She counted twice.</p>"
    for position in (0, 2, 3, 4):
        assert after[position].html == original[position].html


async def test_replace_with_empty_text_removes_the_block():
    db, fake = _rich_env()
    result = await _edit(db, fake, [{"action": "replace", "index": 4, "text": ""}])
    assert result["block_count"] == 4
    assert "cave.png" not in _content_of(fake)


async def test_removing_every_block_leaves_an_empty_chapter():
    db, fake = _rich_env()
    ops = [{"action": "replace", "index": i, "text": ""} for i in range(5)]
    result = await _edit(db, fake, ops)
    assert _content_of(fake) == ""
    assert result["block_count"] == 0
    assert result["word_count"] == 0


async def test_insert_after_places_the_block_in_the_right_seam():
    db, fake = _rich_env()
    await _edit(
        db, fake, [{"action": "insert_after", "index": 1, "text": "Then she stopped."}]
    )
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(fake))]
    assert tags == ["h2", "p", "p", "ul", "blockquote", "img"]
    assert blocks_module.split_blocks(_content_of(fake))[2].html == (
        "<p>Then she stopped.</p>"
    )


async def test_insert_after_minus_one_prepends():
    db, fake = _rich_env()
    await _edit(
        db, fake, [{"action": "insert_after", "index": -1, "text": "Prologue."}]
    )
    assert _content_of(fake).startswith("<p>Prologue.</p>\n<h2>")


async def test_insert_after_last_index_appends():
    db, fake = _rich_env()
    await _edit(db, fake, [{"action": "insert_after", "index": 4, "text": "The end."}])
    assert _content_of(fake).endswith("<p>The end.</p>")


async def test_one_op_may_introduce_several_blocks():
    db, fake = _rich_env()
    result = await _edit(
        db,
        fake,
        [{"action": "replace", "index": 1, "text": "First para.\n\nSecond para."}],
    )
    assert result["block_count"] == 6
    after = blocks_module.split_blocks(_content_of(fake))
    assert [b.html for b in after[1:3]] == ["<p>First para.</p>", "<p>Second para.</p>"]


async def test_replacing_a_heading_keeps_its_level():
    """A heading is rebuilt as itself, not flattened into a paragraph.

    RICH[0] is an <h2>. Routing it through _to_paragraph_html like a paragraph
    would silently demote it, and nothing in the product could restore it.
    """
    db, fake = _rich_env()
    await _edit(db, fake, [{"action": "replace", "index": 0, "text": "The Ascent"}])
    after = blocks_module.split_blocks(_content_of(fake))
    assert after[0] == blocks_module.Block(tag="h2", html="<h2>The Ascent</h2>")


async def test_heading_replacement_escapes_markup():
    db, fake = _rich_env()
    await _edit(
        db, fake, [{"action": "replace", "index": 0, "text": "A <b>bold</b> title"}]
    )
    assert "<h2>A &lt;b&gt;bold&lt;/b&gt; title</h2>" in _content_of(fake)


async def test_multi_paragraph_text_is_refused_for_a_heading():
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="single line"):
        await _edit(
            db, fake, [{"action": "replace", "index": 0, "text": "One.\n\nTwo."}]
        )
    assert _content_of(fake) == RICH


@pytest.mark.parametrize("index", [2, 4])  # <ul>, <img>
async def test_replacing_a_structural_block_is_refused(index):
    """Refused rather than flattened: _to_paragraph_html would turn the list
    into one soft-wrapped <p> and drop every item boundary."""
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="cannot rewrite"):
        await _edit(
            db, fake, [{"action": "replace", "index": index, "text": "rope and lamp"}]
        )
    assert _content_of(fake) == RICH


async def test_a_structural_block_can_still_be_deleted():
    """Deletion stays open for every tag — it is what the caller asked for,
    where a rewrite would be a downgrade they did not ask for."""
    db, fake = _rich_env()
    await _edit(db, fake, [{"action": "replace", "index": 2, "text": ""}])
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(fake))]
    assert tags == ["h2", "p", "blockquote", "img"]


async def test_a_refused_op_abandons_the_whole_call():
    """One bad op must not leave the other ops half-applied."""
    db, fake = _rich_env()
    ops = [
        {"action": "replace", "index": 1, "text": "Rewritten."},
        {"action": "replace", "index": 2, "text": "flattened list"},
    ]
    with pytest.raises(ValueError, match="cannot rewrite"):
        await _edit(db, fake, ops)
    assert _content_of(fake) == RICH


@pytest.mark.parametrize("bad_text", [0, False, [], None, 12])
async def test_a_non_string_text_is_refused_not_treated_as_a_deletion(bad_text):
    """`op.get("text") or ""` used to coerce every falsy non-string to "",
    which is the deletion sentinel — a malformed op silently destroyed a
    paragraph. Pydantic catches these at the tool layer; this is the boundary
    behind it holding on its own."""
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="text must be a string"):
        await _edit(db, fake, [{"action": "replace", "index": 1, "text": bad_text}])
    assert _content_of(fake) == RICH


async def test_an_absent_text_key_still_means_deletion():
    """ "" is the documented schema default, so an omitted key keeps deleting."""
    db, fake = _rich_env()
    await _edit(db, fake, [{"action": "replace", "index": 1}])
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(fake))]
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
    db_one, first = _rich_env()
    await _edit(db_one, first, ops)
    db_two, second = _rich_env()
    await _edit(db_two, second, list(reversed(ops)))
    assert _content_of(first) == _content_of(second)

    after = blocks_module.split_blocks(_content_of(first))
    # Block 0 is the <h2>, so the replacement is rebuilt at that level.
    assert after[0].html == "<h2>New heading.</h2>"
    # Original block 2 was the list; the insert landed straight after it.
    assert after[2].tag == "ul"
    assert after[3].html == "<p>After the list.</p>"


async def test_removal_and_insert_in_one_call_use_original_indices():
    db, fake = _rich_env()
    await _edit(
        db,
        fake,
        [
            {"action": "replace", "index": 3, "text": ""},  # drop the blockquote
            {"action": "insert_after", "index": 0, "text": "Subtitle."},
        ],
    )
    tags = [b.tag for b in blocks_module.split_blocks(_content_of(fake))]
    assert tags == ["h2", "p", "p", "ul", "img"]


async def test_duplicate_index_is_rejected():
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="two operations target block 1"):
        await _edit(
            db,
            fake,
            [
                {"action": "replace", "index": 1, "text": "a"},
                {"action": "insert_after", "index": 1, "text": "b"},
            ],
        )
    assert _content_of(fake) == RICH


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
    db, fake = _rich_env()
    with pytest.raises(ValueError, match="out of range"):
        await _edit(db, fake, [op])
    assert _content_of(fake) == RICH


async def test_empty_chapter_guides_the_caller_to_insert_after_minus_one():
    db, fake = _rich_env()
    fake.seed_chapter("story-a", "blank", title="Blank", position=9.0, content="")
    with pytest.raises(ValueError, match="no blocks yet"):
        await _edit(
            db, fake, [{"action": "replace", "index": 0, "text": "x"}], chapter="blank"
        )
    await _edit(
        db,
        fake,
        [{"action": "insert_after", "index": -1, "text": "First."}],
        chapter="blank",
    )
    assert _content_of(fake, "blank") == "<p>First.</p>"


async def test_op_list_validation():
    db, fake = _rich_env()
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
            await _edit(db, fake, bad)
    assert _content_of(fake) == RICH


async def test_too_many_ops_rejected():
    db, fake = _rich_env()
    ops = [
        {"action": "insert_after", "index": i, "text": "x"}
        for i in range(writes.MAX_OPS_PER_CALL + 1)
    ]
    with pytest.raises(ValueError, match="at most 20 operations"):
        await _edit(db, fake, ops)


async def test_edit_escapes_markup():
    db, fake = _rich_env()
    await _edit(
        db, fake, [{"action": "replace", "index": 1, "text": "<script>x</script>"}]
    )
    assert "<script>" not in _content_of(fake)
    assert "&lt;script&gt;" in _content_of(fake)


async def test_edit_word_cap_is_reported_from_story_data():
    db, fake = _rich_env()
    with pytest.raises(story_data.Rejected, match="5000 words"):
        await _edit(
            db,
            fake,
            [
                {
                    "action": "replace",
                    "index": 1,
                    "text": " ".join(["w"] * (FakeStoryData.WORD_LIMIT + 1)),
                }
            ],
        )
    assert _content_of(fake) == RICH


async def test_edit_enforces_the_char_cap_on_the_rebuilt_string():
    """The one ceiling still applied here, and it is measured on the joined
    blocks rather than the caller's text."""
    db, fake = _rich_env()
    with pytest.raises(writes.LimitExceededError) as exc:
        await _edit(
            db,
            fake,
            [
                {
                    "action": "replace",
                    "index": 1,
                    "text": "x" * writes.MAX_CHAPTER_CONTENT_CHARS,
                }
            ],
        )
    assert exc.value.limit_name == "chapter_content_chars"
    assert _content_of(fake) == RICH


async def test_edit_is_owner_scoped():
    db, fake = _rich_env()
    with pytest.raises(data.StoryNotFoundError):
        await writes.edit_chapter_blocks(
            db,
            UID_A,
            "story-b",
            "chb",
            [{"action": "replace", "index": 0, "text": "mine now"}],
            _revision_of(fake, "chb", "story-b"),
        )
    assert fake.writes == []


async def test_edit_on_a_missing_chapter_raises_entity_not_found():
    db, fake = _rich_env()
    with pytest.raises(data.EntityNotFoundError):
        await writes.edit_chapter_blocks(
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
    db, fake = _rich_env()
    revision = _revision_of(fake)
    first = await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "Once.", revision
    )
    after_first = _content_of(fake)
    second = await writes.append_to_chapter(
        db, UID_A, "story-a", "rich", "Once.", revision
    )
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert second["revision"] == first["revision"]
    assert _content_of(fake) == after_first


async def test_identical_edit_replays_instead_of_editing_twice():
    db, fake = _rich_env()
    revision = _revision_of(fake)
    ops = [{"action": "insert_after", "index": 0, "text": "Inserted."}]
    first = await writes.edit_chapter_blocks(
        db, UID_A, "story-a", "rich", ops, revision
    )
    after_first = _content_of(fake)
    second = await writes.edit_chapter_blocks(
        db, UID_A, "story-a", "rich", ops, revision
    )
    assert second["idempotent_replay"] is True
    assert _content_of(fake) == after_first
    assert _content_of(fake).count("Inserted.") == 1
    assert first["block_count"] == second["block_count"]


async def test_same_ops_against_the_new_revision_is_a_fresh_edit():
    """A replay is anchored to the base version, not to the text of the call."""
    db, fake = _rich_env()
    ops = [{"action": "insert_after", "index": 0, "text": "Again."}]
    await writes.edit_chapter_blocks(
        db, UID_A, "story-a", "rich", ops, _revision_of(fake)
    )
    await writes.edit_chapter_blocks(
        db, UID_A, "story-a", "rich", ops, _revision_of(fake)
    )
    assert _content_of(fake).count("Again.") == 2


async def test_op_order_does_not_change_the_idempotency_key():
    db, fake = _rich_env()
    revision = _revision_of(fake)
    ops = [
        {"action": "replace", "index": 0, "text": "A."},
        {"action": "insert_after", "index": 2, "text": "B."},
    ]
    await writes.edit_chapter_blocks(db, UID_A, "story-a", "rich", ops, revision)
    replay = await writes.edit_chapter_blocks(
        db, UID_A, "story-a", "rich", list(reversed(ops)), revision
    )
    assert replay["idempotent_replay"] is True


async def test_failed_edit_releases_the_reservation():
    """A corrected retry must not be blocked by the failed attempt's claim."""
    db, fake = _rich_env()
    revision = _revision_of(fake)
    with pytest.raises(story_data.Rejected):
        await writes.edit_chapter_blocks(
            db,
            UID_A,
            "story-a",
            "rich",
            [
                {
                    "action": "replace",
                    "index": 1,
                    "text": " ".join(["w"] * (FakeStoryData.WORD_LIMIT + 1)),
                }
            ],
            revision,
        )
    ok = await writes.edit_chapter_blocks(
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
    db, fake = _rich_env()
    story = await writes.create_story(db, UID_B, "B's book")
    chapter = await writes.create_chapter(db, UID_B, story["story_id"], "Ch", "text")
    revision = _revision_of(fake, chapter["chapter_id"], story["story_id"])
    await writes.append_to_chapter(
        db, UID_B, story["story_id"], chapter["chapter_id"], "hi", revision
    )
    with pytest.raises(data.StoryNotFoundError):
        await writes.append_to_chapter(
            db, UID_A, story["story_id"], chapter["chapter_id"], "hi", revision
        )


# ---------------------------------------------------------------------------
# Editing — through the MCP tool layer
# ---------------------------------------------------------------------------


async def test_edit_tools_require_the_write_scope():
    db, fake = _rich_env()
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
                        "revision": _revision_of(fake),
                        **args,
                    },
                )
    assert _content_of(fake) == RICH


async def test_edit_tool_idor_returns_story_not_found():
    db, fake = _rich_env()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="Story not found"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-b",
                    "chapter_id": "chb",
                    "content": "mine now",
                    "revision": _revision_of(fake, "chb", "story-b"),
                },
            )
    assert _content_of(fake, "chb", "story-b") == "secret text"


async def test_edit_tool_missing_chapter_says_chapter_not_found():
    db, fake = _rich_env()
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
    db, fake = _rich_env()
    mcp = _tool_server(db, enable_writes=True)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        with pytest.raises(ToolError, match="changed since you read it"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-a",
                    "chapter_id": "rich",
                    "content": "x",
                    "revision": "999",
                },
            )
    assert _content_of(fake) == RICH


async def test_edit_tools_share_the_write_rate_limiter():
    db, fake = _rich_env()
    mcp = _tool_server(db, enable_writes=True, write_rpm=1)
    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        await _call(
            mcp,
            "append_to_chapter",
            {
                "story_id": "story-a",
                "chapter_id": "rich",
                "content": "one",
                "revision": _revision_of(fake),
            },
        )
        with pytest.raises(ToolError, match="Write rate limit"):
            await mcp.call_tool(
                "append_to_chapter",
                {
                    "story_id": "story-a",
                    "chapter_id": "rich",
                    "content": "two",
                    "revision": _revision_of(fake),
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
            {"position": 4.0, "chapter_count": 4, "attempts": 1},
        ),
    ],
)
async def test_every_write_is_audited_with_the_caller_and_target(
    tool, arguments, expected_extra
):
    """The audit line is the only record of who changed what, so its spine —
    uid, connector, story, replay flag — has to survive refactoring."""
    db, fake = _rich_env()
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
    db, fake = _rich_env()
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
    db, fake = _rich_env()
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
                    "revision": _revision_of(fake),
                },
            )
    line = next(e for e in logs if e["event"] == "mcp_write_chapter_appended")
    assert line["chapter_id"] == "rich"
    assert line["appended_chars"] == len("Postscript.")
    assert line["block_count"] == 6


async def test_full_edit_cycle_through_the_write_tools():
    """Edit by index, chain a second edit on the returned revision, verify bytes.

    Both halves go through the tool layer: the block listing comes from the
    READ tool and its revision drives the write, which is the handshake the
    Firestore write path could not make.
    """
    db, fake = _rich_env()
    mcp = _tool_server(db, enable_writes=True)

    with patch("mcp_server.tools.get_access_token", return_value=_token(UID_A, _RW)):
        listing = await _call(
            mcp, "get_chapter_blocks", {"story_id": "story-a", "chapter_id": "rich"}
        )
        target_index = next(
            block["index"]
            for block in listing["blocks"]
            if block["preview"].startswith("She")
        )
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
                        "index": target_index,
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
        final_read = await _call(
            mcp, "get_chapter", {"story_id": "story-a", "chapter_id": "rich"}
        )

    final = final_read["content"]
    assert "She counted the steps twice." in final
    assert final.endswith("<p>Postscript.</p>")
    # Untouched blocks kept their exact markup through two edits.
    assert (
        '<ul class="list-disc"><li><p>rope</p></li><li><p>lamp</p></li></ul>' in final
    )


# ---------------------------------------------------------------------------
# story_data client — status mapping for the write verbs
#
# The fake reproduces these outcomes everywhere else in this file, so they are
# pinned once against real httpx responses.
# ---------------------------------------------------------------------------


def _client_over(handler) -> story_data.StoryDataClient:
    client = story_data.StoryDataClient("http://sd.test", "tok")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.parametrize(
    "status,expected",
    [
        (404, story_data.NotFound),
        (403, story_data.NotFound),  # "not yours" must not be distinguishable
        (409, story_data.Conflict),
        (428, story_data.Conflict),  # If-Match missing or unparseable
        (422, story_data.Rejected),
        (400, story_data.Rejected),
        (500, story_data.StoryDataError),
    ],
)
async def test_write_status_codes_map_to_their_errors(status, expected):
    client = _client_over(
        lambda request: httpx.Response(status, json={"error": "nope"})
    )
    with pytest.raises(expected):
        await client.create_story("u", {"title": "x"})


async def test_a_rejection_carries_story_datas_own_message():
    """The ceilings live there now, so its wording is what the model must see."""
    client = _client_over(
        lambda request: httpx.Response(
            422, json={"error": "you have reached the limit of 100 stories"}
        )
    )
    with pytest.raises(story_data.Rejected) as exc:
        await client.create_story("u", {"title": "x"})
    assert exc.value.message == "you have reached the limit of 100 stories"


async def test_update_chapter_sends_the_revision_as_if_match():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"id": "c", "revision": 8})

    result = await _client_over(handler).update_chapter(
        "u", "s", "c", {"title": "T", "content": "x", "position": 1}, "7"
    )
    assert seen["if-match"] == "7"
    assert seen["x-user-id"] == "u"
    assert seen["x-service-token"] == "tok"
    assert result["revision"] == 8


async def test_all_write_methods_match_the_story_data_http_contract():
    seen: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content.decode()),
            )
        )
        if request.url.path == "/v1/stories":
            return httpx.Response(201, json={"id": "s"})
        if request.method == "POST":
            return httpx.Response(201, json={"id": "c", "revision": 1})
        return httpx.Response(200, json={"id": "c", "revision": 2})

    client = _client_over(handler)
    story = {"title": "T", "published": False}
    chapter = {"title": "C", "content": "<p>x</p>", "position": 1.0}
    await client.create_story("u", story)
    await client.create_chapter("u", "s", chapter)
    await client.update_chapter("u", "s", "c", chapter, "1")

    assert seen == [
        ("POST", "/v1/stories", story),
        ("POST", "/v1/stories/s/chapters", chapter),
        ("PATCH", "/v1/stories/s/chapters/c", chapter),
    ]


# ---------------------------------------------------------------------------
# Drift guards
# ---------------------------------------------------------------------------


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


def test_ceilings_story_data_owns_are_not_restated_here():
    """writes.py used to re-declare firestore.rules' ceilings because the Admin
    SDK bypassed them. story-data enforces its own, transactionally, so a copy
    here would be a second number to keep in step — and the first one to go
    stale. Only the stored-size bound, which has no counterpart there, remains.
    """
    for gone in (
        "MAX_STORIES_PER_USER",
        "MAX_CHAPTERS_PER_STORY",
        "MAX_CHAPTER_WORDS",
    ):
        assert not hasattr(writes, gone), f"{gone} is story-data's to enforce"
    assert writes.MAX_CHAPTER_CONTENT_CHARS == 100_000
