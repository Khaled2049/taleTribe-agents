"""Tool executors: one happy path each, plus the bounds and the ownership gate.

Deliberately not a matrix. The argument schemas are already covered by
test_assistant_tools.py, and the story-data read semantics by
test_mcp_tools.py -- what is new here is the adapter between them, the second
bounds check at the execution boundary, and the story scoping search_story has
to re-establish by hand.
"""

import json

import pytest

from assistant.errors import ErrorCode
from assistant.executors import (
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    ToolExecutionError,
    ToolRuntime,
    execute_tool,
)
from assistant.protocol import EditorContext, EditorTextWindow, Selection
from assistant.tools import (
    ToolContext,
    available_tools,
    validate_tool_arguments,
)
from mcp_server import story_data
from tests.mcp_fakes import FakeStoryData

OWNER = "uid-owner"
STORY = "story-1"


@pytest.fixture
def fake():
    data = FakeStoryData()
    data.seed_story(STORY, OWNER, title="Saltmarsh", description="A harbour town.")
    data.seed_chapter(
        STORY, "chapter-1", title="The Lamp Room", content="Brass polish. " * 200
    )
    data.seed_chapter(STORY, "chapter-2", title="Low Water", content="The tide went.")
    data.seed_entity(STORY, "characters", "char-1", name="Mina", personality="Guarded.")
    story_data.configure(data)
    yield data
    story_data.configure(None)


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    async def embed(self, text):
        self.queries.append(text)
        return [0.0] * 768


class FakePostgres:
    """Records the story_id the SQL predicate would receive."""

    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.scoped_to: list[str] = []

    async def search_chunks(self, story_id, embedding, top_k=4):
        self.scoped_to.append(story_id)
        return self.chunks[:top_k]


def chunk(chunk_id="chunk-1", text="The lamp room smelled of brass polish."):
    return {
        "chunk_id": chunk_id,
        "kind": "chapter",
        "source_id": "chapter-1",
        "source_revision": 3,
        "current_revision": 3,
        "chunk_index": 0,
        "indexed_at": None,
        "source_updated_at": None,
        "metadata": {"title": "The Lamp Room", "chapterNumber": 1},
        "text": text,
    }


def runtime(**kwargs):
    return ToolRuntime(ctx=ToolContext(user_id=OWNER, story_id=STORY), **kwargs)


async def run(name, arguments, rt):
    """Go through validation the way the loop will, not straight to the executor."""
    validate_tool_arguments(name, arguments)
    from assistant.tools import TOOL_SCHEMAS

    return await execute_tool(name, TOOL_SCHEMAS[name].model_validate(arguments), rt)


async def test_overview_chapter_ids_can_be_read_even_with_duplicate_titles(fake):
    fake.chapters[STORY] = []
    fake.seed_chapter(
        STORY, "chapter-1", title="Two Inches", content="The first chapter."
    )
    fake.seed_chapter(
        STORY, "chapter-2", title="Two Inches", content="The second chapter."
    )
    overview = await run("get_story_overview", {}, runtime())

    for chapter, expected in zip(
        overview.result["chapters"], ["The first chapter.", "The second chapter."]
    ):
        result = await run(
            "read_chapter", {"chapterId": chapter["chapter_id"]}, runtime()
        )
        assert result.result["title"] == "Two Inches"
        assert result.result["content"] == expected
        assert result.references[0].source_id == chapter["chapter_id"]


async def test_shortened_overview_preserves_readable_chapter_ids(fake):
    ids = [f"00000000-0000-4000-8000-{i:012d}" for i in range(50)]
    for chapter_id in ids:
        fake.seed_chapter(
            STORY, chapter_id, title="Long title " * 40, content="A passage."
        )
    result = await run("get_story_overview", {}, runtime(max_result_chars=1_500))

    assert result.truncated is True
    assert len(json.dumps(result.result)) <= 1_500
    assert result.result["chapters"]
    for chapter in result.result["chapters"]:
        assert chapter["chapter_id"] in {"chapter-1", "chapter-2", *ids}
        read = await run(
            "read_chapter", {"chapterId": chapter["chapter_id"]}, runtime()
        )
        assert read.result.get("found") is not False


async def test_list_story_entities_maps_the_singular_kind(fake):
    result = await run("list_story_entities", {"kind": "character"}, runtime())
    assert result.result["kind"] == "character"
    assert [e["name"] for e in result.result["entities"]] == ["Mina"]
    assert result.result["truncated"] is False


async def test_list_story_entities_reports_its_own_limit_as_truncation(fake):
    fake.seed_entity(STORY, "characters", "char-2", name="Ansel")
    result = await run(
        "list_story_entities", {"kind": "character", "limit": 1}, runtime()
    )
    assert len(result.result["entities"]) == 1
    assert result.result["truncated"] is True


async def test_unknown_entity_is_a_result_not_a_failure(fake):
    """A model that guessed an id should be able to recover on the next turn."""
    result = await run(
        "get_story_entity", {"kind": "character", "entityId": "nope"}, runtime()
    )
    assert result.result == {"found": False, "kind": "character", "id": "nope"}


async def test_read_chapter_emits_a_source_and_windows_the_text(fake):
    result = await run(
        "read_chapter", {"chapterId": "chapter-1", "limit": 100}, runtime()
    )
    assert len(result.result["content"]) == 100
    assert result.result["next_offset"] == 100
    (reference,) = result.references
    assert reference.source_id == "chapter-1"
    assert reference.kind == "story"
    assert reference.title == "Chapter 1: The Lamp Room"
    assert reference.snippet


async def test_read_chapter_ceiling_counts_serialized_characters(fake):
    """Legal source text may expand several times over when encoded as JSON."""
    content = '"\\\n字😀' * 6_000
    fake.seed_chapter(STORY, "chapter-json", title="Escapes", content=content)
    result = await run(
        "read_chapter",
        {"chapterId": "chapter-json", "limit": 20_000},
        runtime(),
    )

    assert len(json.dumps(result.result)) <= DEFAULT_MAX_TOOL_RESULT_CHARS
    assert result.result["content"] == content[: len(result.result["content"])]
    assert result.result["next_offset"] == len(result.result["content"])
    assert result.truncated is True


async def test_read_chapter_missing_is_a_result(fake):
    result = await run("read_chapter", {"chapterId": "nope"}, runtime())
    assert result.result["found"] is False


async def test_search_story_scopes_to_the_context_story(fake):
    """The story_id in the SQL predicate comes from ToolContext, never the model."""
    postgres = FakePostgres([chunk()])
    embedder = FakeEmbedder()
    result = await run(
        "search_story",
        {"query": "brass polish"},
        runtime(postgres=postgres, embedder=embedder),
    )
    assert postgres.scoped_to == [STORY]
    assert embedder.queries == ["brass polish"]
    hit = result.result["results"][0]
    assert hit["chunk_id"] == "chunk-1"
    assert hit["title"] == "The Lamp Room"
    assert hit["source_revision"] == 3
    assert hit["stale"] is False
    (reference,) = result.references
    assert reference.source_id == "chunk-1"
    assert reference.kind == "story"


async def test_search_story_marks_an_outdated_chunk_stale(fake):
    old = chunk()
    old["current_revision"] = 4
    result = await run(
        "search_story",
        {"query": "brass polish"},
        runtime(postgres=FakePostgres([old]), embedder=FakeEmbedder()),
    )
    assert result.result["results"][0]["stale"] is True


async def test_search_story_without_retrieval_fails_rather_than_reporting_nothing(
    fake,
):
    with pytest.raises(ToolExecutionError) as excinfo:
        await run("search_story", {"query": "anything"}, runtime())
    assert excinfo.value.code is ErrorCode.INTERNAL_ERROR


async def test_search_story_divides_the_result_budget_across_hits(fake):
    postgres = FakePostgres([chunk(f"chunk-{i}", "y" * 5_000) for i in range(4)])
    result = await run(
        "search_story",
        {"query": "y", "limit": 4},
        runtime(postgres=postgres, embedder=FakeEmbedder(), max_result_chars=2_000),
    )
    assert len(json.dumps(result.result)) <= 2_000
    assert len(result.result["results"]) == 4


async def test_read_current_editor_without_a_buffer(fake):
    result = await run("read_current_editor", {}, runtime())
    assert result.result == {"available": False, "reason": "no active editor"}


async def test_read_current_editor_returns_what_the_browser_sent(fake):
    editor = EditorContext(
        chapter_id="chapter-1",
        persisted_revision=4,
        document_version=11,
        selection=Selection(**{"from": 10, "to": 20, "text": "brass polish"}),
        buffer=EditorTextWindow(text="The lamp room smelled of brass polish."),
        dirty=True,
    )
    result = await run("read_current_editor", {}, runtime(editor_context=editor))
    assert result.result["available"] is True
    assert result.result["dirty"] is True
    assert result.result["selection"] == {
        "from": 10,
        "to": 20,
        "text": "brass polish",
    }
    # The default stays selection-only even when a bounded window was supplied.
    assert result.result["full_document_available"] is False

    window = await run(
        "read_current_editor",
        {"selectionOnly": False},
        runtime(editor_context=editor),
    )
    assert window.result["buffer"] == {
        "text": "The lamp room smelled of brass polish.",
        "truncated": False,
    }
    assert window.result["full_document_available"] is True


async def test_a_non_owner_is_refused_by_the_gate_the_tools_inherit(fake):
    """story-data serves a published story to anyone; the executors must not."""
    fake.stories[STORY]["published"] = True
    intruder = ToolRuntime(ctx=ToolContext(user_id="uid-other", story_id=STORY))
    with pytest.raises(ToolExecutionError) as excinfo:
        await execute_tool(
            "get_story_overview",
            available_tools(edits_enabled=False, research_enabled=False)[
                "get_story_overview"
            ].model_validate({}),
            intruder,
        )
    assert excinfo.value.code is ErrorCode.STORY_ACCESS_DENIED


async def test_story_data_being_down_is_an_internal_error(fake):
    class Broken(FakeStoryData):
        async def get_story(self, uid, story_id):
            raise story_data.StoryDataError("boom")

    story_data.configure(Broken())
    with pytest.raises(ToolExecutionError) as excinfo:
        await run("get_story_overview", {}, runtime())
    assert excinfo.value.code is ErrorCode.INTERNAL_ERROR


# --- the result ceiling ------------------------------------------------------
#
# story-data accepts 200 relationships per character at 20 000 prose characters
# each, 200 chapters at 500-character titles, and a 10 000-character selection
# (internal/store/validate.go, assistant/protocol.py). Every one of those is a
# legal record, so the ceiling has to hold against them rather than against
# tidy fixtures.


async def test_get_story_overview_respects_the_ceiling(fake):
    """A full 50-chapter story must fit without concealing truncation."""
    for i in range(48):
        fake.seed_chapter(STORY, f"big-{i}", title="C" * 500)
    fake.stories[STORY]["description"] = "d" * 5_000
    result = await run("get_story_overview", {}, runtime())
    assert len(json.dumps(result.result)) <= DEFAULT_MAX_TOOL_RESULT_CHARS
    assert result.truncated is True
    assert result.result["chapter_count"] == 50
    assert result.result["chapters"]


async def test_list_story_entities_respects_the_ceiling(fake):
    for i in range(20):
        fake.seed_entity(
            STORY, "places", f"p{i}", name="P" * 200, description="d" * 20_000
        )
    result = await run("list_story_entities", {"kind": "place", "limit": 20}, runtime())
    assert len(json.dumps(result.result)) <= DEFAULT_MAX_TOOL_RESULT_CHARS
    assert result.result["truncated"] is True


async def test_get_story_entity_respects_the_ceiling(fake):
    fake.seed_entity(
        STORY,
        "characters",
        "char-big",
        name="Mina",
        backstory="b" * 20_000,
        relationships=[
            {"type": "ally", "description": "x" * 20_000} for _ in range(200)
        ],
    )
    result = await run(
        "get_story_entity", {"kind": "character", "entityId": "char-big"}, runtime()
    )
    assert len(json.dumps(result.result)) <= DEFAULT_MAX_TOOL_RESULT_CHARS
    assert result.truncated is True


async def test_read_current_editor_respects_the_ceiling(fake):
    """MAX_SELECTION_CHARS is 10 000, which is above the result ceiling."""
    editor = EditorContext(
        selection=Selection(**{"from": 0, "to": 10_000, "text": "s" * 10_000})
    )
    result = await run("read_current_editor", {}, runtime(editor_context=editor))
    assert len(json.dumps(result.result)) <= DEFAULT_MAX_TOOL_RESULT_CHARS
    assert result.truncated is True


# --- citations ---------------------------------------------------------------


async def test_get_story_entity_emits_a_citation(fake):
    """The phase gate wants character/place/plot answers to carry references.

    search_story cites the chunk it retrieved, but a model that goes straight to
    the structured record -- the obvious move for "what is Mina's personality?"
    -- would otherwise assert a fact with nothing to click.
    """
    result = await run(
        "get_story_entity", {"kind": "character", "entityId": "char-1"}, runtime()
    )
    (reference,) = result.references
    assert reference.source_id == "char-1"
    assert reference.kind == "story"
    assert reference.title == "Character: Mina"
    assert reference.snippet == "Guarded."
