"""Chunk retrieval: the projection, and the jsonb decode that sits under it.

No database here. What is worth pinning is that search_chunks keeps metadata in
its own key so a metadata entry can never shadow a column, and the decode, which
is where retrieval had been quietly failing.
"""

import re

from agents.storyAgent.postgres_context import (
    MAX_SLIM_CONTEXT_CHARS,
    MAX_SLIM_DESCRIPTION_CHARS,
    MAX_SLIM_LABEL_CHARS,
    SLIM_ROSTER_LIMIT,
    PostgresStoryContext,
    _jsonb,
)

_MORE = re.compile(r" \(\+\d+ more\)$")


def test_jsonb_decodes_the_text_asyncpg_actually_returns():
    """No codec is registered on the pool, so jsonb arrives as a string."""
    assert _jsonb('{"title": "The Lamp Room", "chapterNumber": 1}') == {
        "title": "The Lamp Room",
        "chapterNumber": 1,
    }
    assert _jsonb({"title": "already decoded"}) == {"title": "already decoded"}


def test_jsonb_is_total_over_the_shapes_a_row_can_carry():
    """A malformed value must degrade to an unlabelled chunk, not kill the read."""
    for value in (None, "", "not json", "[1,2]", b'{"a": 1}', 7):
        assert isinstance(_jsonb(value), dict)
    assert _jsonb(b'{"a": 1}') == {"a": 1}


def test_slim_context_bounds_roster_labels_description_and_total_size():
    huge = "x" * (MAX_SLIM_CONTEXT_CHARS * 2)
    context = PostgresStoryContext.format_slim_context(
        {
            "story": {"title": huge, "description": huge},
            "characters": [{"name": huge} for _ in range(20)],
            "places": [{"name": huge} for _ in range(20)],
            "plots": [{"name": huge} for _ in range(20)],
            "chapters": [{"title": huge} for _ in range(20)],
        }
    )

    assert len(context) <= MAX_SLIM_CONTEXT_CHARS
    lines = context.splitlines()
    assert len(lines[0].removeprefix("Story: ")) <= MAX_SLIM_LABEL_CHARS
    assert len(lines[1].removeprefix("Description: ")) <= MAX_SLIM_DESCRIPTION_CHARS
    for roster in lines[2:]:
        names = _MORE.sub("", roster.split(": ", 1)[1])
        for label in names.split(", "):
            assert len(label) <= MAX_SLIM_LABEL_CHARS
    assert context.count("…") >= 2


def test_slim_roster_marks_how_many_entities_it_left_out():
    """A prefix must say it is one.

    Twelve names with nothing after them read as the complete cast, and that is
    how the assistant comes to answer "there is no such character" about the
    thirteenth -- it has the tool to check and no reason to call it.
    """
    context = PostgresStoryContext.format_slim_context(
        {
            "story": {"title": "Saltmarsh", "description": "Brine."},
            "characters": [{"name": f"C{i}"} for i in range(SLIM_ROSTER_LIMIT)],
            "places": [{"name": "Lamp Room"}],
            "plots": [],
            "chapters": [{"title": "One"}],
            "totals": {"characters": 50, "places": 1, "plots": 0, "chapters": 1},
        }
    )

    assert "(+38 more)" in context
    assert context.count("more)") == 1
    assert "Plot lines" not in context


def test_slim_roster_totals_are_optional():
    """A caller holding a whole collection should not have to count it twice."""
    context = PostgresStoryContext.format_slim_context(
        {
            "story": {"title": "Saltmarsh", "description": "Brine."},
            "characters": [{"name": "Mina"}],
            "places": [],
            "plots": [],
            "chapters": [],
        }
    )

    assert "Characters: Mina" in context
    assert "more)" not in context
