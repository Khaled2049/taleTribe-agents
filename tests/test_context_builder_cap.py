"""Tests for bounding the number of entities rendered into a full generation
prompt (ENTITY_CONTEXT_LIMIT). Large stories accumulate hundreds of entities;
format_context_for_prompt keeps only the most-recently-updated N per kind and
notes the remainder, so a generate prompt doesn't grow without bound."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from agents.storyAgent import context_builder  # noqa: E402
from agents.storyAgent.context_builder import (  # noqa: E402
    ENTITY_CONTEXT_LIMIT,
    StoryContextBuilder,
)


class _FakeFirestoreClient:
    def __init__(self, project="test-project"):
        self.project = project


def _make_builder(project="test-project"):
    with patch.object(
        context_builder.firestore, "Client", return_value=_FakeFirestoreClient(project)
    ):
        return StoryContextBuilder(project_id=project)


def _characters(n, base_time=None):
    """n characters with strictly increasing updatedAt (char 0 oldest).

    Zero-padded names (Char000, Char001, ...) so substring assertions don't
    collide (e.g. "Char1" would otherwise match "Char14")."""
    base = base_time or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {"name": f"Char{i:03d}", "updatedAt": base + timedelta(hours=i)}
        for i in range(n)
    ]


def _context(characters=None, places=None, plots=None):
    return {
        "story": {"id": "s1", "title": "Test", "genre": "SF", "tone": "dark"},
        "characters": characters or [],
        "places": places or [],
        "plots": plots or [],
        "chapters": [],
    }


def test_under_limit_renders_all_and_no_more_line():
    builder = _make_builder()
    out = builder.format_context_for_prompt(_context(characters=_characters(3)))
    for i in range(3):
        assert f"Char{i:03d}" in out
    assert "more characters not shown" not in out


def test_over_limit_caps_and_notes_remainder():
    builder = _make_builder()
    total = ENTITY_CONTEXT_LIMIT + 5
    out = builder.format_context_for_prompt(_context(characters=_characters(total)))
    assert "... and 5 more characters not shown" in out


def test_keeps_most_recently_updated():
    builder = _make_builder()
    total = ENTITY_CONTEXT_LIMIT + 3
    chars = _characters(total)  # Char000 oldest ... Char{total-1} newest
    out = builder.format_context_for_prompt(_context(characters=chars))
    # The three oldest must be dropped; the newest must survive.
    assert "Char000" not in out
    assert "Char001" not in out
    assert "Char002" not in out
    assert f"Char{total - 1:03d}" in out


def test_entity_missing_timestamp_sorts_oldest():
    builder = _make_builder()
    # One character has no updatedAt; fill the rest so we exceed the limit by 1.
    dated = _characters(ENTITY_CONTEXT_LIMIT)
    undated = {"name": "NoTimestamp"}
    out = builder.format_context_for_prompt(_context(characters=[undated] + dated))
    # The undated one is the single overflow entry that gets dropped.
    assert "NoTimestamp" not in out
    assert "... and 1 more characters not shown" in out


def test_falls_back_to_created_at_when_no_updated_at():
    builder = _make_builder()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    newest = {"name": "Newest", "createdAt": base + timedelta(days=10)}
    fillers = _characters(ENTITY_CONTEXT_LIMIT)  # all in early Jan
    out = builder.format_context_for_prompt(_context(characters=fillers + [newest]))
    # Newest (by createdAt) survives even though it has no updatedAt.
    assert "Newest" in out
