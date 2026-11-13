"""Plot events carry their chapter by id; context adds its number and title.

Pure functions only. The number comes from reading order, since story-data
stores no chapter number, so reordering chapters renumbers events with them.
"""

import pytest

from agents.storyAgent.postgres_context import _chapter_labels, _event_chapter

pytestmark = pytest.mark.unit

CHAPTERS = [
    {"id": "c-harbour", "title": "The Harbour"},
    {"id": "c-ford", "title": "The Ford"},
]


def test_event_chapter_numbers_follow_reading_order():
    labels = _chapter_labels(CHAPTERS)
    assert _event_chapter("c-ford", labels) == {
        "chapterId": "c-ford",
        "chapterNumber": 2,
        "chapterTitle": "The Ford",
    }
    # Reordering the chapters renumbers the same event without touching its id.
    swapped = _chapter_labels(list(reversed(CHAPTERS)))
    assert _event_chapter("c-ford", swapped)["chapterNumber"] == 1


def test_unassigned_or_unknown_chapter_has_no_label():
    labels = _chapter_labels(CHAPTERS)
    empty = {"chapterNumber": None, "chapterTitle": None}
    assert _event_chapter(None, labels) == {"chapterId": None, **empty}
    assert _event_chapter("c-gone", labels) == {"chapterId": "c-gone", **empty}
