"""Generation prompts are built around the chapter being written."""

import copy

import pytest

from agents.storyAgent.context_format import (
    format_chapter_plan,
    format_context_for_prompt,
)
from agents.storyAgent.plot_selection import ENTITY_CONTEXT_LIMIT, select_plot_context

pytestmark = pytest.mark.unit

CHAPTERS = [
    {"id": "c1", "title": "The Harbour", "order": 1},
    {"id": "c2", "title": "The Ford", "order": 2},
    {"id": "c3", "title": "The Pass", "order": 3},
    {"id": "c4", "title": "The Keep", "order": 4},
]


def _event(event_id, name, chapter=None, position=None, requires=()):
    numbers = {c["id"]: n for n, c in enumerate(CHAPTERS, start=1)}
    return {
        "id": event_id,
        "name": name,
        "content": f"{name} happens.",
        "chapterId": chapter,
        "chapterNumber": numbers.get(chapter),
        "chapterTitle": next(
            (c["title"] for c in CHAPTERS if c["id"] == chapter), None
        ),
        "chapterPosition": position,
        "status": "planned",
        "tension_level": 6,
        "requiresIds": list(requires),
        "requires": [],
    }


def _story(extra_lines=0):
    """The active chapter's line is the oldest, so recency alone would drop it."""
    plots = [
        {
            "id": "mutiny",
            "name": "The Mutiny",
            "description": "Mara turns the crew.",
            "updatedAt": 1,
            "events": [
                _event("e-cross", "Crossing", "c2", 2, requires=["e-letter"]),
                _event("e-confront", "Mara confronts the captain", "c2", 1),
                _event("e-harbour", "Leaving port", "c1", 1),
                _event("e-pass", "Ambush in the pass", "c3", 1),
                _event("e-keep", "The siege", "c4", 1),
            ],
        },
        {
            "id": "letters",
            "name": "Letters",
            "description": "A trail of letters.",
            "updatedAt": 2,
            "events": [_event("e-letter", "Finds the letter")],
        },
    ]
    for n in range(extra_lines):
        plots.append(
            {
                "id": f"line-{n:02d}",
                "name": f"Side plot {n}",
                "updatedAt": 100 + n,
                "events": [_event(f"x-{n}", f"Side event {n}")],
            }
        )
    return {"chapters": CHAPTERS, "plots": plots}


def _names(section):
    return [event["name"] for _, event in section["events"]]


def test_active_chapter_plan_survives_many_plot_lines():
    sections = select_plot_context(_story(extra_lines=30), "c2")
    plan = sections[0]
    assert plan["kind"] == "chapter_plan"
    assert _names(plan) == ["Mara confronts the captain", "Crossing"]

    lines = sections[-1]
    assert lines["kind"] == "plot_lines" and not lines["with_events"]
    assert len(lines["plots"]) == ENTITY_CONTEXT_LIMIT
    # Lines already shown lead, so the active line's description is never lost.
    assert [p["name"] for p in lines["plots"][:2]] == ["The Mutiny", "Letters"]


def test_prerequisites_from_another_plot_line_are_included():
    sections = select_plot_context(_story(), "c2")
    kinds = {s["kind"]: s for s in sections}
    assert _names(kinds["prerequisites"]) == ["Finds the letter"]
    assert _names(kinds["previous_chapter"]) == ["Leaving port"]
    assert _names(kinds["next_chapter"]) == ["Ambush in the pass"]
    shown = [name for s in sections if "events" in s for name in _names(s)]
    assert "The siege" not in shown
    assert len(shown) == len(set(shown))


def test_same_input_gives_same_output():
    story = _story(extra_lines=20)
    for line in story["plots"][2:]:
        line["updatedAt"] = 5  # all tied: order must come from the id
    shuffled = copy.deepcopy(story)
    shuffled["plots"].reverse()
    assert format_context_for_prompt(story, "c2") == format_context_for_prompt(
        shuffled, "c2"
    )
    assert format_context_for_prompt(story) == format_context_for_prompt(
        copy.deepcopy(story)
    )


def test_without_a_known_chapter_the_prompt_lists_plot_lines_with_events():
    for chapter_id in (None, "not-a-chapter"):
        prompt = format_context_for_prompt(_story(), chapter_id)
        assert "=== PLOTS ===" in prompt
        assert "Event - Crossing: Crossing happens. (Chapter 2 · The Ford)" in prompt
        assert "PLAN FOR THIS CHAPTER" not in prompt


def test_prompt_and_enhance_plan_render_the_chapter():
    prompt = format_context_for_prompt(_story(), "c2")
    assert "=== PLAN FOR THIS CHAPTER (2 · The Ford) ===" in prompt
    assert "Plot line: The Mutiny · Status: planned · Tension: 6/10" in prompt

    plan = format_chapter_plan(_story(), "c2")
    assert plan.startswith("Planned for this chapter (2 · The Ford):")
    assert "Finds the letter" in plan and "Ambush in the pass" not in plan
    assert format_chapter_plan(_story(), None) == ""
