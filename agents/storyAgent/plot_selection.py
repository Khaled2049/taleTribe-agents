"""Choose which plot material reaches a generation prompt.

The prompt can hold only ENTITY_CONTEXT_LIMIT plot lines, so a story with many
lines used to lose the very events planned for the chapter being written. With
an active chapter, selection starts from that chapter and works outwards:

1. the chapter's own planned events, always, in chapter order;
2. the events those require, from any plot line or chapter;
3. the events planned for the previous and next chapter;
4. plot lines by name and description, those already shown first, then by
   most recent update.

Pure and deterministic: equal input gives equal output, with ties broken by id.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

ENTITY_CONTEXT_LIMIT = 12

Event = Dict[str, Any]
Plot = Dict[str, Any]


def _timestamp(item: Dict[str, Any]) -> float:
    """Epoch seconds from updatedAt (fallback createdAt); missing sorts oldest."""
    for field in ("updatedAt", "createdAt"):
        value = item.get(field)
        if value is None:
            continue
        if hasattr(value, "timestamp"):
            try:
                return value.timestamp()
            except (ValueError, OSError):
                continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return float("-inf")


def recent_plots(plots: List[Plot]) -> List[Plot]:
    """Plot lines, most recently updated first, ties broken by id."""
    return sorted(plots, key=lambda p: (-_timestamp(p), str(p.get("id") or "")))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _chapter_order(chapter: Dict[str, Any]) -> Tuple[float, str]:
    return (_number(chapter.get("order")) or 0.0, str(chapter.get("id") or ""))


def _in_chapter_order(events: List[Event]) -> List[Event]:
    def key(event: Event) -> Tuple[float, float, str]:
        position = _number(event.get("chapterPosition"))
        return (
            position if position is not None else float("inf"),
            _number(event.get("orderIndex")) or 0.0,
            str(event.get("id") or ""),
        )

    return sorted(events, key=key)


def _in_reading_order(events: List[Event]) -> List[Event]:
    """Assigned events by chapter then chapter order; unassigned ones last."""

    def key(event: Event) -> Tuple[float, float, float, str]:
        chapter = _number(event.get("chapterNumber"))
        position = _number(event.get("chapterPosition"))
        return (
            chapter if chapter is not None else float("inf"),
            position if position is not None else float("inf"),
            _number(event.get("orderIndex")) or 0.0,
            str(event.get("id") or ""),
        )

    return sorted(events, key=key)


def select_plot_context(
    context: Dict[str, Any], chapter_id: Optional[str]
) -> List[Dict[str, Any]]:
    """Ordered sections of plot material for the prompt.

    Each section is ``{"kind": ..., ...}``:

    - ``{"kind": "chapter_plan" | "prerequisites" | "previous_chapter" |
      "next_chapter", "chapter": {...} | None, "events": [(plot, event), ...]}``
    - ``{"kind": "plot_lines", "plots": [plot, ...], "with_events": bool,
      "omitted": int}``

    Without a ``chapter_id`` that names a chapter of the story, the result is
    one ``plot_lines`` section listing the most recently updated lines with all
    their events, which is what the prompt has always shown.
    """
    plots = [p for p in context.get("plots") or [] if isinstance(p, dict)]
    chapters = sorted(
        (c for c in context.get("chapters") or [] if isinstance(c, dict)),
        key=_chapter_order,
    )
    index = next(
        (i for i, c in enumerate(chapters) if chapter_id and c.get("id") == chapter_id),
        None,
    )
    if index is None:
        # Within the limit nothing is dropped, so keep the stored order.
        ordered = plots if len(plots) <= ENTITY_CONTEXT_LIMIT else recent_plots(plots)
        return [
            {
                "kind": "plot_lines",
                "plots": ordered[:ENTITY_CONTEXT_LIMIT],
                "with_events": True,
                "omitted": max(0, len(ordered) - ENTITY_CONTEXT_LIMIT),
            }
        ]

    line_of: Dict[str, Plot] = {}
    by_id: Dict[str, Event] = {}
    by_chapter: Dict[str, List[Event]] = {}
    for plot in plots:
        for event in plot.get("events") or []:
            if not isinstance(event, dict) or not event.get("id"):
                continue
            event_id = str(event["id"])
            by_id[event_id] = event
            line_of[event_id] = plot
            if event.get("chapterId"):
                by_chapter.setdefault(str(event["chapterId"]), []).append(event)

    shown: set[str] = set()
    sections: List[Dict[str, Any]] = []

    def take(events: List[Event], limit: Optional[int]) -> List[Tuple[Plot, Event]]:
        picked = []
        for event in events:
            event_id = str(event["id"])
            if event_id in shown:
                continue
            if limit is not None and len(picked) >= limit:
                break
            shown.add(event_id)
            picked.append((line_of[event_id], event))
        return picked

    active = chapters[index]
    planned = take(_in_chapter_order(by_chapter.get(str(active["id"]), [])), None)
    sections.append({"kind": "chapter_plan", "chapter": active, "events": planned})

    required = [
        by_id[req]
        for _, event in planned
        for req in event.get("requiresIds") or []
        if req in by_id
    ]
    prerequisites = take(_in_reading_order(required), ENTITY_CONTEXT_LIMIT)
    if prerequisites:
        sections.append(
            {"kind": "prerequisites", "chapter": None, "events": prerequisites}
        )

    for kind, neighbour in (
        ("previous_chapter", index - 1),
        ("next_chapter", index + 1),
    ):
        if not 0 <= neighbour < len(chapters):
            continue
        chapter = chapters[neighbour]
        events = take(
            _in_chapter_order(by_chapter.get(str(chapter["id"]), [])),
            ENTITY_CONTEXT_LIMIT,
        )
        if events:
            sections.append({"kind": kind, "chapter": chapter, "events": events})

    # Lines already represented come first so their descriptions frame the
    # events above; the rest follow by recency.
    seen_lines: List[Plot] = []
    for section in sections:
        for plot, _ in section["events"]:
            if not any(plot is p for p in seen_lines):
                seen_lines.append(plot)
    rest = [p for p in recent_plots(plots) if not any(p is s for s in seen_lines)]
    lines = seen_lines + rest
    if lines:
        sections.append(
            {
                "kind": "plot_lines",
                "plots": lines[:ENTITY_CONTEXT_LIMIT],
                "with_events": False,
                "omitted": max(0, len(lines) - ENTITY_CONTEXT_LIMIT),
            }
        )
    return sections


def chapter_label(chapter: Dict[str, Any], number: int) -> str:
    title = str(chapter.get("title") or "").strip()
    return f"{number} · {title}" if title else str(number)


def chapter_numbers(context: Dict[str, Any]) -> Dict[str, int]:
    """Chapter id -> 1-based number in reading order."""
    chapters = sorted(
        (c for c in context.get("chapters") or [] if isinstance(c, dict)),
        key=_chapter_order,
    )
    return {str(c.get("id")): n for n, c in enumerate(chapters, start=1)}
