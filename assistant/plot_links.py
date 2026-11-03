"""Plot events as the model should read them: chapters by label, not id."""

from __future__ import annotations

from typing import Any


def chapter_labels(chapters: list[Any]) -> dict[str, str]:
    """Chapter id -> ``3 · The Ford``, numbered in reading (``position``) order."""
    rows = [c for c in chapters if isinstance(c, dict) and c.get("id")]

    def order(chapter: dict[str, Any]) -> tuple[float, str]:
        value = chapter.get("position")
        number = (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else 0.0
        )
        return (number, str(chapter["id"]))

    labels = {}
    for number, chapter in enumerate(sorted(rows, key=order), start=1):
        title = str(chapter.get("title") or "").strip()
        labels[str(chapter["id"])] = f"{number} · {title}" if title else str(number)
    return labels


def label_event_chapters(
    events: list[dict[str, Any]], labels: dict[str, str]
) -> list[dict[str, Any]]:
    """Copies of `events` with `chapter` (``3 · The Ford`` or ``Unassigned``)
    in place of the chapter id and its sort key."""
    out = []
    for event in events:
        chapter_id = event.get("chapterId")
        named = {
            key: value
            for key, value in event.items()
            if key not in ("chapterId", "chapterPosition")
        }
        named["chapter"] = (
            labels.get(str(chapter_id), "Unknown chapter")
            if chapter_id
            else "Unassigned"
        )
        out.append(named)
    return out
