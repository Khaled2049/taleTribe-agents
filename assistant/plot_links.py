"""Plot event requirements as the model should read them: by name, not id.

story-data stores an event's prerequisites as `dependencies` of event ids and
derives the reverse as `dependents`. Ids mean nothing in a prompt, so readers
swap both for `requires`: the names of the events this one needs first.
"""

from __future__ import annotations

from typing import Any


def event_names(plots: list[dict[str, Any]]) -> dict[str, str]:
    """Every event id in the story mapped to its name."""
    return {
        str(event["id"]): str(event.get("name") or "Untitled event")
        for line in plots
        if isinstance(line, dict)
        for event in line.get("events") or []
        if isinstance(event, dict) and event.get("id")
    }


def name_requirements(events: list[Any], names: dict[str, str]) -> list[dict[str, Any]]:
    """Copies of `events` with `requires` (names) in place of the id links."""
    out = []
    for event in events:
        if not isinstance(event, dict):
            continue
        requires = [
            names[str(link.get("eventId"))]
            for link in event.get("dependencies") or []
            if isinstance(link, dict) and str(link.get("eventId")) in names
        ]
        named = {
            key: value
            for key, value in event.items()
            if key not in ("dependencies", "dependents")
        }
        if requires:
            named["requires"] = requires
        out.append(named)
    return out


def name_plot_requirements(plots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plot lines whose events name their requirements, across lines."""
    names = event_names(plots)
    return [
        {**line, "events": name_requirements(line.get("events") or [], names)}
        for line in plots
    ]
