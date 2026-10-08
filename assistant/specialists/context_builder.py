"""Build what a specialist reads, on the server, before it is asked anything.

The alternative -- hand a specialist the read tools and let it fetch -- costs a
model call per lookup and lets each specialist wander the whole story. A
deterministic builder makes a consult one model call, and makes "which context
did it see" a property of the code rather than of a transcript: the Character
Editor gets the character, the people it is tied to and the events it appears
in, and never the rest of the manuscript.

Everything is read through the same owner-gated story-data paths the read tools
use, inside the run's server-owned scope. The model supplies focus ids; an id
that does not resolve in this story is a rejected consult, not a wider read.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from assistant.executors import _encode, _fit
from assistant.specialists.base import Specialist
from assistant.tools import FocusRef, ToolContext
from mcp_server import data, story_data

COLLECTION_BY_KIND = {"character": "characters", "place": "places", "plot": "plots"}

# A focused chapter contributes a window, never the whole manuscript.
CHAPTER_WINDOW_CHARS = 6_000
MAX_LISTED_NAMES = 12
MAX_INFERRED_FOCUS = 2
MAX_CAST_WITHOUT_FOCUS = 8

# Fields that are noise to a specialist: media, bookkeeping, back-references.
_DROPPED = frozenset(
    {
        "artUrl",
        "imageUrl",
        "userId",
        "storyId",
        "createdAt",
        "updatedAt",
        "dependents",
        "revision",
    }
)


class ConsultRejected(Exception):
    """The consult cannot be built. The message is safe to show the director."""


def _clean(record: Any) -> Any:
    if isinstance(record, list):
        return [_clean(item) for item in record]
    if not isinstance(record, dict):
        return record
    return {
        key: _clean(value)
        for key, value in record.items()
        if key not in _DROPPED and value not in (None, "", [], {})
    }


def _brief_character(record: dict[str, Any]) -> dict[str, Any]:
    return _clean(
        {
            "id": record.get("id"),
            "name": record.get("name"),
            "soul": record.get("soul"),
            "personality": record.get("personality"),
            "affiliations": record.get("affiliations"),
        }
    )


def _known(rows: list[dict[str, Any]], label: str = "name") -> str:
    names = [str(row.get(label) or "").strip() for row in rows]
    listed = ", ".join(name for name in names[:MAX_LISTED_NAMES] if name)
    more = "" if len(names) <= MAX_LISTED_NAMES else ", and others"
    return f"{listed}{more}"


def _match(rows: list[dict[str, Any]], ref: str, label: str) -> list[dict[str, Any]]:
    """Rows a reference could mean: an exact id, else a case-insensitive name."""
    by_id = [row for row in rows if str(row.get("id")) == ref]
    if by_id:
        return by_id
    wanted = ref.strip().casefold()
    return [
        row for row in rows if str(row.get(label) or "").strip().casefold() == wanted
    ]


def _named_in(brief: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entities a brief refers to: by full name, else by an unambiguous first name."""
    words = set(re.findall(r"[\w'-]+", brief.casefold()))
    text = brief.casefold()
    full = [
        row
        for row in rows
        if (name := str(row.get("name") or "").strip().casefold()) and name in text
    ]
    if full:
        return full[:MAX_INFERRED_FOCUS]
    firsts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tokens = str(row.get("name") or "").casefold().split()
        if tokens and len(tokens[0]) >= 3:
            firsts.setdefault(tokens[0], []).append(row)
    return [
        matches[0]
        for first, matches in firsts.items()
        if first in words and len(matches) == 1
    ][:MAX_INFERRED_FOCUS]


class _Story:
    """Per-consult read cache over story-data, and the only id resolver.

    The director sees a roster of names, not ids, so a reference may be either.
    Resolving here rather than asking the model to look ids up saves a model
    call per consult and removes the ids a small model would otherwise invent.
    """

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._client = story_data.client()
        self._rosters: dict[str, list[dict[str, Any]]] = {}
        self._chapters: Optional[list[dict[str, Any]]] = None
        # References that named nothing, with what the story does have.
        self.unresolved: list[tuple[FocusRef, str]] = []

    async def roster(self, collection: str) -> list[dict[str, Any]]:
        if collection not in self._rosters:
            rows = await self._client.list_entities(
                self._ctx.user_id, self._ctx.story_id, collection
            )
            self._rosters[collection] = [row for row in rows if isinstance(row, dict)]
        return self._rosters[collection]

    async def chapter_index(self) -> list[dict[str, Any]]:
        if self._chapters is None:
            rows = await self._client.list_chapter_index(
                self._ctx.user_id, self._ctx.story_id
            )
            ordered = sorted(
                (row for row in rows if isinstance(row, dict)),
                key=lambda row: float(row.get("position") or 0),
            )
            self._chapters = [
                {**row, "number": str(number)}
                for number, row in enumerate(ordered, start=1)
            ]
        return self._chapters

    async def events(self) -> list[dict[str, Any]]:
        """Every event, tagged with the plot line it belongs to."""
        return [
            {**event, "plotLineId": line.get("id"), "plotLine": line.get("name")}
            for line in await self.roster("plots")
            for event in line.get("events") or []
            if isinstance(event, dict)
        ]

    async def known(self, kind: str) -> str:
        if kind == "chapter":
            return _known(await self.chapter_index(), "title")
        if kind == "event":
            return _known(await self.events())
        return _known(await self.roster(COLLECTION_BY_KIND[kind]))

    async def resolve(self, ref: FocusRef) -> Optional[dict[str, Any]]:
        """The one record a reference names, or None when it names nothing."""
        if ref.kind == "chapter":
            rows = await self.chapter_index()
            matches = _match(rows, ref.ref, "title") or [
                row for row in rows if row["number"] == ref.ref.strip()
            ]
            label = "title"
        elif ref.kind == "event":
            rows = await self.events()
            matches = _match(rows, ref.ref, "name")
            label = "name"
        else:
            rows = await self.roster(COLLECTION_BY_KIND[ref.kind])
            matches = _match(rows, ref.ref, "name")
            label = "name"
        if len(matches) > 1:
            raise ConsultRejected(
                f"More than one {ref.kind} is called {ref.ref!r}. Use an id: "
                + ", ".join(str(row.get("id")) for row in matches[:MAX_LISTED_NAMES])
                + "."
            )
        if not matches:
            self.unresolved.append((ref, _known(rows, label)))
            return None
        return matches[0]

    async def chapter(self, chapter_id: str) -> dict[str, Any]:
        window = await data.get_chapter(
            self._ctx.story_id,
            chapter_id,
            self._ctx.user_id,
            0,
            CHAPTER_WINDOW_CHARS,
        )
        return {
            "id": window.get("chapter_id"),
            "title": window.get("title"),
            "chapterNumber": window.get("chapter_number"),
            "text": window.get("content"),
            "truncated": window.get("next_offset") is not None,
        }


async def _story_header(ctx: ToolContext) -> dict[str, Any]:
    overview = await data.get_story_overview(ctx.story_id, ctx.user_id)
    return _clean(
        {
            "title": overview.get("title"),
            "description": overview.get("description"),
            "category": overview.get("category"),
            "tags": overview.get("tags"),
            "chapters": [
                {"number": chapter.get("chapter_number"), "title": chapter.get("title")}
                for chapter in overview.get("chapters") or []
            ],
        }
    )


def _events_with(plots: list[dict[str, Any]], character_id: str) -> list[dict]:
    found = []
    for line in plots:
        for event in line.get("events") or []:
            if isinstance(event, dict) and character_id in (
                event.get("characterIds") or []
            ):
                found.append(
                    {
                        **event,
                        "plotLineId": line.get("id"),
                        "plotLine": line.get("name"),
                    }
                )
    return found


async def _scene_material(
    story: _Story, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Who is in the focused events, where they happen, and what came before."""
    characters = {str(row.get("id")): row for row in await story.roster("characters")}
    places = {str(row.get("id")): row for row in await story.roster("places")}
    every_event = await story.events()
    cast: dict[str, dict[str, Any]] = {}
    settings: dict[str, dict[str, Any]] = {}
    before: dict[str, dict[str, Any]] = {}
    focus_ids = {str(event.get("id")) for event in events}
    for event in events:
        for character_id in event.get("characterIds") or []:
            if str(character_id) in characters:
                cast[str(character_id)] = characters[str(character_id)]
        if str(event.get("locationId")) in places:
            settings[str(event["locationId"])] = places[str(event["locationId"])]
        # The event just before it on the same line, for continuity.
        earlier = [
            other
            for other in every_event
            if other.get("plotLineId") == event.get("plotLineId")
            and (other.get("orderIndex") or 0) < (event.get("orderIndex") or 0)
            and str(other.get("id")) not in focus_ids
        ]
        if earlier:
            previous = max(earlier, key=lambda other: other.get("orderIndex") or 0)
            before[str(previous.get("id"))] = previous
    return _clean(
        {
            "events": events,
            "charactersInScene": list(cast.values()),
            "setting": list(settings.values()),
            "whatCameBefore": list(before.values()),
        }
    )


def _clip_prose(node: Any, width: int, key: str = "") -> Any:
    """Shorten prose without changing record identities or dropping focus rows."""
    if isinstance(node, dict):
        return {k: _clip_prose(v, width, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_clip_prose(item, width, key) for item in node]
    if (
        isinstance(node, str)
        and key
        not in {"id", "name", "title", "specialist", "plotLine", "operation", "target"}
        and not key.endswith(("Id", "Ids"))
    ):
        return node if len(node) <= width else node[:width] + "…"
    return node


def _fit_context(context: dict[str, Any], cap: int, character_editor: bool) -> dict:
    if len(_encode(context)) <= cap:
        return context

    # Spend on explicit focus and colleagues' reasoning before broad rosters.
    keys = {
        "focusCharacters",
        "focusPlotLines",
        "focusEvents",
        "chapters",
        "places",
        "events",
        "charactersInScene",
        "setting",
        "priorFindings",
        "focusNotFound",
    }
    if character_editor:
        keys.add("characters")
    primary = {key: value for key, value in context.items() if key in keys}
    primary["story"] = {"title": context["story"].get("title")}
    primary["truncated"] = True
    background = {key: value for key, value in context.items() if key not in keys}
    background.pop("story")
    background["storyOverview"] = context["story"]

    if len(_encode(primary)) > cap:
        # Keep all focused rows and their names/IDs, even when prose must shrink.
        lo, hi, fitted = 0, cap, None
        while lo <= hi:
            width = (lo + hi) // 2
            candidate = _clip_prose(primary, width)
            if len(_encode(candidate)) <= cap:
                fitted, lo = candidate, width + 1
            else:
                hi = width - 1
        if fitted is None:
            raise ConsultRejected(
                "The focused material is too large for one consult. "
                "Focus on fewer entities or a single event instead of a whole plot."
            )
        primary = fitted

    remaining = cap - len(_encode(primary))
    if remaining >= 128:
        # Two nonempty JSON objects merge with no additional encoded overhead.
        extra, _ = _fit(background, remaining)
        primary = {**extra, **primary}
    return primary


async def build_context(
    specialist: Specialist,
    focus: list[FocusRef],
    ctx: ToolContext,
    prior_findings: Optional[list[dict[str, Any]]] = None,
    brief: str = "",
) -> dict[str, Any]:
    """The bounded story material for one consult, as JSON-serializable data."""
    story = _Story(ctx)
    # The ownership gate: every later read is scoped to a story the caller owns.
    context: dict[str, Any] = {"story": await _story_header(ctx)}

    focused: dict[str, list[dict[str, Any]]] = {}
    for ref in focus:
        record = await story.resolve(ref)
        if record is None:
            continue
        if ref.kind == "chapter":
            record = await story.chapter(str(record.get("id")))
        rows = focused.setdefault(ref.kind, [])
        if all(row.get("id") != record.get("id") for row in rows):
            rows.append(record)

    any_of = specialist.required_any_focus
    if any_of and not any(focused.get(kind) for kind in any_of):
        wanted = " or ".join(
            f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}" for kind in any_of
        )
        named = [ref.ref for ref, _ in story.unresolved if ref.kind in any_of]
        problem = (
            f"Nothing in this story is called {named[0]!r}."
            if named
            else f"The {specialist.name} needs {wanted} in focus."
        )
        listed = []
        for kind in any_of:
            known = await story.known(kind)
            if known:
                listed.append(f"{kind}s: {known}")
        if not listed:
            raise ConsultRejected(
                f"{problem} This story has none recorded yet, so there is "
                f"nothing for the {specialist.name} to work from."
            )
        raise ConsultRejected(f"{problem} This story's {'; '.join(listed)}.")

    for kind in specialist.required_focus:
        if focused.get(kind):
            continue
        # Tell the director what does exist, so it can retry without a lookup.
        known = await story.known(kind)
        named = [ref.ref for ref, _ in story.unresolved if ref.kind == kind]
        if not known:
            raise ConsultRejected(
                f"This story has no {kind}s recorded yet, so there is nothing "
                f"for the {specialist.name} to assess."
            )
        problem = (
            f"No {kind} is called {named[0]!r}."
            if named
            else f"The {specialist.name} needs a {kind} in focus."
        )
        raise ConsultRejected(f"{problem} This story's {kind}s: {known}.")

    # A wrong optional reference narrows the material; it does not stop the consult.
    if story.unresolved:
        context["focusNotFound"] = [
            {"kind": ref.kind, "ref": ref.ref} for ref, _ in story.unresolved
        ]

    if specialist.id == "character_editor":
        plots = await story.roster("plots")
        characters = await story.roster("characters")
        by_id = {str(row.get("id")): row for row in characters}
        # Nobody in focus: use whoever the brief names, else the whole cast.
        # A question about a new character has no existing one to point at.
        chosen = focused.get("character") or _named_in(brief, characters)
        if not chosen:
            chosen = characters[:MAX_CAST_WITHOUT_FOCUS]
            context["castNote"] = (
                "No single character was named, so this is the cast as recorded."
                if chosen
                else "This story has no characters recorded yet."
            )
        focused["character"] = chosen
        focus_ids = {str(row.get("id")) for row in focused["character"]}
        related: dict[str, dict[str, Any]] = {}
        appearances: list[dict[str, Any]] = []
        for character in focused["character"]:
            for relation in character.get("relationships") or []:
                other = by_id.get(str(relation.get("characterId")))
                if other and str(other.get("id")) not in focus_ids:
                    related[str(other.get("id"))] = _brief_character(other)
            appearances.extend(_events_with(plots, str(character.get("id"))))
        context["characters"] = _clean(focused["character"])
        context["relatedCharacters"] = list(related.values())
        context["eventsTheyAppearIn"] = _clean(appearances)
    elif specialist.mode == "draft":
        # Scene material only: the cast and setting of what is being written,
        # not the whole plot the drafter might be tempted to advance.
        context.update(await _scene_material(story, focused.get("event", [])))
        if focused.get("character"):
            context["focusCharacters"] = _clean(focused["character"])
    else:
        context["plotLines"] = _clean(await story.roster("plots"))
        context["characters"] = [
            _brief_character(row) for row in await story.roster("characters")
        ]
        if focused.get("character"):
            context["focusCharacters"] = _clean(focused["character"])
        if focused.get("plot"):
            context["focusPlotLines"] = _clean(focused["plot"])
        if focused.get("event"):
            context["focusEvents"] = _clean(focused["event"])

    for kind, key in (("place", "places"), ("chapter", "chapters")):
        if focused.get(kind):
            context[key] = _clean(focused[kind])

    if prior_findings:
        context["priorFindings"] = prior_findings

    return _fit_context(
        context, specialist.context_chars, specialist.id == "character_editor"
    )


def context_size(context: Optional[dict[str, Any]]) -> int:
    from assistant.executors import _encode

    return len(_encode(context)) if context is not None else 0
