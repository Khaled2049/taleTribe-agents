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

from typing import Any, Optional

from assistant.executors import _fit
from assistant.specialists.base import Specialist
from assistant.tools import FocusRef, ToolContext
from mcp_server import data, story_data

COLLECTION_BY_KIND = {"character": "characters", "place": "places", "plot": "plots"}

# A focused chapter contributes a window, never the whole manuscript.
CHAPTER_WINDOW_CHARS = 6_000
MAX_LISTED_NAMES = 12

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

    async def resolve(self, ref: FocusRef) -> Optional[dict[str, Any]]:
        """The one record a reference names, or None when it names nothing."""
        if ref.kind == "chapter":
            rows = await self.chapter_index()
            matches = _match(rows, ref.ref, "title") or [
                row for row in rows if row["number"] == ref.ref.strip()
            ]
            label = "title"
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


async def build_context(
    specialist: Specialist, focus: list[FocusRef], ctx: ToolContext
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

    for kind in specialist.required_focus:
        if focused.get(kind):
            continue
        # Tell the director what does exist, so it can retry without a lookup.
        known = _known(await story.roster(COLLECTION_BY_KIND[kind]))
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
    else:
        context["plotLines"] = _clean(await story.roster("plots"))
        context["characters"] = [
            _brief_character(row) for row in await story.roster("characters")
        ]
        if focused.get("character"):
            context["focusCharacters"] = _clean(focused["character"])
        if focused.get("plot"):
            context["focusPlotLines"] = [str(row.get("id")) for row in focused["plot"]]

    for kind, key in (("place", "places"), ("chapter", "chapters")):
        if focused.get(kind):
            context[key] = _clean(focused[kind])

    fitted, _ = _fit(context, specialist.context_chars)
    return fitted if isinstance(fitted, dict) else {"truncated": True}


def context_size(context: Optional[dict[str, Any]]) -> int:
    from assistant.executors import _encode

    return len(_encode(context)) if context is not None else 0
