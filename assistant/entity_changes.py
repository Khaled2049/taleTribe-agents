"""Story-change proposals: binding a model's draft to the story as it is now.

The model writes *what* should change. Everything that says *which row, at
which revision* is read here from story-data, inside the run's server-owned
scope, so a proposal can never name an entity outside the story or carry a
revision the model made up. Nothing in this module writes: an approved proposal
is applied by the browser with the writer's own token and ``If-Match``.

A draft the server cannot bind is the model's recoverable mistake, so it comes
back as ``ProposalRejected`` and the loop hands the reason to the model as a
tool result rather than failing the run.
"""

from __future__ import annotations

from typing import Any, Optional

from assistant.protocol import (
    MAX_ENTITY_NAME_CHARS,
    EntityContinuation,
    ProposeStoryChangesArgs,
    StoryChange,
    StoryChangeDraft,
)
from assistant.tools import ProposeStoryChangesDraft, ToolContext
from mcp_server import data, story_data

COLLECTION_BY_KIND = {"character": "characters", "place": "places", "plot": "plots"}
MAX_LISTED_NAMES = 12


class ProposalRejected(Exception):
    """The draft cannot be bound. The message is safe to show the model."""


def _label(value: Any) -> str:
    text = str(value or "").strip() or "Unnamed"
    return text[:MAX_ENTITY_NAME_CHARS]


def _revision(record: dict[str, Any]) -> int:
    revision = record.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise story_data.StoryDataError("entity has no revision")
    return revision


class _StoryReader:
    """Per-proposal read cache, so five changes do not mean five roster reads."""

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._client = story_data.client()
        self._rosters: dict[str, list[dict[str, Any]]] = {}
        self._plots: dict[str, Optional[dict[str, Any]]] = {}

    async def roster(self, collection: str) -> list[dict[str, Any]]:
        if collection not in self._rosters:
            rows = await self._client.list_entities(
                self._ctx.user_id, self._ctx.story_id, collection
            )
            self._rosters[collection] = [row for row in rows if isinstance(row, dict)]
        return self._rosters[collection]

    async def entity(self, collection: str, entity_id: str) -> Optional[dict[str, Any]]:
        try:
            return await self._client.get_entity(
                self._ctx.user_id, self._ctx.story_id, collection, entity_id
            )
        except story_data.NotFound:
            return None

    async def plot(self, plot_line_id: str) -> Optional[dict[str, Any]]:
        if plot_line_id not in self._plots:
            self._plots[plot_line_id] = await self.entity("plots", plot_line_id)
        return self._plots[plot_line_id]


async def _check_event_references(
    change: StoryChangeDraft, reader: _StoryReader
) -> None:
    for collection, ids in (
        ("characters", change.fields.character_ids or []),
        ("places", [change.fields.location_id] if change.fields.location_id else []),
    ):
        if not ids:
            continue
        known = {str(row.get("id")) for row in await reader.roster(collection)}
        missing = [entity_id for entity_id in ids if entity_id not in known]
        if missing:
            raise ProposalRejected(
                f"Unknown {collection} id(s): {', '.join(missing)}. An event can "
                "only reference entities that already exist; create them in an "
                "earlier proposal."
            )


def _find(rows: list[dict[str, Any]], ref: str) -> list[dict[str, Any]]:
    """Rows a reference could mean: an exact id, else a case-insensitive name."""
    by_id = [row for row in rows if str(row.get("id")) == ref]
    if by_id:
        return by_id
    wanted = ref.strip().casefold()
    return [
        row for row in rows if str(row.get("name") or "").strip().casefold() == wanted
    ]


def _names(rows: list[dict[str, Any]]) -> str:
    return ", ".join(_label(row.get("name")) for row in rows[:MAX_LISTED_NAMES])


def _one(rows: list[dict[str, Any]], ref: str, noun: str) -> Optional[dict[str, Any]]:
    matches = _find(rows, ref)
    if len(matches) > 1:
        raise ProposalRejected(
            f"More than one {noun} is called {ref!r}. Use an id: "
            + ", ".join(str(row.get("id")) for row in matches[:MAX_LISTED_NAMES])
            + "."
        )
    return matches[0] if matches else None


def _plot_holding_event(
    plots: list[dict[str, Any]], ref: str
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """The (plot line, event) a reference names, when exactly one event matches."""
    found = [
        (line, event)
        for line in plots
        for event in _find(
            [e for e in line.get("events") or [] if isinstance(e, dict)], ref
        )
    ]
    return found[0] if len(found) == 1 else None


async def _bind_change(change: StoryChangeDraft, reader: _StoryReader) -> StoryChange:
    """Resolve one change's targets.

    A reference may be an id or an exact name. The model works from a roster of
    names and from search hits whose id is the *event's*, so insisting on the
    right kind of id sent it in circles; the server knows which row was meant.
    """
    base_revision: Optional[int] = None
    label = _label(change.fields.name)
    entity_id = change.entity_id
    plot_line_id = change.plot_line_id

    if change.kind == "event":
        assert plot_line_id is not None  # enforced by the schema
        plots = await reader.roster("plots")
        plot = _one(plots, plot_line_id, "plot line")
        held = None
        if not change.is_create:
            assert entity_id is not None  # enforced by the schema
            held = _plot_holding_event(plots, entity_id)
            # An event names its own plot line, so a wrong plotLineId is fixable.
            if plot is None and held is not None:
                plot = held[0]
        if plot is None:
            raise ProposalRejected(
                f"No plot line is called {plot_line_id!r}. This story's plot "
                f"lines: {_names(plots) or 'none yet'}."
            )
        plot_line_id = str(plot.get("id"))
        await _check_event_references(change, reader)
        events = [e for e in plot.get("events") or [] if isinstance(e, dict)]
        if change.is_create:
            if any(
                str(event.get("name") or "").strip().casefold() == label.casefold()
                for event in events
            ):
                raise ProposalRejected(
                    f"An event named {label!r} already exists in plot line "
                    f"{_label(plot.get('name'))!r}. Propose an update to it instead."
                )
        else:
            assert entity_id is not None
            event = _one(events, entity_id, "event")
            if event is None:
                raise ProposalRejected(
                    f"No event is called {entity_id!r} in plot line "
                    f"{_label(plot.get('name'))!r}. Its events: "
                    f"{_names(events) or 'none yet'}."
                )
            entity_id = str(event.get("id"))
            base_revision = _revision(event)
            label = _label(event.get("name"))
    else:
        collection = COLLECTION_BY_KIND[change.kind]
        rows = await reader.roster(collection)
        if change.is_create:
            wanted = label.casefold()
            for row in rows:
                if str(row.get("name") or "").strip().casefold() == wanted:
                    raise ProposalRejected(
                        f"A {change.kind} named {label!r} already exists. "
                        "Propose an update to it instead."
                    )
        else:
            assert entity_id is not None  # enforced by the schema
            noun = "plot line" if change.kind == "plot" else change.kind
            current = _one(rows, entity_id, noun)
            if current is None:
                hint = ""
                if change.kind == "plot":
                    held = _plot_holding_event(rows, entity_id)
                    if held is not None:
                        hint = (
                            f" That is the event {_label(held[1].get('name'))!r} in "
                            f"plot line {_label(held[0].get('name'))!r}: use "
                            "event.update for the event, or name the plot line."
                        )
                raise ProposalRejected(
                    f"No {noun} is called {entity_id!r}.{hint} This story's "
                    f"{noun}s: {_names(rows) or 'none yet'}."
                )
            entity_id = str(current.get("id"))
            base_revision = _revision(current)
            label = _label(current.get("name"))

    return StoryChange(
        operation=change.operation,
        entity_id=entity_id,
        plot_line_id=plot_line_id,
        fields=change.fields,
        base_revision=base_revision,
        label=label,
    )


async def bind_story_changes(
    draft: ProposeStoryChangesDraft, ctx: ToolContext
) -> ProposeStoryChangesArgs:
    """Resolve every target against story-data and attach its current revision."""
    await data.get_owned_story(ctx.story_id, ctx.user_id)
    reader = _StoryReader(ctx)

    changes = [await _bind_change(change, reader) for change in draft.changes]

    # Checked on resolved ids, so a name and an id for one row still collide.
    created: set[tuple[str, str, str]] = set()
    targets: set[tuple[str, str]] = set()
    for bound in changes:
        if bound.is_create:
            create_key = (bound.kind, bound.plot_line_id or "", bound.label.casefold())
            if create_key in created:
                scope = " in the same plot line" if bound.kind == "event" else ""
                raise ProposalRejected(
                    f"The proposal creates two {bound.kind}s with the same name{scope}."
                )
            created.add(create_key)
            continue
        key = (bound.kind, str(bound.entity_id))
        if key in targets:
            # The first update moves the revision, so the second would be stale.
            raise ProposalRejected(
                f"The proposal updates {bound.kind} {bound.label!r} twice. "
                "Combine the fields into one change."
            )
        targets.add(key)

    return ProposeStoryChangesArgs(
        summary=draft.summary, reason=draft.reason, changes=changes
    )


def validate_entity_continuation(
    continuation: EntityContinuation, *, expected_proposal_id: str
) -> None:
    """Check the decision and the per-change results agree with each other."""
    if continuation.proposal_id != expected_proposal_id:
        raise ValueError("proposal linkage is invalid")
    results = continuation.results
    count = len(continuation.proposal.changes)
    if continuation.decision in ("applied", "apply_failed"):
        if results is None or [item.index for item in results] != list(range(count)):
            raise ValueError("apply continuations report every change in order")
        all_applied = all(item.status == "applied" for item in results)
        if all_applied != (continuation.decision == "applied"):
            raise ValueError("decision does not match the reported results")
    elif results is not None:
        raise ValueError("non-apply continuation cannot carry results")
    if continuation.decision == "revision_requested" and not continuation.feedback:
        raise ValueError("revision feedback is required")


def _noun(change: StoryChange) -> str:
    kind = "plot line" if change.kind == "plot" else change.kind
    return f"{kind} {change.label}"


def describe_change(change: StoryChange) -> str:
    verb = "created" if change.is_create else "updated"
    return f"{verb} {_noun(change)}"


def resolution_text(continuation: EntityContinuation) -> str:
    """The persisted reply for a settled approval.

    History replays text parts only, so this sentence is how the next turn
    knows what was proposed and what actually landed.
    """
    changes = continuation.proposal.changes
    if continuation.decision == "rejected":
        return "Left the story unchanged."
    results = continuation.results or []
    applied = [
        describe_change(changes[item.index])
        for item in results
        if item.status == "applied"
    ]
    if continuation.decision == "applied":
        return f"Saved: {'; '.join(applied)}."
    reasons = {
        "stale": "it changed since I proposed this",
        "failed": "the save failed",
        "skipped": "an earlier change did not save",
    }
    missed = [
        f"{_noun(changes[item.index])} ({reasons[item.status]})"
        for item in results
        if item.status != "applied"
    ]
    saved = f"Saved: {'; '.join(applied)}. " if applied else ""
    return f"{saved}Not saved: {'; '.join(missed)}."
