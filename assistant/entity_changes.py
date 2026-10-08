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


async def _bind_change(change: StoryChangeDraft, reader: _StoryReader) -> StoryChange:
    base_revision: Optional[int] = None
    label = _label(change.fields.name)

    if change.kind == "event":
        assert change.plot_line_id is not None  # enforced by the schema
        plot = await reader.plot(change.plot_line_id)
        if plot is None:
            raise ProposalRejected(f"No plot line with id {change.plot_line_id}.")
        await _check_event_references(change, reader)
        if not change.is_create:
            events = plot.get("events") or []
            event = next(
                (
                    row
                    for row in events
                    if isinstance(row, dict) and row.get("id") == change.entity_id
                ),
                None,
            )
            if event is None:
                raise ProposalRejected(
                    f"No event with id {change.entity_id} in that plot line."
                )
            base_revision = _revision(event)
            label = _label(event.get("name"))
    else:
        collection = COLLECTION_BY_KIND[change.kind]
        if change.is_create:
            wanted = label.casefold()
            for row in await reader.roster(collection):
                if str(row.get("name") or "").strip().casefold() == wanted:
                    raise ProposalRejected(
                        f"A {change.kind} named {label!r} already exists with id "
                        f"{row.get('id')}. Propose an update to it instead."
                    )
        else:
            assert change.entity_id is not None  # enforced by the schema
            current = await reader.entity(collection, change.entity_id)
            if current is None:
                raise ProposalRejected(f"No {change.kind} with id {change.entity_id}.")
            base_revision = _revision(current)
            label = _label(current.get("name"))

    return StoryChange(
        operation=change.operation,
        entity_id=change.entity_id,
        plot_line_id=change.plot_line_id,
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

    targets: set[tuple[str, str]] = set()
    created: set[tuple[str, str]] = set()
    for change in draft.changes:
        if change.is_create:
            key = (change.kind, _label(change.fields.name).casefold())
            if change.kind != "event" and key in created:
                raise ProposalRejected(
                    f"The proposal creates two {change.kind}s with the same name."
                )
            created.add(key)
            continue
        assert change.entity_id is not None
        key = (change.kind, change.entity_id)
        if key in targets:
            # The first update moves the revision, so the second would be stale.
            raise ProposalRejected(
                f"The proposal updates {change.kind} {change.entity_id} twice. "
                "Combine the fields into one change."
            )
        targets.add(key)

    changes = [await _bind_change(change, reader) for change in draft.changes]
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
