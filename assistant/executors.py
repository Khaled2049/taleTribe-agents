"""Executors for the assistant's read tools. Reads only -- nothing here mutates.

Phase 1 shipped the tool *schemas* with no way to run them. This is the other
half, and it is deliberately thin: five of the six read tools are an adapter
over ``mcp_server.data``, which already funnels every read through
``get_owned_story`` and already re-checks ``ownerId == uid`` because story-data
serves a *published* story to any caller. Reimplementing those reads here would
mean maintaining a second ownership gate, and the second one is the one that
rots.

``search_story`` is the exception, and the one place a story-scoped boundary is
re-established by hand: pgvector has no notion of who is asking, so the
``story_id`` in the SQL predicate comes from ``ToolContext`` -- built from the
verified Firebase token and the endpoint's ownership check -- and never from
model output.

Three rules shape everything below.

**Executors are pure functions of (args, runtime).** No event emission, no
orchestrator state, no streaming. References come back as ``SourcePart`` values
in the return, and the loop turns them into ``reference.emitted`` frames. That
is what lets the whole file be tested before the loop exists.

**A result is data, not instructions.** Every executor returns a
JSON-serializable dict. Prose is only ever a *value* inside it, so a chapter
that says "ignore your previous instructions" arrives as a string field rather
than as a line in the prompt.

**Not-found is a result, not a failure.** A model that names a chapter id that
does not exist has made a recoverable mistake, and ``{"found": false}`` lets it
recover on the next turn. ``ToolExecutionError`` is reserved for the server
failing -- story-data unreachable, retrieval unconfigured -- where there is
nothing for the model to do differently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel

from assistant.errors import ErrorCode, safe_message
from assistant.protocol import EditorContext, SourcePart
from assistant.tools import (
    GetStoryEntityArgs,
    GetStoryOverviewArgs,
    ListStoryEntitiesArgs,
    ReadChapterArgs,
    ReadCurrentEditorArgs,
    SearchStoryArgs,
    ToolContext,
    UnknownToolError,
)
from mcp_server import data, story_data

logger = logging.getLogger(__name__)

# Per-result ceiling, before the result re-enters the prompt. This is the bound
# that actually caps a run's cost: read_chapter's schema allows a 20 000-char
# window, and three of those in a growing message list is how a loop with a step
# ceiling still becomes expensive.
DEFAULT_MAX_TOOL_RESULT_CHARS = 8_000

# How much of a source's text is quoted back to the UI as a citation. Well under
# SourcePart's own MAX_PROMPT_CHARS bound: a source is evidence the user can
# click, not a second copy of the manuscript.
MAX_SNIPPET_CHARS = 400

# assistant tools name an entity kind in the singular; story-data (and therefore
# mcp_server.data) addresses the collection. One explicit map rather than an
# ``+ "s"`` that happens to work for these three words.
COLLECTION_BY_ENTITY_KIND = {
    kind: collection for collection, kind in data.ENTITY_KIND_BY_COLLECTION.items()
}


class ToolExecutionError(Exception):
    """The server could not run the tool. Carries a code safe to show a user."""

    def __init__(self, code: ErrorCode) -> None:
        self.code = code
        self.message = safe_message(code)
        super().__init__(code.value)


@dataclass(frozen=True)
class ToolRuntime:
    """What an executor needs beyond its arguments.

    ``ctx`` stays exactly the identity scope Phase 1 defined -- who is asking and
    which story -- and the service handles sit beside it rather than inside it,
    so "never serialized into a prompt" keeps meaning one small, checkable thing.

    Every handle is optional because a run is not entitled to all of them: an
    instance without ``STORY_DATA_DATABASE_URL`` has no retrieval, and a request
    with no open editor has no buffer. Executors turn each absence into an
    explicit answer rather than an AttributeError.

    No story-data client here on purpose. ``mcp_server.data`` reaches the
    process-wide one through ``story_data.client()``, and threading a second
    handle through would mean two ways to reach the same service with only one
    of them carrying the service token. Note that server.py only configures that
    client when MCP is enabled -- widening it is P3-T1's job, not something an
    executor should paper over.
    """

    ctx: ToolContext
    postgres: Optional[Any] = None
    embedder: Optional[Any] = None
    editor_context: Optional[EditorContext] = None
    max_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS


@dataclass(frozen=True)
class ToolResult:
    """What the loop needs back: prompt payload, plus anything worth citing."""

    result: Any
    references: tuple[SourcePart, ...] = ()
    truncated: bool = False


def _snippet(text: Any, limit: int = MAX_SNIPPET_CHARS) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _encode(payload: Any) -> str:
    return json.dumps(payload, default=str)


def _clip(node: Any, max_string: int, max_items: Optional[int]) -> Any:
    """Rebuild a JSON-ish tree with every string and every list bounded."""
    if isinstance(node, str):
        return node if len(node) <= max_string else node[:max_string] + "…"
    if isinstance(node, dict):
        return {k: _clip(v, max_string, max_items) for k, v in node.items()}
    if isinstance(node, list):
        kept = node if max_items is None else node[:max_items]
        return [_clip(v, max_string, max_items) for v in kept]
    return node


# Tried in order once the scaffolding itself is what does not fit. ``None``
# means "keep every element"; the ladder only starts dropping items when empty
# strings alone would still overflow.
_ITEM_LADDER: tuple[Optional[int], ...] = (None, 50, 20, 10, 5, 2, 1, 0)


def _widest_fitting(payload: Any, cap: int, max_items: Optional[int]) -> Optional[Any]:
    """The most text that fits at this list length, or None if even none does."""
    lo, hi, best = 0, cap, None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _clip(payload, mid, max_items)
        if len(_encode(candidate)) <= cap:
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    return best


def _fit(payload: Any, cap: int) -> tuple[Any, bool]:
    """Shrink a result until its JSON encoding fits. Guaranteed, not best-effort.

    The obvious implementation -- halve the longest string, repeat -- does not
    terminate anywhere useful. story-data accepts 200 relationships per character
    at 20 000 prose characters each (``internal/store/validate.go``), and against
    a 4 MB entity a bounded halving loop returns 3.4 MB: it shortens one leaf per
    pass while several hundred others stay untouched. A result ceiling that the
    ceiling function does not enforce is worse than none, because every caller
    then believes it is bounded.

    So: clip *uniformly* and binary search the width. Encoded size is monotone in
    the per-string cap, so the search is sound, and ``max_string=0`` always fits
    once the skeleton does -- which is what makes the result a guarantee rather
    than an attempt. Lists are only shortened when even empty strings overflow,
    since dropping a relationship loses more than shortening one.

    The truncation is marked *inside* the payload as well as returned, because
    the model only ever sees the payload, and a silently short view is how it
    comes to assert that a story does not contain something.
    """
    if len(_encode(payload)) <= cap:
        return payload, False

    # One cheap pass first: no single string can matter beyond the cap itself,
    # and this takes a multi-megabyte tree down to something the search can walk
    # a dozen times without cost. The marker goes on here rather than at the end
    # so that every measurement below includes it -- adding a key after the
    # search has settled is how a "guaranteed" fit lands 18 characters over.
    payload = _mark_truncated(_clip(payload, cap, None))

    for max_items in _ITEM_LADDER:
        if len(_encode(_clip(payload, 0, max_items))) > cap:
            continue  # the scaffolding alone overflows; drop more elements
        fitted = _widest_fitting(payload, cap, max_items)
        if fitted is not None:
            return fitted, True

    # Unreachable for every shape story-data can produce: with no list elements
    # and no string content, what is left is a dozen dict keys. Kept so the
    # function is total rather than total-in-practice.
    return {
        "truncated": True,
        "reason": "result exceeded the tool result ceiling",
    }, True


def _mark_truncated(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload["truncated"] = True
    return payload


def _fit_chapter_window(
    chapter: dict[str, Any], offset: int, cap: int
) -> tuple[dict[str, Any], bool]:
    """Keep the largest content prefix whose serialized result fits ``cap``.

    A raw character allowance is not sufficient here: quotes, backslashes,
    control characters and non-ASCII text expand when JSON-encoded. Paging is
    expressed in source-text offsets, though, so search by prefix length and
    derive ``next_offset`` from the exact prefix the model receives.
    """
    content = chapter.get("content") or ""
    total_chars = int(chapter.get("total_chars") or 0)

    def candidate(length: int) -> dict[str, Any]:
        shown = content[:length]
        end = offset + len(shown)
        return {
            **chapter,
            "content": shown,
            "next_offset": end if end < total_chars else None,
        }

    full = candidate(len(content))
    if len(_encode(full)) <= cap:
        return full, False

    # The encoded size of a non-terminal prefix is monotone: adding source
    # characters can only add JSON characters, while next_offset never shrinks.
    # Checking ``full`` above handles the one terminal discontinuity where the
    # integer next_offset becomes the slightly shorter JSON ``null``.
    empty = candidate(0)
    if len(_encode(empty)) > cap:
        fitted, _ = _fit(empty, cap)
        return fitted, True

    lo, hi, best = 1, len(content) - 1, empty
    while lo <= hi:
        mid = (lo + hi) // 2
        payload = candidate(mid)
        if len(_encode(payload)) <= cap:
            best, lo = payload, mid + 1
        else:
            hi = mid - 1
    return best, True


# Identity and bookkeeping, not evidence -- skipped when picking a citation
# snippet so a source quotes the character rather than their uuid.
_NON_PROSE_FIELDS = frozenset({"entity_id", "name", "updated_at", "kind", "truncated"})


def _first_prose(entity: dict[str, Any]) -> Optional[str]:
    """The first descriptive field, in the projection order get_entity emits.

    Deliberately positional rather than a curated per-kind list: mcp_server.data
    already fixed that order from entity_schema.py, and a second ranking here
    would be one more thing to keep in step with a renamed column.
    """
    for key, value in entity.items():
        if key in _NON_PROSE_FIELDS:
            continue
        snippet = _snippet(value)
        if snippet:
            return snippet
    return None


def _not_found(kind: str, identifier: str) -> dict[str, Any]:
    """A recoverable model mistake, phrased so the next turn can act on it."""
    return {"found": False, "kind": kind, "id": identifier}


async def get_story_overview(
    args: GetStoryOverviewArgs, runtime: ToolRuntime
) -> ToolResult:
    overview = await data.get_story_overview(runtime.ctx.story_id, runtime.ctx.user_id)
    # A 200-chapter story with 500-character titles is a legal story-data
    # record and renders about 118 000 characters here, so the ceiling has to
    # apply to the table of contents too. chapter_count is computed before the
    # clip, so "200 chapters, 50 listed" stays a true statement.
    payload, truncated = _fit(overview, runtime.max_result_chars)
    return ToolResult(payload, truncated=truncated)


async def list_story_entities(
    args: ListStoryEntitiesArgs, runtime: ToolRuntime
) -> ToolResult:
    collection = COLLECTION_BY_ENTITY_KIND[args.kind]
    page = await data.list_entities(
        runtime.ctx.story_id, runtime.ctx.user_id, collection
    )
    # The schema's `limit` bounds what the model asked for; data.list_entities
    # applies its own, larger page cap. Report both truncations as one flag so a
    # partial list is never presented as a complete roster.
    items = page.items[args.offset : args.offset + args.limit]
    # `total` is what the server loaded, so it is itself bounded by
    # COLLECTION_FETCH_LIMIT. That understatement is safe in the direction that
    # matters -- it is `truncated`, not `total`, that says "keep paging".
    payload, clipped = _fit(
        {
            "kind": args.kind,
            "entities": items,
            "offset": args.offset,
            "total": len(page.items),
            "truncated": page.truncated or args.offset + len(items) < len(page.items),
        },
        runtime.max_result_chars,
    )
    # Either kind of shortening is the same fact to a reader of this result: the
    # roster in front of you is not the whole roster.
    payload["truncated"] = bool(payload.get("truncated")) or clipped
    return ToolResult(payload, truncated=payload["truncated"])


async def get_story_entity(
    args: GetStoryEntityArgs, runtime: ToolRuntime
) -> ToolResult:
    collection = COLLECTION_BY_ENTITY_KIND[args.kind]
    try:
        entity = await data.get_entity(
            runtime.ctx.story_id, runtime.ctx.user_id, collection, args.entity_id
        )
    except data.EntityNotFoundError:
        return ToolResult(_not_found(args.kind, args.entity_id))

    name = str(entity.get("name") or "Unnamed")
    reference = SourcePart(
        type="source",
        source_id=str(entity.get("entity_id") or args.entity_id),
        kind="story",
        title=f"{args.kind.capitalize()}: {name}"[:500],
        snippet=_first_prose(entity),
    )
    payload, truncated = _fit({"kind": args.kind, **entity}, runtime.max_result_chars)
    return ToolResult(payload, references=(reference,), truncated=truncated)


async def read_chapter(args: ReadChapterArgs, runtime: ToolRuntime) -> ToolResult:
    # The second of Phase 1's two checks. `limit` was validated against the
    # schema's 20 000-char ceiling on the way in; here it is clamped again to
    # what one result may actually cost, so a legal argument cannot buy an
    # illegal prompt.
    window = max(1, min(args.limit, runtime.max_result_chars))
    try:
        chapter = await data.get_chapter(
            runtime.ctx.story_id,
            args.chapter_id,
            runtime.ctx.user_id,
            args.offset,
            window,
        )
    except data.EntityNotFoundError:
        return ToolResult(_not_found("chapter", args.chapter_id))

    # Trim against the serialized *whole* result rather than raw prose length.
    # JSON escaping can make one source character cost several result characters;
    # the helper also derives next_offset from the exact prefix returned so paging
    # cannot skip text the model never saw.
    chapter, truncated = _fit_chapter_window(
        chapter, args.offset, runtime.max_result_chars
    )
    content = chapter.get("content") or ""

    number = chapter.get("chapter_number")
    title = chapter.get("title") or "Untitled"
    label = f"Chapter {number}: {title}" if number is not None else title
    reference = SourcePart(
        type="source",
        source_id=str(chapter.get("chapter_id") or args.chapter_id),
        kind="story",
        title=label[:500],
        snippet=_snippet(content),
    )
    return ToolResult(chapter, references=(reference,), truncated=truncated)


async def search_story(args: SearchStoryArgs, runtime: ToolRuntime) -> ToolResult:
    """Semantic search over this story's indexed chunks, and nothing else.

    An instance with no pgvector or no embedder raises rather than returning an
    empty list: answering "I found nothing" when the index was never consulted
    invites the model to state that the story does not mention something it
    plainly does.

    Every hit carries its indexed and current revisions and timestamps because
    indexing runs off an outbox. A chapter edited a moment ago may legitimately
    still have an older chunk, so that lag is returned explicitly as ``stale``.
    """
    if runtime.postgres is None or runtime.embedder is None:
        raise ToolExecutionError(ErrorCode.INTERNAL_ERROR)
    try:
        embedding = await runtime.embedder.embed(args.query)
        chunks = await runtime.postgres.search_chunks(
            runtime.ctx.story_id, embedding, top_k=args.limit
        )
    except Exception:
        logger.warning("assistant_search_failed story_scoped=1")
        raise ToolExecutionError(ErrorCode.INTERNAL_ERROR) from None

    budget = max(1, runtime.max_result_chars // max(1, len(chunks)))
    results: list[dict[str, Any]] = []
    references: list[SourcePart] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        label = str(metadata.get("title") or metadata.get("name") or "").strip()
        if not label:
            label = chunk["kind"].replace("_", " ").title()
        text = chunk["text"][:budget]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "kind": chunk["kind"],
                "source_id": chunk["source_id"],
                "title": label,
                "chapter_number": metadata.get("chapterNumber"),
                "source_revision": chunk["source_revision"],
                "current_revision": chunk.get("current_revision"),
                "indexed_at": _isoformat(chunk.get("indexed_at")),
                "source_updated_at": _isoformat(chunk.get("source_updated_at")),
                "stale": _stale_chunk(chunk),
                "text": text,
            }
        )
        references.append(
            SourcePart(
                type="source",
                source_id=chunk["chunk_id"],
                kind="story",
                title=label[:500],
                snippet=_snippet(chunk["text"]),
            )
        )
    payload, truncated = _fit(
        {"query": args.query, "results": results}, runtime.max_result_chars
    )
    return ToolResult(payload, references=tuple(references), truncated=truncated)


async def read_current_editor(
    args: ReadCurrentEditorArgs, runtime: ToolRuntime
) -> ToolResult:
    """Return the bounded send-time snapshot supplied by the active editor."""
    editor = runtime.editor_context
    if editor is None:
        return ToolResult({"available": False, "reason": "no active editor"})

    # Browser context is not an ownership assertion. Re-establish that the
    # chapter belongs to the already-authorized story before reflecting it to
    # the model. The returned canonical text is intentionally discarded.
    if editor.chapter_id:
        await data.get_chapter(
            runtime.ctx.story_id,
            editor.chapter_id,
            runtime.ctx.user_id,
            0,
            1,
        )

    selection = editor.selection
    buffer = editor.buffer if not args.selection_only else None
    payload, truncated = _fit(
        {
            "available": True,
            "chapter_id": editor.chapter_id,
            "dirty": editor.dirty,
            "persisted_revision": editor.persisted_revision,
            "document_version": editor.document_version,
            "selection": (
                None
                if selection is None
                else {
                    "from": selection.from_,
                    "to": selection.to,
                    "text": selection.text,
                }
            ),
            "buffer": (
                None
                if buffer is None
                else {"text": buffer.text, "truncated": buffer.truncated}
            ),
            "full_document_available": buffer is not None,
        },
        runtime.max_result_chars,
    )
    return ToolResult(payload, truncated=truncated)


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _stale_chunk(chunk: dict[str, Any]) -> bool:
    """Whether an indexed chunk predates its canonical source."""
    current_revision = chunk.get("current_revision")
    if current_revision is None:
        return True
    try:
        if int(chunk["source_revision"]) != int(current_revision):
            return True
    except (KeyError, TypeError, ValueError):
        return True

    indexed_at = chunk.get("indexed_at")
    source_updated_at = chunk.get("source_updated_at")
    if indexed_at is not None and source_updated_at is not None:
        try:
            return bool(indexed_at < source_updated_at)
        except TypeError:
            # Revision is the authoritative fallback when a fake or older driver
            # supplies timestamps in incomparable representations.
            return False
    return False


Executor = Callable[[Any, ToolRuntime], Awaitable[ToolResult]]

# Read tools only. The edit and research schemas exist, but a schema with no
# executor offered to a model is a tool call that can only fail, so
# available_tools(edits_enabled=False, research_enabled=False) and this table
# have to agree -- and a test asserts they do.
EXECUTORS: dict[str, Executor] = {
    "get_story_overview": get_story_overview,
    "search_story": search_story,
    "list_story_entities": list_story_entities,
    "get_story_entity": get_story_entity,
    "read_chapter": read_chapter,
    "read_current_editor": read_current_editor,
}


async def execute_tool(name: str, args: BaseModel, runtime: ToolRuntime) -> ToolResult:
    """Run one validated tool call. ``args`` is already a parsed schema instance.

    Validation stays with the caller (``validate_tool_arguments``) so the loop
    can report a malformed call as ``tool.failed`` without having started any
    work, and so this function never sees a raw dict from a model.
    """
    executor = EXECUTORS.get(name)
    if executor is None:
        raise UnknownToolError(f"Unknown tool: {name}")
    try:
        return await executor(args, runtime)
    except (ToolExecutionError, UnknownToolError):
        raise
    except data.StoryNotFoundError:
        # The endpoint's ownership gate passed, so this is a story deleted or
        # unshared mid-run rather than a probe. Same code either way.
        raise ToolExecutionError(ErrorCode.STORY_ACCESS_DENIED) from None
    except story_data.StoryDataError:
        logger.warning("assistant_tool_story_data_unavailable tool=%s", name)
        raise ToolExecutionError(ErrorCode.INTERNAL_ERROR) from None
