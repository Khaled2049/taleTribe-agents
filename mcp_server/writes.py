from __future__ import annotations

import functools
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Optional

import anyio.to_thread
import structlog
from google.api_core import exceptions as gcp_exceptions

from mcp_server import blocks, data, story_data
from mcp_server.oauth_store import as_utc

logger = structlog.get_logger(__name__)

# Product ceilings, applied before the request leaves this service. These are
# deliberately tighter than story-data's own field bounds (internal/store/
# validate.go), which are the outer wall for every client; these are what the
# MCP connector offers. Not config settings — an env var invites raising the
# MCP ceiling above what the editor can comfortably render.
#
# The ceilings story-data enforces itself — 100 stories per user, 50 chapters
# per story, 5 000 words per chapter — are deliberately NOT restated here. It
# counts them transactionally and returns a message written for a caller to
# read, so a copy here would only be a second number to keep in step.
MAX_TITLE_CHARS = 200
MAX_DESCRIPTION_CHARS = 2_000
MAX_TAGS = 10
MAX_TAG_CHARS = 40
MAX_CATEGORY_CHARS = 60
# story-data bounds a chapter in words, not characters, so this one has no
# counterpart there: it keeps a caller from storing megabytes of markup that
# passes the word count.
MAX_CHAPTER_CONTENT_CHARS = 100_000

# Retries for a chapter position lost to a concurrent create.
MAX_CLAIM_ATTEMPTS = 3

# Positional edit operations. Two actions cover the ground: "replace" with
# empty text removes a block, so a third "delete" action would add schema
# without adding reach.
OP_ACTIONS = ("replace", "insert_after")

# Which blocks a `replace` may rewrite, keyed on the tag the splitter reports.
#
# _to_paragraph_html only ever emits <p>, so applying it to whatever block the
# caller named would turn an <h2> into a paragraph and flatten a <ul> into one
# soft-wrapped line — silently, and the product has no undo. Paragraphs are
# rewritten as before; headings keep their level; everything else carries
# structure this module cannot rebuild from plain text and is refused instead
# of downgraded. See _replacement_html.
PLAIN_BLOCK_TAGS = frozenset({"p", blocks.TEXT_TAG})
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# Enough for a thorough revision pass over one chapter, small enough that a
# runaway model cannot smuggle a whole-chapter rewrite through as ops.
MAX_OPS_PER_CALL = 20

# Write-idempotency reservations. The one piece of this module still in
# Firestore, and deliberately so: it is the connector's own bookkeeping, like
# the OAuth records beside it, not story content. Service-only; see the
# firestore.rules deny block and the Terraform TTL policy on `expiresAt`.
WRITES_COLLECTION = "mcpWrites"
IDEMPOTENCY_TTL_SECONDS = 120


class LimitExceededError(Exception):
    """A product ceiling this service applies before calling story-data.

    Carries the numbers as attributes so the audit log can record them without
    parsing the message.
    """

    def __init__(self, limit_name: str, observed: int, ceiling: int, message: str):
        super().__init__(message)
        self.limit_name = limit_name
        self.observed = observed
        self.ceiling = ceiling


class WriteConflictError(Exception):
    """Concurrent chapter creation lost its position too many times."""


class DuplicateInFlightError(Exception):
    """An identical call from the same user is still running."""


class StaleRevisionError(Exception):
    """The chapter moved after the caller read it, so the edit's base is gone.

    Distinct from WriteConflictError, which means "we lost a race we can
    retry". This one is not retryable by the server: the content the caller
    reasoned about no longer exists, and only the caller can decide whether its
    edit still makes sense against the new text.
    """


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _to_thread(fn, *args, **kwargs) -> Any:
    """Run one of the synchronous Firestore idempotency calls off the loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def _expired(record: dict[str, Any]) -> bool:
    # as_utc is shared with oauth_store for the same reason it exists there:
    # the emulator and hand-seeded fixtures don't always attach a tzinfo, and
    # comparing naive to aware raises.
    expires_at = as_utc(record.get("expiresAt"))
    if expires_at is None:
        return False
    return expires_at <= _now()


# ----------------------------------------------------------------------
# Input normalization
# ----------------------------------------------------------------------


def _clean_title(value: Any, field: str = "title") -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(f"{field} must not be empty.")
    if len(text) > MAX_TITLE_CHARS:
        raise ValueError(f"{field} must be at most {MAX_TITLE_CHARS} characters.")
    return text


def _clean_optional(value: Any, limit: int, field: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if len(text) > limit:
        raise ValueError(f"{field} must be at most {limit} characters.")
    return text


def _clean_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("tags must be a list of strings.")
    tags: list[str] = []
    for item in value:
        text = item.strip() if isinstance(item, str) else ""
        if not text:
            continue
        if len(text) > MAX_TAG_CHARS:
            raise ValueError(f"each tag must be at most {MAX_TAG_CHARS} characters.")
        if text not in tags:
            tags.append(text)
    if len(tags) > MAX_TAGS:
        raise ValueError(f"at most {MAX_TAGS} tags are allowed.")
    return tags


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _to_paragraph_html(content: Any) -> str:
    """Escape plain text and wrap blank-line-separated blocks in <p> tags.

    `content` is stored in the same field the TipTap editor loads, which parses
    it as HTML. Storing raw text there would silently collapse every paragraph
    break and swallow any literal '<'. Escaping first means a tool argument can
    never introduce markup — the tools accept prose, not HTML, and say so in
    their docstrings.

    Single newlines are left inside the paragraph: HTML collapses them, which
    is the correct behaviour for a soft-wrapped line.
    """
    text = content if isinstance(content, str) else ""
    if not text.strip():
        return ""
    paragraphs = [block.strip() for block in _PARAGRAPH_BREAK.split(text.strip())]
    return "\n".join(
        f"<p>{escape(block, quote=False)}</p>" for block in paragraphs if block
    )


def _clean_revision(value: Any) -> str:
    """The caller's claimed base version, normalized to story-data's spelling.

    story-data carries an integer `revision` per row and takes it back as an
    `If-Match` header, so this parses rather than merely strips: a caller that
    passes something else gets a clear refusal here instead of a 428 from a
    service it has never heard of.
    """
    text = value.strip() if isinstance(value, str) else str(value or "").strip()
    if not text:
        raise ValueError(
            "revision is required. Call get_chapter_blocks (or get_chapter) "
            "and pass the revision it returns."
        )
    try:
        number = int(text)
    except ValueError:
        raise ValueError(
            "revision must be the value from a recent read of this chapter."
        )
    if number < 1:
        raise ValueError(
            "revision must be the value from a recent read of this chapter."
        )
    return str(number)


def _check_content_size(stored: str) -> None:
    """Apply the stored-string ceiling. Always measured on what is about to be
    written, since an edit's result is the join of blocks the caller never sent."""
    if len(stored) > MAX_CHAPTER_CONTENT_CHARS:
        raise LimitExceededError(
            "chapter_content_chars",
            len(stored),
            MAX_CHAPTER_CONTENT_CHARS,
            f"Chapter content must be at most {MAX_CHAPTER_CONTENT_CHARS} "
            "characters once formatted.",
        )


# ----------------------------------------------------------------------
# Idempotency (Firestore)
# ----------------------------------------------------------------------


def idempotency_key(uid: str, tool: str, args: dict[str, Any]) -> str:
    """Stable key for one logical call.

    The uid is part of the key, not just the record: without it one user could
    probe another's dedup entries for existence, or poison them to deny the
    write outright.

    Time is deliberately NOT part of the key. A time-bucketed id has a boundary
    hole — a retry at t=59.9s and t=60.1s lands in different buckets and both
    calls succeed. Expiry lives in `expiresAt` and is re-checked on read
    instead, exactly as oauth_store does.
    """
    payload = json.dumps(
        args, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(f"{uid}|{tool}|{payload}".encode("utf-8")).hexdigest()


def _claim(db: Any, key: str, uid: str, tool: str) -> Optional[dict]:
    """Reserve the key before doing any work.

    Returns a previous result to replay, or None when the caller owns the claim
    and should proceed. Reserving *before* the write (rather than recording
    after it) is what makes a retry that arrives mid-flight observable at all.
    """
    ref = db.collection(WRITES_COLLECTION).document(key)
    record = {
        "uid": uid,
        "tool": tool,
        "expiresAt": _now() + timedelta(seconds=IDEMPOTENCY_TTL_SECONDS),
    }
    try:
        ref.create(record)
        return None
    except gcp_exceptions.AlreadyExists:
        pass

    snap = ref.get()
    existing = snap.to_dict() if snap.exists else None
    if existing is None:
        # Gone between the create above and this read — a concurrent _release,
        # or the TTL collector. The key is free again, so claim it the same way
        # the first attempt would have.
        _claim_vacant(ref, record, tool)
        return None
    if _expired(existing):
        # TTL collection lags by hours, so an expired reservation is ours to
        # take — but only if we win it. Unconditionally overwriting would let
        # two retries that observe the SAME expired record both proceed and
        # both write, losing the exactly-one-winner property `create` gives the
        # common path for free.
        #
        # `result: None` clears any result recorded in the expired round; the
        # check below already reads a non-dict result as "no result".
        try:
            ref.update(
                {**record, "result": None},
                option=db.write_option(last_update_time=snap.update_time),
            )
        except gcp_exceptions.FailedPrecondition:
            raise DuplicateInFlightError(tool)
        except gcp_exceptions.NotFound:
            _claim_vacant(ref, record, tool)
        return None
    result = existing.get("result")
    if isinstance(result, dict) and result:
        return result
    raise DuplicateInFlightError(tool)


def _claim_vacant(ref: Any, record: dict, tool: str) -> None:
    """Claim a key whose document is absent, losing to anyone who beats us."""
    try:
        ref.create(record)
    except gcp_exceptions.AlreadyExists:
        raise DuplicateInFlightError(tool)


def _record(db: Any, key: str, result: dict) -> None:
    db.collection(WRITES_COLLECTION).document(key).set({"result": result}, merge=True)


def _release(db: Any, key: str) -> None:
    """Drop a reservation whose write failed, so a genuine retry isn't blocked.

    Best-effort, and it must swallow EVERYTHING: this runs inside an `except`
    block, so any exception it lets out replaces the original failure — a
    clean LimitExceededError would surface as an unmapped 500 because the
    cleanup delete hit a transient Firestore error. If the delete fails the
    reservation simply expires on its own (IDEMPOTENCY_TTL_SECONDS).
    """
    try:
        db.collection(WRITES_COLLECTION).document(key).delete()
    except Exception as exc:
        logger.warning(
            "mcp_write_release_failed",
            error_type=type(exc).__name__,
        )


# ----------------------------------------------------------------------
# create_story
# ----------------------------------------------------------------------


async def _author_name(uid: str) -> str:
    """Display name for the story's byline, or "" when there is no profile yet."""
    try:
        profile = await story_data.client().get_my_profile(uid)
    except story_data.StoryDataError:
        return ""
    username = profile.get("username") if isinstance(profile, dict) else None
    return username.strip() if isinstance(username, str) else ""


async def create_story(
    db: Any,
    uid: str,
    title: str,
    description: str = "",
    category: str = "",
    tags: Optional[list[str]] = None,
) -> dict:
    """Create an unpublished story owned by `uid`."""
    title = _clean_title(title)
    description = _clean_optional(description, MAX_DESCRIPTION_CHARS, "description")
    category = _clean_optional(category, MAX_CATEGORY_CHARS, "category")
    tag_list = _clean_tags(tags)

    key = idempotency_key(
        uid,
        "create_story",
        {
            "title": title,
            "description": description,
            "category": category,
            "tags": tag_list,
        },
    )
    replay = await _to_thread(_claim, db, key, uid, "create_story")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        record = await story_data.client().create_story(
            uid,
            {
                "title": title,
                "description": description,
                "authorName": await _author_name(uid),
                "category": category,
                "tags": tag_list,
                "published": False,
            },
        )
        result = {
            "story_id": record.get("id"),
            "title": record.get("title") or title,
            "description": record.get("description") or "",
            "is_published": bool(record.get("published", False)),
        }
    except Exception:
        await _to_thread(_release, db, key)
        raise

    await _to_thread(_record, db, key, result)
    return {**result, "idempotent_replay": False}


# ----------------------------------------------------------------------
# create_chapter
# ----------------------------------------------------------------------


def _next_position(chapters: list[dict]) -> float:
    """max(position) + 1 over the chapters currently listed.

    `position` is a sort key with a UNIQUE (story_id, position) constraint, not
    a count: it keeps gaps after a mid-book delete and takes fractional values
    when the editor inserts between neighbours, so deriving the next one from
    the number of chapters would collide. A concurrent create can still take
    the slot this read saw, which is why _append_chapter treats story-data's
    409 as "re-read and take the next one" rather than an error.
    """
    highest = 0.0
    for chapter in chapters:
        value = chapter.get("position")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            highest = max(highest, float(value))
    return highest + 1.0


async def _append_chapter(uid: str, story_id: str, title: str, stored: str) -> dict:
    client = story_data.client()
    attempts = 0
    while attempts < MAX_CLAIM_ATTEMPTS:
        attempts += 1
        rows = await client.list_chapter_index(uid, story_id)
        chapters = rows if isinstance(rows, list) else []
        try:
            chapter = await client.create_chapter(
                uid,
                story_id,
                {
                    "title": title,
                    "content": stored,
                    "position": _next_position(chapters),
                },
            )
        except story_data.Conflict:
            continue  # someone took that position; re-read and take the next
        except story_data.NotFound as exc:
            raise data.StoryNotFoundError(story_id) from exc
        return {
            "story_id": story_id,
            "chapter_id": chapter.get("id"),
            "title": chapter.get("title") or title,
            "order": chapter.get("position"),
            "word_count": chapter.get("wordCount"),
            # What we saw, plus ours. story-data owns the real ceiling and
            # rejects the create once the story is full.
            "chapter_count": len(chapters) + 1,
            "attempts": attempts,
        }
    raise WriteConflictError(story_id)


async def create_chapter(
    db: Any, uid: str, story_id: str, title: str, content: str = ""
) -> dict:
    """Append a chapter to a story owned by `uid`.

    `content` is plain text; it is escaped and paragraph-wrapped before storage.
    """
    title = _clean_title(title)
    stored = _to_paragraph_html(content)
    _check_content_size(stored)

    # Ownership before the reservation: without this, a caller aiming at a
    # story they don't own still makes the service write (and then release) an
    # mcpWrites document.
    await data.get_owned_story(story_id, uid)

    key = idempotency_key(
        uid,
        "create_chapter",
        {"story_id": story_id, "title": title, "content": stored},
    )
    replay = await _to_thread(_claim, db, key, uid, "create_chapter")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        result = await _append_chapter(uid, story_id, title, stored)
    except Exception:
        await _to_thread(_release, db, key)
        raise

    await _to_thread(_record, db, key, result)
    return {**result, "idempotent_replay": False}


# ----------------------------------------------------------------------
# Editing an existing chapter
#
# Both tools below take a caller-supplied `revision` and make exactly ONE
# attempt. That is the deliberate difference from _append_chapter's retry loop:
# there, the conflict is over a position the server picks and can simply pick
# again. Here it guards the base version the CALLER read and reasoned about. A
# lost If-Match means that text is gone, and re-reading and re-applying
# server-side would silently overwrite whoever changed it — precisely the lost
# update the revision exists to prevent. So a conflict is reported, not
# retried, and the caller re-reads and decides.
# ----------------------------------------------------------------------


async def _chapter_at_revision(
    uid: str, story_id: str, chapter_id: str, revision: str
) -> dict:
    """Read the chapter and verify the caller's base version.

    The check is not made redundant by the If-Match on the write: it turns the
    common case into a cheap, clearly-worded refusal before any content is
    built, and it is also where the current title and position come from, which
    story-data's chapter PATCH replaces wholesale.
    """
    try:
        record = await story_data.client().get_chapter(uid, story_id, chapter_id)
    except story_data.NotFound as exc:
        raise data.EntityNotFoundError(chapter_id) from exc
    if str(record.get("revision", "")) != revision:
        raise StaleRevisionError(chapter_id)
    return record


async def _write_chapter_content(
    uid: str, story_id: str, chapter_id: str, current: dict, stored: str, revision: str
) -> dict:
    """Replace the chapter body under the read revision; return the new record.

    Title and position are carried over from the read: story-data's PATCH takes
    a whole ChapterInput, so omitting them would rename the chapter and move it
    to the front of the book.
    """
    try:
        return await story_data.client().update_chapter(
            uid,
            story_id,
            chapter_id,
            {
                "title": current.get("title") or "",
                "content": stored,
                "position": current.get("position") or 0,
            },
            revision,
        )
    except story_data.Conflict as exc:
        # Someone wrote the chapter between the read above and this update.
        raise StaleRevisionError(chapter_id) from exc
    except story_data.NotFound as exc:
        raise data.EntityNotFoundError(chapter_id) from exc


async def append_to_chapter(
    db: Any,
    uid: str,
    story_id: str,
    chapter_id: str,
    content: str,
    revision: str,
) -> dict:
    """Append paragraphs to the end of a chapter owned by `uid`.

    `content` is plain text; it is escaped and paragraph-wrapped exactly as
    create_chapter does, so a tool argument can never introduce markup.

    Appending needs no block splitting — adding after the last block is
    concatenation with the standard separator, and the existing string is
    carried over byte for byte.
    """
    revision = _clean_revision(revision)
    addition = _to_paragraph_html(content)
    if not addition:
        raise ValueError("content must not be empty.")

    await data.get_owned_story(story_id, uid)

    key = idempotency_key(
        uid,
        "append_to_chapter",
        {
            "story_id": story_id,
            "chapter_id": chapter_id,
            # The base version is part of the key, so a retry replays against
            # the version it was written for, while the same text sent again
            # after a successful append is a different (and genuinely new) call.
            "revision": revision,
            "content": addition,
        },
    )
    replay = await _to_thread(_claim, db, key, uid, "append_to_chapter")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        current = await _chapter_at_revision(uid, story_id, chapter_id, revision)
        existing = current.get("content") or ""
        if existing:
            stored = existing + blocks.BLOCK_SEPARATOR + addition
        else:
            stored = addition
        _check_content_size(stored)
        updated = await _write_chapter_content(
            uid, story_id, chapter_id, current, stored, revision
        )
        result = {
            "story_id": story_id,
            "chapter_id": chapter_id,
            "word_count": updated.get("wordCount"),
            "content_chars": len(stored),
            "appended_blocks": len(blocks.split_blocks(addition)),
            # Re-split of the WHOLE chapter, deliberately, even though summing
            # the two halves looks equivalent and would save a parse of up to
            # MAX_CHAPTER_CONTENT_CHARS. It isn't equivalent: an unclosed tag
            # in the existing content leaves the parser at depth > 0, so the
            # appended <p> opens no new top-level block and the two counts
            # differ —
            #     "<p>unclosed" + "\n" + "<p>new</p>"  ->  1 block, not 2
            # Editor output is well-formed, but this module never gets to
            # assume that about content it did not write. `block_count` is the
            # index range the caller's next edit will address, so it has to be
            # what split_blocks will actually say next time it is asked.
            "block_count": len(blocks.split_blocks(stored)),
            # The version this write produced, so the caller can chain another
            # edit without a re-read.
            "revision": str(updated.get("revision", "")),
        }
    except Exception:
        await _to_thread(_release, db, key)
        raise

    await _to_thread(_record, db, key, result)
    return {**result, "idempotent_replay": False}


def _range_message(action: str, index: int, count: int) -> str:
    if count == 0:
        return (
            f"This chapter has no blocks yet, so {action} at index {index} has "
            "nothing to address. Use insert_after with index -1 to add the "
            "first block."
        )
    high = count - 1
    low = 0 if action == "replace" else -1
    return (
        f"{action} index {index} is out of range: valid indices are {low} to "
        f"{high} ({count} blocks). Call get_chapter_blocks for current indices."
    )


def _normalize_ops(ops: Any) -> list[dict]:
    """Validate and canonicalize the op list. Raises ValueError on bad input.

    Re-validates everything the tool layer's schema already types, because that
    schema is a hint to the model and this module is the boundary that has to
    hold regardless of what reaches it.

    Sorting descending by index is what lets every index mean a position in the
    ORIGINAL block list: applying from the back means no earlier op has shifted
    the positions a later one refers to. It also makes the idempotency key
    insensitive to the order the model happened to list its ops in.
    """
    if not isinstance(ops, (list, tuple)):
        raise ValueError("ops must be a list of edit operations.")
    if not ops:
        raise ValueError("ops must contain at least one operation.")
    if len(ops) > MAX_OPS_PER_CALL:
        raise ValueError(
            f"at most {MAX_OPS_PER_CALL} operations per call; got {len(ops)}."
        )

    normalized: list[dict] = []
    seen: set[int] = set()
    for op in ops:
        if not isinstance(op, dict):
            raise ValueError("each op must be an object with action, index and text.")
        action = op.get("action")
        if action not in OP_ACTIONS:
            raise ValueError(f"action must be one of {', '.join(OP_ACTIONS)}.")
        index = op.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("index must be an integer block index.")
        # `op.get("text") or ""` would coerce every FALSY non-string — 0,
        # False, [], None — to "", which is the block-DELETION sentinel, so a
        # malformed op would quietly destroy a paragraph instead of being
        # refused. An absent key still means "" because that is the schema
        # default; anything present must actually be a string.
        raw_text = op.get("text", "")
        if not isinstance(raw_text, str):
            raise ValueError('text must be a string (pass "" to delete the block).')
        html = _to_paragraph_html(raw_text)
        if action == "insert_after" and not html:
            raise ValueError("insert_after requires non-empty text.")
        if index in seen:
            # Two ops on one block would need a defined relative order, and any
            # pair worth expressing is expressible as a single replace.
            raise ValueError(
                f"two operations target block {index}. Combine them into one "
                "replace whose text contains everything that block should become."
            )
        seen.add(index)
        # `text` rides along beside the rendered html because the final markup
        # of a replace depends on the tag being replaced, which is only known
        # once the chapter is read (see _replacement_html). Both are derived
        # from the same input, so the idempotency key stays deterministic.
        normalized.append(
            {"action": action, "index": index, "html": html, "text": raw_text}
        )

    normalized.sort(key=lambda entry: entry["index"], reverse=True)
    return normalized


def _is_single_paragraph(text: str) -> bool:
    return len(_PARAGRAPH_BREAK.split(text.strip())) == 1


def _replacement_html(op: dict, tag: str, index: int) -> str:
    """The markup a replace should store, given the tag it is replacing.

    Paragraphs (and bare top-level text, which the editor normalises into one)
    keep the existing behaviour. A heading is rebuilt at its own level: heading
    nodes hold INLINE content, so the text goes in bare — <h2><p>x</p></h2> is
    not something the editor's schema accepts — and internal whitespace
    collapses because a heading has no paragraphs to separate.

    Everything else is refused. A <ul> rewritten through _to_paragraph_html
    comes back as one soft-wrapped <p>, which loses every item boundary; the
    same goes for tables, figures and code. Refusing costs the caller a trip to
    the editor, where silently flattening costs the author their formatting
    with nothing to restore it from.
    """
    if tag in PLAIN_BLOCK_TAGS:
        return op["html"]
    if tag in HEADING_TAGS:
        if not _is_single_paragraph(op["text"]):
            raise ValueError(
                f"block {index} is an <{tag}> heading, which holds a single "
                "line. Remove the blank line from `text`, or delete the "
                "heading and insert_after the paragraphs you want."
            )
        inline = escape(" ".join(op["text"].split()), quote=False)
        return f"<{tag}>{inline}</{tag}>"
    raise ValueError(
        f"block {index} is a <{tag}>, which carries formatting this tool "
        "cannot rewrite without flattening it into a plain paragraph. Edit it "
        "in TheTaleTribe's editor, or delete it (replace with empty text) and "
        "insert_after the replacement paragraphs."
    )


def _apply_ops(parts: list[str], tags: list[str], normalized: list[dict]) -> None:
    """Apply canonicalized ops to a block list in place.

    Indices are validated against the ORIGINAL length, never the running one,
    so an out-of-range op is rejected on what the caller actually saw. `tags`
    is the parallel list of original tags and is never mutated, so tags[index]
    keeps naming the block the caller addressed even after a higher-indexed op
    has removed or inserted entries in `parts`.
    """
    count = len(parts)
    for op in normalized:
        index = op["index"]
        action = op["action"]
        if action == "replace":
            if not 0 <= index < count:
                raise ValueError(_range_message("replace", index, count))
            if op["html"]:
                parts[index : index + 1] = [_replacement_html(op, tags[index], index)]
            else:
                # Documented behaviour: replacing with empty text deletes the
                # block. Allowed for EVERY tag, including the structural ones
                # _replacement_html refuses to rewrite: removing a list is an
                # explicit act the caller asked for, where rewriting one would
                # be a silent downgrade they did not.
                del parts[index]
        else:  # insert_after; -1 means "before the first block"
            if not -1 <= index < count:
                raise ValueError(_range_message("insert_after", index, count))
            # One op's text may render as several <p> blocks. Splicing it as a
            # single entry is byte-equivalent under join_blocks and avoids
            # splitting on "\n", which _to_paragraph_html also emits INSIDE a
            # paragraph for a soft-wrapped line.
            parts.insert(index + 1, op["html"])


async def edit_chapter_blocks(
    db: Any,
    uid: str,
    story_id: str,
    chapter_id: str,
    ops: Any,
    revision: str,
) -> dict:
    """Apply positional block edits to a chapter owned by `uid`.

    Blocks the ops do not name are carried through byte for byte (see
    blocks.py), so an edit cannot disturb the author's headings, lists, images
    or inline formatting elsewhere in the chapter.
    """
    revision = _clean_revision(revision)
    normalized = _normalize_ops(ops)

    await data.get_owned_story(story_id, uid)

    key = idempotency_key(
        uid,
        "edit_chapter_blocks",
        {
            "story_id": story_id,
            "chapter_id": chapter_id,
            "revision": revision,
            "ops": normalized,
        },
    )
    replay = await _to_thread(_claim, db, key, uid, "edit_chapter_blocks")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        current = await _chapter_at_revision(uid, story_id, chapter_id, revision)
        split = blocks.split_blocks(current.get("content") or "")
        parts = [block.html for block in split]
        _apply_ops(parts, [block.tag for block in split], normalized)

        # Legitimately "" when every block was removed: that is the state a
        # freshly created chapter is in, and the editor renders it.
        stored = blocks.join_blocks(parts)
        _check_content_size(stored)
        updated = await _write_chapter_content(
            uid, story_id, chapter_id, current, stored, revision
        )
        result = {
            "story_id": story_id,
            "chapter_id": chapter_id,
            "op_count": len(normalized),
            # Re-split rather than len(parts): one op's text can introduce
            # several blocks, and this is the index range the caller's next
            # call must address.
            "block_count": len(blocks.split_blocks(stored)),
            "word_count": updated.get("wordCount"),
            "content_chars": len(stored),
            "revision": str(updated.get("revision", "")),
        }
    except Exception:
        await _to_thread(_release, db, key)
        raise

    await _to_thread(_record, db, key, result)
    return {**result, "idempotent_replay": False}
