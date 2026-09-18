from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Optional

import structlog
from google.api_core import exceptions as gcp_exceptions
from google.cloud.firestore_v1.query import Query

from mcp_server import blocks, data
from mcp_server.oauth_store import as_utc


def _owned_story_snapshot(db: Any, story_id: str, uid: str) -> Any:
    """The ownership gate, Firestore-shaped, returning the raw snapshot.

    Lives here rather than in data.py because the write path is the only thing
    that still needs a Firestore snapshot: it makes the chapterCount bump
    conditional on `snap.update_time`, the version it read. data.py now reads
    through story-data over HTTP and has no snapshot to hand back.

    Kept identical in behaviour to the gate it came from — missing and non-owned
    raise the same error, so story ids cannot be probed for existence.
    """
    snap = db.collection("stories").document(story_id).get()
    record = snap.to_dict() if snap.exists else None
    if not record or record.get("userId") != uid:
        raise data.StoryNotFoundError(story_id)
    return snap


def _revision_token(update_time: Any) -> str:
    """Canonical version string for a Firestore document, from either spelling.

    Opaque to callers: they pass it back unmodified, and the write path compares
    it against a freshly derived one.

    The normalisation is not cosmetic. A read hands back a
    DatetimeWithNanoseconds (DocumentSnapshot.update_time) while a write hands
    back a protobuf Timestamp (WriteResult.update_time), and str() spells the
    same instant completely differently for the two:

        "2026-08-01 13:54:57.745934+00:00"
        "seconds: 1785000000\\nnanos: 745934000\\n"

    Taking the token straight from str() would therefore mean the revision an
    edit returns never equals the one the next read reports, so every chained
    edit would be refused as stale with no concurrent writer anywhere.

    story-data needs none of this — it carries an integer `revision` per row —
    so when the write tools move, this goes away and If-Match replaces it.
    """
    seconds = getattr(update_time, "seconds", None)
    nanos = getattr(update_time, "nanos", None)
    if isinstance(seconds, int) and isinstance(nanos, int):
        return f"{seconds}.{nanos:09d}"  # protobuf Timestamp
    as_pb = getattr(update_time, "timestamp_pb", None)
    if callable(as_pb):
        stamp = as_pb()  # DatetimeWithNanoseconds
        return f"{stamp.seconds}.{stamp.nanos:09d}"
    # The test fake's monotonic counter, and any other opaque version object.
    return str(update_time)


logger = structlog.get_logger(__name__)

# Ceilings the Admin SDK bypasses. Each names the upstream source of truth in
# the frontend repo; test_write_limits_match_the_client_limits parses those
# files and fails when the two drift. Deliberately NOT config settings — an env
# var invites raising the MCP ceiling above what firestore.rules and the editor
# accept, which produces documents the owner cannot save.
MAX_TITLE_CHARS = 200
MAX_DESCRIPTION_CHARS = 2_000
MAX_TAGS = 10
MAX_TAG_CHARS = 40
MAX_CATEGORY_CHARS = 60
# firestore.rules: chapter create/update require content.size() <= 100000.
MAX_CHAPTER_CONTENT_CHARS = 100_000
# StoriesRepo.WORD_LIMIT.
MAX_CHAPTER_WORDS = 5_000
# StoriesRepo.CHAPTER_LIMIT.
MAX_CHAPTERS_PER_STORY = 50
# firestore.rules MAX_STORIES_PER_USER, via the userStoryCount() helper.
MAX_STORIES_PER_USER = 100

# Retries for the chapterCount precondition before giving up.
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

# Write-idempotency reservations. Service-only; see the firestore.rules deny
# block and the Terraform TTL policy on `expiresAt`.
WRITES_COLLECTION = "mcpWrites"
IDEMPOTENCY_TTL_SECONDS = 120


class LimitExceededError(Exception):
    """A ceiling the Admin SDK bypasses but the frontend/rules would enforce.

    Carries the numbers as attributes so the audit log can record them without
    parsing the message.
    """

    def __init__(self, limit_name: str, observed: int, ceiling: int, message: str):
        super().__init__(message)
        self.limit_name = limit_name
        self.observed = observed
        self.ceiling = ceiling


class WriteConflictError(Exception):
    """Concurrent chapter creation lost its precondition too many times."""


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

    Nothing in the frontend currently renders chapter content through
    dangerouslySetInnerHTML, so this is not closing a live XSS hole; it is
    making one structurally impossible to open later, and fixing a real
    rendering bug today.

    Single newlines are left inside the paragraph: HTML collapses them, which
    is the correct behaviour for a soft-wrapped line.
    """
    text = content if isinstance(content, str) else ""
    if not text.strip():
        # Matches StoriesRepo.addChapter, which seeds a new chapter with "".
        return ""
    blocks = [block.strip() for block in _PARAGRAPH_BREAK.split(text.strip())]
    return "\n".join(
        f"<p>{escape(block, quote=False)}</p>" for block in blocks if block
    )


def _clean_revision(value: Any) -> str:
    """The caller's claimed base version, as an opaque string.

    Not parsed or reformatted beyond stripping: it is compared verbatim against
    _revision_token() of a freshly read update_time, so interpreting it as
    a timestamp here would only create ways for an equal version to compare
    unequal.
    """
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(
            "revision is required. Call get_chapter_blocks (or get_chapter) "
            "and pass the revision it returns."
        )
    return text


def _count_words(stored: str) -> int:
    """Word count over the STORED string, matching StoriesRepo.countWords.

    The frontend counts `content.trim().split(/\\s+/)` on the raw HTML, and
    recomputes it on the next editor save. Counting the same string here means
    the number does not jump under the user the first time they edit. The <p>
    wrapping is glued to the adjacent word, so this equals the plain-text count.

    One deliberate divergence: the frontend's split returns 1 for an empty
    string (it has no filter(Boolean)), but addChapter writes 0 for empty
    content. We follow addChapter.
    """
    return len(stored.split())


# ----------------------------------------------------------------------
# Idempotency
# ----------------------------------------------------------------------


def idempotency_key(uid: str, tool: str, args: dict[str, Any]) -> str:
    """Stable key for one logical call.

    The uid is part of the key, not just the record: without it one user could
    probe another's dedup entries for existence, or poison them to deny the
    write outright.

    Time is deliberately NOT part of the key. A time-bucketed id (the shape
    chapterIndexTrigger uses for debouncing) has a boundary hole — a retry at
    t=59.9s and t=60.1s lands in different buckets and both calls succeed.
    Expiry lives in `expiresAt` and is re-checked on read instead, exactly as
    oauth_store does.
    """
    payload = json.dumps(
        args, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(f"{uid}|{tool}|{payload}".encode("utf-8")).hexdigest()
    return digest


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
        # check below already reads a non-dict result as "no result". A field
        # delete would be tidier but needs a sentinel, and this module keeps
        # sentinels out of its writes (see create_story's note on
        # SERVER_TIMESTAMP).
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


def _story_count(db: Any, uid: str) -> int:
    """Denormalized story count, with the same semantics as the rules helper.

    userStoryCount() in firestore.rules treats a missing user doc or missing
    field as 0, so this does too. Counting `stories where userId == uid` would
    be exact but is an unbounded, per-document-billed query on every create —
    a trivial way to run up a bill.

    The counter is maintained by storyCountTrigger and is eventually
    consistent, so a fast enough burst can overshoot. That is why this is
    documented as a soft cap and why the write rate limiter is the real control.
    """
    snap = db.collection("users").document(uid).get()
    if not snap.exists:
        return 0
    value = (snap.to_dict() or {}).get("storyCount")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _author_name(db: Any, uid: str) -> str:
    """Display name, mirroring StoriesRepo.getUserInfo: "" when absent."""
    try:
        snap = db.collection("publicProfiles").document(uid).get()
    except gcp_exceptions.GoogleAPIError:
        return ""
    if not snap.exists:
        return ""
    value = (snap.to_dict() or {}).get("username")
    return value.strip() if isinstance(value, str) else ""


def create_story(
    db: Any,
    uid: str,
    title: str,
    description: str = "",
    category: str = "",
    tags: Optional[list[str]] = None,
) -> dict:
    """Create an empty story owned by `uid`.

    Writes every field StoriesRepo.createStory writes, so the frontend's
    mapStoryDoc never meets an absent one — it calls .toDate() on createdAt and
    updatedAt without guarding.

    Unlike StoriesRepo.createStory this does NOT also create "Chapter 1". A
    zero-chapter story renders fine (the sidebar has an explicit empty state),
    and a second write would introduce a partial-failure state for no gain.
    """
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
    replay = _claim(db, key, uid, "create_story")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        count = _story_count(db, uid)
        if count >= MAX_STORIES_PER_USER:
            raise LimitExceededError(
                "stories_per_user",
                count,
                MAX_STORIES_PER_USER,
                f"You have reached the limit of {MAX_STORIES_PER_USER} stories.",
            )

        ref = db.collection("stories").document()
        # Container clock, not firestore.SERVER_TIMESTAMP: the sentinel can't
        # round-trip through the test fake, and a second or two of Cloud Run
        # clock skew only nudges the story's position in the updatedAt sort.
        now = _now()
        ref.set(
            {
                "id": ref.id,  # denormalized into the body, as StoriesRepo does
                "title": title,
                "description": description,
                "userId": uid,  # REQUIRED: storyCountTrigger no-ops without it
                "author": _author_name(db, uid),
                "isPublished": False,
                "createdAt": now,
                "updatedAt": now,
                "chapterCount": 0,
                "views": 0,
                "likes": 0,
                "category": category,
                "tags": tag_list,
                "targetAudience": "",
                "language": "",
                "copyright": "",
                "coverImageUrl": "",
            }
        )
        result = {
            "story_id": ref.id,
            "title": title,
            "description": description,
            "chapter_count": 0,
            "is_published": False,
        }
    except Exception:
        _release(db, key)
        raise

    _record(db, key, result)
    return {**result, "idempotent_replay": False}


# ----------------------------------------------------------------------
# create_chapter
# ----------------------------------------------------------------------


def _next_order(db: Any, story_id: str) -> int:
    """max(order) + 1, read from the chapters themselves.

    NOT derived from story.chapterCount, which is what StoriesRepo.addChapter
    does. That has two independent failure modes: concurrent callers both read
    the same count and write the same `order`, and deleteChapter decrements the
    counter without renumbering, so after any mid-book delete the counter
    already equals max(order) and the next add collides even single-threaded.

    On its own this read is racy too — a winner that has claimed its slot but
    not yet written the chapter is invisible to it — so _append_chapter only
    uses it as a floor under `nextChapterOrder`, which IS precondition-guarded.
    It bootstraps stories that predate the counter and self-heals if a frontend
    addChapter writes an order the counter doesn't know about.

    Served by the automatic single-field index; reads one document.
    """
    docs = list(
        db.collection("stories")
        .document(story_id)
        .collection("chapters")
        .select(["order"])
        .order_by("order", direction=Query.DESCENDING)
        .limit(1)
        .stream()
    )
    if not docs:
        return 0
    value = (docs[0].to_dict() or {}).get("order")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value) + 1


def _check_content_limits(stored: str) -> int:
    """Enforce the content ceilings on a stored chapter string; return its words.

    Shared by creation and every edit, and always applied to the string that is
    about to be written rather than to the caller's input — the rule the Admin
    SDK bypasses measures the stored value, and an edit's result is the join of
    blocks the caller never sent.
    """
    if len(stored) > MAX_CHAPTER_CONTENT_CHARS:
        # Measured on the stored string because that is what the rule measures.
        raise LimitExceededError(
            "chapter_content_chars",
            len(stored),
            MAX_CHAPTER_CONTENT_CHARS,
            f"Chapter content must be at most {MAX_CHAPTER_CONTENT_CHARS} "
            "characters once formatted.",
        )
    word_count = _count_words(stored)
    if word_count > MAX_CHAPTER_WORDS:
        raise LimitExceededError(
            "chapter_words",
            word_count,
            MAX_CHAPTER_WORDS,
            f"Chapter content must be at most {MAX_CHAPTER_WORDS} words.",
        )
    return word_count


def create_chapter(
    db: Any, uid: str, story_id: str, title: str, content: str = ""
) -> dict:
    """Append a chapter to a story owned by `uid`.

    `content` is plain text; it is escaped and paragraph-wrapped before storage.
    """
    title = _clean_title(title)
    stored = _to_paragraph_html(content)
    word_count = _check_content_limits(stored)

    # Ownership before the reservation: without this, a caller aiming at a
    # story they don't own still makes the service write (and then release) an
    # mcpWrites document. One extra read closes that small write amplifier;
    # _append_chapter re-reads inside its retry loop regardless.
    _owned_story_snapshot(db, story_id, uid)

    key = idempotency_key(
        uid,
        "create_chapter",
        {"story_id": story_id, "title": title, "content": stored},
    )
    replay = _claim(db, key, uid, "create_chapter")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        result = _append_chapter(db, uid, story_id, title, stored, word_count)
    except Exception:
        _release(db, key)
        raise

    _record(db, key, result)
    return {**result, "idempotent_replay": False}


def _append_chapter(
    db: Any, uid: str, story_id: str, title: str, stored: str, word_count: int
) -> dict:
    """Claim the next slot on the story doc, then write the chapter.

    Both counters — chapterCount and nextChapterOrder — are claimed FIRST, in
    one update under a last_update_time precondition. Claiming `order` in that
    same write is what makes it collision-free: exactly one caller per story
    version survives the precondition, so no two callers can leave it holding
    the same slot. (Deriving `order` from a separate max(order) read was racy:
    a winner that had claimed but not yet written its chapter was invisible to
    a second caller's read.)

    If the chapter write then fails, chapterCount over-counts by one and the
    order sequence gains a gap: cosmetic — order has gaps after any mid-book
    delete anyway, and chapterIndexTrigger rebuilds the visible chapter list
    from the subcollection regardless. The reverse order would leave the
    counter un-bumped with a chapter present, which guarantees a duplicate
    `order` on the very next create. Over-counting is the strictly better
    failure.

    A transaction would also work, but there are none in this repo — the
    precondition gives the same exactly-one-winner property oauth_store already
    relies on for single-use codes.
    """
    story_ref = db.collection("stories").document(story_id)
    attempts = 0

    while attempts < MAX_CLAIM_ATTEMPTS:
        attempts += 1
        snap = _owned_story_snapshot(db, story_id, uid)
        story = snap.to_dict() or {}

        raw_count = story.get("chapterCount")
        chapter_count = (
            int(raw_count)
            if isinstance(raw_count, (int, float)) and not isinstance(raw_count, bool)
            else 0
        )
        if chapter_count >= MAX_CHAPTERS_PER_STORY:
            raise LimitExceededError(
                "chapters_per_story",
                chapter_count,
                MAX_CHAPTERS_PER_STORY,
                f"This story already has the maximum of {MAX_CHAPTERS_PER_STORY} "
                "chapters.",
            )

        raw_next = story.get("nextChapterOrder")
        counter_next = (
            int(raw_next)
            if isinstance(raw_next, (int, float)) and not isinstance(raw_next, bool)
            else 0
        )
        # The subcollection read is only a floor (see _next_order); the counter
        # carries slots claimed by writes whose chapter isn't visible yet.
        next_order = max(counter_next, _next_order(db, story_id))
        try:
            story_ref.update(
                {
                    "chapterCount": chapter_count + 1,
                    "nextChapterOrder": next_order + 1,
                    "updatedAt": _now(),
                },
                option=db.write_option(last_update_time=snap.update_time),
            )
        except gcp_exceptions.FailedPrecondition:
            continue  # someone else moved the story; re-read and take the next slot
        except gcp_exceptions.NotFound:
            raise data.StoryNotFoundError(story_id)

        chapter_ref = story_ref.collection("chapters").document()
        chapter_ref.set(
            {
                "id": chapter_ref.id,
                "title": title,
                "content": stored,
                "order": next_order,
                "wordCount": word_count,
                # The story's owner, not the caller — they are equal here
                # because of the gate above, and sourcing it from the story
                # keeps that an invariant rather than a coincidence.
                "userId": story.get("userId", uid),
                "createdAt": _now(),
            }
        )
        # chapterNumber is deliberately omitted, matching StoriesRepo.addChapter.
        # `order` has gaps after a mid-book delete, so chapterNumber = order + 1
        # would mislabel chapters and put chapterIndexTrigger's
        # `order ?? chapterNumber` sort at odds with data._chapter_sort_key.
        return {
            "story_id": story_id,
            "chapter_id": chapter_ref.id,
            "title": title,
            "order": next_order,
            "word_count": word_count,
            "chapter_count": chapter_count + 1,
            "attempts": attempts,
        }

    raise WriteConflictError(story_id)


# ----------------------------------------------------------------------
# Editing an existing chapter
#
# Both tools below take a caller-supplied `revision` and make exactly ONE
# attempt. That is the deliberate difference from _append_chapter's retry loop:
# there, the precondition guards server-derived counters the server can simply
# re-derive, so retrying is transparent to the caller's intent. Here it guards
# the base version the CALLER read and reasoned about. A lost precondition
# means that text is gone, and re-reading and re-applying server-side would
# silently overwrite whoever changed it — precisely the lost update the
# revision exists to prevent. So a precondition failure is reported, not
# retried, and the caller re-reads and decides.
# ----------------------------------------------------------------------


def _chapter_ref(db: Any, story_id: str, chapter_id: str) -> Any:
    return (
        db.collection("stories")
        .document(story_id)
        .collection("chapters")
        .document(chapter_id)
    )


def _load_chapter_for_edit(
    db: Any, story_id: str, chapter_id: str, revision: str
) -> Any:
    """Re-read the chapter and verify the caller's base version.

    Returns the snapshot so the update can precondition on the SAME
    update_time this check saw. The two are not redundant: this check turns the
    common case into a cheap, clearly-worded refusal before any content is
    built or validated, and the precondition closes the window between the
    check and the write, where a concurrent editor save would otherwise land.
    """
    snap = _chapter_ref(db, story_id, chapter_id).get()
    if not snap.exists:
        raise data.EntityNotFoundError(chapter_id)
    if _revision_token(snap.update_time) != revision:
        raise StaleRevisionError(chapter_id)
    return snap


def _touch_story(db: Any, story_id: str) -> None:
    """Bump the parent story's updatedAt. Best effort, failures swallowed.

    `updatedAt` is client-maintained (no trigger writes it), and stories
    missing it drop out of list_my_stories' order_by entirely — so it must be
    written. But it is written AFTER the chapter, and its failure must not
    propagate: the edit is already durable at that point, and raising would
    tell the model its edit failed. The model would then retry, idempotency
    would replay the recorded result, and the human would have been told
    something false for no gain. A stale updatedAt only mis-sorts the story
    until the next save from any path.

    No precondition: this is a pure touch, so contending with a concurrent
    writer over who stamps it last is meaningless.
    """
    try:
        db.collection("stories").document(story_id).update({"updatedAt": _now()})
    except Exception as exc:
        logger.warning(
            "mcp_write_story_touch_failed",
            story_id=story_id,
            error_type=type(exc).__name__,
        )


def _write_chapter_content(
    db: Any,
    story_id: str,
    chapter_id: str,
    snap: Any,
    stored: str,
    word_count: int,
) -> str:
    """Write content+wordCount under the read version; return the new revision.

    Writes exactly the two fields StoriesRepo.updateChapter writes minus the
    title, which no tool here edits. Notably NOT chapterCount: no chapter is
    created or destroyed by an edit.
    """
    try:
        result = _chapter_ref(db, story_id, chapter_id).update(
            {"content": stored, "wordCount": word_count},
            option=db.write_option(last_update_time=snap.update_time),
        )
    except gcp_exceptions.FailedPrecondition:
        # Someone wrote the chapter between the check above and this update.
        raise StaleRevisionError(chapter_id)
    except gcp_exceptions.NotFound:
        raise data.EntityNotFoundError(chapter_id)

    _touch_story(db, story_id)
    # Through the same canonicaliser as the read path: WriteResult.update_time
    # is a protobuf Timestamp where a snapshot's is a DatetimeWithNanoseconds,
    # and str() of the two never matches (see data.revision_token). Getting
    # this wrong would make every chained edit look stale.
    return _revision_token(result.update_time)


def append_to_chapter(
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

    # Ownership before the reservation, same write-amplifier argument as
    # create_chapter: a caller aiming at someone else's story should not make
    # this service write (and then release) an mcpWrites document.
    _owned_story_snapshot(db, story_id, uid)

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
    replay = _claim(db, key, uid, "append_to_chapter")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        snap = _load_chapter_for_edit(db, story_id, chapter_id, revision)
        existing = (snap.to_dict() or {}).get("content") or ""
        if existing:
            stored = existing + blocks.BLOCK_SEPARATOR + addition
        else:
            stored = addition
        word_count = _check_content_limits(stored)
        new_revision = _write_chapter_content(
            db, story_id, chapter_id, snap, stored, word_count
        )
        result = {
            "story_id": story_id,
            "chapter_id": chapter_id,
            "word_count": word_count,
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
            "revision": new_revision,
        }
    except Exception:
        _release(db, key)
        raise

    _record(db, key, result)
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
    the positions an later one refers to. It also makes the idempotency key
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


def edit_chapter_blocks(
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

    _owned_story_snapshot(db, story_id, uid)

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
    replay = _claim(db, key, uid, "edit_chapter_blocks")
    if replay is not None:
        return {**replay, "idempotent_replay": True}

    try:
        snap = _load_chapter_for_edit(db, story_id, chapter_id, revision)
        current = (snap.to_dict() or {}).get("content") or ""
        split = blocks.split_blocks(current)
        parts = [block.html for block in split]
        _apply_ops(parts, [block.tag for block in split], normalized)

        # Legitimately "" when every block was removed: that is the state
        # StoriesRepo.addChapter seeds a chapter in, and the editor renders it.
        stored = blocks.join_blocks(parts)
        word_count = _check_content_limits(stored)
        new_revision = _write_chapter_content(
            db, story_id, chapter_id, snap, stored, word_count
        )
        result = {
            "story_id": story_id,
            "chapter_id": chapter_id,
            "op_count": len(normalized),
            # Re-split rather than len(parts): one op's text can introduce
            # several blocks, and this is the index range the caller's next
            # call must address.
            "block_count": len(blocks.split_blocks(stored)),
            "word_count": word_count,
            "content_chars": len(stored),
            "revision": new_revision,
        }
    except Exception:
        _release(db, key)
        raise

    _record(db, key, result)
    return {**result, "idempotent_replay": False}
