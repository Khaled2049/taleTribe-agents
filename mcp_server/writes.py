"""Owner-enforced Firestore writes backing the MCP write tools.

This module is the entire mutation surface of the MCP server. It is separate
from data.py on purpose: that module's promise is that it cannot change
anything, and a reviewer asking "what can this server mutate?" should have
exactly one file to read. Ownership is not re-implemented here — every
story-scoped write goes through data.get_owned_story_snapshot, the same gate
the read path uses.

Three things this file exists to get right:

1. **The Admin SDK bypasses firestore.rules.** Every ceiling the frontend
   relies on is unenforced on this path, so they are re-declared below as
   module constants and checked in code. Writing a document that violates them
   produces a story the owner can no longer save from the editor.

2. **Some derived fields have no trigger.** chapterIndex, users/{uid}.storyCount
   and re-embedding are maintained by Firestore triggers in the frontend repo,
   which fire on Admin SDK writes too. But `chapterCount`, `wordCount` and
   `stories/{id}.updatedAt` are client-maintained, so this module must write
   them itself. Missing `updatedAt` is the worst of the three: Firestore's
   order_by excludes documents lacking the field, so the story disappears from
   list_my_stories, and the frontend's mapStoryDoc calls .toDate() on it
   unguarded, so story lists throw.

3. **An LLM harness issues tool calls in parallel.** The frontend's own
   addChapter derives `order` from the denormalized `chapterCount`, which is a
   lost update under concurrency and additionally collides after any mid-book
   delete (deleteChapter decrements the counter without renumbering). This
   module claims `order` from a `nextChapterOrder` counter on the story doc,
   bumped in the SAME last_update_time-preconditioned update as chapterCount —
   deriving it from a separate max(order) read would leave a window between a
   winner's claim and its chapter write in which a second caller re-derives
   the same value (the claim moves the story doc, but the chapter that
   justifies the next order does not exist yet).

All functions are synchronous (google-cloud-firestore sync client); tools.py
bridges them with anyio.to_thread, matching data.py.
"""

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

from mcp_server import data
from mcp_server.oauth_store import as_utc

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
# StoriesRepo.WORD_LIMIT, and generateChapterTask.MAX_CHAPTER_WORDS.
MAX_CHAPTER_WORDS = 5_000
# StoriesRepo.CHAPTER_LIMIT.
MAX_CHAPTERS_PER_STORY = 50
# firestore.rules MAX_STORIES_PER_USER, via the userStoryCount() helper.
MAX_STORIES_PER_USER = 100

# Retries for the chapterCount precondition before giving up.
MAX_CLAIM_ATTEMPTS = 3

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
    if existing is None or _expired(existing):
        # TTL collection lags by hours; an expired reservation is ours to take.
        ref.set(record)
        return None
    result = existing.get("result")
    if isinstance(result, dict) and result:
        return result
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


def create_chapter(
    db: Any, uid: str, story_id: str, title: str, content: str = ""
) -> dict:
    """Append a chapter to a story owned by `uid`.

    `content` is plain text; it is escaped and paragraph-wrapped before storage.
    """
    title = _clean_title(title)
    stored = _to_paragraph_html(content)
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

    # Ownership before the reservation: without this, a caller aiming at a
    # story they don't own still makes the service write (and then release) an
    # mcpWrites document. One extra read closes that small write amplifier;
    # _append_chapter re-reads inside its retry loop regardless.
    data.get_owned_story_snapshot(db, story_id, uid)

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
        snap = data.get_owned_story_snapshot(db, story_id, uid)
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
