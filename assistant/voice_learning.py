"""Conservative background learning from successfully saved chapter revisions.

The queue is created by a story-data chapter trigger. This worker only claims
jobs in PostgreSQL; it reads chapters and writes voice notes through story-data.
No raw passage or profile note is logged. A model proposes, and the checks here
and in story-data decide whether anything may change.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agents.storyAgent.llm_provider import _byok_config, _firebase_token, _usage_feature
from mcp_server import story_data

logger = logging.getLogger(__name__)


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain(raw: str) -> str:
    parser = _Text()
    parser.feed(raw)
    return " ".join(" ".join(parser.parts).split())


class VoiceProposal(BaseModel):
    operation: Literal["no_change", "add", "revise", "retire"]
    note_id: str = ""
    text: str = Field(default="", max_length=240)
    reason: str = Field(default="", max_length=500)
    evidence_indices: list[int] = Field(default_factory=list, max_length=3)


@dataclass(frozen=True)
class Passage:
    chapter_id: str
    source_revision: int
    start: int
    end: int
    text: str

    def reference(self) -> dict[str, Any]:
        return {
            "chapterId": self.chapter_id,
            "sourceRevision": self.source_revision,
            "start": self.start,
            "end": self.end,
        }


def _passages(source: dict, other: dict) -> list[Passage]:
    result: list[Passage] = []
    for chapter, ranges in ((source, ((0, 1000), (1300, 2300))), (other, ((0, 1000),))):
        content = chapter.get("content", "")
        if not isinstance(content, str):
            return []
        for start, end in ranges:
            if len(content) < end:
                return []
            text = _plain(content[start:end])[:750]
            if len(text) < 180:
                return []
            result.append(
                Passage(str(chapter["id"]), int(chapter["revision"]), start, end, text)
            )
    return result


def _too_similar(text: str, notes: list[dict]) -> bool:
    normalized = text.casefold().strip()
    words = set(re.findall(r"[a-z]{4,}", normalized))
    for note in notes:
        existing = str(note.get("text", "")).casefold().strip()
        if SequenceMatcher(None, normalized, existing).ratio() >= 0.72:
            return True
        known = set(re.findall(r"[a-z]{4,}", existing))
        if (
            note.get("origin") == "writer"
            and words
            and known
            and len(words & known) / len(words | known) > 0.55
        ):
            return True
    return False


def validate_proposal(proposal: VoiceProposal, notes: list[dict]) -> bool:
    if proposal.operation == "no_change":
        return False
    if sorted(proposal.evidence_indices) != [0, 1, 2]:
        return False
    if len(proposal.reason.strip()) < 8:
        return False
    if proposal.operation in ("add", "revise"):
        if len(proposal.text.strip()) < 12 or _too_similar(proposal.text, notes):
            return False
    if proposal.operation == "add":
        return not proposal.note_id and len(notes) < 12
    return any(
        note.get("id") == proposal.note_id
        and note.get("origin") == "agent"
        and not note.get("pinned")
        for note in notes
    )


class VoiceLearningWorker:
    def __init__(self, pool: Any, provider: Any) -> None:
        self.pool, self.provider = pool, provider

    async def run_once(self) -> int:
        if self.pool is None:
            return 0
        jobs = await self.pool.fetch("""WITH next AS (
                 SELECT id FROM story_voice_learning_jobs WHERE processed_at IS NULL
                   AND available_at <= now()
                   AND (lease_expires_at IS NULL OR lease_expires_at < now())
                   AND attempts < 3 ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 3
               ) UPDATE story_voice_learning_jobs j
               SET lease_expires_at=now()+interval '5 minutes', attempts=attempts+1
               FROM next, stories s WHERE j.id=next.id AND s.id=j.story_id
               RETURNING j.id,j.story_id,j.chapter_id,j.source_revision,s.owner_id""")
        for job in jobs:
            try:
                await self._process(job)
                await self.pool.execute(
                    "UPDATE story_voice_learning_jobs SET processed_at=now(),lease_expires_at=NULL WHERE id=$1",
                    job["id"],
                )
            except Exception:
                # A transient API/model failure can retry. The log never holds text.
                logger.exception(
                    "voice_learning_job_failed", extra={"job_id": str(job["id"])}
                )
                await self.pool.execute(
                    "UPDATE story_voice_learning_jobs SET lease_expires_at=NULL,available_at=now()+interval '30 minutes' WHERE id=$1",
                    job["id"],
                )
        return len(jobs)

    async def eligible(self, job: Any) -> bool:
        """Cheap preflight before a frontend request spends daily AI quota."""
        if await self._material(job) is None:
            return False
        state = await self.pool.fetchrow(
            "SELECT last_analyzed_at < now()-interval '30 minutes' AND "
            "(day <> (now() AT TIME ZONE 'utc')::date OR pass_count < 3) AS allowed "
            "FROM story_voice_learning_state WHERE story_id=$1",
            job["story_id"],
        )
        return state is None or bool(state["allowed"])

    async def _material(self, job: Any):
        uid, story_id, chapter_id = (
            str(job["owner_id"]),
            str(job["story_id"]),
            str(job["chapter_id"]),
        )
        client = story_data.client()
        profile = await client.get_voice_profile(uid, story_id)
        if profile.get("learnFromWriting") is not True:
            return None
        source = await client.get_chapter(uid, story_id, chapter_id)
        if int(source.get("revision", 0)) != int(job["source_revision"]):
            return None
        index = await client.list_chapter_index(uid, story_id)
        candidates = [row for row in index if row.get("id") != chapter_id]
        if not candidates:
            return None
        for candidate in reversed(candidates[-4:]):
            other = await client.get_chapter(uid, story_id, str(candidate["id"]))
            passages = _passages(source, other)
            if len(passages) == 3:
                return profile, passages
        return None

    async def _process(
        self,
        job: Any,
        provider_config: dict | None = None,
        firebase_token: str | None = None,
    ) -> str:
        uid, story_id = str(job["owner_id"]), str(job["story_id"])
        client = story_data.client()
        material = await self._material(job)
        if material is None:
            return "skipped"
        profile, passages = material
        # Cost and frequency are enforced durably before model inference.
        allowed = await self.pool.fetchrow(
            """INSERT INTO story_voice_learning_state(story_id,last_analyzed_at,day,pass_count)
               VALUES($1,now(),(now() AT TIME ZONE 'utc')::date,1)
               ON CONFLICT(story_id) DO UPDATE SET
                 last_analyzed_at=now(), day=(now() AT TIME ZONE 'utc')::date,
                 pass_count=CASE WHEN story_voice_learning_state.day=(now() AT TIME ZONE 'utc')::date
                   THEN story_voice_learning_state.pass_count+1 ELSE 1 END
               WHERE story_voice_learning_state.last_analyzed_at < now()-interval '30 minutes'
                 AND (story_voice_learning_state.day <> (now() AT TIME ZONE 'utc')::date
                   OR story_voice_learning_state.pass_count < 3)
               RETURNING pass_count""",
            job["story_id"],
        )
        if allowed is None:
            return "skipped"
        notes = profile.get("notes") if isinstance(profile.get("notes"), list) else []
        prompt = (
            "Analyze repeated, story-wide prose patterns across three independent saved passages "
            "from two chapters. A single correction, grammar fix, plot detail, or uncertain pattern "
            "means no_change. Do not copy passage text into the note or reason. Respect writer notes. "
            "Return only JSON with operation (no_change|add|revise|retire), note_id, text, reason, "
            "evidence_indices. For any change, use all three indices [0,1,2]. Keep text under 240 chars.\n"
            + json.dumps(
                {"notes": notes, "passages": [p.text for p in passages]},
                ensure_ascii=False,
            )
        )
        billing = _byok_config.set({"user_id": uid, **(provider_config or {})})
        identity = _firebase_token.set(firebase_token)
        feature = _usage_feature.set("voice_analysis")
        try:
            raw = await self.provider.generate_content_async(
                prompt, max_output_tokens=400
            )
        finally:
            _usage_feature.reset(feature)
            _byok_config.reset(billing)
            _firebase_token.reset(identity)
        try:
            proposal = VoiceProposal.model_validate_json(raw)
        except (ValidationError, ValueError):
            return "analyzed"
        if not validate_proposal(proposal, notes):
            return "analyzed"
        # The API rechecks every source revision, the switch, and the target
        # note while holding the story lock. A conflict is skipped, never forced.
        op = {
            "operation": proposal.operation,
            "noteId": proposal.note_id,
            "text": proposal.text.strip(),
            "reason": proposal.reason.strip(),
            "source": "passive",
            "evidence": [p.reference() for p in passages],
            "idempotencyKey": str(uuid.uuid5(uuid.NAMESPACE_URL, str(job["id"]))),
        }
        try:
            await client.update_story_voice_note(
                uid, story_id, int(profile["revision"]), op
            )
        except (story_data.Conflict, story_data.Rejected, story_data.NotFound):
            return "analyzed"
        return "updated"
