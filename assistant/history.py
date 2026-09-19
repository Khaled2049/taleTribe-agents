from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from mcp_server import story_data

logger = logging.getLogger(__name__)

REPLAYABLE_ROLES = ("user", "assistant")
COMPLETE = "complete"


@dataclass(frozen=True)
class HistoryLimits:
    max_messages: int = 10
    max_chars: int = 4_000

    @classmethod
    def from_settings(cls, settings: Any) -> "HistoryLimits":
        return cls(
            max_messages=settings.assistant_history_max_messages,
            max_chars=settings.assistant_history_max_chars,
        )


def _text_of(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks)


def _replayable(messages: list[dict]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in REPLAYABLE_ROLES:
            continue
        if message.get("status") != COMPLETE:
            continue
        text = _text_of(message.get("parts"))
        if text:
            turns.append((role, text))
    while turns and turns[-1][0] == "user":
        turns.pop()
    return turns


def _within_budget(
    turns: list[tuple[str, str]], max_chars: int
) -> list[tuple[str, str]]:
    if max_chars <= 0:
        return []
    kept: list[tuple[str, str]] = []
    spent = 0
    for role, text in reversed(turns):
        spent += len(text)
        if spent > max_chars and kept:
            break
        if spent > max_chars:
            return [(role, text[:max_chars])]
        kept.append((role, text))
    kept.reverse()
    return kept


async def prior_turns(
    *,
    uid: str,
    story_id: str,
    thread_id: Optional[str],
    limits: HistoryLimits,
) -> list[dict[str, Any]]:
    if not thread_id or limits.max_messages <= 0:
        return []
    try:
        client = story_data.client()
        thread = await client.get_assistant_thread(uid, story_id, thread_id)
        count = thread.get("messageCount")
        if not isinstance(count, int) or count <= 0:
            return []
        cursor = max(0, count - limits.max_messages)
        page = await client.list_assistant_messages(
            uid,
            story_id,
            thread_id,
            cursor=cursor,
            limit=limits.max_messages,
        )
    except story_data.NotFound:
        logger.info("assistant_history_thread_absent story_scoped=1")
        return []
    except story_data.StoryDataError:
        logger.warning("assistant_history_unavailable story_scoped=1")
        return []

    raw = page.get("messages") if isinstance(page, dict) else None
    if not isinstance(raw, list):
        return []
    turns = _within_budget(_replayable(raw), limits.max_chars)
    return [
        {"role": role, "parts": [{"type": "text", "text": text}]}
        for role, text in turns
    ]
