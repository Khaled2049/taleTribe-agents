"""Memory reflector — post-response extraction pipeline."""
import asyncio
import json
import logging
import re

from ..types import ReflectionInput
from ..memory.working import WorkingMemoryLayer
from ..memory.procedural import ProceduralMemoryLayer
from ..memory.semantic import SemanticMemoryLayer
from ..memory.episodic import EpisodicMemoryLayer

logger = logging.getLogger(__name__)


class MemoryReflector:
    def __init__(
        self,
        llm_provider,
        working: WorkingMemoryLayer,
        procedural: ProceduralMemoryLayer,
        semantic: SemanticMemoryLayer,
        episodic: EpisodicMemoryLayer,
    ):
        self._llm = llm_provider
        self._working = working
        self._procedural = procedural
        self._semantic = semantic
        self._episodic = episodic

    async def reflect(self, inp: ReflectionInput) -> None:
        results = await asyncio.gather(
            self._update_working(inp),
            self._update_procedural(inp),
            self._extract_semantic(inp),
            self._extract_episodic(inp),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_names = ["working", "procedural", "semantic", "episodic"]
                logger.warning("Brain reflect[%s] failed: %s", task_names[i], result)

    async def _update_working(self, inp: ReflectionInput) -> None:
        prompt = (
            "Extract the current scene state from this story passage.\n"
            "Respond ONLY with valid JSON (no markdown, no extra text):\n"
            '{"current_scene": "...", "active_characters": ["..."], "recent_events": ["..."], "mood": "..."}\n\n'
            f"Passage:\n{inp.assistant_response[:2000]}"
        )
        raw = await self._llm.generate_content_async(prompt)
        data = _parse_json_object(raw)
        if data:
            # Keep recent_events bounded to last 5
            events = data.get("recent_events", [])
            if isinstance(events, list) and len(events) > 5:
                data["recent_events"] = events[-5:]
            await self._working.patch(data)

    async def _update_procedural(self, inp: ReflectionInput) -> None:
        prompt = (
            "Did this story passage reveal any author style preferences "
            "(tone, POV, narrative rules, genre)?\n"
            "If yes, respond with ONLY a JSON object of changed fields (e.g. {\"tone\": \"melancholic\"}).\n"
            "If no style signals, respond with ONLY: {}\n\n"
            f"Passage:\n{inp.assistant_response[:1000]}"
        )
        raw = await self._llm.generate_content_async(prompt)
        data = _parse_json_object(raw)
        if data:
            await self._procedural.write_global(data)

    async def _extract_semantic(self, inp: ReflectionInput) -> None:
        prompt = (
            "Extract NEW factual statements about characters, world, or lore from this text.\n"
            "Each fact must be a single self-contained sentence.\n"
            "Respond ONLY with a JSON array of strings. Max 5 facts. If none, return [].\n\n"
            f"Text:\n{inp.assistant_response[:3000]}"
        )
        raw = await self._llm.generate_content_async(prompt)
        facts = _parse_json_array(raw)
        for fact in facts[:5]:
            if isinstance(fact, str) and fact.strip():
                await self._semantic.store(fact.strip(), type="extracted_fact")

    async def _extract_episodic(self, inp: ReflectionInput) -> None:
        prompt = (
            "Summarize the key narrative event(s) in this story passage as 1-2 sentences in past tense.\n"
            "This will be stored as a memory entry.\n"
            "If no significant event occurred, respond with an empty string.\n\n"
            f"Passage:\n{inp.assistant_response[:3000]}"
        )
        summary = (await self._llm.generate_content_async(prompt)).strip()
        if summary:
            await self._episodic.store(
                text=inp.assistant_response[:500],
                summary=summary,
            )


def _parse_json_object(raw: str) -> dict | None:
    try:
        text = _strip_fences(raw)
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_json_array(raw: str) -> list:
    try:
        text = _strip_fences(raw)
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
