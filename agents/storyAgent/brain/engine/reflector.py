"""Memory reflector — post-response extraction pipeline."""

import asyncio
import json
import logging
import re

from ..memory.episodic import EpisodicMemoryLayer
from ..memory.procedural import ProceduralMemoryLayer
from ..memory.semantic import SemanticMemoryLayer
from ..memory.working import WorkingMemoryLayer
from ..types import ReflectionInput

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
        try:
            payload = await self._extract_reflection(inp)
        except Exception as exc:
            logger.warning("Brain reflect extraction failed: %s", exc, exc_info=True)
            return

        # Salvage: proceed if ANY layer key carries content. Old behavior required
        # a fully-parseable top-level object; partial JSON now still writes what it can.
        if not any(
            payload.get(k)
            for k in ("working", "procedural", "semantic_facts", "episodic_summary")
        ):
            return

        await self._apply_reflection(inp, payload)

    async def _extract_reflection(self, inp: ReflectionInput) -> dict:
        """Single LLM call to extract all memory-layer updates."""
        passage = inp.assistant_response[:3000]
        prompt = (
            "Analyze this story assistant passage and propose memory updates.\n"
            "Respond ONLY with valid JSON (no markdown, no extra text):\n"
            "{\n"
            '  "working": {\n'
            '    "current_scene": "...",\n'
            '    "active_characters": ["..."],\n'
            '    "recent_events": ["..."],\n'
            '    "mood": "..."\n'
            "  },\n"
            '  "procedural": {},\n'
            '  "semantic_facts": ["self-contained fact sentence", ...],\n'
            '  "episodic_summary": "1-2 sentence past-tense summary or empty string"\n'
            "}\n\n"
            "Rules:\n"
            "- working: current scene state from the passage.\n"
            "- procedural: only changed author style fields (tone, pov, style, genre, "
            "narrative_rules, preferences); use {} if none.\n"
            "- semantic_facts: up to 5 NEW factual sentences about characters/world/lore; [] if none.\n"
            '- episodic_summary: key narrative event in past tense, or "" if none.\n\n'
            f"Passage:\n{passage}"
        )
        raw = await self._llm.generate_content_async(prompt)
        return _parse_reflection_payload(raw)

    async def _apply_reflection(self, inp: ReflectionInput, payload: dict) -> None:
        tasks: list[tuple[str, object]] = []

        working = payload.get("working")
        if isinstance(working, dict) and working:
            events = working.get("recent_events", [])
            if isinstance(events, list) and len(events) > 5:
                working = {**working, "recent_events": events[-5:]}
            tasks.append(("working", self._working.patch(working)))

        procedural = payload.get("procedural")
        if isinstance(procedural, dict) and procedural:
            tasks.append(("procedural", self._procedural.write_global(procedural)))

        facts = payload.get("semantic_facts", [])
        if isinstance(facts, list):
            for fact in facts[:5]:
                if isinstance(fact, str) and fact.strip():
                    tasks.append(
                        (
                            "semantic",
                            self._semantic.store(fact.strip(), type="extracted_fact"),
                        )
                    )

        summary = payload.get("episodic_summary", "")
        if isinstance(summary, str) and summary.strip():
            tasks.append(
                (
                    "episodic",
                    self._episodic.store(
                        text=inp.assistant_response[:500],
                        summary=summary.strip(),
                    ),
                )
            )

        if not tasks:
            return

        results = await asyncio.gather(
            *(coro for _, coro in tasks), return_exceptions=True
        )
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning("Brain reflect[%s] failed: %s", name, result)


def _parse_reflection_payload(raw: str) -> dict:
    data = _parse_json_object(raw)
    if not data:
        return {}

    working = data.get("working")
    procedural = data.get("procedural")
    facts = data.get("semantic_facts", [])
    summary = data.get("episodic_summary", "")

    if not isinstance(working, dict):
        working = {}
    if not isinstance(procedural, dict):
        procedural = {}
    if not isinstance(facts, list):
        facts = _parse_json_array(json.dumps(facts)) if facts else []
    if not isinstance(summary, str):
        summary = str(summary) if summary else ""

    return {
        "working": working,
        "procedural": procedural,
        "semantic_facts": facts,
        "episodic_summary": summary,
    }


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
