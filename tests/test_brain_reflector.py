"""Unit tests for MemoryReflector with mocked dependencies."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.storyAgent.brain.engine.reflector import MemoryReflector, _parse_json_object, _parse_json_array, _strip_fences
from agents.storyAgent.brain.types import AssembledPrompt, ReflectionInput


def _make_reflector(llm_response: str = "{}"):
    llm = MagicMock()
    llm.generate_content_async = AsyncMock(return_value=llm_response)

    working = MagicMock()
    working.patch = AsyncMock()

    procedural = MagicMock()
    procedural.write_global = AsyncMock()

    semantic = MagicMock()
    semantic.store = AsyncMock(return_value="doc-id")

    episodic = MagicMock()
    episodic.store = AsyncMock(return_value="ep-id")

    return MemoryReflector(llm, working, procedural, semantic, episodic), working, procedural, semantic, episodic


def _make_input(response="Elena found the letter in the library."):
    assembled = AssembledPrompt(text="context", working_injected=True, procedural_injected=True)
    return ReflectionInput(
        user_message="Write the next scene",
        assistant_response=response,
        assembled_prompt=assembled,
    )


@pytest.mark.asyncio
async def test_reflect_calls_all_four_layers():
    working_json = '{"current_scene": "Library", "active_characters": ["Elena"], "recent_events": ["Found letter"], "mood": "tense"}'
    reflector, working, procedural, semantic, episodic = _make_reflector(working_json)

    # Override each call to return appropriate values
    call_count = [0]
    responses = [
        working_json,           # _update_working
        "{}",                   # _update_procedural — no style change
        '["Elena is brave"]',   # _extract_semantic
        "Elena found a letter.", # _extract_episodic
    ]

    async def side_effect(prompt):
        r = responses[call_count[0] % len(responses)]
        call_count[0] += 1
        return r

    reflector._llm.generate_content_async.side_effect = side_effect

    await reflector.reflect(_make_input())

    working.patch.assert_awaited_once()
    # procedural.write_global not called — empty dict returned
    semantic.store.assert_awaited_once()
    episodic.store.assert_awaited_once()


@pytest.mark.asyncio
async def test_reflect_survives_llm_failure():
    reflector, working, procedural, semantic, episodic = _make_reflector()
    reflector._llm.generate_content_async.side_effect = RuntimeError("API down")

    # Should not raise — return_exceptions=True in gather
    await reflector.reflect(_make_input())


@pytest.mark.asyncio
async def test_reflect_skips_episodic_on_empty_summary():
    reflector, working, procedural, semantic, episodic = _make_reflector()
    responses = [
        '{"current_scene": "x", "active_characters": [], "recent_events": [], "mood": ""}',
        "{}",
        "[]",
        "",  # empty summary — no episodic store
    ]
    call_count = [0]

    async def side_effect(prompt):
        r = responses[call_count[0] % len(responses)]
        call_count[0] += 1
        return r

    reflector._llm.generate_content_async.side_effect = side_effect
    await reflector.reflect(_make_input())
    episodic.store.assert_not_awaited()


# --- Helper function tests ---

def test_parse_json_object_clean():
    assert _parse_json_object('{"tone": "dark"}') == {"tone": "dark"}


def test_parse_json_object_with_fences():
    assert _parse_json_object('```json\n{"tone": "dark"}\n```') == {"tone": "dark"}


def test_parse_json_object_invalid():
    assert _parse_json_object("not json") is None


def test_parse_json_array_clean():
    assert _parse_json_array('["fact one", "fact two"]') == ["fact one", "fact two"]


def test_parse_json_array_wrapped():
    assert _parse_json_array('{"facts": ["a", "b"]}') == ["a", "b"]


def test_parse_json_array_empty():
    assert _parse_json_array("[]") == []


def test_strip_fences_removes_markdown():
    assert _strip_fences("```json\n{}\n```").strip() == "{}"


def test_strip_fences_passthrough():
    assert _strip_fences('{"x": 1}') == '{"x": 1}'
