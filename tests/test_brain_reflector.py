"""Unit tests for MemoryReflector with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.storyAgent.brain.engine.reflector import (
    MemoryReflector,
    _parse_json_array,
    _parse_json_object,
    _parse_reflection_payload,
    _strip_fences,
)
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

    return (
        MemoryReflector(llm, working, procedural, semantic, episodic),
        working,
        procedural,
        semantic,
        episodic,
    )


def _make_input(response="Elena found the letter in the library."):
    assembled = AssembledPrompt(
        text="context", working_injected=True, procedural_injected=True
    )
    return ReflectionInput(
        user_message="Write the next scene",
        assistant_response=response,
        assembled_prompt=assembled,
    )


_COMBINED_PAYLOAD = """{
  "working": {
    "current_scene": "Library",
    "active_characters": ["Elena"],
    "recent_events": ["Found letter"],
    "mood": "tense"
  },
  "procedural": {},
  "semantic_facts": ["Elena is brave"],
  "episodic_summary": "Elena found a letter."
}"""


@pytest.mark.asyncio
async def test_reflect_calls_all_four_layers():
    reflector, working, procedural, semantic, episodic = _make_reflector(
        _COMBINED_PAYLOAD
    )

    await reflector.reflect(_make_input())

    reflector._llm.generate_content_async.assert_awaited_once()
    working.patch.assert_awaited_once()
    semantic.store.assert_awaited_once()
    episodic.store.assert_awaited_once()
    procedural.write_global.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflect_single_llm_call():
    reflector, *_ = _make_reflector(_COMBINED_PAYLOAD)
    await reflector.reflect(_make_input())
    assert reflector._llm.generate_content_async.await_count == 1


@pytest.mark.asyncio
async def test_reflect_survives_llm_failure():
    reflector, working, procedural, semantic, episodic = _make_reflector()
    reflector._llm.generate_content_async.side_effect = RuntimeError("API down")

    await reflector.reflect(_make_input())

    working.patch.assert_not_awaited()
    semantic.store.assert_not_awaited()
    episodic.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflect_writes_procedural_when_present():
    payload = """{
      "working": {},
      "procedural": {"tone": "melancholic", "pov": "first"},
      "semantic_facts": [],
      "episodic_summary": ""
    }"""
    reflector, working, procedural, semantic, episodic = _make_reflector(payload)
    await reflector.reflect(_make_input())
    procedural.write_global.assert_awaited_once_with(
        {"tone": "melancholic", "pov": "first"}
    )
    working.patch.assert_not_awaited()
    semantic.store.assert_not_awaited()
    episodic.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflect_salvages_partial_payload():
    # Only semantic_facts present — other keys missing entirely. Old code returned
    # early; new salvage logic should still write the one good layer.
    payload = '{"semantic_facts": ["Marcus drinks black coffee"]}'
    reflector, working, procedural, semantic, episodic = _make_reflector(payload)
    await reflector.reflect(_make_input())
    semantic.store.assert_awaited_once()
    working.patch.assert_not_awaited()
    procedural.write_global.assert_not_awaited()
    episodic.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflect_skips_episodic_on_empty_summary():
    payload = """{
      "working": {"current_scene": "x", "active_characters": [], "recent_events": [], "mood": ""},
      "procedural": {},
      "semantic_facts": [],
      "episodic_summary": ""
    }"""
    reflector, working, procedural, semantic, episodic = _make_reflector(payload)
    await reflector.reflect(_make_input())
    working.patch.assert_awaited_once()
    episodic.store.assert_not_awaited()


# --- Helper function tests ---


def test_parse_reflection_payload_combined():
    parsed = _parse_reflection_payload(_COMBINED_PAYLOAD)
    assert parsed["working"]["current_scene"] == "Library"
    assert parsed["semantic_facts"] == ["Elena is brave"]
    assert parsed["episodic_summary"] == "Elena found a letter."


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
