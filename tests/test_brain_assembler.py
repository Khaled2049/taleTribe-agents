"""Unit tests for PromptAssembler — no I/O required."""

from datetime import datetime, timezone

from agents.storyAgent.brain.engine.assembler import PromptAssembler
from agents.storyAgent.brain.types import (
    MemoryDocument,
    ProceduralMemoryState,
    WorkingMemoryState,
)


def _doc(text, id="test-id"):
    return MemoryDocument(
        id=id, text=text, embedding=[], created_at=datetime.now(timezone.utc)
    )


def test_assemble_includes_all_sections():
    assembler = PromptAssembler()
    result = assembler.assemble(
        user_message="Write the next scene",
        working=WorkingMemoryState(current_scene="Library", mood="tense"),
        procedural=ProceduralMemoryState(tone="dark", genre="gothic"),
        semantic_docs=[_doc("Elena fears fire")],
        episodic_docs=[_doc("Elena fled the tower", id="ep-1")],
    )
    assert "WRITER STYLE" in result.text
    assert "dark" in result.text
    assert "gothic" in result.text
    assert "CURRENT SCENE" in result.text
    assert "Library" in result.text
    assert "RELEVANT FACTS" in result.text
    assert "Elena fears fire" in result.text
    assert "RELEVANT PAST EVENTS" in result.text
    assert "Elena fled the tower" in result.text
    assert "CURRENT REQUEST" in result.text
    assert "Write the next scene" in result.text
    assert result.semantic_count == 1
    assert result.episodic_count == 1
    assert result.working_injected is True
    assert result.procedural_injected is True


def test_assemble_empty_layers():
    assembler = PromptAssembler()
    result = assembler.assemble(
        user_message="Hello",
        working=None,
        procedural=None,
        semantic_docs=[],
        episodic_docs=[],
    )
    assert "CURRENT REQUEST" in result.text
    assert "Hello" in result.text
    assert result.semantic_count == 0
    assert result.episodic_count == 0
    assert result.working_injected is False
    assert result.procedural_injected is False


def test_assemble_skips_empty_procedural():
    assembler = PromptAssembler()
    result = assembler.assemble(
        user_message="Test",
        working=None,
        procedural=ProceduralMemoryState(),  # all empty
        semantic_docs=[],
        episodic_docs=[],
    )
    assert "WRITER STYLE" not in result.text
    assert result.procedural_injected is False


def test_assemble_uses_summary_for_episodic():
    assembler = PromptAssembler()
    doc = MemoryDocument(
        id="ep-1",
        text="Long full text of the episode...",
        embedding=[],
        created_at=datetime.now(timezone.utc),
        summary="Elena escaped the burning tower",
    )
    result = assembler.assemble("Q", None, None, [], [doc])
    assert "Elena escaped the burning tower" in result.text
    assert "Long full text" not in result.text


def test_assemble_narrative_rules_listed():
    assembler = PromptAssembler()
    result = assembler.assemble(
        user_message="Write",
        working=None,
        procedural=ProceduralMemoryState(
            tone="dark", narrative_rules=["No fourth wall", "Villain sympathetic"]
        ),
        semantic_docs=[],
        episodic_docs=[],
    )
    assert "No fourth wall" in result.text
    assert "Villain sympathetic" in result.text
