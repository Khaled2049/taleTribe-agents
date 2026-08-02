"""Unit tests for chapter RAG helpers (pure functions, no Firestore)."""

import pytest

from agents.storyAgent.brain.vector_store import _to_list
from agents.storyAgent.chapter_rag import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    _chunk_text,
    compose_entity_text,
    format_excerpts,
)

pytestmark = pytest.mark.unit


def test_chunk_empty_returns_empty():
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []


def test_chunk_short_text_single_chunk():
    chunks = _chunk_text("a b c d e")
    assert chunks == ["a b c d e"]


def test_chunk_long_text_overlaps_and_covers_all_words():
    words = [f"w{i}" for i in range(CHUNK_WORDS * 2 + 30)]
    text = " ".join(words)
    chunks = _chunk_text(text)

    assert len(chunks) >= 2
    # Every chunk is at most CHUNK_WORDS words.
    for c in chunks:
        assert len(c.split()) <= CHUNK_WORDS
    # Consecutive chunks overlap by CHUNK_OVERLAP_WORDS.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    assert first_words[step:] == second_words[:CHUNK_OVERLAP_WORDS]
    # All original words are present across chunks (no data dropped).
    assert words[-1] in chunks[-1].split()


def test_to_list_handles_list_and_vector_like():
    assert _to_list([1.0, 2.0]) == [1.0, 2.0]
    assert _to_list(None) == []

    class FakeVector:
        value = [3.0, 4.0]

    assert _to_list(FakeVector()) == [3.0, 4.0]


def test_format_excerpts_empty():
    assert format_excerpts([]) == ""


def test_format_excerpts_renders_chapter_and_entity_labels():
    out = format_excerpts(
        [
            {
                "kind": "chapter",
                "chapterNumber": 3,
                "title": "The Storm",
                "text": "Rain fell.",
            },
            {"kind": "character", "name": "Mara", "text": "A jaded smuggler."},
            {"kind": "place", "name": "Dockside", "text": "Foggy harbor."},
            {"kind": "plot", "name": "The Heist", "text": "Steal the relic."},
        ]
    )
    assert "RELEVANT STORY DETAILS" in out
    assert "[Ch3: The Storm] Rain fell." in out
    assert "[Character: Mara] A jaded smuggler." in out
    assert "[Place: Dockside] Foggy harbor." in out
    assert "[Plot: The Heist] Steal the relic." in out


def test_format_excerpts_defaults_unknown_kind_to_chapter():
    # legacy chunks written before `kind` existed have no kind field
    out = format_excerpts([{"chapterNumber": 1, "title": "Intro", "text": "Hi."}])
    assert "[Ch1: Intro] Hi." in out


def test_compose_entity_text_character():
    text = compose_entity_text(
        "character",
        {
            "name": "Mara",
            "personality": "Cunning and loyal.",
            "voice": "Clipped, wry.",
            "backstory": "Raised on the docks.",
            "affiliations": "The Dockside Crew.",
            "relationships": [
                {"name": "Theo", "type": "family", "description": "Her brother."}
            ],
        },
    )
    assert "Character: Mara" in text
    assert "Personality: Cunning and loyal." in text
    assert "Voice: Clipped, wry." in text
    assert "Backstory: Raised on the docks." in text
    assert "Affiliations: The Dockside Crew." in text
    assert "Relationship - Theo (family): Her brother." in text


def test_compose_entity_text_place_and_plot():
    place = compose_entity_text(
        "place",
        {"name": "Dockside", "description": "A foggy harbor.", "atmosphere": "tense"},
    )
    assert "Place: Dockside" in place
    assert "Description: A foggy harbor." in place
    assert "Atmosphere: tense" in place

    plot = compose_entity_text(
        "plot",
        {
            "title": "The Heist",
            "description": "Steal the relic.",
            "events": [{"name": "Recon", "content": "Scout the vault."}],
        },
    )
    assert "Plot: The Heist" in plot
    assert "Event - Recon: Scout the vault." in plot


def test_compose_entity_text_empty_returns_empty():
    assert compose_entity_text("character", {}) == "Character: Untitled"
    assert compose_entity_text("place", {}) == "Place: Untitled"


# The embedded-field set per kind MUST match the frontend's SIGNATURE_FIELDS in
# taleTribe-frontend/functions/src/entityIndexTrigger.ts, which decides when a
# re-embed fires. This mirror catches Python-side drift (a TS-side change still needs
# a human to update both). If you change one, change the other.
_FRONTEND_SIGNATURE_FIELDS = {
    "character": [
        "name",
        "age",
        "soul",
        "personality",
        "voice",
        "backstory",
        "affiliations",
        "notes",
        "relationships",
    ],
    "place": [
        "name",
        "description",
        "atmosphere",
        "geography",
        "history",
        "significance",
        "notes",
    ],
    "plot": ["name", "description", "events"],
}


@pytest.mark.parametrize("kind", ["character", "place", "plot"])
def test_entity_schema_matches_frontend_signature_fields(kind):
    from agents.storyAgent.entity_schema import embedded_field_names

    assert embedded_field_names(kind) == _FRONTEND_SIGNATURE_FIELDS[kind]
