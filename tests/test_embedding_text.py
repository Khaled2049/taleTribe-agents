"""Unit tests for the embedding/chunking helpers (pure functions, no database)."""

import pytest

from agents.storyAgent.embedding_text import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    _chunk_text,
    compose_entity_text,
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
