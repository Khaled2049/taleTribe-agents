"""Chunk retrieval: the two shapes, and the jsonb decode that sat between them.

No database here. What is worth pinning is the projection -- search_chunks keeps
metadata in its own key so a metadata entry can never shadow a column, retrieve
flattens the same rows into the shape excerpts.format_excerpts renders -- and
the decode, which is where retrieval had been quietly failing.
"""

from agents.storyAgent.excerpts import format_excerpts
from agents.storyAgent.postgres_context import PostgresStoryContext, _jsonb


def test_jsonb_decodes_the_text_asyncpg_actually_returns():
    """No codec is registered on the pool, so jsonb arrives as a string."""
    assert _jsonb('{"title": "The Lamp Room", "chapterNumber": 1}') == {
        "title": "The Lamp Room",
        "chapterNumber": 1,
    }
    assert _jsonb({"title": "already decoded"}) == {"title": "already decoded"}


def test_jsonb_is_total_over_the_shapes_a_row_can_carry():
    """A malformed value must degrade to an unlabelled chunk, not kill the read."""
    for value in (None, "", "not json", "[1,2]", b'{"a": 1}', 7):
        assert isinstance(_jsonb(value), dict)
    assert _jsonb(b'{"a": 1}') == {"a": 1}


async def test_retrieve_flattens_search_chunks_for_the_chat_prompt(monkeypatch):
    store = PostgresStoryContext(dsn="postgres://unused")
    chunks = [
        {
            "chunk_id": "chunk-1",
            "kind": "chapter",
            "source_id": "chapter-1",
            "source_revision": 3,
            "chunk_index": 0,
            "indexed_at": None,
            "metadata": {"title": "The Lamp Room", "chapterNumber": 1},
            "text": "Brass polish.",
        }
    ]

    async def fake_search(story_id, embedding, top_k=4):
        assert story_id == "story-1"
        return chunks

    monkeypatch.setattr(store, "search_chunks", fake_search)
    excerpts = await store.retrieve("story-1", [0.0] * 768)

    assert excerpts == [
        {
            "kind": "chapter",
            "title": "The Lamp Room",
            "chapterNumber": 1,
            "text": "Brass polish.",
        }
    ]
    # The whole point of the flattening: format_excerpts can label the chunk.
    assert "[Ch1: The Lamp Room] Brass polish." in format_excerpts(excerpts)
