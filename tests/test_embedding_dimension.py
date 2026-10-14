"""Tests for the single embedding-dimension contract (#1).

Guards the invariant that every vector store and the Firestore vector index agree
on one dimension, and that a mismatch is surfaced loudly at startup instead of
silently rotting recall.
"""

import pytest

from agents.storyAgent.embedding_provider import (
    EXPECTED_EMBEDDING_DIM,
    EmbeddingProvider,
    MockEmbeddingProvider,
    verify_embedding_dimension,
)

pytestmark = pytest.mark.unit


class _WrongDimProvider(EmbeddingProvider):
    """A provider whose output dim deliberately disagrees with the index."""

    @property
    def dimension(self) -> int:
        return EXPECTED_EMBEDDING_DIM - 1  # e.g. a 384-dim model vs a 768-dim index

    async def embed(self, text):  # pragma: no cover - not exercised here
        return [0.0] * self.dimension


@pytest.mark.asyncio
async def test_mock_provider_matches_expected_dimension():
    provider = MockEmbeddingProvider()
    assert provider.dimension == EXPECTED_EMBEDDING_DIM
    emb = await provider.embed("hello")
    assert len(emb) == EXPECTED_EMBEDDING_DIM


def test_verify_warns_but_does_not_raise_by_default(monkeypatch, caplog):
    monkeypatch.delenv("STRICT_EMBEDDING_DIM", raising=False)
    with caplog.at_level("ERROR"):
        verify_embedding_dimension(_WrongDimProvider())  # must not raise
    assert any("dimension mismatch" in r.message.lower() for r in caplog.records)


def test_verify_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("STRICT_EMBEDDING_DIM", "true")
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        verify_embedding_dimension(_WrongDimProvider())
