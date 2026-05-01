"""Embedding provider abstraction — mirrors the LLM provider pattern."""
import hashlib
import logging
import os
from abc import ABC, abstractmethod

import anyio
import httpx

logger = logging.getLogger(__name__)

_GOOGLE_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)
_GOOGLE_DEFAULT_MODEL = "text-embedding-004"
_MOCK_DIM = 768  # matches text-embedding-004 so tests use the same dimension


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class GoogleAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = _GOOGLE_DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
        self._url = _GOOGLE_EMBED_URL.format(model=model)

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._url,
                headers={"x-goog-api-key": self._api_key},
                json={
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                },
            )
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)

    async def embed(self, text: str) -> list[float]:
        result = await anyio.to_thread.run_sync(
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return result.tolist()


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings for tests — no real API calls."""

    async def embed(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [((seed >> i) & 0xFF) / 255.0 for i in range(_MOCK_DIM)]


def get_embedding_provider(api_key: str | None = None) -> EmbeddingProvider | None:
    """Select embedding provider using the same env-var priority as the LLM provider."""
    if os.getenv("USE_MOCK", "").lower() == "true":
        return MockEmbeddingProvider()

    if api_key:
        model = os.getenv("GOOGLE_AI_STUDIO_MODEL", _GOOGLE_DEFAULT_MODEL)
        # text-embedding-004 is the dedicated embedding model; don't use a generative model name
        embed_model = os.getenv("GOOGLE_EMBEDDING_MODEL", _GOOGLE_DEFAULT_MODEL)
        logger.info("Embedding provider: GoogleAI (%s)", embed_model)
        return GoogleAIEmbeddingProvider(api_key, embed_model)

    try:
        provider = SentenceTransformerEmbeddingProvider()
        logger.info("Embedding provider: SentenceTransformer (local)")
        return provider
    except ImportError:
        logger.warning("No embedding provider available; brain memory retrieval disabled")
        return None
