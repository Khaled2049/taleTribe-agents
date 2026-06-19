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
_GOOGLE_DEFAULT_MODEL = "gemini-embedding-001"

# Single source of truth for the embedding dimension. EVERY vector we write
# (chapter_chunks AND semantic_memory) and EVERY native vector index must use this.
# Keep it in lockstep with:
#   novelsync-frontend/firestore.indexes.json
#     → fieldOverrides[chapter_chunks].vectorConfig.dimension
# If the embedder's output dim != this, native find_nearest can't match the index's
# vector dimension and queries fail — retrieval silently returns nothing (there is no
# brute-force fallback). That is exactly why this lives in one place and is checked at
# startup (see verify_embedding_dimension). See wiki/chat-scaling-design.md.
EXPECTED_EMBEDDING_DIM = 768

# Known output dimensions per Google embedding model (no network call needed).
# gemini-embedding-001 is Matryoshka (MRL): it natively emits 3072 dims but can be
# truncated via the `outputDimensionality` request param. We pin it to
# EXPECTED_EMBEDDING_DIM so it matches the 768-dim Firestore vector index (Firestore
# native KNN caps at 2048 dims, so the full 3072 isn't indexable anyway).
_GOOGLE_MODEL_DIMS = {
    "gemini-embedding-001": EXPECTED_EMBEDDING_DIM,
    "text-embedding-004": 768,  # retired on AI Studio; kept for reference
}

_MOCK_DIM = EXPECTED_EMBEDDING_DIM  # tests embed at the production dimension


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector length. Must equal EXPECTED_EMBEDDING_DIM for native
        Firestore vector search to work against the provisioned index."""
        ...


class GoogleAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = _GOOGLE_DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
        self._url = _GOOGLE_EMBED_URL.format(model=model)
        # One process-lifetime client (connection pool reuse). Creating a fresh
        # AsyncClient per embed() added TLS/handshake latency on every call, which
        # is now multiplied by several retrievals per chat turn. Closed via aclose().
        self._client = httpx.AsyncClient(timeout=30)

    @property
    def dimension(self) -> int:
        return _GOOGLE_MODEL_DIMS.get(self._model, EXPECTED_EMBEDDING_DIM)

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            self._url,
            headers={"x-goog-api-key": self._api_key},
            json={
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                # Truncate MRL models (e.g. gemini-embedding-001, native 3072) to
                # the dim our vector index expects. Sourced from self.dimension so
                # the request can never drift from the declared/verified dimension.
                "outputDimensionality": self.dimension,
            },
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    async def aclose(self) -> None:
        await self._client.aclose()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    async def embed(self, text: str) -> list[float]:
        result = await anyio.to_thread.run_sync(
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return result.tolist()


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings for tests — no real API calls."""

    @property
    def dimension(self) -> int:
        return _MOCK_DIM

    async def embed(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [((seed >> i) & 0xFF) / 255.0 for i in range(_MOCK_DIM)]


def get_embedding_provider(api_key: str | None = None) -> EmbeddingProvider | None:
    """Select embedding provider using the same env-var priority as the LLM provider."""
    if os.getenv("USE_MOCK", "").lower() == "true":
        return MockEmbeddingProvider()

    if api_key:
        # text-embedding-004 is the dedicated embedding model; don't use a generative model name
        embed_model = os.getenv("GOOGLE_EMBEDDING_MODEL", _GOOGLE_DEFAULT_MODEL)
        logger.info("Embedding provider: GoogleAI (%s)", embed_model)
        return GoogleAIEmbeddingProvider(api_key, embed_model)

    try:
        provider = SentenceTransformerEmbeddingProvider()
        logger.info("Embedding provider: SentenceTransformer (local)")
        return provider
    except ImportError:
        logger.warning(
            "No embedding provider available; brain memory retrieval disabled"
        )
        return None


def verify_embedding_dimension(embedder: EmbeddingProvider | None) -> None:
    """Assert the active embedder matches the dimension every vector store + index
    expects. Call once at startup so a misconfigured embedder surfaces loudly
    instead of silently rotting recall.

    Default behavior is a loud log.error (so dev with a 384-dim local model still
    runs); set STRICT_EMBEDDING_DIM=true to hard-fail instead — recommended in
    production, where a mismatch means the vector index is effectively dead.
    """
    if embedder is None:
        logger.warning(
            "No embedding provider available — chapter RAG and brain memory "
            "retrieval are DISABLED. Set GOOGLE_AI_STUDIO_API_KEY to enable them."
        )
        return

    dim = embedder.dimension
    if dim == EXPECTED_EMBEDDING_DIM:
        logger.info(
            "Embedding provider %s verified at %d dims (matches vector index).",
            type(embedder).__name__,
            dim,
        )
        return

    msg = (
        "Embedding dimension mismatch: provider %s outputs %d-dim vectors but every "
        "vector store and the Firestore vector index expect %d. Native vector search "
        "(find_nearest) will fail and retrieval will silently return nothing (there is "
        "no brute-force fallback). Use a %d-dim embedding model or "
        "reprovision the index to %d. See wiki/chat-scaling-design.md."
        % (
            type(embedder).__name__,
            dim,
            EXPECTED_EMBEDDING_DIM,
            EXPECTED_EMBEDDING_DIM,
            dim,
        )
    )
    if os.getenv("STRICT_EMBEDDING_DIM", "").lower() == "true":
        raise RuntimeError(msg)
    logger.error(msg)
