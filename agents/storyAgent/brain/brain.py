"""Brain — four-layer cognitive memory system public API."""
import asyncio
import logging
from typing import Optional

from google.cloud import firestore

from .types import BrainConfig, AssembledPrompt, ReflectionInput
from .engine.router import MemoryRouter
from .engine.assembler import PromptAssembler
from .engine.reflector import MemoryReflector
from .memory.working import WorkingMemoryLayer
from .memory.procedural import ProceduralMemoryLayer
from .memory.semantic import SemanticMemoryLayer
from .memory.episodic import EpisodicMemoryLayer

logger = logging.getLogger(__name__)


class Brain:
    """
    Four-layer cognitive memory: working, procedural, semantic, episodic.

    One Brain instance per request. Heavy shared objects (embedder, Firestore
    client, LLM provider) are injected from StoryAgent and loaded once per process.
    """

    def __init__(
        self,
        config: BrainConfig,
        llm_provider,
        embedder,
        db: Optional[firestore.Client] = None,
    ):
        if db is None:
            db = firestore.Client(project=config.project_id)

        self._config = config
        self._router = MemoryRouter()
        self._assembler = PromptAssembler()

        working = WorkingMemoryLayer(db, config.context_id)
        procedural = ProceduralMemoryLayer(db, config.user_id, config.context_id)
        semantic = SemanticMemoryLayer(db, config.context_id, embedder)
        episodic = EpisodicMemoryLayer(db, config.context_id, embedder)

        self._working = working
        self._procedural = procedural
        self._semantic = semantic
        self._episodic = episodic

        self._reflector = MemoryReflector(llm_provider, working, procedural, semantic, episodic)

    async def assemble(self, user_message: str, action_hint: str = "") -> AssembledPrompt:
        """Fetch all relevant memory layers concurrently and build layered prompt."""
        decision = self._router.route(user_message, action_hint)

        async def _noop_docs():
            return []

        semantic_coro = (
            self._semantic.retrieve(decision.semantic_query, self._config.semantic_top_k)
            if decision.fetch_semantic else _noop_docs()
        )
        episodic_coro = (
            self._episodic.retrieve(decision.episodic_query, self._config.episodic_top_k)
            if decision.fetch_episodic else _noop_docs()
        )
        working_coro = self._working.read() if decision.fetch_working else None
        procedural_coro = self._procedural.read() if decision.fetch_procedural else None

        results = await asyncio.gather(
            working_coro or _noop_result(None),
            procedural_coro or _noop_result(None),
            semantic_coro,
            episodic_coro,
            return_exceptions=True,
        )

        working = results[0] if not isinstance(results[0], Exception) else None
        procedural = results[1] if not isinstance(results[1], Exception) else None
        semantic_docs = results[2] if not isinstance(results[2], Exception) else []
        episodic_docs = results[3] if not isinstance(results[3], Exception) else []

        return self._assembler.assemble(user_message, working, procedural, semantic_docs, episodic_docs)

    async def reflect(self, reflection_input: ReflectionInput) -> None:
        """Update all memory layers based on assistant response. Never raises."""
        try:
            await self._reflector.reflect(reflection_input)
        except Exception:
            logger.exception(
                "Brain.reflect failed for context_id=%s", self._config.context_id
            )


async def _noop_result(value):
    return value
