"""Main ADK agent implementation for story generation."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Handle imports for both direct execution and module import
try:
    from .brain import Brain, BrainConfig, ReflectionInput
    from .chapter_rag import ChapterRAG, format_excerpts
    from .llm_provider import get_llm_provider
    from .tools import (
        BrainstormingTool,
        ChapterGenerationTool,
        CharacterBrainstormingTool,
        ChatWithContextTool,
        EnhanceTextTool,
        EnhanceWizardInputTool,
        NextLineGenerationTool,
        PlotBrainstormingTool,
        StoryChoicesTool,
    )
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.brain import Brain, BrainConfig, ReflectionInput
    from agents.storyAgent.chapter_rag import ChapterRAG, format_excerpts
    from agents.storyAgent.llm_provider import get_llm_provider
    from agents.storyAgent.tools import (
        BrainstormingTool,
        ChapterGenerationTool,
        CharacterBrainstormingTool,
        ChatWithContextTool,
        EnhanceTextTool,
        EnhanceWizardInputTool,
        NextLineGenerationTool,
        PlotBrainstormingTool,
        StoryChoicesTool,
    )


class StoryAgent:
    """Main agent for story generation and brainstorming."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        """
        Initialize the story agent.

        Args:
            project_id: GCP project ID (defaults to environment variable, required for Firestore)
            location: GCP location (not used, kept for backward compatibility)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("project_id must be provided or set GOOGLE_CLOUD_PROJECT")

        self.location = location

        # Shared brain resources — loaded once per process
        self._llm_provider = get_llm_provider(self.project_id, self.location)
        self._embedder = _load_embedder()
        self._db = _get_firestore_client(self.project_id)

        # Initialize tools
        self.chapter_tool = ChapterGenerationTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.brainstorm_tool = BrainstormingTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.character_tool = CharacterBrainstormingTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.plot_tool = PlotBrainstormingTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.next_line_tool = NextLineGenerationTool(
            self.project_id, self.location, llm_provider=self._llm_provider, db=self._db
        )
        self.chat_tool = ChatWithContextTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.enhance_text_tool = EnhanceTextTool(
            self.project_id, self.location, llm_provider=self._llm_provider, db=self._db
        )
        self.enhance_wizard_tool = EnhanceWizardInputTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.story_choices_tool = StoryChoicesTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        # Chapter RAG shares the process-wide embedder + Firestore client.
        self.chapter_rag = ChapterRAG(self._db, self._embedder)

    @property
    def llm_provider(self):
        """The shared CreditProxyProvider — also used for credit balance/top-up."""
        return self._llm_provider

    async def aclose(self) -> None:
        """Release process-lifetime resources (e.g. the LLM + embedding HTTP clients)."""
        for provider in (self._llm_provider, self._embedder):
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()

    def _make_brain(self, user_id: str, context_id: str) -> Brain:
        return Brain(
            config=BrainConfig(
                user_id=user_id,
                context_id=context_id,
                project_id=self.project_id,
            ),
            llm_provider=self._llm_provider,
            embedder=self._embedder,
            db=self._db,
        )

    async def generate_next_lines(
        self,
        story_id: str,
        content: str,
        cursorPosition: int,
        chapter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate 3 next line suggestions based on chapter content and cursor position.

        Args:
            story_id: Firestore story document ID
            content: Current content of the chapter being edited
            cursorPosition: Character index where the new line should be inserted
            chapter_id: Optional chapter document ID for better context and validation

        Returns:
            Dictionary containing the suggestions array.
        """
        return await self.next_line_tool.execute(
            story_id, content, cursorPosition, chapter_id
        )

    async def generate_chapter(
        self,
        story_id: str,
        chapter_number: int,
        previous_chapters: Optional[list] = None,
        plot_context: Optional[str] = None,
        order: Optional[float] = None,
        prev_chapter: Optional[Dict[str, Any]] = None,
        next_chapter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chapter.

        Args:
            story_id: Firestore story document ID
            chapter_number: Chapter number to generate
            previous_chapters: Legacy full list of previous chapters
            order: Float ordering key of the chapter being generated
            prev_chapter: Full immediate previous neighbor
            next_chapter: Full immediate next neighbor (for mid-story inserts)

        Returns:
            Generated chapter content
        """
        return await self.chapter_tool.execute(
            story_id,
            chapter_number,
            previous_chapters=previous_chapters,
            plot_context=plot_context,
            order=order,
            prev_chapter=prev_chapter,
            next_chapter=next_chapter,
        )

    async def brainstorm_ideas(
        self,
        story_id: str,
        idea_type: str,
        prompt: Optional[str] = None,
        count: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate brainstorming ideas.

        Args:
            story_id: Firestore story document ID
            idea_type: Type of idea (characters/plots/places/themes)
            prompt: Optional specific prompt
            count: Number of ideas to generate

        Returns:
            Dictionary with generated ideas
        """
        return await self.brainstorm_tool.execute(story_id, idea_type, prompt, count)

    async def brainstorm_character(
        self,
        story_id: str,
        role: Optional[str] = None,
        archetype: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate character ideas.

        Args:
            story_id: Firestore story document ID
            role: Optional character role
            archetype: Optional character archetype

        Returns:
            Dictionary with character profile
        """
        return await self.character_tool.execute(story_id, role, archetype)

    async def brainstorm_plot(
        self,
        story_id: str,
        plot_type: str = "conflict",
    ) -> Dict[str, Any]:
        """
        Generate plot ideas.

        Args:
            story_id: Firestore story document ID
            plot_type: Type of plot element

        Returns:
            Dictionary with plot suggestions
        """
        return await self.plot_tool.execute(story_id, plot_type)

    async def chat_with_context(
        self,
        story_id: str,
        message: str,
        chat_history: Optional[list] = None,
        user_id: str = "anonymous",
        background_tasks=None,
    ) -> Dict[str, Any]:
        """
        Generate chat response using story context with optional brain-augmented memory.

        Args:
            story_id: Firestore story document ID
            message: User's message
            chat_history: Optional list of previous messages for conversational context
            user_id: User identifier for procedural memory scoping
            background_tasks: FastAPI BackgroundTasks for async reflection

        Returns:
            Dictionary containing response and context usage
        """
        brain_context = None
        brain = None
        assembled = None
        chapter_excerpts = None

        if self._embedder is not None:
            log = logging.getLogger(__name__)

            # Embed the message ONCE. Chapter retrieval and semantic memory both query
            # on the full message, so they share this vector (episodic uses a truncated
            # query and embeds its own). Saves one embedding API call per chat turn.
            query_vec: Optional[List[float]] = None
            try:
                query_vec = await self._embedder.embed(message)
            except Exception:
                log.warning(
                    "query embedding failed for story_id=%s; retrieval degraded",
                    story_id,
                )

            try:
                brain = self._make_brain(user_id, story_id)
            except Exception:
                brain = None
                log.warning("make_brain failed for story_id=%s", story_id)

            # Chapter retrieval and brain assembly are independent → run concurrently.
            async def _retrieve_excerpts():
                excerpts = await self.chapter_rag.retrieve(
                    story_id, message, top_k=4, query_embedding=query_vec
                )
                return format_excerpts(excerpts) or None

            async def _assemble_brain():
                if brain is None:
                    return None
                return await brain.assemble(
                    message,
                    action_hint="chatWithContext",
                    query_embedding=query_vec,
                )

            excerpts_res, assemble_res = await asyncio.gather(
                _retrieve_excerpts(), _assemble_brain(), return_exceptions=True
            )

            if isinstance(excerpts_res, Exception):
                log.warning(
                    "chapter_rag.retrieve failed for story_id=%s, continuing without excerpts",
                    story_id,
                )
            else:
                chapter_excerpts = excerpts_res

            if isinstance(assemble_res, Exception):
                log.warning(
                    "Brain.assemble failed for story_id=%s, falling back", story_id
                )
                brain = None
            elif assemble_res is not None:
                assembled = assemble_res
                brain_context = (
                    assembled.text if _assembled_has_memory(assembled) else None
                )
                if brain_context:
                    brain_context = (
                        brain_context.split("\n=== CURRENT REQUEST ===")[0].strip()
                        or None
                    )
                log.info(
                    "Full brain_context for chat story_id=%s:\n%s",
                    story_id,
                    brain_context,
                )

        result = await self.chat_tool.execute(
            story_id,
            message,
            chat_history,
            brain_context=brain_context,
            chapter_excerpts=chapter_excerpts,
        )

        if brain is not None and assembled is not None and background_tasks is not None:
            response_text = result.get("response", "")
            if response_text:
                ri = ReflectionInput(
                    user_message=message,
                    assistant_response=response_text,
                    assembled_prompt=assembled,
                )
                background_tasks.add_task(brain.reflect, ri)

        return result

    async def enhance_text(
        self,
        story_id: str,
        action: str,
        selected_text: str,
        chapter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance selected text based on action type.

        Args:
            story_id: Firestore story document ID
            action: Action type (expand, dialogue, rewrite)
            selected_text: The text to enhance
            chapter_id: Optional chapter document ID for better context

        Returns:
            Dictionary containing enhanced text
        """
        return await self.enhance_text_tool.execute(
            story_id, action, selected_text, chapter_id
        )

    async def summarize_chapter(
        self,
        story_id: str,
        content: str,
        chapter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Summarize a chapter's content for story continuity.

        Returns a short summary used to keep long-range continuity cheap on
        future chapter generations. No extra context build — operates directly
        on the supplied chapter text.
        """
        text = (content or "").strip()
        if not text:
            return {"summary": ""}

        prompt = (
            "Summarize the following chapter in 2-3 sentences for story "
            "continuity. Capture key plot events, character developments, and "
            "unresolved threads. Return ONLY the summary text, with no preamble "
            "or labels.\n\n"
            f"CHAPTER:\n{text}"
        )
        summary = await self.chapter_tool.llm_provider.generate_content_async(
            prompt, max_output_tokens=256
        )
        return {"summary": (summary or "").strip()}

    async def generate_story_choices(
        self,
        story_id: str,
        mode: str,
        current_content: str = "",
        chapter_id: Optional[str] = None,
        turn_count: int = 0,
        user_id: str = "anonymous",
        background_tasks=None,
    ) -> Dict[str, Any]:
        """
        Generate interactive story choices for the co-write feature.

        Args:
            story_id: Firestore story document ID
            mode: "opening", "continuation", or "ending"
            current_content: HTML already in the editor (empty string for opening)
            chapter_id: Optional chapter document ID for chapter-specific context
            turn_count: How many choices the user has selected (used for arc-aware prompting)
            user_id: User identifier for procedural memory scoping
            background_tasks: FastAPI BackgroundTasks for async brain reflection

        Returns:
            For opening: {"storyId", "openingScene", "choices": [{label, sceneText}, ...]}
            For continuation: {"storyId", "choices": [{label, sceneText}, ...]}
            For ending: {"storyId", "choices": [{label, sceneText, isFinal: true}]}
        """
        logger = logging.getLogger(__name__)
        brain_context = None
        brain = None
        assembled = None

        if self._embedder is not None:
            try:
                logger.info(
                    "Generating story choices for story_id=%s mode=%s", story_id, mode
                )
                brain = self._make_brain(user_id, story_id)
                query = (
                    f"{mode} scene. {current_content[:200]}"
                    if current_content
                    else f"{mode} scene"
                )
                assembled = await brain.assemble(
                    query, action_hint="generateStoryChoices"
                )
                brain_context = (
                    assembled.text if _assembled_has_memory(assembled) else None
                )
                if brain_context:
                    brain_context = (
                        brain_context.split("\n=== CURRENT REQUEST ===")[0].strip()
                        or None
                    )
            except Exception:
                logger.warning(
                    "Brain assembly failed for generateStoryChoices story_id=%s, falling back to legacy context",
                    story_id,
                )

        result = await self.story_choices_tool.execute(
            story_id,
            mode,
            current_content,
            chapter_id,
            turn_count,
            brain_context=brain_context,
        )

        if brain is not None and assembled is not None and background_tasks is not None:
            prose = _extract_choices_prose(result)
            if prose:
                ri = ReflectionInput(
                    user_message=f"Generate {mode} story choices",
                    assistant_response=prose,
                    assembled_prompt=assembled,
                )
                background_tasks.add_task(brain.reflect, ri)
                logger.info(
                    "Brain reflection scheduled for generateStoryChoices story_id=%s mode=%s",
                    story_id,
                    mode,
                )

        return result

    async def enhance_wizard_input(
        self,
        user_id: str,
        wizard_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhance wizard input across premise/character/place/conflict/blueprint."""
        return await self.enhance_wizard_tool.execute(user_id, wizard_type, data)

    async def index_chapter(
        self,
        story_id: str,
        chapter_id: str,
        title: str = "",
        content: str = "",
        chapter_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Embed a chapter's body into the vector index. Called by the chapter-write
        trigger so retrieval cost is paid once per edit, not per chat message."""
        if self._embedder is None:
            logging.getLogger(__name__).info(
                "index_chapter skipped (no embedder) story_id=%s chapter_id=%s",
                story_id,
                chapter_id,
            )
            return {"indexed": False, "chunks": 0, "reason": "no_embedder"}
        chunks = await self.chapter_rag.index_chapter(
            story_id, chapter_id, title, content, chapter_number
        )
        return {"indexed": True, "chunks": chunks, "chapterId": chapter_id}

    async def delete_chapter_chunks(
        self, story_id: str, chapter_id: str
    ) -> Dict[str, Any]:
        """Remove a chapter's chunks from the vector index (chapter deleted)."""
        removed = await self.chapter_rag.delete_chapter(story_id, chapter_id)
        return {"deleted": True, "chunks": removed, "chapterId": chapter_id}

    async def index_entity(
        self,
        story_id: str,
        kind: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Embed a metadata entity (character/place/plot) so chat can retrieve its
        details on demand. Called by the entity-write trigger, once per edit."""
        if self._embedder is None:
            return {"indexed": False, "chunks": 0, "reason": "no_embedder"}
        chunks = await self.chapter_rag.index_entity(
            story_id, kind, entity_id, data or {}
        )
        return {"indexed": True, "chunks": chunks, "entityId": entity_id, "kind": kind}

    async def delete_entity_chunks(
        self, story_id: str, entity_id: str
    ) -> Dict[str, Any]:
        """Remove a metadata entity's chunks from the vector index (entity deleted)."""
        removed = await self.chapter_rag.delete_entity(story_id, entity_id)
        return {"deleted": True, "chunks": removed, "entityId": entity_id}

    async def clear_memory(
        self, story_id: str, user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Clear all story-scoped brain memory. Global procedural is kept."""
        brain = self._make_brain(user_id, story_id)
        await brain.clear()
        logging.getLogger(__name__).info(
            "Brain memory cleared story_id=%s user_id=%s", story_id, user_id
        )
        return {"cleared": True, "storyId": story_id}

    async def execute_agent(
        self,
        action: str,
        parameters: Dict[str, Any],
        background_tasks=None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute agent action dynamically.

        Args:
            action: Action to perform (generateChapter/brainstorm/etc.)
            parameters: Parameters for the action
            background_tasks: Optional FastAPI BackgroundTasks for async brain reflection

        Returns:
            Result from the agent execution
        """
        logger = logging.getLogger(__name__)
        logger.info(
            "Executing action=%s with parameter_keys=%s",
            action,
            sorted(parameters.keys()),
        )

        # effective_user_id is only plumbed to actions that personalize via the brain
        # memory system or are billed per user (chat, story choices, wizard input,
        # clear memory). The other actions (generateChapter,
        # brainstorm*, generateNextLines, enhanceText) are stateless from the brain's
        # perspective and don't take a user_id parameter — adding one here would be
        # dead plumbing until those actions opt in.
        param_user_id = self._param(parameters, "userId", "user_id")
        effective_user_id = param_user_id or user_id

        if action == "generateChapter":
            return await self.generate_chapter(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "chapterNumber", "chapter_number"),
                self._param(parameters, "previousChapters", "previous_chapters"),
                self._param(parameters, "plotContext", "plot_context"),
                order=self._param(parameters, "order"),
                prev_chapter=self._param(parameters, "prevChapter", "prev_chapter"),
                next_chapter=self._param(parameters, "nextChapter", "next_chapter"),
            )
        if action == "summarizeChapter":
            return await self.summarize_chapter(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "content"),
                self._param(parameters, "chapterId", "chapter_id"),
            )
        if action == "brainstormIdeas":
            return await self.brainstorm_ideas(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "type", "idea_type"),
                self._param(parameters, "prompt"),
                self._param(parameters, "count", default=5),
            )
        if action == "brainstormCharacter":
            return await self.brainstorm_character(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "role"),
                self._param(parameters, "archetype"),
            )
        if action == "brainstormPlot":
            return await self.brainstorm_plot(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "plotType", "plot_type", "conflict"),
            )
        if action == "generateNextLines":
            return await self.generate_next_lines(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "content"),
                self._param(parameters, "cursorPosition", "cursor_position"),
                self._param(parameters, "chapterId", "chapter_id"),
            )
        if action == "chatWithContext":
            return await self.chat_with_context(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "message"),
                self._param(parameters, "chatHistory", "chat_history"),
                user_id=effective_user_id,
                background_tasks=background_tasks,
            )
        if action == "enhanceText":
            return await self.enhance_text(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "action"),
                self._param(parameters, "selectedText", "selected_text"),
                self._param(parameters, "chapterId", "chapter_id"),
            )
        if action == "enhanceWizardInput":
            return await self.enhance_wizard_input(
                effective_user_id,
                self._param(parameters, "type", "wizard_type"),
                self._param(parameters, "data", default={}) or {},
            )

        if action == "generateStoryChoices":
            return await self.generate_story_choices(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "mode"),
                self._param(parameters, "currentContent", "current_content", ""),
                self._param(parameters, "chapterId", "chapter_id"),
                int(self._param(parameters, "turnCount", "turn_count", 0) or 0),
                user_id=effective_user_id,
                background_tasks=background_tasks,
            )

        if action == "clearMemory":
            return await self.clear_memory(
                self._param(parameters, "storyId", "story_id"),
                user_id=effective_user_id,
            )

        if action == "indexChapter":
            return await self.index_chapter(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "chapterId", "chapter_id"),
                self._param(parameters, "title", default="") or "",
                self._param(parameters, "content", default="") or "",
                self._param(parameters, "chapterNumber", "chapter_number"),
            )

        if action == "deleteChapterChunks":
            return await self.delete_chapter_chunks(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "chapterId", "chapter_id"),
            )

        if action == "indexEntity":
            return await self.index_entity(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "kind"),
                self._param(parameters, "entityId", "entity_id"),
                self._param(parameters, "data", default={}) or {},
            )

        if action == "deleteEntityChunks":
            return await self.delete_entity_chunks(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "entityId", "entity_id"),
            )

        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _param(
        parameters: Dict[str, Any],
        camel: str,
        snake: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """Read an action parameter from camelCase and snake_case names."""
        if camel in parameters:
            return parameters[camel]
        if snake and snake in parameters:
            return parameters[snake]
        return default


def _assembled_has_memory(assembled) -> bool:
    """Return True if the assembled prompt contains at least one real memory layer."""
    return bool(
        assembled.semantic_count
        or assembled.episodic_count
        or assembled.working_injected
        or assembled.procedural_injected
    )


def _extract_choices_prose(result: dict) -> str:
    """Extract narrative prose from a generateStoryChoices result for brain reflection."""
    parts = []
    if result.get("openingScene"):
        parts.append(result["openingScene"])
    for choice in result.get("choices", []):
        if choice.get("sceneText"):
            parts.append(choice["sceneText"])
    return "\n\n".join(parts)


def _load_embedder():
    """Load embedding provider once. Returns None if unavailable."""
    from agents.storyAgent.brain.embedding_provider import (  # noqa: PLC0415
        get_embedding_provider,
        verify_embedding_dimension,
    )

    embedder = get_embedding_provider(os.getenv("GOOGLE_AI_STUDIO_API_KEY"))
    # One dimension contract for both chapter RAG and brain memory: fail loud here
    # rather than silently lose recall later (mixed-dim vectors score 0.0).
    verify_embedding_dimension(embedder)
    return embedder


def _get_firestore_client(project_id: str):
    """Get a shared Firestore client."""
    from google.cloud import firestore as _fs

    return _fs.Client(project=project_id)
