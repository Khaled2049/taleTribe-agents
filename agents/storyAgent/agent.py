"""Main ADK agent implementation for story generation."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Handle imports for both direct execution and module import
try:
    from .excerpts import format_excerpts
    from .llm_provider import get_llm_provider
    from .postgres_context import PostgresIndexWorker, PostgresStoryContext
    from .tools import (
        BrainstormingTool,
        ChatWithContextTool,
        EnhanceTextTool,
        EnhanceWizardInputTool,
        NextLineGenerationTool,
        StoryChoicesTool,
    )
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.excerpts import format_excerpts
    from agents.storyAgent.llm_provider import get_llm_provider
    from agents.storyAgent.postgres_context import (
        PostgresIndexWorker,
        PostgresStoryContext,
    )
    from agents.storyAgent.tools import (
        BrainstormingTool,
        ChatWithContextTool,
        EnhanceTextTool,
        EnhanceWizardInputTool,
        NextLineGenerationTool,
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

        # Shared providers — loaded once per process
        self._llm_provider = get_llm_provider(self.project_id, self.location)
        self._embedder = _load_embedder()
        self._db = _get_firestore_client(self.project_id)

        # Initialize tools
        self.brainstorm_tool = BrainstormingTool(
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
        self.postgres_context = PostgresStoryContext()
        self.index_worker = PostgresIndexWorker(self.postgres_context, self._embedder)

    async def start(self) -> None:
        await self.postgres_context.start()

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
        await self.postgres_context.close()

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
        context = await self.postgres_context.context(story_id)
        return await self.next_line_tool.execute(
            story_id, content, cursorPosition, chapter_id, context
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
        context = await self.postgres_context.context(story_id)
        return await self.brainstorm_tool.execute(
            story_id, idea_type, prompt, count, context
        )

    async def chat_with_context(
        self,
        story_id: str,
        message: str,
        chat_history: Optional[list] = None,
        user_id: str = "anonymous",
        background_tasks=None,
    ) -> Dict[str, Any]:
        """
        Generate a chat response from story context and retrieved excerpts.

        Args:
            story_id: Firestore story document ID
            message: User's message
            chat_history: Optional list of previous messages for conversational context
            user_id: User identifier for procedural memory scoping
            background_tasks: FastAPI BackgroundTasks for async reflection

        Returns:
            Dictionary containing response and context usage
        """
        chapter_excerpts = None
        context = await self.postgres_context.context(story_id)

        if self._embedder is not None:
            log = logging.getLogger(__name__)
            try:
                query_vec = await self._embedder.embed(message)
            except Exception:
                query_vec = None
                log.warning(
                    "query embedding failed for story_id=%s; retrieval degraded",
                    story_id,
                )
            if query_vec is not None:
                try:
                    excerpts = await self.postgres_context.retrieve(
                        story_id, query_vec, top_k=4
                    )
                    chapter_excerpts = format_excerpts(excerpts) or None
                except Exception:
                    log.warning(
                        "vector retrieval failed for story_id=%s, continuing "
                        "without excerpts",
                        story_id,
                    )

        return await self.chat_tool.execute(
            story_id,
            message,
            chat_history,
            chapter_excerpts=chapter_excerpts,
            context_override=context,
        )

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
        context = await self.postgres_context.context(story_id)
        return await self.enhance_text_tool.execute(
            story_id, action, selected_text, chapter_id, context
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
        summary = await self._llm_provider.generate_content_async(
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

        Returns:
            For opening: {"storyId", "openingScene", "choices": [{label, sceneText}, ...]}
            For continuation: {"storyId", "choices": [{label, sceneText}, ...]}
            For ending: {"storyId", "choices": [{label, sceneText, isFinal: true}]}
        """
        context = await self.postgres_context.context(story_id)
        return await self.story_choices_tool.execute(
            story_id,
            mode,
            current_content,
            chapter_id,
            turn_count,
            context_override=context,
        )

    async def enhance_wizard_input(
        self,
        user_id: str,
        wizard_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhance wizard input across premise/character/place/conflict/blueprint."""
        return await self.enhance_wizard_tool.execute(user_id, wizard_type, data)

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
            action: Action to perform (brainstorm/enhanceText/etc.)
            parameters: Parameters for the action

        Returns:
            Result from the agent execution
        """
        logger = logging.getLogger(__name__)
        logger.info(
            "Executing action=%s with parameter_keys=%s",
            action,
            sorted(parameters.keys()),
        )

        # effective_user_id is only plumbed to actions that need the caller
        # memory system or are billed per user (chat, story choices, wizard input,
        # clear memory). The other actions (summarizeChapter,
        # brainstorm*, generateNextLines, enhanceText) are stateless from the brain's
        # perspective and don't take a user_id parameter — adding one here would be
        # dead plumbing until those actions opt in.
        param_user_id = self._param(parameters, "userId", "user_id")
        effective_user_id = param_user_id or user_id

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


def _load_embedder():
    """Load embedding provider once. Returns None if unavailable."""
    from agents.storyAgent.embedding_provider import (  # noqa: PLC0415
        get_embedding_provider,
        verify_embedding_dimension,
    )

    embedder = get_embedding_provider(os.getenv("GOOGLE_AI_STUDIO_API_KEY"))
    # One dimension contract for every embedding consumer: fail loud here
    # rather than silently lose recall later (mixed-dim vectors score 0.0).
    verify_embedding_dimension(embedder)
    return embedder


def _get_firestore_client(project_id: str):
    """Get a shared Firestore client."""
    from google.cloud import firestore as _fs

    return _fs.Client(project=project_id)
