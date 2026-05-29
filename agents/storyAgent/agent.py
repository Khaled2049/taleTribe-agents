"""Main ADK agent implementation for story generation."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Handle imports for both direct execution and module import
try:
    from .brain import Brain, BrainConfig, ReflectionInput
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
        StoryGenerationTool,
    )
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.brain import Brain, BrainConfig, ReflectionInput
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
        StoryGenerationTool,
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
        self.story_tool = StoryGenerationTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
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

    async def aclose(self) -> None:
        """Release process-lifetime resources (e.g. the LLM HTTP client)."""
        close = getattr(self._llm_provider, "aclose", None)
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

    async def generate_story(
        self,
        story_id: str,
        genre: Optional[str] = None,
        tone: Optional[str] = None,
        length: Optional[str] = None,
        generate_first_chapter_only: bool = True,
        plot_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete story.

        Args:
            story_id: Firestore story document ID
            genre: Story genre
            tone: Story tone
            length: Story length
            generate_first_chapter_only: Whether to generate only the first chapter
            plot_context: Optional plot context
        Returns:
            Generated story content
        """
        return await self.story_tool.execute(
            story_id, genre, tone, length, generate_first_chapter_only, plot_context
        )

    async def generate_chapter(
        self,
        story_id: str,
        chapter_number: int,
        previous_chapters: Optional[list] = None,
        plot_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chapter.

        Args:
            story_id: Firestore story document ID
            chapter_number: Chapter number to generate
            previous_chapters: Optional list of previous chapters

        Returns:
            Generated chapter content
        """
        return await self.chapter_tool.execute(
            story_id, chapter_number, previous_chapters, plot_context
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

        if self._embedder is not None:
            try:
                brain = self._make_brain(user_id, story_id)
                assembled = await brain.assemble(message, action_hint="chatWithContext")
                brain_context = (
                    assembled.text if _assembled_has_memory(assembled) else None
                )
                if brain_context:
                    brain_context = (
                        brain_context.split("\n=== CURRENT REQUEST ===")[0].strip()
                        or None
                    )
                logging.getLogger(__name__).info(
                    "Full brain_context for chat story_id=%s:\n%s",
                    story_id,
                    brain_context,
                )
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Brain.assemble failed for story_id=%s, falling back", story_id
                )

        result = await self.chat_tool.execute(
            story_id, message, chat_history, brain_context=brain_context
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
            action: Action to perform (generateStory/generateChapter/brainstorm/etc.)
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
        # clear memory). The other actions (generateStory, generateChapter,
        # brainstorm*, generateNextLines, enhanceText) are stateless from the brain's
        # perspective and don't take a user_id parameter — adding one here would be
        # dead plumbing until those actions opt in.
        param_user_id = self._param(parameters, "userId", "user_id")
        effective_user_id = param_user_id or user_id

        if action == "generateStory":
            return await self.generate_story(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "genre"),
                self._param(parameters, "tone"),
                self._param(parameters, "length"),
                self._param(
                    parameters,
                    "generateFirstChapterOnly",
                    "generate_first_chapter_only",
                    True,
                ),
                self._param(parameters, "plotContext", "plot_context"),
            )
        if action == "generateChapter":
            return await self.generate_chapter(
                self._param(parameters, "storyId", "story_id"),
                self._param(parameters, "chapterNumber", "chapter_number"),
                self._param(parameters, "previousChapters", "previous_chapters"),
                self._param(parameters, "plotContext", "plot_context"),
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
    )

    return get_embedding_provider(os.getenv("GOOGLE_AI_STUDIO_API_KEY"))


def _get_firestore_client(project_id: str):
    """Get a shared Firestore client."""
    from google.cloud import firestore as _fs

    return _fs.Client(project=project_id)
