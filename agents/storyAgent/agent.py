"""Main ADK agent implementation for story generation."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Handle imports for both direct execution and module import
try:
    from .tools import (
        StoryGenerationTool,
        ChapterGenerationTool,
        BrainstormingTool,
        CharacterBrainstormingTool,
        PlotBrainstormingTool,
        NextLineGenerationTool,
        ChatWithContextTool,
    )
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.tools import (
        StoryGenerationTool,
        ChapterGenerationTool,
        BrainstormingTool,
        CharacterBrainstormingTool,
        PlotBrainstormingTool,
        NextLineGenerationTool,
        ChatWithContextTool,
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

        # Initialize tools
        self.story_tool = StoryGenerationTool(self.project_id, self.location)
        self.chapter_tool = ChapterGenerationTool(self.project_id, self.location)
        self.brainstorm_tool = BrainstormingTool(self.project_id, self.location)
        self.character_tool = CharacterBrainstormingTool(self.project_id, self.location)
        self.plot_tool = PlotBrainstormingTool(self.project_id, self.location)
        self.next_line_tool = NextLineGenerationTool(self.project_id, self.location)
        self.chat_tool = ChatWithContextTool(self.project_id, self.location)

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
        return await self.next_line_tool.execute(story_id, content, cursorPosition, chapter_id)

        
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
        return await self.story_tool.execute(story_id, genre, tone, length, generate_first_chapter_only, plot_context)

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
        return await self.chapter_tool.execute(story_id, chapter_number, previous_chapters, plot_context)

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
    ) -> Dict[str, Any]:
        """
        Generate chat response using story context (RAG).

        Args:
            story_id: Firestore story document ID
            message: User's message
            chat_history: Optional list of previous messages for conversational context

        Returns:
            Dictionary containing response and context usage
        """
        return await self.chat_tool.execute(story_id, message, chat_history)

    async def execute_agent(
        self,
        action: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute agent action dynamically.

        Args:
            action: Action to perform (generateStory/generateChapter/brainstorm/etc.)
            parameters: Parameters for the action

        Returns:
            Result from the agent execution
        """
        if action == "generateStory":
            return await self.generate_story(
                parameters.get("storyId"),
                parameters.get("genre"),
                parameters.get("tone"),
                parameters.get("length"),
            )
        elif action == "generateChapter":
            return await self.generate_chapter(
                parameters.get("storyId"),
                parameters.get("chapterNumber"),
                parameters.get("previousChapters"),
            )
        elif action == "brainstormIdeas":
            return await self.brainstorm_ideas(
                parameters.get("storyId"),
                parameters.get("type"),
                parameters.get("prompt"),
                parameters.get("count", 5),
            )
        elif action == "brainstormCharacter":
            return await self.brainstorm_character(
                parameters.get("storyId"),
                parameters.get("role"),
                parameters.get("archetype"),
            )
        elif action == "brainstormPlot":
            return await self.brainstorm_plot(
                parameters.get("storyId"),
                parameters.get("plotType", "conflict"),
            )
        elif action == "generateNextLines":
            import logging
            logger = logging.getLogger(__name__)
            story_id = parameters.get('storyId')
            cursor_pos = parameters.get('cursorPosition')
            has_chapter_id = bool(parameters.get('chapterId'))
            content_length = len(parameters.get('content', ''))

            logger.info(f"generateNextLines called with storyId={story_id}, cursorPosition={cursor_pos}, content_length={content_length}, hasChapterId={has_chapter_id}")
            print(f"[AGENT] generateNextLines called: storyId={story_id}, cursorPosition={cursor_pos}, content_length={content_length}, hasChapterId={has_chapter_id}")

            result = await self.generate_next_lines(
                parameters.get("storyId"),
                parameters.get("content"),
                parameters.get("cursorPosition"),
                parameters.get("chapterId"),  # Optional
            )

            result_info = f"result keys: {list(result.keys())}" if isinstance(result, dict) else f"result type: {type(result)}"
            logger.info(f"generateNextLines completed, {result_info}")
            print(f"[AGENT] generateNextLines completed, {result_info}")
            return result
        elif action == "chatWithContext":
            return await self.chat_with_context(
                parameters.get("storyId"),
                parameters.get("message"),
                parameters.get("chatHistory"),
            )
        else:
            raise ValueError(f"Unknown action: {action}")

