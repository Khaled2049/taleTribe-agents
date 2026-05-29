"""Tool for generating individual chapters with continuity."""

from typing import Any, Dict, List, Optional

from ..context_builder import StoryContextBuilder
from ..llm_provider import LLMProvider, get_llm_provider


class ChapterGenerationTool:
    """Tool for generating individual chapters with continuity."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        llm_provider: Optional[LLMProvider] = None,
    ):
        """Initialize the chapter generation tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = llm_provider or get_llm_provider(
            project_id, location
        )
        self.context_builder = StoryContextBuilder(project_id)

    async def execute(
        self,
        story_id: str,
        chapter_number: int,
        previous_chapters: Optional[List[Dict[str, Any]]] = None,
        plot_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chapter with continuity.

        Args:
            story_id: Firestore story document ID
            chapter_number: The chapter number to generate
            previous_chapters: Optional list of previous chapter contents
            plot_context: Optional plot context
        Returns:
            Dictionary with generated chapter content
        """
        # Build context from Firestore
        context = self.context_builder.build_story_context(story_id)
        formatted_context = self.context_builder.format_context_for_prompt(context)

        # Get existing chapters for continuity
        existing_chapters = context.get("chapters", [])
        if previous_chapters is None:
            previous_chapters = existing_chapters[: chapter_number - 1]

        # Build continuity summary
        continuity_text = self._build_enhanced_continuity(previous_chapters)

        plot_section = ""
        if plot_context:
            plot_section = f"""
*** CURRENT PLOT DIRECTION ***
{plot_context}
IMPORTANT: You are writing a specific segment of a larger arc.
- If the stage says "Frustration", the character must fail or face obstacles.
- If the stage says "Nightmare", the situation must look hopeless.
- Do not rush to the ending unless the stage instructions say "Resolution".
"""

        prompt = f"""You are an expert novelist. Generate Chapter {chapter_number} for this story.

    {formatted_context}
    {plot_section}
    {continuity_text}

    Generate Chapter {chapter_number} that:
    1. Continues naturally from Chapter {chapter_number - 1}
    2. Advances at least one plot thread
    3. Develops character relationships and growth
    4. Maintains consistent tone and pacing
    5. Ends with a hook for the next chapter
    6. References and builds upon events from previous chapters

    Length: Approximately {self._get_chapter_length(context)} words

    Return in this format:
    - Title: [Compelling Chapter Title]
    - Content: [Full chapter text]
    """

        generated_text = await self.llm_provider.generate_content_async(prompt)

        return {
            "storyId": story_id,
            "chapterNumber": chapter_number,
            "content": generated_text,
        }

    def _build_enhanced_continuity(self, previous_chapters: List[Dict]) -> str:
        """Build comprehensive continuity context."""
        if not previous_chapters:
            return ""

        text = "\n=== STORY SO FAR ===\n"

        # Overall summary of ALL chapters
        text += "\nStory Summary:\n"
        for ch in previous_chapters:
            num = ch.get("chapterNumber", 0)
            title = ch.get("title", "Untitled")
            # Use more content for summary (1000 chars per chapter)
            content = ch.get("content", "")[:1000]
            text += f"• Chapter {num} ({title}): {content}...\n"

        # Detailed view of last chapter
        if previous_chapters:
            last_ch = previous_chapters[-1]
            text += "\n=== PREVIOUS CHAPTER (Full) ===\n"
            text += (
                f"Chapter {last_ch.get('chapterNumber')}: {last_ch.get('title')}\n\n"
            )
            text += last_ch.get("content", "")[:2000]  # More context
            text += "\n\n"

        # Key plot threads to continue
        text += "\n=== KEY ELEMENTS TO CONTINUE ===\n"
        text += "Ensure you:\n"
        text += "- Reference events and character decisions from previous chapters\n"
        text += "- Maintain character personality and development\n"
        text += "- Continue unresolved plot threads\n"
        text += "- Keep consistent world-building details\n\n"

        return text

    def _get_chapter_length(self, context: Dict) -> str:
        """Determine appropriate chapter length."""
        story = context.get("story", {})
        length_pref = story.get("length", "medium")

        lengths = {"short": "1500-2000", "medium": "2500-3500", "long": "4000-5500"}
        return lengths.get(length_pref, "2500-3500")
