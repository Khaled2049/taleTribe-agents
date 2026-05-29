"""Tool for generating complete stories."""
from typing import Dict, Any, Optional

from ..context_builder import StoryContextBuilder
from ..llm_provider import get_llm_provider, LLMProvider


class StoryGenerationTool:
    """Tool for generating complete stories."""

    def __init__(self, project_id: str, location: str = "us-central1", llm_provider: Optional[LLMProvider] = None):
        """Initialize the story generation tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = llm_provider or get_llm_provider(project_id, location)
        self.context_builder = StoryContextBuilder(project_id)

    async def execute(
        self, story_id: str, 
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
            genre: Story genre (optional, uses story data if not provided)
            tone: Story tone (optional, uses story data if not provided)
            length: Story length (short/medium/long)
            generate_first_chapter_only: Whether to generate only the first chapter
            plot_context: Optional plot context
        Returns:
            Dictionary with generated story content
        """
        # Build context from Firestore
        context = self.context_builder.build_story_context(story_id)
        formatted_context = self.context_builder.format_context_for_prompt(context)

        story = context["story"]
        genre = genre or story.get("genre", "general fiction")
        tone = tone or story.get("tone", "neutral")
        length = length or "medium"

        plot_section = ""
        if plot_context:
            plot_section = f"""
        *** CURRENT PLOT DIRECTION ***
        {plot_context}
        IMPORTANT: The events of this chapter MUST align with the narrative stage described above.
        """

        if generate_first_chapter_only:
            prompt = f"""You are an expert novelist. Generate the FIRST CHAPTER of a {genre} story with a {tone} tone.

{formatted_context}
{plot_section}
Generate Chapter 1 that:
1. Introduces the main characters and setting
2. Establishes the story's tone and atmosphere
3. Presents the initial situation or conflict
4. Creates hooks that make readers want to continue
5. ENDS with clear story threads to continue in future chapters
6. Does NOT resolve the main conflict
7. Strictly follows the 'Current Plot Direction' provided above (if any)

Length: Approximately {self._get_chapter_length(length)} words

Return in this format:
- Title: [Chapter Title]
- Content: [Full chapter text]
"""
            # Generate using LLM provider
            generated_text = await self.llm_provider.generate_content_async(prompt)

            # Parse response (simple extraction)
            return {
                "storyId": story_id,
                "content": generated_text,
                "metadata": {
                    "genre": genre,
                    "tone": tone,
                    "length": length,
                },
            }
        else:
        # Build prompt
            prompt = f"""You are an expert novelist. Generate a complete {genre} story with a {tone} tone.

    {formatted_context}
    {plot_section}
    Generate a complete story that:
    1. Incorporates all the characters, places, and plot elements provided
    2. Maintains consistency with the story's genre and tone
    3. Is approximately {length} length
    4. Has a clear beginning, middle, and end
    5. Includes character development and plot progression
    6. Follows the narrative arc described in the Plot Direction

    Return the story in the following format:
    - Title: [Story Title]
    - Story: [Full story text]
    - Summary: [Brief summary]
    """

            # Generate using LLM provider
            generated_text = await self.llm_provider.generate_content_async(prompt)

            # Parse response (simple extraction)
            return {
                "storyId": story_id,
                "content": generated_text,
                "metadata": {
                    "genre": genre,
                    "tone": tone,
                    "length": length,
                },
            }

    def _get_chapter_length(self, length: str) -> str:
        """Determine appropriate chapter length based on story length preference."""
        lengths = {
            "short": "1500-2000",
            "medium": "2500-3500",
            "long": "4000-5500"
        }
        return lengths.get(length.lower(), "2500-3500")
