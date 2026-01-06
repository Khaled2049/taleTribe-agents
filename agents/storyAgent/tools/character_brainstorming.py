"""Specialized tool for character brainstorming."""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Handle imports for both direct execution and module import
try:
    from ..context_builder import StoryContextBuilder
    from ..llm_provider import get_llm_provider, LLMProvider
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = Path(__file__).parent.parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.context_builder import StoryContextBuilder
    from agents.storyAgent.llm_provider import get_llm_provider, LLMProvider


class CharacterBrainstormingTool:
    """Specialized tool for character brainstorming."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the character brainstorming tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = get_llm_provider(project_id, location)
        self.context_builder = StoryContextBuilder(project_id)

    async def execute(
        self,
        story_id: str,
        role: Optional[str] = None,
        archetype: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate character ideas.

        Args:
            story_id: Firestore story document ID
            role: Optional character role (protagonist, antagonist, etc.)
            archetype: Optional character archetype

        Returns:
            Dictionary with character profiles
        """
        context = self.context_builder.build_story_context(story_id)
        formatted_context = self.context_builder.format_context_for_prompt(context)

        story = context["story"]
        genre = story.get("genre", "general fiction")
        tone = story.get("tone", "neutral")

        role_text = f"Role: {role}" if role else "Any role"
        archetype_text = f"Archetype: {archetype}" if archetype else "Any archetype"

        prompt = f"""Generate a detailed character profile for this {genre} story with a {tone} tone.

{formatted_context}

Character Requirements:
- {role_text}
- {archetype_text}

Provide a complete character profile with:
- Name
- Age and physical description
- Personality traits (3-5 key traits)
- Backstory (detailed, 2-3 paragraphs)
- Motivations and goals
- Fears and weaknesses
- Relationships with other characters
- How they fit into the story's plot
- Character arc potential

Make the character compelling and well-developed."""

        generated_text = self.llm_provider.generate_content(prompt)

        return {
            "storyId": story_id,
            "character": {
                "role": role,
                "archetype": archetype,
                "profile": generated_text,
            },
        }

