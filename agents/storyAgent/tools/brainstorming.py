"""Tool for brainstorming ideas."""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

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


class BrainstormingTool:
    """Tool for brainstorming ideas."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the brainstorming tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = get_llm_provider(project_id, location)
        self.context_builder = StoryContextBuilder(project_id)

    async def execute(
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
            prompt: Optional specific prompt or requirement
            count: Number of ideas to generate (default 5)

        Returns:
            Dictionary with list of generated ideas
        """
        # Build context from Firestore
        context = self.context_builder.build_story_context(story_id)
        formatted_context = self.context_builder.format_context_for_prompt(context)

        story = context["story"]
        genre = story.get("genre", "general fiction")
        tone = story.get("tone", "neutral")

        # Build type-specific prompt
        type_prompts = {
            "characters": f"""Generate {count} unique character ideas for this {genre} story with a {tone} tone.

{formatted_context}

For each character, provide:
- Name
- Role in the story
- Key personality traits
- Backstory (2-3 sentences)
- Motivations and goals
- How they fit into the existing story context""",

            "plots": f"""Generate {count} plot ideas or plot developments for this {genre} story with a {tone} tone.

{formatted_context}

For each plot idea, provide:
- Plot title/name
- Description of the plot element
- How it connects to existing story elements
- Potential conflicts or tensions
- How it advances the story""",

            "places": f"""Generate {count} location or setting ideas for this {genre} story with a {tone} tone.

{formatted_context}

For each place, provide:
- Name of the location
- Description and atmosphere
- Key features or landmarks
- How it fits into the story
- Potential events that could happen there""",

            "themes": f"""Generate {count} theme ideas for this {genre} story with a {tone} tone.

{formatted_context}

For each theme, provide:
- Theme name
- Description
- How it relates to the existing story elements
- Ways to explore this theme in the narrative""",
        }

        base_prompt = type_prompts.get(idea_type, type_prompts["characters"])
        if prompt:
            base_prompt += f"\n\nAdditional requirements: {prompt}"

        # Generate using LLM provider
        generated_text = self.llm_provider.generate_content(base_prompt)

        # Parse ideas (simple extraction - could be improved)
        ideas = self._parse_ideas(generated_text, count)

        return {
            "storyId": story_id,
            "type": idea_type,
            "ideas": ideas,
            "rawResponse": generated_text,
        }

    def _parse_ideas(self, text: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse ideas from generated text."""
        # Simple parsing - split by numbered items or dashes
        lines = text.split("\n")
        ideas = []
        current_idea = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_idea:
                    ideas.append(current_idea)
                    current_idea = {}
                continue

            # Check if it's a new idea (starts with number or dash)
            if line[0].isdigit() or line.startswith("-"):
                if current_idea:
                    ideas.append(current_idea)
                current_idea = {"text": line}
            else:
                if current_idea:
                    current_idea["text"] += " " + line
                else:
                    current_idea = {"text": line}

        if current_idea:
            ideas.append(current_idea)

        # Limit to expected count
        return ideas[:expected_count]

