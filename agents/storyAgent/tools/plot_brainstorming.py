"""Specialized tool for plot brainstorming."""
import sys
from pathlib import Path
from typing import Dict, Any

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


class PlotBrainstormingTool:
    """Specialized tool for plot brainstorming."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the plot brainstorming tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = get_llm_provider(project_id, location)
        self.context_builder = StoryContextBuilder(project_id)

    async def execute(
        self,
        story_id: str,
        plot_type: str = "conflict",
    ) -> Dict[str, Any]:
        """
        Generate plot ideas.

        Args:
            story_id: Firestore story document ID
            plot_type: Type of plot element (conflict/twist/subplot/development)

        Returns:
            Dictionary with plot suggestions
        """
        context = self.context_builder.build_story_context(story_id)
        formatted_context = self.context_builder.format_context_for_prompt(context)

        story = context["story"]
        genre = story.get("genre", "general fiction")
        tone = story.get("tone", "neutral")

        type_descriptions = {
            "conflict": "a major conflict or obstacle",
            "twist": "a plot twist or unexpected development",
            "subplot": "a subplot that complements the main story",
            "development": "a plot development that advances the story",
        }

        plot_desc = type_descriptions.get(plot_type, "a plot element")

        prompt = f"""Generate {plot_desc} for this {genre} story with a {tone} tone.

{formatted_context}

Provide:
- Title/name for this plot element
- Detailed description
- How it connects to existing story elements
- Characters involved
- Potential consequences or outcomes
- How it advances the overall narrative
- Tension or conflict it creates

Make it compelling and well-integrated with the existing story."""

        generated_text = self.llm_provider.generate_content(prompt)

        return {
            "storyId": story_id,
            "plotType": plot_type,
            "plot": generated_text,
        }

