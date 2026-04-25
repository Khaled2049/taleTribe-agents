"""Tool for chat with RAG (Retrieval-Augmented Generation) using story context."""
import logging
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

logger = logging.getLogger(__name__)


class ChatWithContextTool:
    """Tool for chatting with context-aware AI assistant."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the chat tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = get_llm_provider(project_id, location)
        self.context_builder = StoryContextBuilder(project_id)

    async def execute(
        self,
        story_id: str,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        brain_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate chat response using story context (RAG).

        Args:
            story_id: Firestore story document ID
            message: User's message
            chat_history: List of previous messages for conversational context
                         Each message should have "role" ("user" or "assistant") and "content"
            brain_context: Optional pre-assembled brain memory context; replaces
                           the default Firestore context string when provided

        Returns:
            Dictionary containing:
            - response: AI-generated response
            - contextUsed: Counts of story elements used
        """
        # Build context from Firestore — slim format keeps prompt focused
        context = self.context_builder.build_story_context(story_id)
        slim_firestore = self.context_builder.format_slim_context_for_chat(context)

        # Brain context (style/memory) prepended to slim Firestore summary
        context_text = (brain_context + "\n\n" + slim_firestore) if brain_context else slim_firestore

        system_prompt = f"""You are a writing assistant inside NovelSync. You know this story.

Rules:
- Reply in 1-3 sentences unless the user asks for more, a list, or prose help.
- No filler openers ("Great question!", "Of course!", "Sure!").
- For prose help: show a rewritten example, not just advice.
- For story questions: answer directly from context.
- For brainstorming: give 2-3 specific ideas, not a numbered essay.

STORY CONTEXT:
{context_text}
"""

        # Build conversation history with system prompt
        full_prompt = system_prompt + "\n\n"

        # Add chat history if provided (last 10 messages)
        if chat_history:
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    full_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    full_prompt += f"Assistant: {content}\n\n"

        # Add current user message
        full_prompt += f"User: {message}\n\nAssistant:"

        # Generate response using LLM provider
        logger.info("Full chat prompt story_id=%s:\n%s", story_id, full_prompt)
        response = await self.llm_provider.generate_content_async(full_prompt)

        # Calculate context usage
        context_used = {
            "chapters": len(context.get("chapters", [])),
            "characters": len(context.get("characters", [])),
            "plots": len(context.get("plots", [])),
            "places": len(context.get("places", [])),
        }
        

        return {
            "response": response.strip(),
            "contextUsed": context_used,
        }

    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build formatted context string from story data."""
        parts = []

        # Story metadata
        story = context.get("story", {})
        if story:
            parts.append(f"STORY: {story.get('title', 'Untitled')}")
            if story.get("description"):
                parts.append(f"Description: {story.get('description')}")
            if story.get("genre"):
                parts.append(f"Genre: {story.get('genre')}")
            if story.get("tone"):
                parts.append(f"Tone: {story.get('tone')}")
            parts.append("")

        # Characters
        characters = context.get("characters", [])
        if characters:
            parts.append("CHARACTERS:")
            for char in characters:
                name = char.get("name", "Unknown")
                backstory = char.get("backstory", "No backstory")
                parts.append(f"- {name}: {backstory}")
            parts.append("")

        # Plots
        plots = context.get("plots", [])
        if plots:
            parts.append("PLOT LINES:")
            for plot in plots:
                plot_name = plot.get("name", "Unnamed plot")
                description = plot.get("description", "")
                parts.append(f"- {plot_name}: {description}")

                # Include first few events
                events = plot.get("events", [])
                for event in events[:5]:  # First 5 events
                    event_name = event.get("name", "")
                    event_content = event.get("content", "")
                    if event_name or event_content:
                        parts.append(f"  * {event_name}: {event_content}")
            parts.append("")

        # Places
        places = context.get("places", [])
        if places:
            parts.append("LOCATIONS:")
            for place in places:
                place_name = place.get("name", "Unknown location")
                description = place.get("description", "")
                parts.append(f"- {place_name}: {description}")
            parts.append("")

        # Chapters (summarize older, full-text recent)
        chapters = context.get("chapters", [])
        if chapters:
            parts.append("CHAPTERS:")
            total_chapters = len(chapters)

            # Last 3 chapters: full content
            recent_chapters = chapters[-3:] if total_chapters > 3 else chapters
            older_chapters = chapters[:-3] if total_chapters > 3 else []

            # Summarize older chapters
            if older_chapters:
                chapter_titles = [c.get("title", "Untitled") for c in older_chapters]
                parts.append(f"[Earlier chapters 1-{len(older_chapters)}: {' | '.join(chapter_titles)}]")
                parts.append("")

            # Full text for recent chapters
            for chapter in recent_chapters:
                title = chapter.get("title", "Untitled Chapter")
                content = chapter.get("content", "")
                # Truncate if too long (keep first 2000 chars)
                truncated_content = content[:2000] + "..." if len(content) > 2000 else content
                parts.append(f"Chapter: {title}")
                parts.append(truncated_content)
                parts.append("")

        return "\n".join(parts)
