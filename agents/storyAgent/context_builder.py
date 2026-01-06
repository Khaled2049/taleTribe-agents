"""Context builder for aggregating story context from Firestore."""
import os
from typing import Dict, List, Any, Optional
from google.cloud import firestore


class StoryContextBuilder:
    """Builds comprehensive context from Firestore for story generation."""

    def __init__(self, project_id: Optional[str] = None):
        """Initialize Firestore client."""
        # Check if running with emulator
        emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST")
        
        if project_id:
            self.db = firestore.Client(project=project_id)
        else:
            self.db = firestore.Client()
        
        # Configure emulator if FIRESTORE_EMULATOR_HOST is set
        # The Firestore client automatically uses the emulator when 
        # FIRESTORE_EMULATOR_HOST environment variable is set
        if emulator_host:
            os.environ["FIRESTORE_EMULATOR_HOST"] = emulator_host

    def build_story_context(self, story_id: str) -> Dict[str, Any]:
        """
        Build complete context for a story from Firestore.

        Args:
            story_id: The Firestore document ID of the story

        Returns:
            Dictionary containing story data, characters, places, plots, and chapters
        """
        story_ref = self.db.collection("stories").document(story_id)
        story_doc = story_ref.get()

        if not story_doc.exists:
            raise ValueError(f"Story {story_id} not found")

        story_data = story_doc.to_dict()
        story_data["id"] = story_doc.id

        # Fetch all subcollections in parallel
        characters = self._fetch_collection(story_ref.collection("characters"))
        places = self._fetch_collection(story_ref.collection("places"))
        plots = self._fetch_collection(story_ref.collection("plots"))
        chapters = self._fetch_collection(story_ref.collection("chapters"))

        # Sort chapters by number if available
        chapters.sort(key=lambda x: x.get("chapterNumber", 0))

        return {
            "story": story_data,
            "characters": characters,
            "places": places,
            "plots": plots,
            "chapters": chapters,
        }

    def _fetch_collection(self, collection_ref) -> List[Dict[str, Any]]:
        """Fetch all documents from a collection."""
        docs = collection_ref.stream()
        return [{"id": doc.id, **doc.to_dict()} for doc in docs]

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format context into a readable prompt string for the AI model.

        Args:
            context: The context dictionary from build_story_context

        Returns:
            Formatted string with all context information
        """
        story = context.get("story", {})
        characters = context.get("characters", [])
        places = context.get("places", [])
        plots = context.get("plots", [])
        chapters = context.get("chapters", [])

        prompt_parts = []

        # Story metadata
        prompt_parts.append("=== STORY CONTEXT ===")
        prompt_parts.append(f"Title: {story.get('title', 'Untitled')}")
        prompt_parts.append(f"Genre: {story.get('genre', 'Not specified')}")
        prompt_parts.append(f"Tone: {story.get('tone', 'Not specified')}")
        if story.get("description"):
            prompt_parts.append(f"Description: {story.get('description')}")

        # Characters
        if characters:
            prompt_parts.append("\n=== CHARACTERS ===")
            for char in characters:
                char_info = f"- {char.get('name', 'Unnamed')}"
                if char.get("role"):
                    char_info += f" (Role: {char.get('role')})"
                if char.get("backstory"):
                    char_info += f"\n  Backstory: {char.get('backstory')}"
                if char.get("traits"):
                    char_info += f"\n  Traits: {char.get('traits')}"
                if char.get("motivations"):
                    char_info += f"\n  Motivations: {char.get('motivations')}"
                prompt_parts.append(char_info)

        # Places
        if places:
            prompt_parts.append("\n=== PLACES ===")
            for place in places:
                place_info = f"- {place.get('name', 'Unnamed')}"
                if place.get("description"):
                    place_info += f": {place.get('description')}"
                if place.get("atmosphere"):
                    place_info += f"\n  Atmosphere: {place.get('atmosphere')}"
                prompt_parts.append(place_info)

        # Plots
        if plots:
            prompt_parts.append("\n=== PLOTS ===")
            for plot in plots:
                plot_info = f"- {plot.get('title', 'Untitled Plot')}"
                if plot.get("description"):
                    plot_info += f": {plot.get('description')}"
                if plot.get("type"):
                    plot_info += f"\n  Type: {plot.get('type')}"
                prompt_parts.append(plot_info)

        # Existing chapters summary
        if chapters:
            prompt_parts.append(f"\n=== EXISTING CHAPTERS ({len(chapters)} total) ===")
            for chapter in chapters[:5]:  # Show first 5 chapters
                chapter_num = chapter.get("chapterNumber", "?")
                title = chapter.get("title", "Untitled")
                prompt_parts.append(f"Chapter {chapter_num}: {title}")
            if len(chapters) > 5:
                prompt_parts.append(f"... and {len(chapters) - 5} more chapters")

        return "\n".join(prompt_parts)

