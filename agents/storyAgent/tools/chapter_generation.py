"""Tool for generating individual chapters with continuity."""

import json
import re
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
        order: Optional[float] = None,
        prev_chapter: Optional[Dict[str, Any]] = None,
        next_chapter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chapter with continuity.

        Continuity sources, in priority order:
          1. Bounded context (`prev_chapter`/`next_chapter`) — the scalable path;
             payload stays small regardless of story length and supports mid-story
             inserts (chapters on both sides).
          2. Legacy `previous_chapters` full dump.
          3. Chapters loaded from Firestore as a last resort.
        """
        # Build context for story metadata (characters, places, plots, title).
        context = self.context_builder.build_story_context(story_id)
        formatted_context = self.context_builder.format_context_for_prompt(context)

        has_bounded = prev_chapter is not None or next_chapter is not None
        if has_bounded:
            continuity_text = self._build_bounded_continuity(prev_chapter, next_chapter)
        else:
            existing_chapters = context.get("chapters", [])
            if previous_chapters is None:
                previous_chapters = existing_chapters[: chapter_number - 1]
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

        bridge_clause = (
            "Lead naturally INTO the following chapter without contradicting it"
            if next_chapter
            else "Ends with a hook for the next chapter"
        )

        prompt = f"""You are an expert novelist. Generate Chapter {chapter_number} for this story.

    {formatted_context}
    {plot_section}
    {continuity_text}

    Generate Chapter {chapter_number} that:
    1. Continues naturally from the previous chapter
    2. Advances at least one plot thread
    3. Develops character relationships and growth
    4. Maintains consistent tone and pacing
    5. {bridge_clause}
    6. References and builds upon events from previous chapters

    Length: Approximately {self._get_chapter_length(context)} words

    Return ONLY valid JSON (no markdown code fences, no extra prose) with this shape:
    {{"title": "<compelling chapter title>", "content": "<full chapter text>", "summary": "<2-3 sentence summary of this chapter for future continuity>"}}
    """

        generated_text = await self.llm_provider.generate_content_async(
            prompt, max_output_tokens=8192
        )
        title, content, summary = self._parse_generated(generated_text, chapter_number)

        return {
            "storyId": story_id,
            "chapterNumber": chapter_number,
            "order": order,
            "title": title,
            "content": content,
            "summary": summary,
        }

    def _build_bounded_continuity(
        self,
        prev_chapter: Optional[Dict[str, Any]],
        next_chapter: Optional[Dict[str, Any]],
    ) -> str:
        """Build continuity from bounded neighbor context."""
        if not prev_chapter and not next_chapter:
            return ""

        text = "\n=== STORY SO FAR ===\n"

        if prev_chapter:
            text += "\n=== PREVIOUS CHAPTER (Full) ===\n"
            text += (
                f"Chapter {prev_chapter.get('chapterNumber')}: "
                f"{prev_chapter.get('title')}\n\n"
            )
            text += (prev_chapter.get("content") or "")[:2500]
            text += "\n\n"

        if next_chapter:
            text += "\n=== NEXT CHAPTER (Full) — your chapter must lead INTO this ===\n"
            text += (
                f"Chapter {next_chapter.get('chapterNumber')}: "
                f"{next_chapter.get('title')}\n\n"
            )
            text += (next_chapter.get("content") or "")[:2500]
            text += (
                "\n\nIMPORTANT: End your chapter so it connects naturally to the "
                "NEXT chapter above. Do not contradict events that occur in it.\n"
            )

        text += "\n=== KEY ELEMENTS TO CONTINUE ===\n"
        text += "Ensure you:\n"
        text += "- Reference events and character decisions from previous chapters\n"
        text += "- Maintain character personality and development\n"
        text += "- Continue unresolved plot threads\n"
        text += "- Keep consistent world-building details\n\n"

        return text

    def _build_enhanced_continuity(self, previous_chapters: List[Dict]) -> str:
        """Build comprehensive continuity context (legacy full-dump path)."""
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

    def _parse_generated(
        self, text: str, chapter_number: int
    ) -> tuple[str, str, Optional[str]]:
        """
        Parse the model output into (title, content, summary).

        Prefers structured JSON; tolerates code fences; falls back to the legacy
        "Title:" line scan so older/looser responses still produce content.
        """
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                content = (obj.get("content") or "").strip()
                if content:
                    title = (
                        obj.get("title") or ""
                    ).strip() or f"Chapter {chapter_number}"
                    summary = (obj.get("summary") or "").strip() or None
                    return title, content, summary
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: legacy "Title:" line scan.
        lines = (text or "").split("\n")
        title = f"Chapter {chapter_number}"
        content = text or ""
        for line in lines:
            if line.lower().startswith("title:"):
                parsed = line.split(":", 1)[1].strip()
                if parsed:
                    title = parsed
                idx = lines.index(line)
                content = "\n".join(lines[idx + 1 :]).strip()
                break

        return title, content, None

    def _get_chapter_length(self, context: Dict) -> str:
        """Determine appropriate chapter length."""
        story = context.get("story", {})
        length_pref = story.get("length", "medium")

        lengths = {"short": "1500-2000", "medium": "2500-3500", "long": "4000-5500"}
        return lengths.get(length_pref, "2500-3500")
