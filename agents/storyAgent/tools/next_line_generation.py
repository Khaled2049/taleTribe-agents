"""Specialized tool for plot brainstorming."""
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    from ..context_builder import StoryContextBuilder
    from ..llm_provider import get_llm_provider, LLMProvider
except ImportError:    
    current_dir = Path(__file__).parent.parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.context_builder import StoryContextBuilder
    from agents.storyAgent.llm_provider import get_llm_provider, LLMProvider


PREFIX_CHAR_LENGTH = 1200 
SUFFIX_CHAR_LENGTH = 300  
NUMBER_OF_SUGGESTIONS = 3 

class NextLineGenerationTool:
    """Specialized tool for generating next lines."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the next line generation tool."""
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = get_llm_provider(project_id, location)
        self.context_builder = StoryContextBuilder(project_id)

    def _slice_content(self, content: str, cursor_pos: int) -> Tuple[str, str]:
        """Slices the chapter content into a prefix and suffix based on cursor position."""
        
        # 1. Calculate Prefix (Text before cursor)
        prefix_start = max(0, cursor_pos - PREFIX_CHAR_LENGTH)
        prefix = content[prefix_start:cursor_pos]

        # 2. Calculate Suffix (Text after cursor)
        suffix_end = min(len(content), cursor_pos + SUFFIX_CHAR_LENGTH)
        suffix = content[cursor_pos:suffix_end]

        return prefix, suffix

    def _get_chapter(self, story_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific chapter from Firestore."""
        try:
            from google.cloud import firestore
            import os
            
            # Create Firestore client (same way as context_builder)
            emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST")
            if self.project_id:
                db = firestore.Client(project=self.project_id)
            else:
                db = firestore.Client()
            
            if emulator_host:
                os.environ["FIRESTORE_EMULATOR_HOST"] = emulator_host
            
            chapter_ref = db.collection("stories").document(story_id).collection("chapters").document(chapter_id)
            chapter_doc = chapter_ref.get()
            if chapter_doc.exists:
                return {"id": chapter_doc.id, **chapter_doc.to_dict()}
        except Exception as e:
            # Log error but don't fail - chapter_id is optional
            print(f"Warning: Could not fetch chapter {chapter_id}: {e}")
        return None

    def _get_previous_chapters_context(self, chapters: List[Dict[str, Any]], current_chapter_number: Optional[int]) -> str:
        """Build context from previous chapters for continuity."""
        if not current_chapter_number:
            return ""
        
        # Filter and sort previous chapters
        previous_chapters = [
            ch for ch in chapters 
            if (ch.get("chapterNumber") or ch.get("order", 0)) < current_chapter_number
        ]
        previous_chapters.sort(key=lambda x: x.get("chapterNumber") or x.get("order", 0))
        
        if not previous_chapters:
            return ""
        
        # Get the last 2-3 previous chapters for context (to avoid token limits)
        recent_chapters = previous_chapters[-3:]
        context_parts = []
        
        for chapter in recent_chapters:
            chapter_num = chapter.get("chapterNumber") or chapter.get("order", "?")
            title = chapter.get("title", "Untitled")
            content = chapter.get("content", "")
            # Include first 500 chars of each previous chapter
            content_preview = content[:500] + "..." if len(content) > 500 else content
            context_parts.append(f"Chapter {chapter_num}: {title}\n{content_preview}")
        
        if context_parts:
            return "\n\n--- PREVIOUS CHAPTERS (for continuity) ---\n" + "\n\n".join(context_parts)
        return ""

    def _build_system_prompt(self) -> str:
        """Defines the AI's role, rules, and constraints."""
        return f"""
You are a highly skilled, creative, and observant Novelist Assistant AI. Your sole task is to provide seamless, in-context line continuations for a user who is actively writing a novel.

### RULES AND CONSTRAINTS:
1.  **Output Format MUST BE JSON:** Your entire response must be a single, valid JSON array that strictly follows the provided JSON Schema.
2.  **Number of Suggestions:** You MUST generate EXACTLY {NUMBER_OF_SUGGESTIONS} distinct, high-quality line suggestions.
3.  **Suggestion Length:** Each suggestion must be a single, logical sentence or a short, cohesive thought, ready to be dropped directly into the text. Do NOT output full paragraphs or multiple sentences.
4.  **Coherence:** Maintain the established story's Genre, Tone, and the immediate preceding text's flow, pacing, and point-of-view.
5.  **Focus:** Use the "CURRENT CHAPTER CONTEXT" as the primary guide. Use the "GLOBAL STORY CONTEXT" only for world/character consistency.
6.  **DO NOT output any pre-amble, explanation, or text outside of the required JSON object.**
"""

    def _build_user_prompt(self, formatted_context: str, prefix_text: str, suffix_text: str, previous_chapters_text: str = "") -> str:
        """Assembles the dynamic user prompt with all context and the task."""
        previous_section = f"\n{previous_chapters_text}\n" if previous_chapters_text else ""
        
        return f"""
### A. GLOBAL STORY CONTEXT 
{formatted_context}{previous_section}
### B. CURRENT CHAPTER CONTEXT

The user is currently editing a chapter. The following text excerpt provides the immediate context for where the next line should be inserted.

The [INSERTION_POINT] marks the exact spot where the next line should logically begin.

--- START OF EXCERPT ---
{prefix_text}
[INSERTION_POINT]
{suffix_text}
--- END OF EXCERPT ---

### YOUR TASK:
Generate {NUMBER_OF_SUGGESTIONS} unique, short line suggestions that flow naturally from the text preceding the [INSERTION_POINT] and smoothly transition into the text that follows it.

Respond ONLY with the JSON array containing the {NUMBER_OF_SUGGESTIONS} generated lines.
"""

    def _get_response_schema(self) -> Dict[str, Any]:
        """Defines the required JSON schema for the output."""
        return {
            "type": "array",
            "description": f"An array containing exactly {NUMBER_OF_SUGGESTIONS} distinct sentence or line suggestions.",
            "items": {
                "type": "string",
                "description": "A single, coherent sentence or line continuation."
            },
            "minItems": NUMBER_OF_SUGGESTIONS,
            "maxItems": NUMBER_OF_SUGGESTIONS,
        }

    async def execute(
        self,
        story_id: str,
        content: str,
        cursorPosition: int,
        chapter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates 3 next line suggestions based on story context and cursor position.

        Args:
            story_id: Firestore story document ID
            content: Current content of the chapter
            cursorPosition: Character index where the new line should be inserted
            chapter_id: Optional chapter document ID for better context and validation

        Returns:
            Dictionary containing the suggestions array.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"NextLineGenerationTool.execute called: story_id={story_id}, content_length={len(content)}, cursorPosition={cursorPosition}, chapter_id={chapter_id}")
        print(f"[NEXT_LINE_TOOL] execute called: story_id={story_id}, content_length={len(content)}, cursorPosition={cursorPosition}, chapter_id={chapter_id}")
        
        try:
            # Build Macro Context
            logger.info("Building story context...")
            print("[NEXT_LINE_TOOL] Building story context...")
            context = self.context_builder.build_story_context(story_id)
            chapters_count = len(context.get('chapters', []))
            logger.info(f"Story context built, chapters count: {chapters_count}")
            print(f"[NEXT_LINE_TOOL] Story context built, chapters count: {chapters_count}")
    
            # If chapter_id is provided, enhance context with chapter-specific information
            current_chapter_number = None
            previous_chapters_text = ""
            if chapter_id:
                logger.info(f"Fetching chapter {chapter_id}...")
                print(f"[NEXT_LINE_TOOL] Fetching chapter {chapter_id}...")
                current_chapter = self._get_chapter(story_id, chapter_id)
                if current_chapter:
                    current_chapter_number = current_chapter.get("chapterNumber") or current_chapter.get("order")
                    logger.info(f"Found chapter, number: {current_chapter_number}")
                    print(f"[NEXT_LINE_TOOL] Found chapter, number: {current_chapter_number}")
                    # Get previous chapters for continuity
                    previous_chapters_text = self._get_previous_chapters_context(
                        context.get("chapters", []), 
                        current_chapter_number
                    )
                    prev_len = len(previous_chapters_text)
                    logger.info(f"Previous chapters context length: {prev_len}")
                    print(f"[NEXT_LINE_TOOL] Previous chapters context length: {prev_len}")
            
            logger.info("Formatting context for prompt...")
            print("[NEXT_LINE_TOOL] Formatting context for prompt...")
            formatted_context = self.context_builder.format_context_for_prompt(context)
            
            # Build Micro Context
            logger.info("Slicing content...")
            print("[NEXT_LINE_TOOL] Slicing content...")
            prefix_text, suffix_text = self._slice_content(content, cursorPosition)
            logger.info(f"Prefix length: {len(prefix_text)}, Suffix length: {len(suffix_text)}")
            print(f"[NEXT_LINE_TOOL] Prefix length: {len(prefix_text)}, Suffix length: {len(suffix_text)}")
            
            logger.info("Building prompts...")
            print("[NEXT_LINE_TOOL] Building prompts...")
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(formatted_context, prefix_text, suffix_text, previous_chapters_text)
            response_schema = self._get_response_schema()
            
            logger.info("Calling LLM provider...")
            print("[NEXT_LINE_TOOL] Calling LLM provider...")

            # log out the system_prompt, user_prompt, and response_schema
            logger.info(f"System prompt: {system_prompt}")
            print(f"[NEXT_LINE_TOOL] System prompt: {system_prompt}")
            logger.info(f"User prompt: {user_prompt}")
            print(f"[NEXT_LINE_TOOL] User prompt: {user_prompt}")
            logger.info(f"Response schema: {response_schema}")
            print(f"[NEXT_LINE_TOOL] Response schema: {response_schema}")

            try:
                generated_suggestions = await self.llm_provider.generate_structured_content(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=response_schema
                )
                logger.info(f"Generated suggestions: {generated_suggestions}")
                print(f"[NEXT_LINE_TOOL] Generated suggestions: {generated_suggestions}")
                suggestions_count = len(generated_suggestions) if isinstance(generated_suggestions, list) else 'non-list'
                logger.info(f"LLM returned {suggestions_count} suggestions")
                print(f"[NEXT_LINE_TOOL] LLM returned {suggestions_count} suggestions")
                
                if not isinstance(generated_suggestions, list) or len(generated_suggestions) != NUMBER_OF_SUGGESTIONS:
                     raise ValueError("LLM returned improperly formatted or missing suggestions.")

                result = {
                    "storyId": story_id,                
                    "suggestions": generated_suggestions,
                }
                logger.info(f"Successfully generated {len(generated_suggestions)} suggestions")
                print(f"[NEXT_LINE_TOOL] Successfully generated {len(generated_suggestions)} suggestions")
                return result

            except Exception as error:
                logger.error(f"Error in LLM generation: {error}", exc_info=True)
                print(f"[NEXT_LINE_TOOL ERROR] Error in LLM generation: {error}")
                import traceback
                print(f"[NEXT_LINE_TOOL ERROR] Traceback: {traceback.format_exc()}")
                return {
                    "storyId": story_id,
                    "suggestions": [],
                    "error": f"Failed to generate lines: {error}"
                }
        except Exception as error:
            logger.error(f"Error in NextLineGenerationTool.execute: {error}", exc_info=True)
            print(f"[NEXT_LINE_TOOL ERROR] Error in execute: {error}")
            import traceback
            print(f"[NEXT_LINE_TOOL ERROR] Traceback: {traceback.format_exc()}")
            return {
                "storyId": story_id,
                "suggestions": [],
                "error": f"Failed to execute next line generation: {error}"
            }