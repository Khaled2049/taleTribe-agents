"""Tool for generating interactive story choices for the co-write feature."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..context_format import format_context_for_prompt
from ..llm_provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

NUMBER_OF_CHOICES = 3

# Output caps per mode. Opening (an opening scene + 3 prose branches) and ending
# (a multi-paragraph closing) generate the most prose, and 4096 tokens can
# truncate their JSON mid-string — the model returns invalid JSON and the parse
# fails ("unexpected structure"). Give them more headroom. Continuation (3 short
# branches) fits comfortably in less. All stay <= the gateway's MAX_OUTPUT_TOKENS.
MAX_OUTPUT_TOKENS_OPENING = 6144
MAX_OUTPUT_TOKENS_ENDING = 6144
MAX_OUTPUT_TOKENS_CONTINUATION = 4096


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode basic entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return re.sub(r" {2,}", " ", text).strip()


def _safe_json_parse(text: str) -> Any:
    """Parse JSON, stripping markdown fences if present, with regex fallback."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned).rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class StoryChoicesTool:
    """Generate opening scene + choices (opening mode) or continuation choices (co-write mode)."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.project_id = project_id
        self.location = location
        self.llm_provider: LLMProvider = llm_provider or get_llm_provider(
            project_id, location
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_opening_prompt(self, formatted_context: str) -> str:
        return (
            "You are a master novelist writing immersive, publication-quality prose.\n"
            "Output ONLY valid JSON — no preamble, no markdown fences, no extra text.\n\n"
            f"### STORY BLUEPRINT\n{formatted_context}\n\n"
            "### YOUR TASK — OPENING MODE\n"
            "The story has not started yet. Produce:\n"
            f"1. `openingScene` — a 2–3 paragraph opening that establishes tone, setting, and introduces the protagonist.\n"
            f"2. `choices` — exactly {NUMBER_OF_CHOICES} ways the first scene could develop. "
            "Each choice must be a distinct direction (different action, location, or character focus).\n\n"
            "Quality constraints:\n"
            "- Choices must diverge meaningfully — not minor variations of the same beat.\n"
            "- `sceneText` must be actual prose ready to insert into the editor, not a summary.\n"
            "- `label` must be 5–8 words, action-oriented, spoiler-free.\n"
            "- Maintain the genre's tone throughout.\n"
            "- Use only characters and places from the blueprint.\n\n"
            "Respond with this exact JSON shape (no extra keys):\n"
            "{\n"
            '  "openingScene": "<2-3 paragraph prose>",\n'
            '  "choices": [\n'
            '    { "label": "<5-8 word action label>", "sceneText": "<prose paragraph>" },\n'
            '    { "label": "...", "sceneText": "..." },\n'
            '    { "label": "...", "sceneText": "..." }\n'
            "  ]\n"
            "}"
        )

    def _build_continuation_prompt(
        self, formatted_context: str, current_text: str, turn_count: int = 0
    ) -> str:
        arc_guidance = ""
        if turn_count >= 10:
            arc_guidance = (
                "\n### STORY ARC — APPROACHING CLIMAX\n"
                "The story has been developing for many turns. One of your three choices MUST be a "
                "conclusive direction that moves decisively toward resolving the central conflict and "
                "providing emotional payoff. Label it clearly (e.g. 'Begin the final reckoning') and "
                'mark it with `"isFinal": true` in the JSON.\n'
            )
        elif turn_count >= 8:
            arc_guidance = (
                "\n### STORY ARC — NEARING RESOLUTION\n"
                "The story is reaching its natural length. Include at least one choice that begins "
                "moving toward a resolution or climax. Label it to signal this (e.g. 'Confront the truth at last').\n"
            )

        return (
            "You are a master novelist writing immersive, publication-quality prose.\n"
            "Output ONLY valid JSON — no preamble, no markdown fences, no extra text.\n\n"
            f"### STORY BLUEPRINT\n{formatted_context}\n\n"
            f"### STORY SO FAR\n{current_text}\n\n"
            f"{arc_guidance}"
            "### YOUR TASK — CONTINUATION MODE\n"
            f"Produce exactly {NUMBER_OF_CHOICES} choices for what logically happens next.\n"
            "Each `sceneText` must be a self-contained prose paragraph that can be appended directly after the story so far.\n\n"
            "Quality constraints:\n"
            "- Choices must diverge meaningfully — not minor variations of the same beat.\n"
            "- `sceneText` must be actual prose ready to insert into the editor, not a summary.\n"
            "- `label` must be 5–8 words, action-oriented, spoiler-free.\n"
            "- Maintain the genre's tone and character consistency throughout.\n\n"
            'Respond with this exact JSON shape (each choice may optionally include `"isFinal": true`):\n'
            "{\n"
            '  "choices": [\n'
            '    { "label": "<5-8 word action label>", "sceneText": "<prose paragraph>" },\n'
            '    { "label": "...", "sceneText": "..." },\n'
            '    { "label": "...", "sceneText": "..." }\n'
            "  ]\n"
            "}"
        )

    def _build_ending_prompt(self, formatted_context: str, current_text: str) -> str:
        return (
            "You are a master novelist writing immersive, publication-quality prose.\n"
            "Output ONLY valid JSON — no preamble, no markdown fences, no extra text.\n\n"
            f"### STORY BLUEPRINT\n{formatted_context}\n\n"
            f"### STORY SO FAR\n{current_text}\n\n"
            "### YOUR TASK — ENDING MODE\n"
            "Write a single conclusive closing scene that:\n"
            "- Resolves the central conflict established in the story blueprint.\n"
            "- Provides emotional payoff and closure for the main characters.\n"
            "- Feels like a satisfying, complete ending — not a cliffhanger.\n"
            "- Is 3–5 paragraphs of polished prose ready to insert into the editor.\n\n"
            "Respond with this exact JSON shape:\n"
            "{\n"
            '  "choices": [\n'
            '    { "label": "The story reaches its end", "sceneText": "<3-5 paragraph closing prose>", "isFinal": true }\n'
            "  ]\n"
            "}"
        )

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        story_id: str,
        mode: str,
        current_content: str = "",
        chapter_id: Optional[str] = None,
        turn_count: int = 0,
        context_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate story choices for opening, continuation, or ending mode.

        Args:
            story_id: Firestore story document ID
            mode: "opening", "continuation", or "ending"
            current_content: HTML content already in the editor (empty for opening)
            chapter_id: Optional chapter document ID (reserved for future chapter-scoped context)
            turn_count: How many choices the user has selected so far (used for arc guidance)

        Returns:
            For opening: {"storyId": ..., "openingScene": ..., "choices": [...]}
            For continuation: {"storyId": ..., "choices": [...]}
            For ending: {"storyId": ..., "choices": [{"label": ..., "sceneText": ..., "isFinal": true}]}
        """
        logger.info(
            "StoryChoicesTool story_id=%s mode=%s turn_count=%s",
            story_id,
            mode,
            turn_count,
        )

        if mode not in ("opening", "continuation", "ending"):
            return {
                "storyId": story_id,
                "choices": [],
                "error": f"Invalid mode '{mode}'. Must be 'opening', 'continuation', or 'ending'.",
            }

        if context_override is None:
            raise ValueError("story context is required")
        context = context_override
        formatted_context = format_context_for_prompt(context)
        plain_text = _strip_html(current_content) if current_content else ""

        if mode == "opening":
            prompt = self._build_opening_prompt(formatted_context)
            max_output_tokens = MAX_OUTPUT_TOKENS_OPENING
        elif mode == "ending":
            prompt = self._build_ending_prompt(formatted_context, plain_text)
            max_output_tokens = MAX_OUTPUT_TOKENS_ENDING
        else:
            prompt = self._build_continuation_prompt(
                formatted_context, plain_text, turn_count
            )
            max_output_tokens = MAX_OUTPUT_TOKENS_CONTINUATION

        raw_response = await self.llm_provider.generate_content_async(
            prompt, max_output_tokens=max_output_tokens
        )

        parsed = _safe_json_parse(raw_response or "")
        if not isinstance(parsed, dict) or "choices" not in parsed:
            raise ValueError(
                f"LLM returned unexpected structure: {(raw_response or '')[:200]}"
            )

        choices: List[Dict[str, Any]] = parsed.get("choices", [])
        expected = 1 if mode == "ending" else NUMBER_OF_CHOICES
        if len(choices) != expected:
            raise ValueError(
                f"Expected {expected} choice(s) for mode '{mode}', got {len(choices)}."
            )

        output: Dict[str, Any] = {"storyId": story_id, "choices": choices}
        if mode == "opening":
            output["openingScene"] = parsed.get("openingScene", "")

        return output
