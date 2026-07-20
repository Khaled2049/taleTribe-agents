"""Tool for enhancing wizard input into richer story scaffolding."""

import json
import logging
import re
from typing import Any, Dict, Optional

from ..llm_provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


class EnhanceWizardInputTool:
    """Enhance wizard inputs for premise, character, place, conflict, and blueprint."""

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

    async def execute(
        self, user_id: str, wizard_type: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance wizard input payload into API contract output.

        Args:
            user_id: User identifier for traceability/context
            wizard_type: One of premise|character|place|conflict|blueprint
            data: Wizard payload for the given type

        Returns:
            {"enhanced": "..."} or {"blueprint": {...}}
        """
        logger.info("EnhanceWizardInputTool user_id=%s type=%s", user_id, wizard_type)
        prompt = self._build_prompt(user_id=user_id, wizard_type=wizard_type, data=data)

        try:
            response = await self.llm_provider.generate_content_async(
                prompt, max_output_tokens=512
            )
        except Exception:
            logger.exception(
                "EnhanceWizardInputTool LLM call failed user_id=%s type=%s",
                user_id,
                wizard_type,
            )
            raise

        if wizard_type == "blueprint":
            return {"blueprint": self._parse_blueprint(response)}

        return {"enhanced": response.strip()}

    def _build_prompt(
        self, user_id: str, wizard_type: str, data: Dict[str, Any]
    ) -> str:
        type_descriptions = {
            "premise": (
                "Expand and improve a rough story premise in 2-4 sentences. Keep the core idea, "
                "but add concrete stakes and atmosphere."
            ),
            "character": (
                "Enrich a character description in 2-4 sentences with vivid details, implied motivation, "
                "and personality without over-explaining."
            ),
            "place": (
                "Expand a setting in 2-4 sentences with sensory details, atmosphere, and implied history or threat."
            ),
            "conflict": (
                "Sharpen a core conflict in 2-4 sentences, tying external stakes to a personal internal cost."
            ),
            "blueprint": (
                "Return a JSON object with key 'blueprint'. Enrich only what is present in the input. "
                "Output ONLY valid JSON with no markdown or extra commentary."
            ),
        }

        if wizard_type not in type_descriptions:
            logger.error(
                "EnhanceWizardInputTool unsupported wizard type=%s", wizard_type
            )
            raise ValueError(f"Unsupported wizard type: {wizard_type}")

        payload_json = json.dumps(data, ensure_ascii=True, indent=2)

        if wizard_type == "blueprint":
            return (
                "You are a creative writing assistant for NovelSync.\n"
                f"Task: {type_descriptions[wizard_type]}\n"
                "Blueprint schema (all fields optional based on provided input):\n"
                "{\n"
                '  "blueprint": {\n'
                '    "premise": "string",\n'
                '    "conflict": "string",\n'
                '    "characters": [{"name":"string","description":"string","personality":"string","backstory":"string"}],\n'
                '    "places": [{"name":"string","description":"string","atmosphere":"string","history":"string"}]\n'
                "  }\n"
                "}\n\n"
                f"type: {wizard_type}\n"
                f"data:\n{payload_json}\n"
            )

        return (
            "You are a creative writing assistant for NovelSync.\n"
            f"Task: {type_descriptions[wizard_type]}\n"
            "Return ONLY the final prose text. No bullet points, no labels, no markdown.\n\n"
            f"type: {wizard_type}\n"
            f"data:\n{payload_json}\n"
        )

    def _parse_blueprint(self, response_text: str) -> Dict[str, Any]:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        parsed = self._safe_json_parse(cleaned)
        if isinstance(parsed, dict):
            blueprint = parsed.get("blueprint", parsed)
            if isinstance(blueprint, dict):
                return blueprint
            logger.warning(
                "EnhanceWizardInputTool blueprint field is not object type=%s",
                type(blueprint).__name__,
            )
        else:
            logger.warning("EnhanceWizardInputTool could not parse blueprint JSON")
        return {}

    def _safe_json_parse(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.debug(
                "EnhanceWizardInputTool JSON extraction failed: no object found"
            )
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.debug(
                "EnhanceWizardInputTool JSON extraction failed: invalid JSON in matched object"
            )
            return None
