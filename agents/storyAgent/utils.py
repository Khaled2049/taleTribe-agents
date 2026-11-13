"""Shared utilities for the storyAgent package."""

import json
import re
from typing import Any


def sanitize_for_prompt(value: Any, max_chars: int = 800) -> str:
    """Render user-authored content as inert prompt text.

    Strips non-printable control characters (preserving newline/tab/CR),
    escapes backtick fences, and hard-caps length so attacker-controlled
    fields cannot dominate LLM instructions.
    """
    if value is None:
        return ""
    text = str(value)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")
    text = text.replace("```", "\\`\\`\\`").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def safe_json_parse(text: str) -> Any:
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
