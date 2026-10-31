"""The saved story voice reaches prose prompts without changing their contracts."""

import pytest

from agents.storyAgent.context_format import format_context_for_prompt
from agents.storyAgent.tools.enhance_text import EnhanceTextTool
from assistant.prose_voice import voice_context


class CaptureProvider:
    def __init__(self):
        self.prompt = ""

    async def generate_content_async(self, prompt, **_kwargs):
        self.prompt = prompt
        return "The lamp guttered."


@pytest.mark.asyncio
async def test_voice_guides_enhancement_and_is_bounded():
    voice = voice_context(
        {
            "pointOfView": "first",
            "tense": "past",
            "notes": [
                {
                    "id": "note-1",
                    "text": "Keep the unease beneath the dialogue.",
                    "origin": "writer",
                    "pinned": True,
                }
            ],
        }
    )
    assert voice == {
        "pointOfView": "first person",
        "tense": "past",
        "notes": "Pinned: Keep the unease beneath the dialogue.",
    }
    assert (
        voice_context(
            {"pointOfView": "unspecified", "tense": "unspecified", "notes": []}
        )
        is None
    )

    provider = CaptureProvider()
    tool = EnhanceTextTool("test", llm_provider=provider)
    result = await tool.execute(
        "story-1",
        "rewrite",
        "The lantern was dim.",
        context_override={"story": {"title": "Saltmarsh"}, "storyVoice": voice},
    )
    assert result["enhancedText"] == "The lamp guttered."
    assert "Keep the unease beneath the dialogue." in provider.prompt
    assert "lower priority than the selected passage" in provider.prompt
    assert "The lantern was dim." in provider.prompt

    formatted = format_context_for_prompt(
        {"story": {"title": "Saltmarsh"}, "storyVoice": voice}
    )
    assert "<untrusted_story_data>" in formatted
    assert "Point of view: first person" in formatted
    assert "Keep the unease beneath the dialogue." in formatted
