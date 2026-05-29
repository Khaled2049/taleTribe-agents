"""Tests for brain integration in generateStoryChoices."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["USE_MOCK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

from agents.storyAgent.agent import StoryAgent, _extract_choices_prose

# ---------------------------------------------------------------------------
# _extract_choices_prose helper
# ---------------------------------------------------------------------------


class TestExtractChoicesProse:
    def test_extracts_opening_scene_and_choices(self):
        result = {
            "storyId": "s1",
            "openingScene": "The rain had been falling for three days.",
            "choices": [
                {
                    "label": "Elena finds letter",
                    "sceneText": "She found it under the floorboard.",
                },
                {"label": "Stranger arrives", "sceneText": "The door swung open."},
            ],
        }
        prose = _extract_choices_prose(result)
        assert "The rain had been falling" in prose
        assert "She found it under the floorboard." in prose
        assert "The door swung open." in prose

    def test_extracts_continuation_choices_without_opening_scene(self):
        result = {
            "storyId": "s1",
            "choices": [
                {"label": "Confront Marcus", "sceneText": "She stepped into his path."},
                {"label": "Follow the shadow", "sceneText": "The figure slipped away."},
            ],
        }
        prose = _extract_choices_prose(result)
        assert "She stepped into his path." in prose
        assert "The figure slipped away." in prose
        assert "\n\n" in prose  # joined with double newline

    def test_empty_choices_returns_empty_string(self):
        assert _extract_choices_prose({"storyId": "s1", "choices": []}) == ""

    def test_error_result_returns_empty_string(self):
        result = {"storyId": "s1", "choices": [], "error": "LLM failed"}
        assert _extract_choices_prose(result) == ""

    def test_skips_choices_without_scene_text(self):
        result = {
            "choices": [
                {"label": "Option A"},  # no sceneText
                {"label": "Option B", "sceneText": "Valid prose."},
            ]
        }
        prose = _extract_choices_prose(result)
        assert "Valid prose." in prose
        assert "Option A" not in prose

    def test_ending_mode_single_choice(self):
        result = {
            "storyId": "s1",
            "choices": [
                {
                    "label": "The story reaches its end",
                    "sceneText": "The long silence finally broke.",
                    "isFinal": True,
                },
            ],
        }
        prose = _extract_choices_prose(result)
        assert "The long silence finally broke." in prose


# ---------------------------------------------------------------------------
# Brain wiring in generate_story_choices
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateStoryChoicesBrain:
    def _make_agent(self):
        agent = StoryAgent(project_id="test-project")
        # Mock the story choices tool to avoid Firestore
        agent.story_choices_tool = MagicMock()
        agent.story_choices_tool.execute = AsyncMock(
            return_value={
                "storyId": "s1",
                "openingScene": "Opening prose.",
                "choices": [
                    {"label": "A", "sceneText": "Scene A."},
                    {"label": "B", "sceneText": "Scene B."},
                    {"label": "C", "sceneText": "Scene C."},
                ],
            }
        )
        return agent

    async def test_falls_back_gracefully_when_embedder_none(self):
        agent = self._make_agent()
        agent._embedder = None  # simulate sentence-transformers not installed

        result = await agent.generate_story_choices("s1", "opening", user_id="u1")

        assert result["storyId"] == "s1"
        # Tool called with brain_context=None
        agent.story_choices_tool.execute.assert_awaited_once()
        _, kwargs = agent.story_choices_tool.execute.await_args
        assert kwargs.get("brain_context") is None

    async def test_passes_brain_context_to_tool_when_embedder_available(self):
        agent = self._make_agent()

        mock_assembled = MagicMock()
        mock_assembled.text = "=== WRITER STYLE ===\nTone: dark"
        mock_assembled.semantic_count = 3
        mock_assembled.episodic_count = 2

        mock_brain = MagicMock()
        mock_brain.assemble = AsyncMock(return_value=mock_assembled)
        mock_brain.reflect = AsyncMock()

        with patch.object(agent, "_make_brain", return_value=mock_brain):
            result = await agent.generate_story_choices("s1", "opening", user_id="u1")

        assert result["storyId"] == "s1"
        _, kwargs = agent.story_choices_tool.execute.await_args
        assert kwargs["brain_context"] == "=== WRITER STYLE ===\nTone: dark"

    async def test_schedules_reflection_when_background_tasks_provided(self):
        agent = self._make_agent()

        mock_assembled = MagicMock()
        mock_assembled.text = "brain context"
        mock_assembled.semantic_count = 1
        mock_assembled.episodic_count = 0

        mock_brain = MagicMock()
        mock_brain.assemble = AsyncMock(return_value=mock_assembled)
        mock_brain.reflect = AsyncMock()

        background_tasks = MagicMock()
        background_tasks.add_task = MagicMock()

        with patch.object(agent, "_make_brain", return_value=mock_brain):
            await agent.generate_story_choices(
                "s1", "opening", user_id="u1", background_tasks=background_tasks
            )

        background_tasks.add_task.assert_called_once()
        call_args = background_tasks.add_task.call_args
        assert (
            call_args[0][0] == mock_brain.reflect
        )  # first positional arg is brain.reflect

    async def test_no_reflection_when_no_background_tasks(self):
        agent = self._make_agent()

        mock_brain = MagicMock()
        mock_brain.assemble = AsyncMock(
            return_value=MagicMock(text="ctx", semantic_count=0, episodic_count=0)
        )
        mock_brain.reflect = AsyncMock()

        with patch.object(agent, "_make_brain", return_value=mock_brain):
            await agent.generate_story_choices(
                "s1", "opening", user_id="u1", background_tasks=None
            )

        mock_brain.reflect.assert_not_awaited()

    async def test_brain_assembly_failure_falls_back_silently(self):
        agent = self._make_agent()

        mock_brain = MagicMock()
        mock_brain.assemble = AsyncMock(side_effect=RuntimeError("Firestore down"))

        with patch.object(agent, "_make_brain", return_value=mock_brain):
            result = await agent.generate_story_choices("s1", "opening", user_id="u1")

        # Tool still called, brain_context falls back to None
        assert result["storyId"] == "s1"
        _, kwargs = agent.story_choices_tool.execute.await_args
        assert kwargs.get("brain_context") is None

    async def test_assemble_uses_current_content_in_query(self):
        agent = self._make_agent()

        mock_brain = MagicMock()
        mock_brain.assemble = AsyncMock(
            return_value=MagicMock(text="ctx", semantic_count=0, episodic_count=0)
        )

        with patch.object(agent, "_make_brain", return_value=mock_brain):
            await agent.generate_story_choices(
                "s1",
                "continuation",
                current_content="<p>Elena reached the door.</p>",
                user_id="u1",
            )

        assemble_call = mock_brain.assemble.await_args
        query_arg = assemble_call[0][0]
        assert "continuation" in query_arg
        assert "Elena reached the door." in query_arg

    async def test_no_reflection_when_result_has_empty_choices(self):
        agent = self._make_agent()
        agent.story_choices_tool.execute = AsyncMock(
            return_value={
                "storyId": "s1",
                "choices": [],
                "error": "LLM failed",
            }
        )

        mock_brain = MagicMock()
        mock_brain.assemble = AsyncMock(
            return_value=MagicMock(text="ctx", semantic_count=0, episodic_count=0)
        )

        background_tasks = MagicMock()

        with patch.object(agent, "_make_brain", return_value=mock_brain):
            await agent.generate_story_choices(
                "s1", "opening", user_id="u1", background_tasks=background_tasks
            )

        background_tasks.add_task.assert_not_called()
