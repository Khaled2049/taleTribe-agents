"""Tests for StoryAgent action dispatch and parameter handling."""

import os
from unittest.mock import AsyncMock

import pytest

os.environ["USE_MOCK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

from agents.storyAgent.agent import StoryAgent


class TestStoryAgentInitialization:
    def test_agent_initializes_with_mock_provider(self):
        agent = StoryAgent(project_id="test-project")
        assert agent.project_id == "test-project"


@pytest.mark.asyncio
class TestActionDispatch:
    async def test_execute_agent_generate_chapter_dispatch(self):
        agent = StoryAgent(project_id="test-project")
        agent.generate_chapter = AsyncMock(return_value={"ok": True})

        result = await agent.execute_agent(
            "generateChapter",
            {
                "storyId": "s1",
                "chapterNumber": 2,
            },
        )

        assert result == {"ok": True}
        agent.generate_chapter.assert_awaited_once_with(
            "s1",
            2,
            None,
            None,
            order=None,
            prev_chapter=None,
            next_chapter=None,
        )

    async def test_execute_agent_accepts_snake_case(self):
        agent = StoryAgent(project_id="test-project")
        agent.generate_next_lines = AsyncMock(return_value={"suggestions": []})

        await agent.execute_agent(
            "generateNextLines",
            {
                "story_id": "s1",
                "content": "Once",
                "cursor_position": 4,
                "chapter_id": "c1",
            },
        )

        agent.generate_next_lines.assert_awaited_once_with("s1", "Once", 4, "c1")

    async def test_execute_agent_unknown_action_raises(self):
        agent = StoryAgent(project_id="test-project")

        with pytest.raises(ValueError):
            await agent.execute_agent("doesNotExist", {})

    async def test_execute_agent_enhance_wizard_input_dispatch(self):
        agent = StoryAgent(project_id="test-project")
        agent.enhance_wizard_input = AsyncMock(return_value={"enhanced": "ok"})

        result = await agent.execute_agent(
            "enhanceWizardInput",
            {
                "userId": "u1",
                "type": "premise",
                "data": {"title": "T", "premise": "P", "genre": "fantasy"},
            },
        )

        assert result == {"enhanced": "ok"}
        agent.enhance_wizard_input.assert_awaited_once_with(
            "u1",
            "premise",
            {"title": "T", "premise": "P", "genre": "fantasy"},
        )

    async def test_execute_agent_enhance_wizard_input_accepts_snake_case(self):
        agent = StoryAgent(project_id="test-project")
        agent.enhance_wizard_input = AsyncMock(return_value={"blueprint": {}})

        await agent.execute_agent(
            "enhanceWizardInput",
            {
                "user_id": "u2",
                "wizard_type": "blueprint",
                "data": {"title": "The Last Lantern"},
            },
        )

        agent.enhance_wizard_input.assert_awaited_once_with(
            "u2",
            "blueprint",
            {"title": "The Last Lantern"},
        )

    async def test_execute_agent_generate_story_choices_opening(self):
        agent = StoryAgent(project_id="test-project")
        mock_result = {
            "storyId": "s1",
            "openingScene": "The rain had been falling for three days...",
            "choices": [
                {"label": "Elena discovers the letter", "sceneText": "She found it..."},
                {
                    "label": "A stranger arrives at the inn",
                    "sceneText": "The door swung...",
                },
                {
                    "label": "The market erupts in chaos",
                    "sceneText": "First came the sound...",
                },
            ],
        }
        agent.generate_story_choices = AsyncMock(return_value=mock_result)

        result = await agent.execute_agent(
            "generateStoryChoices",
            {"storyId": "s1", "mode": "opening"},
        )

        assert result == mock_result
        agent.generate_story_choices.assert_awaited_once_with(
            "s1",
            "opening",
            "",
            None,
            0,
            user_id="anonymous",
            background_tasks=None,
        )

    async def test_execute_agent_generate_story_choices_continuation(self):
        agent = StoryAgent(project_id="test-project")
        mock_result = {
            "storyId": "s1",
            "choices": [
                {
                    "label": "Confront Marcus directly",
                    "sceneText": "She stepped forward...",
                },
                {
                    "label": "Follow the shadow into alley",
                    "sceneText": "The figure vanished...",
                },
                {"label": "Return to the archive", "sceneText": "The old building..."},
            ],
        }
        agent.generate_story_choices = AsyncMock(return_value=mock_result)

        result = await agent.execute_agent(
            "generateStoryChoices",
            {
                "storyId": "s1",
                "mode": "continuation",
                "currentContent": "<p>Some existing prose.</p>",
                "chapterId": "ch1",
            },
        )

        assert result == mock_result
        agent.generate_story_choices.assert_awaited_once_with(
            "s1",
            "continuation",
            "<p>Some existing prose.</p>",
            "ch1",
            0,
            user_id="anonymous",
            background_tasks=None,
        )

    async def test_execute_agent_generate_story_choices_accepts_snake_case(self):
        agent = StoryAgent(project_id="test-project")
        agent.generate_story_choices = AsyncMock(
            return_value={"storyId": "s2", "choices": []}
        )

        await agent.execute_agent(
            "generateStoryChoices",
            {
                "story_id": "s2",
                "mode": "opening",
                "current_content": "",
                "chapter_id": None,
            },
        )

        agent.generate_story_choices.assert_awaited_once_with(
            "s2",
            "opening",
            "",
            None,
            0,
            user_id="anonymous",
            background_tasks=None,
        )

    async def test_execute_agent_generate_story_choices_passes_user_id(self):
        agent = StoryAgent(project_id="test-project")
        agent.generate_story_choices = AsyncMock(
            return_value={"storyId": "s1", "choices": []}
        )

        await agent.execute_agent(
            "generateStoryChoices",
            {"storyId": "s1", "mode": "opening", "userId": "user-42"},
        )

        _, kwargs = agent.generate_story_choices.await_args
        assert kwargs["user_id"] == "user-42"
