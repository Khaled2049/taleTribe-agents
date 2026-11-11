"""Tests for StoryAgent action dispatch and parameter handling."""

import os
from unittest.mock import AsyncMock

import pytest

os.environ["USE_MOCK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

from agents.storyAgent.agent import StoryAgent


@pytest.mark.asyncio
class TestActionDispatch:
    async def test_execute_agent_unknown_action_raises(self):
        agent = StoryAgent(project_id="test-project")

        with pytest.raises(ValueError):
            await agent.execute_agent("doesNotExist", {})

    @pytest.mark.parametrize(
        "action",
        [
            "generateNextLines",
            "enhanceText",
            "brainstormIdeas",
            "generateStoryChoices",
            "summarizeChapter",
        ],
    )
    async def test_removed_story_actions_are_unknown(self, action):
        """These read a whole story by id; nothing may dispatch them again."""
        agent = StoryAgent(project_id="test-project")

        with pytest.raises(ValueError):
            await agent.execute_agent(action, {"storyId": "s1"})

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
