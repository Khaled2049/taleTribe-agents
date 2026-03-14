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
    async def test_execute_agent_generate_story_dispatch(self):
        agent = StoryAgent(project_id="test-project")
        agent.generate_story = AsyncMock(return_value={"ok": True})

        result = await agent.execute_agent(
            "generateStory",
            {
                "storyId": "s1",
                "genre": "fantasy",
                "generateFirstChapterOnly": False,
            },
        )

        assert result == {"ok": True}
        agent.generate_story.assert_awaited_once_with("s1", "fantasy", None, None, False, None)

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
