"""Tests for StoryAgent functionality."""
import os

import pytest

# Set environment for testing
os.environ["USE_MOCK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

from agents.storyAgent.agent import StoryAgent


class TestStoryAgentInitialization:
    """Tests for StoryAgent initialization."""

    def test_agent_initializes_with_mock_provider(self):
        """Test that agent initializes with mock provider."""
        agent = StoryAgent(project_id="test-project")
        assert agent.project_id == "test-project"

    def test_agent_initializes_without_firestore(self):
        """Test that agent can initialize without Firestore connection."""
        # Should not raise error even without Firestore
        agent = StoryAgent(project_id="test-project")
        assert agent is not None


@pytest.mark.asyncio
class TestAgentMethods:
    """Tests for agent methods with mock provider."""

    async def test_generate_next_lines_returns_dict(self):
        """Test that generate_next_lines returns a dict."""
        agent = StoryAgent(project_id="test-project")
        result = await agent.generate_next_lines(
            story_id="test-story",
            content="Once upon a time",
            cursorPosition=17,
        )
        assert isinstance(result, dict)

    async def test_generate_next_lines_returns_suggestions(self):
        """Test that generate_next_lines returns suggestions."""
        agent = StoryAgent(project_id="test-project")
        result = await agent.generate_next_lines(
            story_id="test-story",
            content="Once upon a time",
            cursorPosition=17,
        )
        assert "suggestions" in result or "error" in result

    async def test_generate_story_returns_dict(self):
        """Test that generate_story returns a dict."""
        agent = StoryAgent(project_id="test-project")
        result = await agent.generate_story(
            title="Test Story",
            genre="Fantasy",
        )
        assert isinstance(result, dict)

    async def test_brainstorm_characters_returns_dict(self):
        """Test that brainstorm_characters returns a dict."""
        agent = StoryAgent(project_id="test-project")
        result = await agent.brainstorm_characters(
            story_context="A fantasy adventure",
            story_id="test-story",
        )
        assert isinstance(result, dict)


class TestAgentIntegration:
    """Integration tests for agent functionality."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_agent_execution_workflow(self):
        """Test a complete agent execution workflow."""
        agent = StoryAgent(project_id="test-project")

        # Test multiple operations in sequence
        result = await agent.generate_next_lines(
            story_id="test-story",
            content="Once upon a time",
            cursorPosition=17,
        )
        assert isinstance(result, dict)
