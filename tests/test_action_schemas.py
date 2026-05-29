"""Tests for action parameter length limits."""
import pytest
from pydantic import ValidationError

from agents.storyAgent.action_schemas import (
    MAX_CONTENT_CHARS,
    MAX_ID_CHARS,
    MAX_PROMPT_CHARS,
    validate_action_parameters,
)


def test_generate_next_lines_rejects_oversized_content():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "generateNextLines",
            {
                "storyId": "s1",
                "content": "x" * (MAX_CONTENT_CHARS + 1),
                "cursorPosition": 0,
            },
        )


def test_generate_next_lines_accepts_content_at_limit():
    result = validate_action_parameters(
        "generateNextLines",
        {
            "storyId": "s1",
            "content": "x" * MAX_CONTENT_CHARS,
            "cursorPosition": 0,
        },
    )
    assert len(result["content"]) == MAX_CONTENT_CHARS


def test_chat_rejects_oversized_message():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "chatWithContext",
            {"storyId": "s1", "message": "m" * (MAX_CONTENT_CHARS + 1)},
        )


def test_enhance_text_rejects_oversized_selected_text():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "enhanceText",
            {
                "storyId": "s1",
                "action": "expand",
                "selectedText": "t" * (MAX_PROMPT_CHARS + 1),
            },
        )


def test_enhance_text_accepts_selected_text_at_prompt_limit():
    result = validate_action_parameters(
        "enhanceText",
        {
            "storyId": "s1",
            "action": "expand",
            "selectedText": "t" * MAX_PROMPT_CHARS,
        },
    )
    assert len(result["selectedText"]) == MAX_PROMPT_CHARS


def test_story_id_max_length():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "generateStory",
            {"storyId": "s" * (MAX_ID_CHARS + 1)},
        )


def test_chapter_id_max_length():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "generateNextLines",
            {
                "storyId": "s1",
                "content": "hello",
                "cursorPosition": 0,
                "chapterId": "c" * (MAX_ID_CHARS + 1),
            },
        )


def test_brainstorm_prompt_max_length():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "brainstormIdeas",
            {
                "storyId": "s1",
                "type": "theme",
                "prompt": "p" * (MAX_PROMPT_CHARS + 1),
            },
        )


def test_plot_context_max_length():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "generateStory",
            {
                "storyId": "s1",
                "plotContext": "p" * (MAX_PROMPT_CHARS + 1),
            },
        )


def test_story_choices_current_content_uses_content_limit():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "generateStoryChoices",
            {
                "storyId": "s1",
                "mode": "continuation",
                "currentContent": "c" * (MAX_CONTENT_CHARS + 1),
            },
        )
