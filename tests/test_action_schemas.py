"""Tests for action parameter validation."""

import pytest
from pydantic import ValidationError

from agents.storyAgent.action_schemas import (
    MAX_ID_CHARS,
    validate_action_parameters,
)

WIZARD = {"type": "premise", "data": {"title": "The Last Lantern"}}


def test_wizard_input_is_normalized_to_camel_case():
    result = validate_action_parameters(
        "enhanceWizardInput",
        {"wizard_type": "premise", "data": WIZARD["data"], "user_id": "u1"},
    )
    assert result == {**WIZARD, "userId": "u1"}


def test_user_id_max_length():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "enhanceWizardInput", {**WIZARD, "userId": "u" * (MAX_ID_CHARS + 1)}
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        validate_action_parameters(
            "enhanceWizardInput", {**WIZARD, "userId": "u1", "storyId": "s1"}
        )


def test_an_action_without_a_schema_is_unknown():
    with pytest.raises(ValueError, match="Unknown action"):
        validate_action_parameters("generateNextLines", {"storyId": "s1"})
