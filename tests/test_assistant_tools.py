"""Tool schema invariants. The first test is the one that matters most."""

import pytest
from pydantic import ValidationError

from assistant.tools import (
    APPROVAL_REQUIRED,
    IDENTITY_FIELD_NAMES,
    MODEL_EDIT_TOOLS,
    READ_TOOLS,
    RESEARCH_TOOLS,
    TOOL_SCHEMAS,
    ToolContext,
    UnknownToolError,
    available_tools,
    validate_tool_arguments,
)


@pytest.mark.parametrize("name", sorted(TOOL_SCHEMAS))
def test_no_tool_argument_model_declares_an_identity_field(name):
    """The structural invariant: a model cannot say whose story to read.

    Identity lives on ToolContext, which is built from the verified token. If
    this fails, a tool has reopened the hole rather than a check having been
    skipped -- which is exactly why the field is absent instead of validated.
    """
    schema = TOOL_SCHEMAS[name]
    declared = set(schema.model_fields)
    aliases = {
        field.alias for field in schema.model_fields.values() if field.alias is not None
    }
    assert not (declared | aliases) & IDENTITY_FIELD_NAMES


@pytest.mark.parametrize("name", sorted(TOOL_SCHEMAS))
def test_tool_arguments_reject_unknown_fields(name):
    with pytest.raises(ValidationError):
        TOOL_SCHEMAS[name].model_validate({"totallyUnexpected": 1})


@pytest.mark.parametrize(
    "name", sorted(available_tools(edits_enabled=True, research_enabled=True))
)
def test_every_model_facing_tool_carries_a_description(name):
    """An undescribed tool is close to an absent one.

    run.py sources each tool's description from its schema's docstring, so a
    schema without one reaches the provider as a bare name and the model has to
    guess when it applies. list_story_entities sat in the allowlist that way
    while the roster's cut-off entities looked unreachable.
    """
    schema = available_tools(edits_enabled=True, research_enabled=True)[name]
    description = str(schema.model_json_schema().get("description", "")).strip()
    assert description, f"{name} has no docstring to describe it to the model"


def test_tool_context_is_not_constructible_from_model_output():
    """A frozen dataclass has no model_validate, so no JSON path builds one."""
    context = ToolContext(user_id="uid-1", story_id="story-1")
    assert not hasattr(ToolContext, "model_validate")
    with pytest.raises(AttributeError):
        context.user_id = "someone-else"


def test_read_tools_are_always_available_and_others_are_gated():
    assert set(available_tools(edits_enabled=False, research_enabled=False)) == set(
        READ_TOOLS
    )
    both = available_tools(edits_enabled=True, research_enabled=True)
    assert set(both) == set(READ_TOOLS) | set(MODEL_EDIT_TOOLS) | set(RESEARCH_TOOLS)
    assert "apply_editor_edit" not in both
    assert set(available_tools(edits_enabled=True, research_enabled=False)) == set(
        READ_TOOLS
    ) | set(MODEL_EDIT_TOOLS)
    assert set(available_tools(edits_enabled=False, research_enabled=True)) == set(
        READ_TOOLS
    ) | set(RESEARCH_TOOLS)


def test_applying_an_edit_requires_approval_but_proposing_does_not():
    assert "apply_editor_edit" in APPROVAL_REQUIRED
    assert "propose_editor_edit" not in APPROVAL_REQUIRED
    assert APPROVAL_REQUIRED <= set(TOOL_SCHEMAS)


def test_model_edit_draft_cannot_supply_editor_coordinates():
    schema = MODEL_EDIT_TOOLS["propose_editor_edit"]
    assert (
        schema.model_validate(
            {"summary": "Tighten it.", "replacementText": "Sharper."}
        ).replacement_text
        == "Sharper."
    )
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "summary": "Tighten it.",
                "replacementText": "Sharper.",
                "chapterId": "model-controlled-chapter",
            }
        )


def test_unknown_tool_is_refused():
    with pytest.raises(UnknownToolError):
        validate_tool_arguments("run_sql", {})


def test_validate_returns_camel_case_payload():
    assert validate_tool_arguments("search_story", {"query": "harbour"}) == {
        "query": "harbour",
        "limit": 8,
    }


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("search_story", {"query": ""}),
        ("search_story", {"query": "x", "limit": 0}),
        ("search_story", {"query": "x", "limit": 999}),
        ("list_story_entities", {"kind": "spaceship"}),
        ("read_chapter", {"chapterId": "c", "limit": 10_000_000}),
        ("read_chapter", {"chapterId": "c", "offset": -1}),
        ("research_web", {"query": "x", "maxResults": 50}),
        (
            "propose_editor_edit",
            {
                "chapterId": "c",
                "baseRevision": 1,
                "baseDocumentVersion": 1,
                "summary": "s",
                "operations": [],
            },
        ),
    ],
)
def test_bounds_are_enforced(name, arguments):
    with pytest.raises(ValidationError):
        validate_tool_arguments(name, arguments)


def test_edit_operations_accept_replace_and_insert_and_reject_others():
    payload = {
        "chapterId": "chapter-2",
        "baseRevision": 7,
        "baseDocumentVersion": 42,
        "summary": "Tighten the exchange.",
        "operations": [
            {
                "type": "replace",
                "from": 0,
                "to": 5,
                "originalText": "hello",
                "replacementText": "",
            },
            {"type": "insert", "at": 5, "text": " there"},
        ],
    }
    result = validate_tool_arguments("propose_editor_edit", payload)
    assert result["operations"][0]["from"] == 0
    # Deletion is a replace with empty text, not its own operation type.
    assert result["operations"][0]["replacementText"] == ""

    with pytest.raises(ValidationError):
        validate_tool_arguments(
            "propose_editor_edit",
            {**payload, "operations": [{"type": "delete", "from": 0, "to": 5}]},
        )
