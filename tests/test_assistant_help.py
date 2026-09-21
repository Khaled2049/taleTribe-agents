"""Help-catalog invariants. The first test is the one that keeps /help honest.

The browser prints this catalog without asking the model, so a capability that
does not exist is a lie told with total confidence. These tests make the
catalog a partition of the tool registry: nothing described that is not
registered, nothing registered that is not described.
"""

import pytest

from assistant.help import (
    HELP_BOUNDARIES,
    HELP_CATALOG,
    HELP_PREAMBLE,
    catalog_payload,
)
from assistant.protocol import MAX_MESSAGE_CHARS, MAX_SUMMARY_CHARS
from assistant.tools import (
    APPROVAL_REQUIRED,
    MODEL_EDIT_TOOLS,
    READ_TOOLS,
    RESEARCH_TOOLS,
    TOOL_SCHEMAS,
    available_tools,
)
from capability_catalog import ASSISTANT_TOOL_NAMES, CAPABILITY_TOOLS


def test_catalog_partitions_the_tool_registry():
    """Add a tool without help copy and this fails, which is the whole point."""
    described = [name for capability in HELP_CATALOG for name in capability.tools]
    assert sorted(described) == sorted(TOOL_SCHEMAS)
    assert set(described) == ASSISTANT_TOOL_NAMES
    assert len(described) == len(set(described)), "a tool is described twice"


def test_browser_only_capabilities_are_explicit_channel_extensions():
    browser_only = {
        capability_id
        for capability_id, tools in CAPABILITY_TOOLS.items()
        if tools.browser_extension
    }
    assert browser_only == {"read_current_editor", "propose_edit"}
    assert all(not CAPABILITY_TOOLS[name].mcp_read for name in browser_only)
    assert all(not CAPABILITY_TOOLS[name].mcp_write for name in browser_only)


@pytest.mark.parametrize("capability", HELP_CATALOG, ids=lambda c: c.id)
def test_gate_matches_what_a_run_would_actually_offer(capability):
    """``gate`` mirrors available_tools, not the registry it is drawn from.

    ``apply_editor_edit`` is the one exception and is spelled out rather than
    waved through: it is never provider-facing, because the browser -- not the
    model -- decides that an approved proposal is applied.
    """
    read_only = available_tools(edits_enabled=False, research_enabled=False)
    ungated = {
        "always": read_only,
        "edits": available_tools(edits_enabled=True, research_enabled=False),
        "research": available_tools(edits_enabled=False, research_enabled=True),
    }[capability.gate]
    for name in capability.tools:
        if name == "apply_editor_edit":
            assert name not in ungated and name in APPROVAL_REQUIRED
            continue
        assert name in ungated
        if capability.gate != "always":
            assert name not in read_only, "a gated tool is offered ungated"


def test_research_stays_gated_until_it_has_an_executor():
    """run.py hardcodes research_enabled=False, so /help must not advertise it."""
    research = {
        name
        for capability in HELP_CATALOG
        if capability.gate == "research"
        for name in capability.tools
    }
    assert research == set(RESEARCH_TOOLS)


def test_the_edit_capability_covers_both_halves_of_one_feature():
    edits = [c for c in HELP_CATALOG if c.gate == "edits"]
    assert len(edits) == 1
    assert set(edits[0].tools) == set(MODEL_EDIT_TOOLS) | {"apply_editor_edit"}


@pytest.mark.parametrize("capability", HELP_CATALOG, ids=lambda c: c.id)
def test_copy_is_present_and_sendable(capability):
    """``example`` is sent verbatim as a user message, so it must fit one."""
    assert capability.title.strip()
    assert capability.summary.strip()
    assert 0 < len(capability.example.strip()) <= MAX_MESSAGE_CHARS
    assert len(capability.summary) <= MAX_SUMMARY_CHARS
    if capability.limits is not None:
        assert capability.limits.strip()


def test_ids_are_unique_and_stable_keys():
    ids = [capability.id for capability in HELP_CATALOG]
    assert len(ids) == len(set(ids))
    assert all(id_.replace("_", "").isalnum() for id_ in ids)


def test_read_tools_are_all_described_as_always_available():
    always = {
        name
        for capability in HELP_CATALOG
        if capability.gate == "always"
        for name in capability.tools
    }
    assert always == set(READ_TOOLS)


def test_payload_is_json_ready_and_drops_absent_limits():
    payload = catalog_payload()
    assert [item["id"] for item in payload] == [c.id for c in HELP_CATALOG]
    by_id = {item["id"]: item for item in payload}
    assert "limits" not in by_id["entity_detail"]
    assert isinstance(by_id["search_story"]["tools"], (list, tuple))


def test_preamble_and_boundaries_are_present():
    assert HELP_PREAMBLE.strip()
    assert len(HELP_BOUNDARIES) >= 3
    assert all(line.strip() for line in HELP_BOUNDARIES)
