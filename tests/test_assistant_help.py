"""Help-catalog invariants. The first test is the one that keeps /help honest.

The browser prints this catalog without asking the model, so a capability that
does not exist is a lie told with total confidence. These tests make the
catalog a partition of the tool registry: nothing described that is not
registered, nothing registered that is not described.
"""

import json

from assistant.help import (
    HELP_CATALOG,
    catalog_payload,
)
from assistant.tools import (
    TOOL_SCHEMAS,
)
from capability_catalog import ASSISTANT_TOOL_NAMES


def test_catalog_partitions_the_tool_registry():
    """Add a tool without help copy and this fails, which is the whole point."""
    described = [name for capability in HELP_CATALOG for name in capability.tools]
    assert sorted(described) == sorted(TOOL_SCHEMAS)
    assert set(described) == ASSISTANT_TOOL_NAMES
    assert len(described) == len(set(described)), "a tool is described twice"


def test_help_payload_can_be_sent_as_json():
    payload = catalog_payload()
    decoded = json.loads(json.dumps(payload))
    assert decoded
    assert len({item["id"] for item in decoded}) == len(decoded)
