"""Fixture round-trip: the Python half of the three-language contract check.

The TypeScript half lives in taleTribe-frontend/packages/assistant-contracts and
the Go half in creditProxy/pkg/contracts. All three read the same bytes, which
is the point -- the fixtures are the fixed point the languages agree on.
"""

import json
from pathlib import Path
from typing import get_args

import pytest

from assistant.events import (
    TERMINAL_EVENT_TYPES,
    AssistantEventAdapter,
    encode_sse,
    validate_event_sequence,
)
from scripts.export_assistant_schema import build as build_schema

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "assistant" / "fixtures"
FIXTURES = sorted(p for p in FIXTURE_DIR.glob("*.json") if p.name != "MANIFEST.json")
EXPECTED = {
    "approval-pause-resume",
    "cancellation",
    "max-steps",
    "multi-tool",
    "provider-error",
    "research-citations",
    "single-tool-round",
    "stale-edit",
    "text-only",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_the_agreed_fixture_set_is_present():
    """Named explicitly so deleting a case is a decision, not an accident."""
    assert {p.stem for p in FIXTURES} == EXPECTED


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_round_trips_without_loss(path):
    document = load(path)
    events = document["events"]
    parsed = validate_event_sequence(events)
    redumped = [
        event.model_dump(by_alias=True, exclude_none=True, mode="json")
        for event in parsed
    ]
    assert redumped == events


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_sse_matches_its_events(path):
    """The wire bytes are part of the contract, not just the parsed objects."""
    document = load(path)
    parsed = [AssistantEventAdapter.validate_python(e) for e in document["events"]]
    assert "".join(encode_sse(event) for event in parsed) == document["sse"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_declares_the_current_protocol_version(path):
    document = load(path)
    assert document["protocolVersion"] == 1
    assert {event["v"] for event in document["events"]} == {1}


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_has_exactly_one_terminal_event_last(path):
    events = load(path)["events"]
    terminals = [e for e in events if e["type"] in TERMINAL_EVENT_TYPES]
    assert len(terminals) == 1
    assert events[-1]["type"] in TERMINAL_EVENT_TYPES


def test_every_event_type_appears_in_at_least_one_fixture():
    """A type no fixture covers is a type the other two languages never check."""
    from assistant import events as events_module

    declared = {
        get_args(cls.model_fields["type"].annotation)[0]
        for cls in vars(events_module).values()
        if isinstance(cls, type)
        and issubclass(cls, events_module.BaseEvent)
        and cls is not events_module.BaseEvent
    }
    covered = {e["type"] for path in FIXTURES for e in load(path)["events"]}
    assert declared - covered == set()


def test_manifest_matches_the_fixture_files():
    """Guards the vendored copies in the other two repositories."""
    import hashlib

    manifest = json.loads((FIXTURE_DIR / "MANIFEST.json").read_text())
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in FIXTURES
    }
    assert manifest == actual


def test_exported_schema_matches_the_pydantic_source():
    schema_path = FIXTURE_DIR.parent / "schema" / "v1.json"
    assert json.loads(schema_path.read_text()) == build_schema()
