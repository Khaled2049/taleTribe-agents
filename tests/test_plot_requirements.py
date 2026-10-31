"""Plot event requirements reach the model by name, never as bare ids."""

import pytest

from agents.storyAgent.context_format import format_context_for_prompt
from agents.storyAgent.postgres_context import _requirement_names
from assistant.plot_links import name_plot_requirements

pytestmark = pytest.mark.unit

PLOTS = [
    {
        "id": "main",
        "name": "Main",
        "events": [
            {"id": "letter", "name": "Finds the letter", "dependencies": []},
            {
                "id": "confront",
                "name": "Confronts the captain",
                "dependencies": [{"eventId": "letter", "plotLineId": "main"}],
                "dependents": [{"eventId": "rumour", "plotLineId": "sub"}],
            },
        ],
    },
    {
        "id": "sub",
        "name": "Sub",
        "events": [
            {
                "id": "rumour",
                "name": "Rumour spreads",
                "dependencies": [{"eventId": "confront", "plotLineId": "main"}],
            }
        ],
    },
]


def test_requirements_are_named_across_plot_lines():
    named = name_plot_requirements(PLOTS)
    confront = named[0]["events"][1]
    rumour = named[1]["events"][0]
    assert confront["requires"] == ["Finds the letter"]
    assert rumour["requires"] == ["Confronts the captain"]


def test_id_links_are_dropped_and_events_without_requirements_stay_clean():
    named = name_plot_requirements(PLOTS)
    for line in named:
        for event in line["events"]:
            assert "dependencies" not in event
            assert "dependents" not in event
    assert "requires" not in named[0]["events"][0]


def test_input_is_not_modified():
    name_plot_requirements(PLOTS)
    assert "dependencies" in PLOTS[0]["events"][1]


def test_context_loader_maps_links_to_names():
    events = [
        {"id": "letter", "name": "Finds the letter"},
        {"id": "confront", "name": "Confronts the captain"},
    ]
    links = [
        {"plot_event_id": "confront", "depends_on_event_id": "letter"},
        {"plot_event_id": "confront", "depends_on_event_id": "vanished"},
    ]
    assert _requirement_names(events, links) == {"confront": ["Finds the letter"]}


def test_prompt_lists_an_events_requirements():
    prompt = format_context_for_prompt(
        {
            "plots": [
                {
                    "name": "Main",
                    "events": [
                        {
                            "name": "Confronts the captain",
                            "content": "She shows him the letter.",
                            "requires": ["Finds the letter"],
                        }
                    ],
                }
            ]
        }
    )
    assert "Requires: Finds the letter" in prompt
