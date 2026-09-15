"""Writer-facing descriptions of what the assistant can do, one per capability.

The browser answers ``/help`` locally -- no model call, no credits -- so the
text it prints has to come from somewhere that cannot drift from the tool
registry. That somewhere is here: ``HELP_CATALOG`` is exported into
``assistant/schema/v1.json`` alongside the argument schemas, vendored into the
frontend by ``scripts/sync_assistant_fixtures.py``, and compiled into a typed
constant by the contracts package. ``tests/test_assistant_help.py`` asserts the
catalog's tool names partition ``TOOL_SCHEMAS`` exactly, so adding a tool
without help copy fails rather than producing a quietly incomplete help text.

Two shapes worth naming, because they are the reason this is a catalog of
*capabilities* rather than a list of tools:

``propose_editor_edit`` and ``apply_editor_edit`` are one capability. The first
drafts a revision and the second is synthesized for the browser's approval
round; to a writer that is a single "suggest a change I approve", and splitting
it in half would describe the implementation instead of the feature.

``gate`` mirrors ``tools.available_tools``, not the schema. ``research_web`` is
registered in ``TOOL_SCHEMAS`` but has no executor and ``run.py`` passes
``research_enabled=False``, so its entry is gated ``research`` and no consumer
shows it. Describing a tool the run loop will not offer is worse than saying
nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

from capability_catalog import assistant_tools

Gate = Literal["always", "edits", "research"]


@dataclass(frozen=True)
class Capability:
    """One thing the assistant can do, in the writer's vocabulary.

    ``summary`` describes what the executor actually returns. ``example`` is a
    prompt a consumer may send verbatim on the user's behalf, so it must read
    as something a writer would type, not as a tool invocation.
    """

    id: str
    tools: tuple[str, ...]
    gate: Gate
    title: str
    summary: str
    example: str
    limits: Optional[str] = None


HELP_PREAMBLE = (
    "I read this story and answer questions about it. Here is what I can do."
)

# The same three boundaries the system prompt asserts, phrased for a reader.
# One copy, so the promise and the rules cannot disagree.
HELP_BOUNDARIES = (
    "I only see the story you have open, and only your own stories.",
    "This conversation is saved with this story and restored when you reopen the panel.",
    "I have no web access, and I never change your manuscript on my own.",
)

HELP_CATALOG: tuple[Capability, ...] = (
    Capability(
        id="story_overview",
        tools=assistant_tools("story_overview"),
        gate="always",
        title="Survey the story",
        summary=(
            "Read the story's details and its chapters in order, so I can talk "
            "about structure and pacing."
        ),
        example="Give me an overview of this story and its chapters.",
        limits="Titles and metadata only -- not the prose of every chapter.",
    ),
    Capability(
        id="list_entities",
        tools=assistant_tools("list_entities"),
        gate="always",
        title="List your cast, places and plot lines",
        summary=(
            "Pull the roster of characters, places or plot threads you have "
            "recorded for this story."
        ),
        example="List the characters in this story.",
        limits="Up to 20 at a time; I tell you when a roster is cut short.",
    ),
    Capability(
        id="entity_detail",
        tools=assistant_tools("entity_detail"),
        gate="always",
        title="Look up one character, place or plot line",
        summary=(
            "Read everything you have written on a single entity, and cite it "
            "so you can jump to it."
        ),
        example="Tell me everything recorded about the protagonist.",
    ),
    Capability(
        id="search_story",
        tools=assistant_tools("search_story"),
        gate="always",
        title="Search the manuscript",
        summary=(
            "Search your prose by meaning rather than exact words, and show "
            "the passages I found."
        ),
        example="Search the story for the protagonist's central conflict.",
        limits=(
            "Search runs on the indexed copy, so a passage written moments ago "
            "may be flagged as out of date."
        ),
    ),
    Capability(
        id="read_chapter",
        tools=assistant_tools("read_chapter"),
        gate="always",
        title="Read a chapter closely",
        summary=(
            "Read a chapter's actual text when a précis is not enough, a "
            "window at a time."
        ),
        example="Read the opening chapter and tell me where the tension drops.",
        limits="A bounded window per read, not the whole manuscript at once.",
    ),
    Capability(
        id="read_current_editor",
        tools=assistant_tools("read_current_editor"),
        gate="always",
        title="See what you have open",
        summary=(
            "Look at the chapter you are editing and the text you have "
            "selected, as it stands right now."
        ),
        example="What is wrong with the paragraph I have selected?",
        limits="Only while the editor is open, and only a bounded window of it.",
    ),
    Capability(
        id="propose_edit",
        tools=assistant_tools("propose_edit"),
        gate="edits",
        title="Suggest a revision you approve",
        summary=(
            "Draft a replacement for the text you have selected. You see it "
            "first and nothing changes until you accept it."
        ),
        example="Suggest a tighter revision for the text I selected in the editor.",
        limits="One selection at a time, as plain text in a single paragraph.",
    ),
    Capability(
        id="research_web",
        tools=assistant_tools("research_web"),
        gate="research",
        title="Look something up on the web",
        summary="Check an outside source and cite where the answer came from.",
        example="Look up how long a letter took to cross the Atlantic in 1890.",
        limits="Not available yet.",
    ),
)


def catalog_payload() -> list[dict[str, Any]]:
    """The catalog as plain JSON, in declaration order, for the schema export.

    ``tools`` becomes a list rather than a tuple so that this round-trips
    through ``json.loads`` unchanged -- ``test_assistant_contracts`` compares
    the written file against this structure, and a tuple would fail that
    equality while producing identical bytes.
    """
    return [
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(capability).items()
            if value is not None
        }
        for capability in HELP_CATALOG
    ]
