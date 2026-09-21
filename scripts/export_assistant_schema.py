"""Export the assistant protocol as JSON Schema for the non-Python consumers.

Pydantic is the source of truth; this is the hand-off. The frontend generates
TypeScript from ``assistant/schema/v1.json``. Go is deliberately not a consumer
of this file -- creditProxy owns a different contract (agents to provider, not
browser to agents), so generating Go from here would either fuse the two
contracts or emit types nobody calls.

    python -m scripts.export_assistant_schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from assistant.errors import ErrorCode
from assistant.events import TERMINAL_EVENT_TYPES, AssistantEvent
from assistant.help import HELP_BOUNDARIES, HELP_PREAMBLE, catalog_payload
from assistant.protocol import (
    MAX_CONTENT_CHARS,
    MAX_EDIT_OPERATIONS,
    MAX_EDITOR_WINDOW_CHARS,
    MAX_ID_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_PARTS_PER_MESSAGE,
    MAX_SELECTION_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_TOOL_NAME_CHARS,
    MAX_URL_CHARS,
    AgentRunRequest,
    RunRequest,
)
from assistant.tools import (
    APPROVAL_REQUIRED,
    MAX_CHAPTER_WINDOW_CHARS,
    MAX_QUERY_CHARS,
    MAX_RESEARCH_RESULTS,
    MAX_TOOL_RESULTS,
    TOOL_SCHEMAS,
)
from assistant.version import ASSISTANT_PROTOCOL_VERSION

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "assistant" / "schema"


def build() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TheTaleTribe assistant protocol",
        "protocolVersion": ASSISTANT_PROTOCOL_VERSION,
        "terminalEventTypes": sorted(TERMINAL_EVENT_TYPES),
        "errorCodes": [code.value for code in ErrorCode],
        "approvalRequiredTools": sorted(APPROVAL_REQUIRED),
        # Runtime validators in non-Python consumers import generated constants
        # from this map. Keeping limits here prevents a fixture-only check from
        # missing a bound change that no canonical fixture happens to exercise.
        "limits": {
            "contentChars": MAX_CONTENT_CHARS,
            "idChars": MAX_ID_CHARS,
            "messageChars": MAX_MESSAGE_CHARS,
            "partsPerMessage": MAX_PARTS_PER_MESSAGE,
            "selectionChars": MAX_SELECTION_CHARS,
            "editorWindowChars": MAX_EDITOR_WINDOW_CHARS,
            "summaryChars": MAX_SUMMARY_CHARS,
            "toolNameChars": MAX_TOOL_NAME_CHARS,
            "urlChars": MAX_URL_CHARS,
            "queryChars": MAX_QUERY_CHARS,
            "toolResults": MAX_TOOL_RESULTS,
            "chapterWindowChars": MAX_CHAPTER_WINDOW_CHARS,
            "editOperations": MAX_EDIT_OPERATIONS,
            "researchResults": MAX_RESEARCH_RESULTS,
        },
        # Writer-facing copy, not JSON Schema, so it sits beside "limits"
        # rather than under "definitions". The browser renders /help from this
        # instead of asking the model what it can do.
        "capabilities": {
            "preamble": HELP_PREAMBLE,
            "boundaries": list(HELP_BOUNDARIES),
            "items": catalog_payload(),
        },
        "definitions": {
            "RunRequest": RunRequest.model_json_schema(by_alias=True),
            "AgentRunRequest": AgentRunRequest.model_json_schema(by_alias=True),
            "AssistantEvent": TypeAdapter(AssistantEvent).json_schema(by_alias=True),
            "tools": {
                name: schema.model_json_schema(by_alias=True)
                for name, schema in sorted(TOOL_SCHEMAS.items())
            },
        },
    }


def main() -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    path = SCHEMA_DIR / f"v{ASSISTANT_PROTOCOL_VERSION}.json"
    path.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
