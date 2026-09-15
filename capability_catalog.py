"""Canonical story capability-to-tool mappings for every assistant channel.

The browser assistant and remote MCP connector expose different tool names and
different mutation UX, but they describe the same product capabilities. Keeping
the mappings here lets both adapters prove that every registered tool belongs to
one catalog. Browser-only editor context is marked explicitly rather than being
mistaken for an MCP omission.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityTools:
    assistant: tuple[str, ...] = ()
    mcp_read: tuple[str, ...] = ()
    mcp_write: tuple[str, ...] = ()
    browser_extension: bool = False


CAPABILITY_TOOLS: dict[str, CapabilityTools] = {
    "story_list": CapabilityTools(mcp_read=("list_my_stories",)),
    "story_overview": CapabilityTools(
        assistant=("get_story_overview",), mcp_read=("get_story_overview",)
    ),
    "chapter_list": CapabilityTools(mcp_read=("list_chapters",)),
    "read_chapter": CapabilityTools(
        assistant=("read_chapter",),
        mcp_read=("get_chapter", "get_chapter_blocks"),
    ),
    "list_entities": CapabilityTools(
        assistant=("list_story_entities",), mcp_read=("list_entities",)
    ),
    "entity_detail": CapabilityTools(
        assistant=("get_story_entity",), mcp_read=("get_entity",)
    ),
    "search_story": CapabilityTools(assistant=("search_story",)),
    "read_current_editor": CapabilityTools(
        assistant=("read_current_editor",), browser_extension=True
    ),
    "propose_edit": CapabilityTools(
        assistant=("propose_editor_edit", "apply_editor_edit"),
        browser_extension=True,
    ),
    "research_web": CapabilityTools(assistant=("research_web",)),
    "create_story": CapabilityTools(mcp_write=("create_story",)),
    "create_chapter": CapabilityTools(mcp_write=("create_chapter",)),
    "append_chapter": CapabilityTools(mcp_write=("append_to_chapter",)),
    "edit_chapter_blocks": CapabilityTools(mcp_write=("edit_chapter_blocks",)),
}


def assistant_tools(capability_id: str) -> tuple[str, ...]:
    return CAPABILITY_TOOLS[capability_id].assistant


def _flatten(field: str) -> tuple[str, ...]:
    return tuple(
        name
        for capability in CAPABILITY_TOOLS.values()
        for name in getattr(capability, field)
    )


ASSISTANT_TOOL_NAMES = frozenset(_flatten("assistant"))
MCP_READ_TOOL_NAMES = frozenset(_flatten("mcp_read"))
MCP_WRITE_TOOL_NAMES = frozenset(_flatten("mcp_write"))
