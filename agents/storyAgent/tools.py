"""ADK tools for story generation, chapter creation, and brainstorming.

This module maintains backward compatibility by re-exporting all tools from the tools package.
The tools have been refactored into separate modules in the tools/ subdirectory for better organization.
"""

try:
    from .tools.brainstorming import BrainstormingTool
    from .tools.chat_with_context import ChatWithContextTool
    from .tools.enhance_text import EnhanceTextTool
except ImportError:
    # Handle direct execution case
    import sys
    from pathlib import Path

    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from agents.storyAgent.tools.brainstorming import BrainstormingTool
    from agents.storyAgent.tools.chat_with_context import ChatWithContextTool
    from agents.storyAgent.tools.enhance_text import EnhanceTextTool

__all__ = [
    "BrainstormingTool",
    "ChatWithContextTool",
    "EnhanceTextTool",
]
