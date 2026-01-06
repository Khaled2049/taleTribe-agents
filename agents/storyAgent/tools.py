"""ADK tools for story generation, chapter creation, and brainstorming.

This module maintains backward compatibility by re-exporting all tools from the tools package.
The tools have been refactored into separate modules in the tools/ subdirectory for better organization.
"""


try:    
    from .tools.story_generation import StoryGenerationTool
    from .tools.chapter_generation import ChapterGenerationTool
    from .tools.brainstorming import BrainstormingTool
    from .tools.character_brainstorming import CharacterBrainstormingTool
    from .tools.plot_brainstorming import PlotBrainstormingTool
except ImportError:
    # Handle direct execution case
    import sys
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from agents.storyAgent.tools.story_generation import StoryGenerationTool
    from agents.storyAgent.tools.chapter_generation import ChapterGenerationTool
    from agents.storyAgent.tools.brainstorming import BrainstormingTool
    from agents.storyAgent.tools.character_brainstorming import CharacterBrainstormingTool
    from agents.storyAgent.tools.plot_brainstorming import PlotBrainstormingTool

__all__ = [
    "StoryGenerationTool",
    "ChapterGenerationTool",
    "BrainstormingTool",
    "CharacterBrainstormingTool",
    "PlotBrainstormingTool",
]

