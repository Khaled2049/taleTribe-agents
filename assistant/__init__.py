"""Versioned assistant protocol: runs, events, and tool schemas.

A peer subsystem to ``mcp_server``, not a story-agent tool. Phase 1 defines the
contracts only -- there is no orchestration loop and no tool executor here. See
``docs/assistant-ui-phase-1-plan.md`` in the frontend repository.
"""

from assistant.version import ASSISTANT_PROTOCOL_VERSION

__all__ = ["ASSISTANT_PROTOCOL_VERSION"]
