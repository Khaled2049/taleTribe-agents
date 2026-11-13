"""Main ADK agent implementation for story generation."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Handle imports for both direct execution and module import
try:
    from .llm_provider import get_llm_provider
    from .postgres_context import PostgresIndexWorker, PostgresStoryContext
    from .tools import EnhanceWizardInputTool
except ImportError:
    # Add parent directory to path for direct execution
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.llm_provider import get_llm_provider
    from agents.storyAgent.postgres_context import (
        PostgresIndexWorker,
        PostgresStoryContext,
    )
    from agents.storyAgent.tools import EnhanceWizardInputTool


class StoryAgent:
    """Runs the /agent/execute actions and owns the shared providers."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        """
        Initialize the story agent.

        Args:
            project_id: GCP project ID (defaults to environment variable, required for Firestore)
            location: GCP location (not used, kept for backward compatibility)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("project_id must be provided or set GOOGLE_CLOUD_PROJECT")

        self.location = location

        # Shared providers — loaded once per process
        self._llm_provider = get_llm_provider(self.project_id, self.location)
        self._embedder = _load_embedder()

        # Initialize tools
        self.enhance_wizard_tool = EnhanceWizardInputTool(
            self.project_id, self.location, llm_provider=self._llm_provider
        )
        self.postgres_context = PostgresStoryContext()
        self.index_worker = PostgresIndexWorker(self.postgres_context, self._embedder)

    async def start(self) -> None:
        await self.postgres_context.start()

    @property
    def llm_provider(self):
        """The shared CreditProxyProvider — also used for credit balance/top-up."""
        return self._llm_provider

    @property
    def embedding_provider(self):
        """The process-wide embedder used by indexing and assistant retrieval."""
        return self._embedder

    async def aclose(self) -> None:
        """Release process-lifetime resources (e.g. the LLM + embedding HTTP clients)."""
        for provider in (self._llm_provider, self._embedder):
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
        await self.postgres_context.close()

    async def enhance_wizard_input(
        self,
        user_id: str,
        wizard_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhance wizard input across premise/character/place/conflict/blueprint."""
        return await self.enhance_wizard_tool.execute(user_id, wizard_type, data)

    async def execute_agent(
        self,
        action: str,
        parameters: Dict[str, Any],
        background_tasks=None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute agent action dynamically.

        Args:
            action: Action to perform (see action_schemas.ActionName)
            parameters: Parameters for the action

        Returns:
            Result from the agent execution
        """
        logger = logging.getLogger(__name__)
        logger.info(
            "Executing action=%s with parameter_keys=%s",
            action,
            sorted(parameters.keys()),
        )

        param_user_id = self._param(parameters, "userId", "user_id")
        effective_user_id = param_user_id or user_id

        if action == "enhanceWizardInput":
            return await self.enhance_wizard_input(
                effective_user_id,
                self._param(parameters, "type", "wizard_type"),
                self._param(parameters, "data", default={}) or {},
            )

        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _param(
        parameters: Dict[str, Any],
        camel: str,
        snake: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """Read an action parameter from camelCase and snake_case names."""
        if camel in parameters:
            return parameters[camel]
        if snake and snake in parameters:
            return parameters[snake]
        return default


def _load_embedder():
    """Load embedding provider once. Returns None if unavailable."""
    from agents.storyAgent.embedding_provider import (  # noqa: PLC0415
        get_embedding_provider,
        verify_embedding_dimension,
    )

    embedder = get_embedding_provider(os.getenv("GOOGLE_AI_STUDIO_API_KEY"))
    # One dimension contract for every embedding consumer: fail loud here
    # rather than silently lose recall later (mixed-dim vectors score 0.0).
    verify_embedding_dimension(embedder)
    return embedder
