"""Scheduling logic for bot actions."""
import asyncio
import random
import logging
from typing import Callable, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Scheduler:
    """Handles scheduling and action selection for bots."""
    
    def __init__(
        self,
        interval_seconds: int = 60,  # 60 seconds default (for testing)
        continue_probability: float = 0.7,  # 70% chance to continue existing story
        min_chapters_for_continuation: int = 1,
        max_chapters_per_story: int = 10,
    ):
        """
        Initialize scheduler.
        
        Args:
            interval_seconds: Seconds between actions (default: 60 = 1 minute for testing)
            continue_probability: Probability of continuing existing story vs creating new (0.0-1.0)
            min_chapters_for_continuation: Minimum chapters a story needs to be considered for continuation
            max_chapters_per_story: Maximum chapters before story is considered complete
        """
        self.interval_seconds = interval_seconds
        self.continue_probability = continue_probability
        self.min_chapters_for_continuation = min_chapters_for_continuation
        self.max_chapters_per_story = max_chapters_per_story
        self.last_action_time: Optional[datetime] = None
    
    def should_continue_existing(self) -> bool:
        """
        Decide whether to continue an existing story or create a new one.
        
        Returns:
            True if should continue existing story, False if should create new
        """
        return random.random() < self.continue_probability
    
    def select_story_for_continuation(
        self,
        stories: list,
        min_chapters: Optional[int] = None,
        max_chapters: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Select a story from the list that should be continued.
        
        Args:
            stories: List of story dictionaries
            min_chapters: Minimum chapter count (uses instance default if None)
            max_chapters: Maximum chapter count (uses instance default if None)
            
        Returns:
            Selected story dictionary or None if no suitable story found
        """
        min_ch = min_chapters or self.min_chapters_for_continuation
        max_ch = max_chapters or self.max_chapters_per_story
        
        # Filter stories that can be continued
        continuable_stories = [
            s for s in stories
            if min_ch <= s.get("chapterCount", 0) < max_ch
        ]
        
        if not continuable_stories:
            return None
        
        # Prefer stories with fewer chapters (to balance story development)
        # But also consider recency (stories updated more recently are more active)
        continuable_stories.sort(
            key=lambda s: (
                s.get("chapterCount", 0),  # Fewer chapters first
                -s.get("updatedAt", datetime.min).timestamp() if isinstance(s.get("updatedAt"), datetime) else 0
            )
        )
        
        # Select from top 3 candidates randomly for variety
        candidates = continuable_stories[:min(3, len(continuable_stories))]
        return random.choice(candidates)
    
    async def wait_for_next_action(self) -> None:
        """
        Wait until it's time for the next action.
        Uses the configured interval.
        """
        if self.last_action_time:
            elapsed = (datetime.now(timezone.utc) - self.last_action_time).total_seconds()
            remaining = max(0, self.interval_seconds - elapsed)
            if remaining > 0:
                logger.info(f"Waiting {remaining:.0f} seconds until next action...")
                await asyncio.sleep(remaining)
        
        self.last_action_time = datetime.now(timezone.utc)
    
    async def run_daily(
        self,
        action_callback: Callable,
        *args,
        **kwargs,
    ) -> None:
        """
        Run actions on a daily schedule.
        
        Args:
            action_callback: Async function to call for each action
            *args: Positional arguments to pass to callback
            **kwargs: Keyword arguments to pass to callback
        """
        logger.info(f"Starting daily scheduler (interval: {self.interval_seconds}s)")
        
        while True:
            try:
                await self.wait_for_next_action()
                logger.info("Executing scheduled action...")
                await action_callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in scheduled action: {e}", exc_info=True)
                # Wait a bit before retrying to avoid rapid error loops
                await asyncio.sleep(60)

