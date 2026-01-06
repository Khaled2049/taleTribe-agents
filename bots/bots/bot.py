"""Sci-Fi bot implementation."""
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from bots.core.agent_runner import AgentRunner
from bots.core.firebase_client import FirebaseClient
from bots.core.personality import Personality
from bots.core.scheduler import Scheduler
from bots.config import BotConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main bot execution."""
    # Load configuration
    config = BotConfig()
    
    # Validate configuration
    is_valid, error = config.validate()
    if not is_valid:
        logger.error(f"Configuration error: {error}")
        logger.error("Please set the following environment variables:")
        logger.error("  - BOT_EMAIL: Bot user email")
        logger.error("  - BOT_PASSWORD: Bot user password")
        logger.error("  - GOOGLE_CLOUD_PROJECT: GCP project ID")
        logger.error("  - FIRESTORE_EMULATOR_HOST: Firestore emulator host (optional, for local dev)")
        sys.exit(1)
    
    logger.info(f"Starting bot with config: {config}")
    
    # Initialize Firebase client
    firebase_client = FirebaseClient(project_id=config.project_id)
    
    # Select a random personality for this bot instance
    selected_personality = Personality.get_random_bot()
    logger.info(f"Bot initialized with personality: {selected_personality.name} ({selected_personality.genre})")
    
    # Initialize agent runner
    agent_runner = AgentRunner(
        firebase_client=firebase_client,
        personality=selected_personality,
        project_id=config.project_id,
        story_length=config.story_length,
        max_chapters_per_story=config.max_chapters_per_story,
    )
    
    # Authenticate bot
    try:
        await agent_runner.authenticate(config.email, config.password)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("Make sure the bot user account exists in Firebase Auth")
        sys.exit(1)
    
    # Create scheduler
    scheduler = Scheduler(
        interval_seconds=config.interval_seconds,
        continue_probability=config.continue_probability,
        max_chapters_per_story=config.max_chapters_per_story,
    )
    
    # Set scheduler and run
    agent_runner.set_scheduler(scheduler)
    
    logger.info("Bot started. Running on scheduled intervals...")
    await agent_runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)