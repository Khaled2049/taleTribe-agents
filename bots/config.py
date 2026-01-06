"""Configuration management for bots."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from python directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path}")


class BotConfig:
    """Bot configuration loaded from environment variables."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Bot credentials
        self.email = os.getenv("BOT_EMAIL", "")
        self.password = os.getenv("BOT_PASSWORD", "")
        
        # Firebase configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.firestore_emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST")
        self.auth_emulator_host = os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "localhost:9099")
        
        # Ollama configuration
        self.use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        # Scheduling configuration
        self.interval_seconds = int(os.getenv("BOT_INTERVAL_SECONDS", "60"))  # Default: 60 seconds (for testing)
        self.continue_probability = float(os.getenv("BOT_CONTINUE_PROBABILITY", "1"))  # 70% continue
        
        # Story generation preferences
        self.story_length = os.getenv("BOT_STORY_LENGTH", "medium")
        self.max_chapters_per_story = int(os.getenv("BOT_MAX_CHAPTERS", "8"))
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.email:
            return False, "BOT_EMAIL environment variable is required"
        if not self.password:
            return False, "BOT_PASSWORD environment variable is required"
        if not self.project_id:
            return False, "GOOGLE_CLOUD_PROJECT environment variable is required"
        
        return True, None
    
    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"BotConfig("
            f"email={self.email}, "
            f"project_id={self.project_id}, "
            f"use_emulator={bool(self.firestore_emulator_host)}, "
            f"use_ollama={self.use_ollama}, "
            f"interval={self.interval_seconds}s"
            f")"
        )

