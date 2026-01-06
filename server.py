"""Unified HTTP server for NovelSync services (agents and image generation)."""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
current_dir = Path(__file__).parent
env_path = current_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")

# Set FIRESTORE_EMULATOR_HOST for local development only
if os.getenv("ENVIRONMENT") != "production" and not os.getenv("FIRESTORE_EMULATOR_HOST"):
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"

# Add current directory to path for imports
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import agent
try:
    from agents.storyAgent.agent import StoryAgent
except ImportError as e:
    logger.error(f"Failed to import StoryAgent: {e}")
    raise

# Try to load image generation routes if enabled
IMAGE_GENERATION_AVAILABLE = False
image_router = None

if os.getenv("ENABLE_LOCAL_IMAGE_GENERATION", "true").lower() == "true":
    try:
        image_gen_path = current_dir / "image-generation"
        if str(image_gen_path) not in sys.path:
            sys.path.insert(0, str(image_gen_path))
        from app.api.routes import router as image_router  # type: ignore
        IMAGE_GENERATION_AVAILABLE = True
        logger.info("Image generation routes loaded successfully")
    except ImportError as e:
        logger.warning(f"Image generation not available: {e}")
else:
    logger.info("Image generation disabled (ENABLE_LOCAL_IMAGE_GENERATION=false)")

# Initialize FastAPI app
app = FastAPI(
    title="NovelSync Unified Service",
    description="Unified API for story agents and image generation"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")

agent = StoryAgent(
    project_id=PROJECT_ID,
    location=os.getenv("VERTEX_AI_LOCATION", "us-central1")  # Legacy parameter
)

# Include image generation routes if available
if IMAGE_GENERATION_AVAILABLE and image_router:
    app.include_router(image_router, tags=["Image Generation"])


class AgentRequest(BaseModel):
    """Request model for agent execution."""
    action: str
    parameters: Dict[str, Any]


class AgentResponse(BaseModel):
    """Response model for agent execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


@app.post("/agent/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest) -> AgentResponse:
    """
    Execute an agent action.

    Available actions:
    - generateStory: Generate a complete story
    - generateChapter: Generate a chapter
    - brainstormIdeas: Generate brainstorming ideas
    - brainstormCharacter: Generate character ideas
    - brainstormPlot: Generate plot ideas
    - generateNextLines: Generate next line suggestions
    """
    try:
        logger.info(
            f"Executing agent action: {request.action}, "
            f"parameters: {list(request.parameters.keys())}"
        )
        result = await agent.execute_agent(request.action, request.parameters)
        logger.info(f"Agent action '{request.action}' completed successfully")
        return AgentResponse(success=True, data=result)
    except Exception as e:
        logger.error(
            f"Error executing agent action '{request.action}': {str(e)}",
            exc_info=True
        )
        return AgentResponse(success=False, error=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project_id": PROJECT_ID,
        "services": {
            "agent": "available",
            "image_generation": "available" if IMAGE_GENERATION_AVAILABLE else "unavailable"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
