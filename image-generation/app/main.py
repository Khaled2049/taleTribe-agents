"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from app.api.routes import router
from app.config import settings
from app.services.image_service import ImageService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Loads the model on startup and cleans up on shutdown.
    """
    # Startup: Load model
    logger.info("Starting application...")
    try:
        logger.info("Loading image generation model...")
        ImageService()  # warm-load the model into the process at startup
        logger.info("Model loaded successfully. Application ready.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        logger.error(
            "Application will start but image generation will fail until model is loaded."
        )

    yield

    # Shutdown: Cleanup (if needed)
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["Image Generation"])


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.

    Returns:
        Welcome message and API information
    """
    return {
        "message": "Local Image Generator API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "generate_cover": "/api/v1/generate-cover",
    }


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "").strip() or "127.0.0.1",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
