"""Unified HTTP server for NovelSync services (agents and optional image generation)."""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from agents.storyAgent.action_schemas import ActionName, validate_action_parameters
from agents.storyAgent.agent import StoryAgent

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Stable error payload returned to API clients."""

    code: str
    message: str
    details: Optional[Any] = None


class AgentRequest(BaseModel):
    """Request model for agent execution."""

    action: ActionName
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response model for agent execution."""

    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorDetail] = None


def _configure_environment() -> Path:
    """Load env variables and local emulator defaults."""
    current_dir = Path(__file__).parent
    env_path = current_dir / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    if os.getenv("ENVIRONMENT") != "production" and not os.getenv("FIRESTORE_EMULATOR_HOST"):
        os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"

    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    return current_dir


def _try_load_image_router(current_dir: Path):
    """Try to load image generation router when enabled."""
    if os.getenv("ENABLE_LOCAL_IMAGE_GENERATION", "true").lower() != "true":
        logger.info("Image generation disabled (ENABLE_LOCAL_IMAGE_GENERATION=false)")
        return None

    try:
        image_gen_path = current_dir / "image-generation"
        if str(image_gen_path) not in sys.path:
            sys.path.insert(0, str(image_gen_path))
        from app.api.routes import router as image_router  # type: ignore

        logger.info("Image generation routes loaded successfully")
        return image_router
    except ImportError as exc:
        logger.warning("Image generation routes not available: %s", exc)
        return None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    current_dir = _configure_environment()

    app = FastAPI(
        title="NovelSync Unified Service",
        description="Unified API for story agents and optional image generation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")

    app.state.project_id = project_id
    app.state.agent = StoryAgent(
        project_id=project_id,
        location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
    )

    image_router = _try_load_image_router(current_dir)
    app.state.image_generation_available = image_router is not None
    if image_router is not None:
        app.include_router(image_router, tags=["Image Generation"])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=AgentResponse(
                success=False,
                error=ErrorDetail(code="VALIDATION_ERROR", message="Invalid request", details=exc.errors()),
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and {"code", "message"}.issubset(detail.keys()):
            error = ErrorDetail(**detail)
        else:
            error = ErrorDetail(code="HTTP_ERROR", message=str(detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=AgentResponse(success=False, error=error).model_dump(),
        )

    @app.post("/agent/execute", response_model=AgentResponse)
    async def execute_agent(request: AgentRequest, background_tasks: BackgroundTasks) -> AgentResponse:
        try:
            validated_params = validate_action_parameters(request.action, request.parameters)
            result = await app.state.agent.execute_agent(
                request.action, validated_params, background_tasks=background_tasks
            )
            return AgentResponse(success=True, data=result)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "VALIDATION_ERROR", "message": "Invalid parameters", "details": exc.errors()},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "BAD_REQUEST", "message": str(exc), "details": None},
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("Unhandled error for action=%s", request.action)
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR", "message": "Internal server error", "details": None},
            ) from exc

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "project_id": app.state.project_id,
            "services": {
                "agent": "available",
                "image_generation": "available" if app.state.image_generation_available else "unavailable",
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
