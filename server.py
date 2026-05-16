"""Unified HTTP server for NovelSync services (agents and optional image generation)."""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field, ValidationError

from agents.storyAgent.action_schemas import ActionName, validate_action_parameters
from agents.storyAgent.agent import StoryAgent
from agents.storyAgent.llm_provider import (
    _byok_config,
    _firebase_token,
    BackendUnavailableError,
    InsufficientCreditsError,
    LLMProviderError,
    LLMTimeoutError,
    ProviderAuthError,
    ProviderNotFoundError,
    RateLimitedError,
)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)


async def _verify_internal_token(request: Request) -> None:
    """Verify that the request comes from Firebase Functions via Google OIDC token.

    No-op outside production so local dev works without credentials.
    In production: validates Bearer token signature + expiry, then checks the
    caller email matches FIREBASE_FUNCTIONS_SERVICE_ACCOUNT if that env var is set.
    """
    if os.getenv("ENVIRONMENT") != "production":
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Missing Authorization header", "details": None},
        )

    token = auth_header.split(" ", 1)[1]
    service_url = os.getenv("AGENT_SERVICE_URL") or None
    expected_sa = os.getenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", "")

    try:
        claims = google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            audience=service_url,
        )
        if expected_sa and claims.get("email") != expected_sa:
            raise ValueError(f"Unexpected caller email: {claims.get('email')}")
    except Exception as exc:
        logger.warning("Token validation failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Invalid or unauthorized token", "details": None},
        )


class ErrorDetail(BaseModel):
    """Stable error payload returned to API clients."""

    code: str
    message: str
    details: Optional[Any] = None


class ProviderConfig(BaseModel):
    """Per-request BYOK provider override."""

    provider: str  # "gemini" | "claude" | "openai"
    api_key: str
    model: Optional[str] = None


class AgentRequest(BaseModel):
    """Request model for agent execution."""

    action: ActionName
    parameters: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    firebase_token: Optional[str] = None
    provider_config: Optional[ProviderConfig] = None


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

    cors_origins_raw = os.getenv("CORS_ORIGINS", "[]")
    try:
        cors_origins = json.loads(cors_origins_raw)
        if not isinstance(cors_origins, list):
            cors_origins = []
    except (json.JSONDecodeError, ValueError):
        cors_origins = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "Authorization"],
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
    async def execute_agent(
        request: AgentRequest,
        raw_request: Request,
        background_tasks: BackgroundTasks,
        _: None = Depends(_verify_internal_token),
    ) -> AgentResponse:
        try:
            validated_params = validate_action_parameters(request.action, request.parameters)

            # Set per-request config in ContextVar so CreditProxyProvider picks it up.
            # Always set user_id so platform users are billed individually, not to shared "platform" pool.
            # ContextVar is async-safe: this context copy is isolated to this request's task.
            pc = request.provider_config
            byok_token = _byok_config.set({
                "user_id": request.user_id or "anonymous",
                "provider": pc.provider if pc else "",
                "api_key": pc.api_key if pc else "",
                "model": pc.model or "" if pc else "",
            })
            incoming_firebase_token = (request.firebase_token or raw_request.headers.get("X-Firebase-Token", "")).strip() or None
            firebase_token = _firebase_token.set(incoming_firebase_token)

            try:
                result = await app.state.agent.execute_agent(
                    request.action,
                    validated_params,
                    background_tasks=background_tasks,
                    user_id=request.user_id or "anonymous",
                )
            finally:
                _byok_config.reset(byok_token)
                _firebase_token.reset(firebase_token)
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
        except InsufficientCreditsError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "INSUFFICIENT_CREDITS", "message": "Insufficient AI credits. Please add your own API key in Settings.", "details": None},
            ) from exc
        except ProviderAuthError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "UNAUTHORIZED", "message": "AI provider authentication failed. Please check your API key in Settings.", "details": None},
            ) from exc
        except BackendUnavailableError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "BACKEND_UNAVAILABLE", "message": "AI backend is unreachable. Please try again later.", "details": None},
            ) from exc
        except ProviderNotFoundError as exc:
            if not exc.model:
                msg = f'No model selected for provider "{exc.provider}". Please choose a model in Settings.' if exc.provider else "No AI model configured. Please add your API key and select a model in Settings."
            else:
                model_label = f"{exc.provider}/{exc.model}" if exc.provider else exc.model
                msg = f'AI model "{model_label}" not found. Please check your model name in Settings.'
            raise HTTPException(
                status_code=500,
                detail={"code": "PROVIDER_NOT_FOUND", "message": msg, "details": None},
            ) from exc
        except RateLimitedError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "RATE_LIMITED", "message": "AI provider rate limit reached. Please try again in a few minutes.", "details": None},
            ) from exc
        except LLMTimeoutError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "TIMEOUT", "message": "AI request timed out. Please try again.", "details": None},
            ) from exc
        except (LLMProviderError, Exception) as exc:
            logger.exception("Unhandled error for action=%s", request.action)
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL_ERROR", "message": "AI service is temporarily unavailable. Please try again.", "details": None},
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
