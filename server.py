"""Unified HTTP server for NovelSync services (agents and optional image generation)."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
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
    BackendUnavailableError,
    BillingCommitError,
    InsufficientCreditsError,
    InvalidRequestError,
    LLMProviderError,
    LLMTimeoutError,
    ProviderAuthError,
    ProviderNotFoundError,
    RateLimitedError,
    _byok_config,
    _firebase_token,
)
from config import Settings
from rate_limit import PerUserRateLimiter

logger = structlog.get_logger(__name__)


def _configure_environment() -> Path:
    """Load env variables and local emulator defaults."""
    current_dir = Path(__file__).parent
    env_path = current_dir / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    return current_dir


def _configure_logging() -> None:
    """Set up structlog with JSON output for Cloud Run / local dev."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    # Also configure stdlib logging for third-party libs (uvicorn, google-auth, etc.).
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


async def _verify_internal_token(request: Request) -> None:
    """Verify that the request comes from Firebase Functions via Google OIDC token.

    No-op outside production so local dev works without credentials.
    In production: validates Bearer token signature, expiry, audience (AGENT_SERVICE_URL),
    and requires the token email claim to be on the configured caller allowlist.
    """
    audience: Optional[str] = request.app.state.oidc_audience
    allowed_callers: frozenset = request.app.state.allowed_callers
    if not audience:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Missing Authorization header",
                "details": None,
            },
        )

    token = auth_header.split(" ", 1)[1]

    try:
        claims = google_id_token.verify_oauth2_token(
            token,
            request.app.state.google_auth_request,  # cached — not created per call
            audience=audience,
        )
        caller_email = claims.get("email")
        if caller_email not in allowed_callers:
            raise ValueError(f"Unexpected caller email: {caller_email}")
    except Exception as exc:
        logger.warning("token_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or unauthorized token",
                "details": None,
            },
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
    """Request model for agent execution.

    `user_id` is the end-user identifier from Firebase Functions (the upstream
    OIDC service-account caller is validated separately in _verify_internal_token).
    Required so the rate limiter and billing pool are keyed per actual user — not
    a shared "anonymous" bucket.
    """

    action: ActionName
    parameters: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = Field(min_length=1, max_length=128)
    firebase_token: Optional[str] = None
    provider_config: Optional[ProviderConfig] = None


class AgentResponse(BaseModel):
    """Response model for agent execution."""

    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorDetail] = None


def _try_load_image_router(current_dir: Path, enabled: bool):
    """Try to load image generation router when enabled."""
    if not enabled:
        logger.info("image_generation_disabled")
        return None

    try:
        image_gen_path = current_dir / "image-generation"
        if str(image_gen_path) not in sys.path:
            sys.path.insert(0, str(image_gen_path))
        from app.api.routes import router as image_router  # type: ignore

        logger.info("image_generation_loaded")
        return image_router
    except ImportError as exc:
        logger.warning("image_generation_unavailable", error=str(exc))
        return None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    _configure_logging()
    current_dir = _configure_environment()

    # Instantiate settings here (not at module level) so tests can monkeypatch env vars first.
    settings = Settings()

    # Set FIRESTORE_EMULATOR_HOST for non-production if not already set.
    if settings.environment != "production" and not settings.firestore_emulator_host:
        os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: nothing to do — state is populated below before the app accepts
        # traffic. Shutdown: release the LLM HTTP client.
        yield
        await app.state.agent.aclose()

    app = FastAPI(
        title="NovelSync Unified Service",
        description="Unified API for story agents and optional image generation",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_origins,
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Cache auth helpers on app.state — one instance for the process lifetime.
    app.state.oidc_audience = settings.oidc_audience
    app.state.allowed_callers = settings.allowed_callers
    app.state.google_auth_request = google_requests.Request()

    if app.state.oidc_audience:
        logger.info("oidc_audience_set", audience=app.state.oidc_audience)
    if app.state.allowed_callers:
        logger.info(
            "oidc_allowed_callers",
            count=len(app.state.allowed_callers),
            callers=sorted(app.state.allowed_callers),
        )

    app.state.project_id = settings.google_cloud_project
    app.state.rate_limiter = PerUserRateLimiter(
        settings.max_requests_per_minute_per_user
    )
    if settings.max_requests_per_minute_per_user > 0:
        logger.info("rate_limit_enabled", rpm=settings.max_requests_per_minute_per_user)

    app.state.agent = StoryAgent(
        project_id=settings.google_cloud_project,
        location=settings.vertex_ai_location,
    )

    image_router = _try_load_image_router(
        current_dir, settings.enable_local_image_generation
    )
    app.state.image_generation_available = image_router is not None
    if image_router is not None:
        app.include_router(image_router, tags=["Image Generation"])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=AgentResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Invalid request",
                    details=exc.errors(),
                ),
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
        user_id = request.user_id
        if not await raw_request.app.state.rate_limiter.allow(user_id):
            logger.warning("rate_limit_exceeded", user_id=user_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "RATE_LIMITED",
                    "message": "Too many AI requests. Please try again in a minute.",
                    "details": None,
                },
            )

        try:
            validated_params = validate_action_parameters(
                request.action, request.parameters
            )

            # Set per-request config in ContextVar so CreditProxyProvider picks it up.
            # ContextVar is async-safe: this context copy is isolated to this request's task.
            pc = request.provider_config
            byok_token = _byok_config.set(
                {
                    "user_id": request.user_id,
                    "provider": pc.provider if pc else "",
                    "api_key": pc.api_key if pc else "",
                    "model": pc.model or "" if pc else "",
                }
            )
            incoming_firebase_token = (
                request.firebase_token
                or raw_request.headers.get("X-Firebase-Token", "")
            ).strip() or None
            firebase_token = _firebase_token.set(incoming_firebase_token)

            try:
                result = await app.state.agent.execute_agent(
                    request.action,
                    validated_params,
                    background_tasks=background_tasks,
                    user_id=request.user_id,
                )
            finally:
                _byok_config.reset(byok_token)
                _firebase_token.reset(firebase_token)
            return AgentResponse(success=True, data=result)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid parameters",
                    "details": exc.errors(),
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "BAD_REQUEST", "message": str(exc), "details": None},
            ) from exc
        except InsufficientCreditsError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INSUFFICIENT_CREDITS",
                    "message": "Insufficient AI credits. Please add your own API key in Settings.",
                    "details": None,
                },
            ) from exc
        except ProviderAuthError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "UNAUTHORIZED",
                    "message": "AI provider authentication failed. Please check your API key in Settings.",
                    "details": None,
                },
            ) from exc
        except BackendUnavailableError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "BACKEND_UNAVAILABLE",
                    "message": "AI backend is unreachable. Please try again later.",
                    "details": None,
                },
            ) from exc
        except ProviderNotFoundError as exc:
            if not exc.model:
                msg = (
                    f'No model selected for provider "{exc.provider}". Please choose a model in Settings.'
                    if exc.provider
                    else "No AI model configured. Please add your API key and select a model in Settings."
                )
            else:
                model_label = (
                    f"{exc.provider}/{exc.model}" if exc.provider else exc.model
                )
                msg = f'AI model "{model_label}" not found. Please check your model name in Settings.'
            raise HTTPException(
                status_code=500,
                detail={"code": "PROVIDER_NOT_FOUND", "message": msg, "details": None},
            ) from exc
        except RateLimitedError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "RATE_LIMITED",
                    "message": "AI provider rate limit reached. Please try again in a few minutes.",
                    "details": None,
                },
            ) from exc
        except LLMTimeoutError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "TIMEOUT",
                    "message": "AI request timed out. Please try again.",
                    "details": None,
                },
            ) from exc
        except InvalidRequestError as exc:
            # creditProxy rejected the request itself (e.g. prompt/params too
            # large) — not a provider or credits issue, and NOT fixed by
            # retrying with the same input.
            logger.warning(
                "invalid_request_to_credit_proxy",
                action=request.action,
                error_type=type(exc).__name__,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_REQUEST",
                    "message": "Your request couldn't be processed — try shortening the input.",
                    "details": None,
                },
            ) from exc
        except BillingCommitError as exc:
            # The LLM call succeeded (content was generated, provider cost
            # already incurred) but billing the reservation afterward failed,
            # so no response was ever returned to the caller. This is a real
            # cost leak worth being loud about — log at error level, not warn.
            logger.error(
                "billing_commit_failed",
                action=request.action,
                error_type=type(exc).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "BILLING_ERROR",
                    "message": "AI service hit a billing error. Please try again.",
                    "details": None,
                },
            ) from exc
        except LLMProviderError as exc:
            logger.exception(
                "llm_provider_error",
                action=request.action,
                error_type=type(exc).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INTERNAL_ERROR",
                    "message": "AI service is temporarily unavailable. Please try again.",
                    "details": None,
                },
            ) from exc
        except Exception as exc:
            logger.exception(
                "unhandled_error", action=request.action, error_type=type(exc).__name__
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INTERNAL_ERROR",
                    "message": "AI service is temporarily unavailable. Please try again.",
                    "details": None,
                },
            ) from exc

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "project_id": app.state.project_id,
            "services": {
                "agent": "available",
                "image_generation": (
                    "available"
                    if app.state.image_generation_available
                    else "unavailable"
                ),
            },
        }

    return app


try:
    app = create_app()
except Exception as exc:
    # _configure_logging is the first line of create_app, so the structlog
    # JSON renderer is wired up by the time most failures happen. Worst case
    # (logging itself failing) the stderr fallback still captures the message
    # before the worker exits. Without this guard, missing prod env vars
    # surface as a raw Pydantic stacktrace easy to miss in Cloud Run logs.
    logger.critical("startup_failed", error=str(exc), error_type=type(exc).__name__)
    raise


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
