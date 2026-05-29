"""Configuration settings for the image generation API."""

from typing import Literal

try:
    from pydantic_settings import BaseSettings

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover - compatibility for partially provisioned envs
    from pydantic import BaseModel as BaseSettings

    _HAS_PYDANTIC_SETTINGS = False

# Auto-detect CUDA availability
try:
    import torch

    _cuda_available = torch.cuda.is_available()
except ImportError:
    torch = None
    _cuda_available = False


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model configuration
    model_name: str = "stabilityai/sd-turbo"
    device: Literal["cpu", "cuda", "auto"] = "auto"  # "auto" will use CUDA if available

    # Image generation parameters
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 4  # Low steps for turbo models on low-end machines
    default_guidance_scale: float = 0.0  # Turbo models don't use guidance scale

    # Performance optimizations
    enable_model_caching: bool = True
    torch_dtype: str = "auto"  # "auto" will use float16 for CUDA, float32 for CPU

    # API configuration
    api_title: str = "Local Image Generator API"
    api_description: str = (
        "A FastAPI application for generating images locally using Stable Diffusion"
    )
    api_version: str = "1.0.0"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Logging
    log_level: str = "INFO"

    if _HAS_PYDANTIC_SETTINGS:
        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "case_sensitive": False,
            "extra": "ignore",
        }
    else:

        class Config:
            """Pydantic v1-style config used by the compatibility fallback."""

            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"  # Ignore extra environment variables not in this model


# Global settings instance
settings = Settings()


def get_device() -> str:
    """
    Get the actual device to use based on settings and availability.

    Returns:
        "cuda" if CUDA is available and requested, otherwise "cpu"
    """
    if settings.device == "auto":
        return "cuda" if _cuda_available else "cpu"
    elif settings.device == "cuda":
        return "cuda" if _cuda_available else "cpu"
    else:
        return "cpu"


def get_torch_dtype():
    """
    Get the torch dtype to use based on settings and device.

    Returns:
        torch.float16 for CUDA, torch.float32 for CPU
    """
    device = get_device()

    if torch is None:
        return None

    if settings.torch_dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    else:
        return getattr(torch, settings.torch_dtype, torch.float32)
