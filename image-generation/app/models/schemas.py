"""Request/response schemas for image generation endpoints."""

from pydantic import BaseModel, Field


class CoverRequest(BaseModel):
    """Text-to-image generation request."""

    prompt: str = Field(..., min_length=1, max_length=500)


class CoverResponse(BaseModel):
    """Generated image payload."""

    image: str
    prompt: str
    model: str
    generation_time: float
