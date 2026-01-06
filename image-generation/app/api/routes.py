"""API route handlers."""

import logging
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import CoverRequest, CoverResponse
from app.services.image_service import ImageService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize image service (singleton)
image_service = ImageService()


@router.post(
    "/generate-cover",
    response_model=CoverResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate an image from a text prompt",
    description="""
    Generate a cover image from a text prompt using Stable Diffusion.
    
    This endpoint accepts a text prompt and returns a base64-encoded PNG image.
    The model is optimized for low-end machines and runs efficiently on CPU.
    
    **Parameters:**
    - `prompt`: A text description of the image you want to generate (1-500 characters)
    
    **Returns:**
    - Base64-encoded PNG image
    - The prompt used
    - Model name
    - Generation time in seconds
    
    **Example:**
    ```json
    {
        "prompt": "A beautiful sunset over mountains"
    }
    ```
    """,
    responses={
        200: {
            "description": "Image generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "image": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "prompt": "A beautiful sunset over mountains",
                        "model": "stabilityai/sd-turbo",
                        "generation_time": 1.23
                    }
                }
            }
        },
        422: {
            "description": "Validation error - invalid request format",
        },
        500: {
            "description": "Internal server error - model not loaded or generation failed",
        }
    }
)
async def generate_cover(request: CoverRequest) -> CoverResponse:
    """
    Generate a cover image from a text prompt.
    
    Args:
        request: CoverRequest containing the text prompt
    
    Returns:
        CoverResponse with the generated image and metadata
    
    Raises:
        HTTPException: If image generation fails
    """
    try:
        logger.info(f"Received request to generate cover with prompt: {request.prompt[:50]}...")
        
        # Generate image
        image, generation_time = await image_service.generate_image(
            prompt=request.prompt
        )
        
        # Encode image to base64
        image_base64 = image_service.encode_image(image)
        
        # Create response
        response = CoverResponse(
            image=image_base64,
            prompt=request.prompt,
            model=settings.model_name,
            generation_time=round(generation_time, 2)
        )
        
        logger.info(f"Successfully generated image in {generation_time:.2f} seconds")
        
        return response
        
    except RuntimeError as e:
        logger.error(f"Runtime error during image generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during image generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        ) from e


@router.get(
    "/image-health",
    summary="Image generation health check endpoint",
    description="Check if the image generation API is running and the model is loaded",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model": "stabilityai/sd-turbo",
                        "device": "cpu"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        Dictionary with status and model information
    """
    try:
        # Check if model is loaded
        model_loaded = image_service._pipeline is not None
        
        return {
            "status": "healthy" if model_loaded else "model_not_loaded",
            "model": settings.model_name,
            "device": settings.device,
            "model_loaded": model_loaded
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

