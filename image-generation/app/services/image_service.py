"""Image generation service using Stable Diffusion."""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch
from PIL import Image

from app.config import settings

if TYPE_CHECKING:
    from diffusers import AutoPipelineForText2Image

logger = logging.getLogger(__name__)


class ImageService:
    """Service for generating images from text prompts."""
    
    _instance: Optional['ImageService'] = None
    _pipeline: Optional["AutoPipelineForText2Image"] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the image service."""
        if self._pipeline is None:
            self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the Stable Diffusion model.
        
        This method loads the model once at startup and keeps it in memory
        for subsequent requests, which is critical for performance on low-end machines.
        """
        try:
            from diffusers import AutoPipelineForText2Image

            logger.info(f"Loading model: {settings.model_name}")
            
            # Determine device (auto-detect CUDA if available)
            from app.config import get_device, get_torch_dtype
            device = get_device()
            torch_dtype = get_torch_dtype()
            
            logger.info(f"Using device: {device}")
            if device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"Using dtype: {torch_dtype}")

            # Load the pipeline
            pipeline_kwargs: dict[str, Any] = {}
            if settings.enable_model_caching:
                pipeline_kwargs["cache_dir"] = "./models"
            if torch_dtype is not None:
                pipeline_kwargs["torch_dtype"] = torch_dtype

            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                settings.model_name,
                **pipeline_kwargs,
            )
            
            # Move to device
            self._pipeline = self._pipeline.to(device)
            
            # Enable inference mode for better performance
            self._pipeline.set_progress_bar_config(disable=True)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    async def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[Image.Image, float]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            width: Image width (defaults to config value)
            height: Image height (defaults to config value)
            num_inference_steps: Number of inference steps (defaults to config value)
        
        Returns:
            Tuple of (PIL Image, generation_time_in_seconds)
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Please restart the application.")
        
        try:
            # Use defaults from config if not provided
            width = width or settings.default_width
            height = height or settings.default_height
            num_inference_steps = num_inference_steps or settings.default_steps
            
            # Validate dimensions (must be multiples of 8 for Stable Diffusion)
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            logger.info(f"Dimensions: {width}x{height}, Steps: {num_inference_steps}")
            
            start_time = time.time()
            
            # Generate image
            with torch.inference_mode():
                result = self._pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=settings.default_guidance_scale,
                )
            
            generation_time = time.time() - start_time
            
            image = result.images[0]
            
            logger.info(f"Image generated successfully in {generation_time:.2f} seconds")
            
            return image, generation_time
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}") from e
    
    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """
        Encode a PIL Image to base64 string.
        
        Args:
            image: PIL Image object
        
        Returns:
            Base64-encoded string of the image
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return base64_string
