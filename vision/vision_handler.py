"""
Vision handler - Main interface for image processing.
Combines image processing and captioning into a unified interface.
"""
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .image_processor import ImageProcessor, ProcessedImage
from .caption_model import CaptionModel, CaptionResult

logger = logging.getLogger(__name__)


@dataclass
class VisionResponse:
    """Complete vision processing response."""
    success: bool
    caption: str
    tags: List[str]
    error: Optional[str]
    processing_time: float
    image_info: Dict[str, Any]
    model_info: Dict[str, Any]
    
    def format_response(self) -> str:
        """Format response for user display."""
        if not self.success:
            return f"❌ Error processing image: {self.error}"
        
        lines = [
            "🖼️ **Image Description**",
            "",
            f"📝 **Caption:** {self.caption}",
            "",
            f"🏷️ **Tags:** {', '.join(f'#{tag}' for tag in self.tags)}",
        ]
        
        return "\n".join(lines)


class VisionHandler:
    """
    Main vision system handler.
    Provides a unified interface for image captioning and tagging.
    """
    
    def __init__(
        self,
        model_type: str = "blip",
        model_id: Optional[str] = None,
        device: str = "cpu",
        max_image_size: tuple = (1024, 1024),
        num_tags: int = 5
    ):
        """
        Initialize vision handler.
        
        Args:
            model_type: Vision model type ('blip', 'blip2', 'git')
            model_id: Custom model ID
            device: Device for inference ('cpu' or 'cuda')
            max_image_size: Maximum image dimensions
            num_tags: Number of tags to generate
        """
        self.device = device
        
        # Initialize components
        self.image_processor = ImageProcessor(
            max_size=max_image_size,
            convert_to_rgb=True
        )
        
        self.caption_model = CaptionModel(
            model_type=model_type,
            model_id=model_id,
            device=device,
            num_tags=num_tags
        )
        
        self._model_loaded = False
        logger.info(f"VisionHandler initialized with {model_type} model")
    
    def preload_model(self):
        """Preload the vision model for faster inference."""
        try:
            self.caption_model._load_model()
            self._model_loaded = True
            logger.info("Vision model preloaded")
        except Exception as e:
            logger.error(f"Error preloading model: {e}")
    
    def process_image_bytes(
        self,
        image_bytes: bytes,
        generate_tags: bool = True
    ) -> VisionResponse:
        """
        Process image from raw bytes.
        
        Args:
            image_bytes: Raw image bytes
            generate_tags: Whether to generate tags
            
        Returns:
            VisionResponse with results
        """
        import time
        start_time = time.time()
        
        # Validate image
        is_valid, error_msg = self.image_processor.validate_image(image_bytes)
        if not is_valid:
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error=error_msg,
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
        
        # Process image
        processed = self.image_processor.process_bytes(image_bytes)
        if not processed:
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error="Failed to process image",
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
        
        return self._generate_response(processed, generate_tags, start_time)
    
    def process_image_file(
        self,
        file_path: str,
        generate_tags: bool = True
    ) -> VisionResponse:
        """
        Process image from file.
        
        Args:
            file_path: Path to image file
            generate_tags: Whether to generate tags
            
        Returns:
            VisionResponse with results
        """
        import time
        start_time = time.time()
        
        processed = self.image_processor.process_file(file_path)
        if not processed:
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error=f"Failed to load image: {file_path}",
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
        
        return self._generate_response(processed, generate_tags, start_time)
    
    async def process_image_url(
        self,
        url: str,
        generate_tags: bool = True
    ) -> VisionResponse:
        """
        Process image from URL.
        
        Args:
            url: Image URL
            generate_tags: Whether to generate tags
            
        Returns:
            VisionResponse with results
        """
        import time
        start_time = time.time()
        
        processed = await self.image_processor.process_url(url)
        if not processed:
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error=f"Failed to download image from URL",
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
        
        return self._generate_response(processed, generate_tags, start_time)
    
    def _generate_response(
        self,
        processed: ProcessedImage,
        generate_tags: bool,
        start_time: float
    ) -> VisionResponse:
        """
        Generate caption and tags for processed image.
        
        Args:
            processed: ProcessedImage object
            generate_tags: Whether to generate tags
            start_time: Processing start time
            
        Returns:
            VisionResponse with results
        """
        import time
        
        try:
            # Generate caption
            result = self.caption_model.process_image(processed.image)
            
            image_info = {
                "original_size": processed.original_size,
                "processed_size": processed.processed_size,
                "format": processed.format,
                "source": processed.source
            }
            
            model_info = {
                "model": result.model,
                "model_type": result.metadata.get("model_type", "unknown"),
                "device": self.device
            }
            
            return VisionResponse(
                success=True,
                caption=result.caption,
                tags=result.tags if generate_tags else [],
                error=None,
                processing_time=time.time() - start_time,
                image_info=image_info,
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error=str(e),
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
    
    def process_with_prompt(
        self,
        image_bytes: bytes,
        prompt: str
    ) -> VisionResponse:
        """
        Process image with a custom prompt for guided captioning.
        
        Args:
            image_bytes: Raw image bytes
            prompt: Custom prompt to guide caption
            
        Returns:
            VisionResponse with results
        """
        import time
        start_time = time.time()
        
        processed = self.image_processor.process_bytes(image_bytes)
        if not processed:
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error="Failed to process image",
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
        
        try:
            caption = self.caption_model.generate_caption(processed.image, prompt)
            
            return VisionResponse(
                success=True,
                caption=caption,
                tags=[],
                error=None,
                processing_time=time.time() - start_time,
                image_info={
                    "original_size": processed.original_size,
                    "processed_size": processed.processed_size
                },
                model_info={"model": self.caption_model.model_id}
            )
            
        except Exception as e:
            logger.error(f"Error with prompted caption: {e}")
            return VisionResponse(
                success=False,
                caption="",
                tags=[],
                error=str(e),
                processing_time=time.time() - start_time,
                image_info={},
                model_info={}
            )
    
    def get_image_info(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Get information about an image without processing."""
        return self.image_processor.get_image_info(image_bytes)
    
    def unload_model(self):
        """Unload the vision model to free memory."""
        self.caption_model.unload()
        self._model_loaded = False
    
    @property
    def is_ready(self) -> bool:
        """Check if vision system is ready for inference."""
        return self._model_loaded
    
    def get_status(self) -> Dict[str, Any]:
        """Get vision system status."""
        return {
            "model_loaded": self._model_loaded,
            "model_type": self.caption_model.model_type.value,
            "model_id": self.caption_model.model_id,
            "device": self.device
        }
