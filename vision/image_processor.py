"""
Image processing utilities for the vision system.
Handles image loading, validation, and preprocessing.
"""
import io
import logging
from typing import Optional, Tuple, Union, List
from pathlib import Path
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Processed image ready for model inference."""
    image: 'PIL.Image.Image'
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    format: str
    source: str  # 'file', 'bytes', 'url', 'base64'
    
    def to_base64(self) -> str:
        """Convert image to base64 string."""
        buffer = io.BytesIO()
        self.image.save(buffer, format=self.format.upper() or 'PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class ImageProcessor:
    """
    Image processing utilities for loading, validating, and preprocessing images.
    """
    
    SUPPORTED_FORMATS = {'JPEG', 'JPG', 'PNG', 'WEBP', 'GIF', 'BMP'}
    
    def __init__(
        self,
        max_size: Tuple[int, int] = (1024, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        convert_to_rgb: bool = True
    ):
        """
        Initialize image processor.
        
        Args:
            max_size: Maximum image dimensions (width, height)
            target_size: Target size for model input (if None, just limit to max_size)
            convert_to_rgb: Convert images to RGB mode
        """
        self.max_size = max_size
        self.target_size = target_size
        self.convert_to_rgb = convert_to_rgb
        
        # Lazy import PIL
        self._pil_imported = False
    
    def _ensure_pil(self):
        """Ensure PIL is imported."""
        if not self._pil_imported:
            try:
                global Image, ImageOps
                from PIL import Image, ImageOps
                self._pil_imported = True
            except ImportError:
                raise ImportError(
                    "Pillow is required for image processing. "
                    "Install with: pip install Pillow"
                )
    
    def process_file(self, file_path: str) -> Optional[ProcessedImage]:
        """
        Process an image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            ProcessedImage or None if processing fails
        """
        self._ensure_pil()
        
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Image file not found: {file_path}")
            return None
        
        try:
            image = Image.open(path)
            return self._process_image(image, source='file')
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            return None
    
    def process_bytes(self, image_bytes: bytes) -> Optional[ProcessedImage]:
        """
        Process image from bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            ProcessedImage or None if processing fails
        """
        self._ensure_pil()
        
        if not image_bytes:
            logger.error("Empty image bytes provided")
            return None
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self._process_image(image, source='bytes')
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return None
    
    def process_base64(self, base64_string: str) -> Optional[ProcessedImage]:
        """
        Process image from base64 string.
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            ProcessedImage or None if processing fails
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            result = self.process_bytes(image_bytes)
            if result:
                result.source = 'base64'
            return result
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return None
    
    async def process_url(self, url: str) -> Optional[ProcessedImage]:
        """
        Download and process image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            ProcessedImage or None if processing fails
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download image: HTTP {response.status}")
                        return None
                    
                    image_bytes = await response.read()
            
            result = self.process_bytes(image_bytes)
            if result:
                result.source = 'url'
            return result
            
        except ImportError:
            # Fallback to sync requests
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                result = self.process_bytes(response.content)
                if result:
                    result.source = 'url'
                return result
            except Exception as e:
                logger.error(f"Error downloading image from {url}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error processing URL image {url}: {e}")
            return None
    
    def _process_image(
        self,
        image: 'Image.Image',
        source: str
    ) -> Optional[ProcessedImage]:
        """
        Internal method to process PIL Image.
        
        Args:
            image: PIL Image object
            source: Source identifier
            
        Returns:
            ProcessedImage or None
        """
        try:
            original_size = image.size
            image_format = image.format or 'PNG'
            
            # Validate format
            if image_format.upper() not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported image format: {image_format}")
                # Try to continue anyway
            
            # Convert mode if needed
            if self.convert_to_rgb and image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Handle transparency
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Resize if needed
            if self.target_size:
                image = self._resize_image(image, self.target_size)
            elif image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
                image = self._resize_image(image, self.max_size, maintain_aspect=True)
            
            # Apply EXIF orientation
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass  # Ignore EXIF errors
            
            return ProcessedImage(
                image=image,
                original_size=original_size,
                processed_size=image.size,
                format=image_format,
                source=source
            )
            
        except Exception as e:
            logger.error(f"Error in _process_image: {e}")
            return None
    
    def _resize_image(
        self,
        image: 'Image.Image',
        target_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> 'Image.Image':
        """
        Resize image to target size.
        
        Args:
            image: PIL Image
            target_size: Target (width, height)
            maintain_aspect: Maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        if maintain_aspect:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def validate_image(self, image_bytes: bytes) -> Tuple[bool, str]:
        """
        Validate image bytes without full processing.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        self._ensure_pil()
        
        if not image_bytes:
            return False, "Empty image data"
        
        # Check minimum size
        if len(image_bytes) < 100:
            return False, "Image data too small"
        
        # Check maximum size (10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return False, "Image too large (max 10MB)"
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify image integrity
            
            # Check format
            image = Image.open(io.BytesIO(image_bytes))
            if image.format and image.format.upper() not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported format: {image.format}"
            
            # Check dimensions
            if image.size[0] < 10 or image.size[1] < 10:
                return False, "Image too small (minimum 10x10)"
            
            if image.size[0] > 10000 or image.size[1] > 10000:
                return False, "Image dimensions too large (max 10000x10000)"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def get_image_info(self, image_bytes: bytes) -> Optional[dict]:
        """
        Get basic information about an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with image info or None
        """
        self._ensure_pil()
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.size[0],
                "height": image.size[1],
                "file_size": len(image_bytes)
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return None
