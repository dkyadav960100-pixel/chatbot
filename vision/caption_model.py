"""
Caption model module for generating image descriptions.
Supports multiple vision models: BLIP, BLIP-2, GIT.
"""
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported vision model types."""
    BLIP = "blip"
    BLIP2 = "blip2"
    GIT = "git"


@dataclass
class CaptionResult:
    """Result from caption generation."""
    caption: str
    tags: List[str]
    confidence: float
    model: str
    processing_time: float
    metadata: Dict[str, Any]


class CaptionModel:
    """
    Image captioning model supporting multiple backends.
    Supports BLIP, BLIP-2, and GIT models from Hugging Face.
    """
    
    MODEL_CONFIGS = {
        ModelType.BLIP: {
            "model_id": "Salesforce/blip-image-captioning-base",
            "large_model_id": "Salesforce/blip-image-captioning-large",
            "processor_class": "BlipProcessor",
            "model_class": "BlipForConditionalGeneration"
        },
        ModelType.BLIP2: {
            "model_id": "Salesforce/blip2-opt-2.7b",
            "processor_class": "Blip2Processor", 
            "model_class": "Blip2ForConditionalGeneration"
        },
        ModelType.GIT: {
            "model_id": "microsoft/git-base-coco",
            "large_model_id": "microsoft/git-large-coco",
            "processor_class": "AutoProcessor",
            "model_class": "AutoModelForCausalLM"
        }
    }
    
    def __init__(
        self,
        model_type: str = "blip",
        model_id: Optional[str] = None,
        device: str = "cpu",
        use_large: bool = False,
        max_length: int = 100,
        num_beams: int = 4,
        num_tags: int = 5
    ):
        """
        Initialize caption model.
        
        Args:
            model_type: Type of model ('blip', 'blip2', 'git')
            model_id: Custom model ID (overrides default)
            device: Device to run on ('cpu' or 'cuda')
            use_large: Use large variant if available
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            num_tags: Number of tags to generate
        """
        self.model_type = ModelType(model_type.lower())
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_tags = num_tags
        
        # Get model config
        config = self.MODEL_CONFIGS[self.model_type]
        
        if model_id:
            self.model_id = model_id
        elif use_large and "large_model_id" in config:
            self.model_id = config["large_model_id"]
        else:
            self.model_id = config["model_id"]
        
        self._processor = None
        self._model = None
        self._loaded = False
        
        logger.info(f"Initialized CaptionModel with {self.model_id}")
    
    def _load_model(self):
        """Load model and processor."""
        if self._loaded:
            return
        
        import time
        start_time = time.time()
        
        try:
            import torch
            
            config = self.MODEL_CONFIGS[self.model_type]
            
            logger.info(f"Loading vision model: {self.model_id}")
            
            if self.model_type == ModelType.BLIP:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                self._processor = BlipProcessor.from_pretrained(self.model_id)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32
                ).to(self.device)
                
            elif self.model_type == ModelType.BLIP2:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                
                self._processor = Blip2Processor.from_pretrained(self.model_id)
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                
            elif self.model_type == ModelType.GIT:
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                self._processor = AutoProcessor.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id
                ).to(self.device)
            
            # Set eval mode
            self._model.eval()
            self._loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_caption(
        self,
        image: 'PIL.Image.Image',
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate caption for an image.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt to guide caption generation
            
        Returns:
            Generated caption string
        """
        self._load_model()
        
        import torch
        
        try:
            with torch.no_grad():
                if self.model_type == ModelType.BLIP:
                    if prompt:
                        # Conditional captioning
                        inputs = self._processor(
                            image,
                            text=prompt,
                            return_tensors="pt"
                        ).to(self.device)
                    else:
                        # Unconditional captioning
                        inputs = self._processor(
                            image,
                            return_tensors="pt"
                        ).to(self.device)
                    
                    outputs = self._model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams
                    )
                    
                elif self.model_type == ModelType.BLIP2:
                    inputs = self._processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
                    
                    outputs = self._model.generate(
                        **inputs,
                        max_length=self.max_length
                    )
                    
                elif self.model_type == ModelType.GIT:
                    inputs = self._processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self._model.generate(
                        pixel_values=inputs.pixel_values,
                        max_length=self.max_length,
                        num_beams=self.num_beams
                    )
                
                caption = self._processor.decode(outputs[0], skip_special_tokens=True)
                return caption.strip()
                
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Unable to generate caption"
    
    def generate_tags(
        self,
        image: 'PIL.Image.Image',
        num_tags: Optional[int] = None
    ) -> List[str]:
        """
        Generate descriptive tags for an image.
        
        Args:
            image: PIL Image object
            num_tags: Number of tags to generate (default: self.num_tags)
            
        Returns:
            List of tag strings
        """
        num_tags = num_tags or self.num_tags
        
        # Generate detailed caption first
        caption = self.generate_caption(image)
        
        # Extract tags from caption
        tags = self._extract_tags_from_caption(caption, num_tags)
        
        # If not enough tags, generate more with prompts
        if len(tags) < num_tags:
            additional = self._generate_tags_with_prompts(image, num_tags - len(tags))
            tags.extend(additional)
        
        return tags[:num_tags]
    
    def _extract_tags_from_caption(self, caption: str, max_tags: int) -> List[str]:
        """Extract relevant tags from a caption."""
        import re
        
        # Common stop words to filter
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'but', 'and',
            'or', 'if', 'then', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'there',
            'here', 'image', 'picture', 'photo', 'shows', 'showing'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', caption.lower())
        
        # Filter and deduplicate
        tags = []
        seen = set()
        
        for word in words:
            if word not in stop_words and word not in seen:
                seen.add(word)
                tags.append(word)
                
                if len(tags) >= max_tags:
                    break
        
        return tags
    
    def _generate_tags_with_prompts(
        self,
        image: 'PIL.Image.Image',
        num_tags: int
    ) -> List[str]:
        """Generate additional tags using prompted captioning."""
        prompts = [
            "The main subject is",
            "The colors are",
            "The setting is",
            "The mood is",
            "Objects include"
        ]
        
        tags = []
        
        for prompt in prompts[:num_tags]:
            try:
                response = self.generate_caption(image, prompt)
                # Extract first meaningful word after prompt
                words = response.lower().split()
                for word in words:
                    clean_word = ''.join(c for c in word if c.isalpha())
                    if len(clean_word) >= 3 and clean_word not in tags:
                        tags.append(clean_word)
                        break
            except Exception:
                continue
        
        return tags
    
    def process_image(
        self,
        image: 'PIL.Image.Image'
    ) -> CaptionResult:
        """
        Complete image processing: generate caption and tags.
        
        Args:
            image: PIL Image object
            
        Returns:
            CaptionResult with caption, tags, and metadata
        """
        import time
        start_time = time.time()
        
        try:
            caption = self.generate_caption(image)
            tags = self.generate_tags(image)
            
            processing_time = time.time() - start_time
            
            return CaptionResult(
                caption=caption,
                tags=tags,
                confidence=0.8,  # Default confidence
                model=self.model_id,
                processing_time=processing_time,
                metadata={
                    "model_type": self.model_type.value,
                    "device": self.device,
                    "image_size": image.size
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return CaptionResult(
                caption="Unable to process image",
                tags=[],
                confidence=0.0,
                model=self.model_id,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        self._loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
