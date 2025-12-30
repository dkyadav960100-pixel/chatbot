"""
Embedding model module for generating text embeddings.
Supports sentence-transformers with caching.
"""
import logging
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    text: str
    embedding: np.ndarray
    model: str
    dimension: int
    
    @property
    def embedding_list(self) -> List[float]:
        """Get embedding as Python list."""
        return self.embedding.tolist()


class EmbeddingModel:
    """
    Text embedding model using sentence-transformers.
    Includes caching to avoid re-computing embeddings.
    """
    
    # Model dimension lookup
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "snowflake-arctic-embed": 384,
    }
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_enabled: bool = True,
        normalize: bool = True
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run model on ('cpu' or 'cuda')
            cache_enabled: Whether to cache embeddings
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.cache_enabled = cache_enabled
        self.normalize = normalize
        
        self._model = None
        self._cache = {} if cache_enabled else None
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 384)
        
        logger.info(f"Initialized EmbeddingModel with {model_name}")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Update dimension from actual model
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        result = EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            dimension=self._dimension
        )
        
        # Cache result
        if self._cache is not None:
            self._cache[cache_key] = result
        
        return result
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if self._cache is not None and cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            embeddings = self.model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )
            
            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, embeddings):
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self.model_name,
                    dimension=self._dimension
                )
                
                # Cache result
                if self._cache is not None:
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = result
                
                results.append((idx, result))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (convenience method).
        
        Args:
            query: Query text
            
        Returns:
            Embedding as numpy array
        """
        result = self.embed(query)
        return result.embedding
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for documents (convenience method).
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embeddings as numpy arrays
        """
        results = self.embed_batch(documents)
        return [r.embedding for r in results]
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # If already normalized, just dot product
        if self.normalize:
            return float(np.dot(embedding1, embedding2))
        
        # Otherwise, compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(
            (self.model_name + text).encode()
        ).hexdigest()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._cache),
            "model": self.model_name
        }


# Global instance for convenience
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> EmbeddingModel:
    """Get or create the default embedding model."""
    global _default_model
    
    if _default_model is None or _default_model.model_name != model_name:
        _default_model = EmbeddingModel(model_name)
    
    return _default_model
