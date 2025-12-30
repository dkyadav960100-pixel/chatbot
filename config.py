"""
Configuration module for GenAI Telegram Bot.
Handles all settings, environment variables, and model configurations.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    allowed_users: List[int] = field(default_factory=list)  # Empty = allow all
    max_message_length: int = 4096
    rate_limit_messages: int = 30  # Max messages per minute
    rate_limit_window: int = 60  # Window in seconds


@dataclass
class RAGConfig:
    """RAG system configuration."""
    # Embedding model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Retrieval settings
    top_k: int = 3
    similarity_threshold: float = 0.3
    
    # Knowledge base path
    knowledge_base_path: str = field(
        default_factory=lambda: os.getenv(
            "KNOWLEDGE_BASE_PATH", 
            str(Path(__file__).parent / "knowledge_base")
        )
    )
    
    # Vector store path
    vector_store_path: str = field(
        default_factory=lambda: os.getenv(
            "VECTOR_STORE_PATH",
            str(Path(__file__).parent / "data" / "vector_store.db")
        )
    )


@dataclass
class VisionConfig:
    """Vision system configuration."""
    # Model selection: 'blip', 'blip2', 'git'
    model_name: str = field(
        default_factory=lambda: os.getenv("VISION_MODEL", "blip")
    )
    
    # Model paths/identifiers
    blip_model: str = "Salesforce/blip-image-captioning-base"
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    git_model: str = "microsoft/git-base-coco"
    
    # Processing settings
    max_image_size: tuple = (1024, 1024)
    supported_formats: List[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif"]
    )
    num_tags: int = 5
    max_caption_length: int = 100
    
    # Device settings
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cpu")
    )


@dataclass
class LLMConfig:
    """LLM configuration for answer generation."""
    # Primary: Ollama
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    )
    
    # Fallback: OpenAI
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: int = 30  # seconds
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    cache_path: str = field(
        default_factory=lambda: str(Path(__file__).parent / "data" / "cache.db")
    )
    query_cache_ttl: int = 3600  # 1 hour
    embedding_cache_ttl: int = 86400  # 24 hours
    max_cache_size: int = 1000  # Max entries


@dataclass
class SessionConfig:
    """User session configuration."""
    history_length: int = 3  # Number of interactions to remember
    session_timeout: int = 3600  # 1 hour
    max_sessions: int = 1000


@dataclass
class AppConfig:
    """Main application configuration."""
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    
    # App settings
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    def validate(self) -> bool:
        """Validate configuration and return True if valid."""
        errors = []
        
        if not self.telegram.bot_token:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_llm_provider(self) -> str:
        """Determine which LLM provider to use."""
        # Check if Ollama is available
        try:
            import requests
            response = requests.get(
                f"{self.llm.ollama_base_url}/api/tags",
                timeout=2
            )
            if response.status_code == 200:
                return "ollama"
        except Exception:
            pass
        
        # Check if OpenAI API key is available
        if self.llm.openai_api_key:
            return "openai"
        
        return "none"


# Global configuration instance
config = AppConfig()


def load_config() -> AppConfig:
    """Load and validate configuration."""
    global config
    config = AppConfig()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    
    return config


def get_config() -> AppConfig:
    """Get current configuration."""
    return config
