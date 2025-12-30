"""
LLM Handler - Manages LLM interactions with Ollama and OpenAI fallback.
Provides answer generation for RAG queries.
"""
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    NONE = "none"


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    provider: str
    model: str
    success: bool
    error: Optional[str]
    generation_time: float
    tokens_used: Optional[int]
    metadata: Dict[str, Any]


class LLMHandler:
    """
    LLM handler with Ollama primary and OpenAI fallback.
    Handles answer generation for RAG queries.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Follow these guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain relevant information, say so
3. Be concise but thorough
4. Cite relevant parts of the context when appropriate
5. Use clear, professional language"""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:1b",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        openai_base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize LLM handler.
        
        Args:
            ollama_base_url: Ollama API base URL
            ollama_model: Ollama model name
            openai_api_key: OpenAI API key (for fallback)
            openai_model: OpenAI model name
            openai_base_url: OpenAI API base URL
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._available_provider: Optional[LLMProvider] = None
        
        logger.info(f"LLMHandler initialized (Ollama: {ollama_model})")
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is available and responsive."""
        try:
            import requests
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                # Check if desired model is available
                data = response.json()
                models = [m.get('name', '') for m in data.get('models', [])]
                
                # Check for exact match or base model match
                model_base = self.ollama_model.split(':')[0]
                if any(self.ollama_model in m or model_base in m for m in models):
                    logger.info(f"Ollama available with model {self.ollama_model}")
                    return True
                else:
                    logger.warning(f"Ollama model {self.ollama_model} not found. Available: {models}")
                    return len(models) > 0  # Use any available model
            
            return False
            
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    def check_openai_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.openai_api_key)
    
    def get_available_provider(self, force_check: bool = False) -> LLMProvider:
        """
        Get the available LLM provider.
        
        Args:
            force_check: Force re-check availability
            
        Returns:
            Available LLMProvider
        """
        if self._available_provider is not None and not force_check:
            return self._available_provider
        
        if self.check_ollama_available():
            self._available_provider = LLMProvider.OLLAMA
        elif self.check_openai_available():
            self._available_provider = LLMProvider.OPENAI
        else:
            self._available_provider = LLMProvider.NONE
        
        return self._available_provider
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response using the available LLM.
        
        Args:
            prompt: User prompt/question
            system_prompt: System instructions
            context: Retrieved context for RAG
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Build full prompt with context
        if context:
            full_prompt = f"""CONTEXT:
{context}

QUESTION: {prompt}

ANSWER:"""
        else:
            full_prompt = prompt
        
        provider = self.get_available_provider()
        
        if provider == LLMProvider.OLLAMA:
            return self._generate_ollama(full_prompt, system_prompt, start_time)
        elif provider == LLMProvider.OPENAI:
            return self._generate_openai(full_prompt, system_prompt, start_time)
        else:
            return LLMResponse(
                content="No LLM provider available. Please configure Ollama or provide an OpenAI API key.",
                provider="none",
                model="none",
                success=False,
                error="No LLM provider available",
                generation_time=time.time() - start_time,
                tokens_used=None,
                metadata={}
            )
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> LLMResponse:
        """Async version of generate."""
        # Run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, system_prompt, context)
        )
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: str,
        start_time: float
    ) -> LLMResponse:
        """Generate using Ollama."""
        import requests
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get('response', '').strip()
                    
                    return LLMResponse(
                        content=content,
                        provider="ollama",
                        model=self.ollama_model,
                        success=True,
                        error=None,
                        generation_time=time.time() - start_time,
                        tokens_used=data.get('eval_count'),
                        metadata={
                            "prompt_eval_count": data.get('prompt_eval_count'),
                            "total_duration": data.get('total_duration')
                        }
                    )
                else:
                    error = f"Ollama error: HTTP {response.status_code}"
                    logger.warning(f"Attempt {attempt + 1}: {error}")
                    
            except requests.Timeout:
                logger.warning(f"Attempt {attempt + 1}: Ollama timeout")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Ollama error - {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(1)  # Wait before retry
        
        # Fallback to OpenAI if available
        if self.check_openai_available():
            logger.info("Falling back to OpenAI")
            return self._generate_openai(prompt, system_prompt, start_time)
        
        return LLMResponse(
            content="Unable to generate response. Please try again later.",
            provider="ollama",
            model=self.ollama_model,
            success=False,
            error="Max retries exceeded",
            generation_time=time.time() - start_time,
            tokens_used=None,
            metadata={}
        )
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        start_time: float
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.openai_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                usage = data.get('usage', {})
                
                return LLMResponse(
                    content=content,
                    provider="openai",
                    model=self.openai_model,
                    success=True,
                    error=None,
                    generation_time=time.time() - start_time,
                    tokens_used=usage.get('total_tokens'),
                    metadata={
                        "prompt_tokens": usage.get('prompt_tokens'),
                        "completion_tokens": usage.get('completion_tokens')
                    }
                )
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
                
                return LLMResponse(
                    content="Unable to generate response from OpenAI.",
                    provider="openai",
                    model=self.openai_model,
                    success=False,
                    error=error_msg,
                    generation_time=time.time() - start_time,
                    tokens_used=None,
                    metadata={}
                )
                
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return LLMResponse(
                content="Error connecting to OpenAI API.",
                provider="openai",
                model=self.openai_model,
                success=False,
                error=str(e),
                generation_time=time.time() - start_time,
                tokens_used=None,
                metadata={}
            )
    
    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response with conversation history.
        
        Args:
            prompt: Current user message
            history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            system_prompt: System instructions
            context: Retrieved context
            
        Returns:
            LLMResponse
        """
        start_time = time.time()
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Build context-aware prompt
        if context:
            augmented_prompt = f"""Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION: {prompt}"""
        else:
            augmented_prompt = prompt
        
        provider = self.get_available_provider()
        
        if provider == LLMProvider.OPENAI:
            return self._generate_openai_with_history(
                augmented_prompt, history, system_prompt, start_time
            )
        elif provider == LLMProvider.OLLAMA:
            # Ollama doesn't natively support history, so we format it into the prompt
            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history[-3:]  # Last 3 messages
            )
            
            full_prompt = f"""Previous conversation:
{history_text}

Current question: {augmented_prompt}"""
            
            return self._generate_ollama(full_prompt, system_prompt, start_time)
        else:
            return LLMResponse(
                content="No LLM provider available.",
                provider="none",
                model="none",
                success=False,
                error="No provider",
                generation_time=time.time() - start_time,
                tokens_used=None,
                metadata={}
            )
    
    def _generate_openai_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: str,
        start_time: float
    ) -> LLMResponse:
        """Generate with OpenAI using conversation history."""
        try:
            import requests
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add history (limit to last 6 messages)
            for msg in history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Add current message
            messages.append({"role": "user", "content": prompt})
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.openai_model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                
                return LLMResponse(
                    content=content,
                    provider="openai",
                    model=self.openai_model,
                    success=True,
                    error=None,
                    generation_time=time.time() - start_time,
                    tokens_used=data.get('usage', {}).get('total_tokens'),
                    metadata={}
                )
            else:
                return LLMResponse(
                    content="Error generating response.",
                    provider="openai",
                    model=self.openai_model,
                    success=False,
                    error=f"HTTP {response.status_code}",
                    generation_time=time.time() - start_time,
                    tokens_used=None,
                    metadata={}
                )
                
        except Exception as e:
            logger.error(f"OpenAI with history error: {e}")
            return LLMResponse(
                content="Error connecting to API.",
                provider="openai",
                model=self.openai_model,
                success=False,
                error=str(e),
                generation_time=time.time() - start_time,
                tokens_used=None,
                metadata={}
            )
    
    def summarize(self, text: str, max_length: int = 100) -> LLMResponse:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Target summary length
            
        Returns:
            LLMResponse with summary
        """
        prompt = f"""Summarize the following text in about {max_length} words:

{text}

Summary:"""
        
        return self.generate(
            prompt,
            system_prompt="You are a helpful assistant that creates concise summaries."
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM handler status."""
        provider = self.get_available_provider(force_check=True)
        
        return {
            "available_provider": provider.value,
            "ollama_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "openai_configured": bool(self.openai_api_key),
            "openai_model": self.openai_model
        }
