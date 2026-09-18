"""Base LLM client interface and factory."""

from abc import ABC, abstractmethod
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Provider-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        pass


class GroqClient(LLMClient):
    """Groq LLM provider client."""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """Initialize Groq client.

        Args:
            api_key: Groq API key
            model: Model identifier
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq library is required. Install with: pip install groq"
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.provider = "groq"
        logger.info(f"Initialized Groq client with model: {model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate text using Groq.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str) and attempt < max_retries - 1:
                    wait_sec = 3.5 * (attempt + 1)
                    logger.warning(
                        f"Rate limit encountered on Groq; pausing {wait_sec:.1f}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_sec)
                    continue
                logger.error(f"Error generating text with Groq: {str(e)}")
                raise


    def get_model_info(self) -> dict:
        """Get Groq model information."""
        return {
            "provider": self.provider,
            "model": self.model,
        }


class OpenAIClient(LLMClient):
    """OpenAI LLM provider client."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model identifier
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai library is required. Install with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.provider = "openai"
        logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate text using OpenAI.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """Get OpenAI model information."""
        return {
            "provider": self.provider,
            "model": self.model,
        }


class LaminiClient(LLMClient):
    """Lamini LLM provider client."""

    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """Initialize Lamini client.

        Args:
            api_key: Lamini API key
            model: Model identifier
        """
        try:
            import lamini
        except ImportError:
            raise ImportError(
                "lamini library is required. Install with: pip install lamini"
            )

        lamini.api_key = api_key
        self.model = model
        self.provider = "lamini"
        logger.info(f"Initialized Lamini client with model: {model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate text using Lamini.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            import lamini

            llm = lamini.Lamini(self.model)
            response = llm.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating text with Lamini: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """Get Lamini model information."""
        return {
            "provider": self.provider,
            "model": self.model,
        }


class LLMFactory:
    """Factory for creating LLM clients based on provider."""

    PROVIDERS = {
        "groq": GroqClient,
        "openai": OpenAIClient,
        "lamini": LaminiClient,
    }

    @classmethod
    def create(
        cls, provider: str, api_key: str, model: str
    ) -> LLMClient:
        """Create LLM client for specified provider.

        Args:
            provider: Provider name ("groq", "openai", "lamini")
            api_key: API key for provider
            model: Model identifier

        Returns:
            LLMClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Available: {list(cls.PROVIDERS.keys())}"
            )

        client_class = cls.PROVIDERS[provider]
        logger.info(f"Creating {provider} LLM client")
        return client_class(api_key, model)
