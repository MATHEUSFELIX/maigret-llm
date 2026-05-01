"""LLM provider factory."""
import os
from typing import Literal, Optional

from .base import BaseLLMClient, LLMConfig, LLMResponse

ProviderName = Literal["anthropic", "openai", "gemini", "ollama"]


def create_llm(
    provider: Optional[ProviderName] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> BaseLLMClient:
    """
    Factory function — returns the right LLM client.

    Provider is resolved in this order:
      1. ``provider`` argument
      2. ``MAIGRET_LLM_PROVIDER`` environment variable
      3. Auto-detect from available API keys
    """
    if provider is None:
        provider = os.environ.get("MAIGRET_LLM_PROVIDER")  # type: ignore[assignment]

    if provider is None:
        # Auto-detect from environment
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = "ollama"

    config = LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if provider == "anthropic":
        from .anthropic import AnthropicClient
        return AnthropicClient(config)
    elif provider == "openai":
        from .openai import OpenAIClient
        return OpenAIClient(config)
    elif provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(config)
    elif provider == "ollama":
        from .ollama import OllamaClient
        return OllamaClient(config)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose: anthropic, openai, gemini, ollama")


__all__ = ["create_llm", "BaseLLMClient", "LLMConfig", "LLMResponse"]
