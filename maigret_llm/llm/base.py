"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMConfig:
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract LLM client — implement one method to add a new provider."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a prompt and return the response."""
        ...

    def complete_text(self, prompt: str, system: Optional[str] = None) -> str:
        """Convenience wrapper — returns only the text content."""
        return self.complete(prompt, system=system).content
