"""Ollama local provider (OpenAI-compatible API)."""
import os
from typing import Optional

from .base import BaseLLMClient, LLMConfig, LLMResponse

_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaClient(BaseLLMClient):
    """
    Ollama via its OpenAI-compatible REST API.
    Requires Ollama running locally: https://ollama.com/
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install: pip install openai  (used for Ollama OpenAI-compat endpoint)")

        base_url = config.base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        # Ollama doesn't require a real API key but the client needs one
        self._client = OpenAI(api_key="ollama", base_url=base_url)
        self._model = config.model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
        )

        text = resp.choices[0].message.content or ""
        in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
        out_tok = getattr(resp.usage, "completion_tokens", 0) or 0

        return LLMResponse(
            content=text,
            model=self._model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,  # Local inference — no cost
        )
