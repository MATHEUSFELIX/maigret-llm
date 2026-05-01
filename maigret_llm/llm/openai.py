"""OpenAI GPT provider."""
import os
from typing import Optional

from .base import BaseLLMClient, LLMConfig, LLMResponse

_PRICING = {
    "gpt-4o":       {"input": 2.5,  "output": 10.0},
    "gpt-4o-mini":  {"input": 0.15, "output": 0.6},
    "o3-mini":      {"input": 1.1,  "output": 4.4},
    "o4-mini":      {"input": 1.1,  "output": 4.4},
}
_DEFAULT_MODEL = "gpt-4o"


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT via the official SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install: pip install openai")

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = config.base_url  # None = default OpenAI endpoint
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = config.model or _DEFAULT_MODEL

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
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens

        pricing = _PRICING.get(self._model, {"input": 2.5, "output": 10.0})
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

        return LLMResponse(
            content=text,
            model=self._model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
