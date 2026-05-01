"""Anthropic Claude provider."""
import os
from typing import Optional

from .base import BaseLLMClient, LLMConfig, LLMResponse

# Pricing per million tokens (as of mid-2025)
_PRICING = {
    "claude-opus-4-6":   {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.8, "output": 4.0},
}
_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude via the official SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("Install: pip install anthropic")

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = config.model or _DEFAULT_MODEL

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        msg = self._client.messages.create(**kwargs)
        text = msg.content[0].text
        in_tok = msg.usage.input_tokens
        out_tok = msg.usage.output_tokens

        pricing = _PRICING.get(self._model, {"input": 3.0, "output": 15.0})
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

        return LLMResponse(
            content=text,
            model=self._model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
