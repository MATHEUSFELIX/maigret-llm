"""Google Gemini provider."""
import os
from typing import Optional

from .base import BaseLLMClient, LLMConfig, LLMResponse

_PRICING = {
    "gemini-2.5-pro":    {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash":  {"input": 0.075,"output": 0.30},
    "gemini-1.5-pro":    {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash":  {"input": 0.075,"output": 0.30},
}
_DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient(BaseLLMClient):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install: pip install google-generativeai")

        api_key = config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = config.model or _DEFAULT_MODEL

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        generation_config = {
            "max_output_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n---\n\n{prompt}"

        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
        )
        resp = model.generate_content(full_prompt)
        text = resp.text

        # Gemini token counting (usage_metadata may not always be populated)
        in_tok = getattr(getattr(resp, "usage_metadata", None), "prompt_token_count", 0) or 0
        out_tok = getattr(getattr(resp, "usage_metadata", None), "candidates_token_count", 0) or 0

        pricing = _PRICING.get(self._model_name, {"input": 0.075, "output": 0.30})
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

        return LLMResponse(
            content=text,
            model=self._model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
