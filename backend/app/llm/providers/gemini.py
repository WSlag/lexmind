"""Google Gemini provider."""
from __future__ import annotations

import logging
from typing import Any

from app.llm.ir import parse_ir
from app.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, *, model: str, api_key: str) -> None:
        self._model_name = model
        self._api_key = api_key

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
    ) -> dict:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat = ChatGoogleGenerativeAI(
            model=self._model_name,
            api_key=self._api_key,
            temperature=0,
        )
        raw = chat.invoke([("system", system), ("human", user)])
        content = getattr(raw, "content", raw)
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "".join(parts)
        result = parse_ir(content or "")
        if not isinstance(result, dict):
            raise ValueError("gemini returned non-object structure")
        return result
