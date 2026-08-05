"""OpenAI-compatible provider (OpenAI, Groq, Agnes, DeepSeek).

All four expose the same ``/v1/chat/completions`` protocol, so a single
LangChain adapter parameterised by base URL and model covers them all.
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm.ir import parse_ir
from app.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    def __init__(self, *, model: str, api_key: str, base_url: str, name: str) -> None:
        self.name = name
        self._model_name = model
        self._api_key = api_key
        self._base_url = base_url

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
    ) -> dict:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            model=self._model_name,
            api_key=self._api_key,
            base_url=self._base_url or None,
            temperature=0,
            max_tokens=max_tokens,
        )
        raw = chat.invoke(self._messages(system, user))
        content = self._extract_text(raw)
        result = parse_ir(content)
        if not isinstance(result, dict):
            raise ValueError(f"{self.name} returned non-object structure")
        return result

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _extract_text(raw: Any) -> str:
        content = getattr(raw, "content", raw)
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "".join(parts)
        return content or ""
