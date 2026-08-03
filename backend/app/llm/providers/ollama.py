"""Ollama (local) provider."""
from __future__ import annotations

import logging
from typing import Any

from app.llm.ir import parse_ir
from app.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, *, model: str, base_url: str) -> None:
        self._model_name = model
        self._base_url = base_url

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
    ) -> dict:
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            model=self._model_name,
            base_url=self._base_url,
            temperature=0,
        )
        raw = chat.invoke([("system", system), ("human", user)])
        result = parse_ir(getattr(raw, "content", "") or "")
        if not isinstance(result, dict):
            raise ValueError("ollama returned non-object structure")
        return result
