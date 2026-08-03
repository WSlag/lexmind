"""Build concrete providers from application settings."""
from __future__ import annotations

from typing import Any

from app.llm.providers.base import LLMProvider
from app.llm.providers.mock import MockProvider
from app.llm.providers.openai_compat import OpenAICompatProvider


def build_provider(settings: Any, provider: str) -> LLMProvider:
    """Construct the provider named by ``provider`` using ``settings``.

    ``openai_compat`` accepts an optional ``target`` hint used by the manager
    to point it at a specific OpenAI-compatible backend.
    """
    target = getattr(settings, "openai_compat_target", None)
    if provider in ("openai", "groq", "agnes", "deepseek"):
        target = target or provider
    if provider == "openai":
        return OpenAICompatProvider(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url="https://api.openai.com/v1",
            name="openai",
        )
    if provider == "groq":
        return OpenAICompatProvider(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            name="groq",
        )
    if provider == "agnes":
        return OpenAICompatProvider(
            model=settings.agnes_model,
            api_key=settings.agnes_api_key,
            base_url=settings.agnes_base_url,
            name="agnes",
        )
    if provider == "deepseek":
        return OpenAICompatProvider(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            name="deepseek",
        )
    if provider == "anthropic":
        from app.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
        )
    if provider == "gemini":
        from app.llm.providers.gemini import GeminiProvider

        return GeminiProvider(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
        )
    if provider == "ollama":
        from app.llm.providers.ollama import OllamaProvider

        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported LLM provider: {provider}")
