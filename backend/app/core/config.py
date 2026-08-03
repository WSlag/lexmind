"""Application settings loaded from environment variables and `.env`."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "LexMind AI"
    debug: bool = Field(default=False)

    # LLM provider: "mock" | "anthropic" | "openai" | "ollama" | "groq" | "gemini" | "agnes" | "deepseek"
    llm_provider: str = "mock"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    agnes_api_key: str = ""
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1"
    agnes_model: str = "agnes-2.0-flash"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Pipeline tuning
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_clauses: int = Field(default=200)
    max_context_chars: int = Field(default=120_000)
    evidence_snippet_chars: int = Field(default=300)
    max_review_tokens_per_clause: int = 4000

    # LLM robustness: retry transient failures (429, 503, timeouts) with backoff.
    llm_max_retries: int = Field(default=3)
    llm_backoff_base_seconds: float = Field(default=2.0)

    # Comma-separated backup providers tried after the primary, e.g.
    # LLM_FALLBACK_PROVIDERS="groq,agnes" to degrade gracefully when DeepSeek is down.
    llm_fallback_providers: str = ""

    # Batch reviews of multiple clauses per LLM call to reduce latency/cost.
    clauses_per_batch: int = Field(default=5)

    @property
    def is_mock(self) -> bool:
        return self.llm_provider == "mock"


@lru_cache
def get_settings() -> Settings:
    return Settings()