"""LLM manager: ordered provider chain with failover.

The manager holds a primary provider plus optional fallbacks configured via
``LLM_FALLBACK_PROVIDERS`` (a comma-separated list). On a transient failure
from the primary it retries the same call against the next provider, so the
pipeline keeps working when a rate-limited or flaky backend degrades.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.llm.factory import build_provider
from app.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """Runs structured completions across an ordered provider chain."""

    def __init__(
        self,
        settings: Any,
        provider_factory: Any | None = None,
    ) -> None:
        self.settings = settings
        self.providers: list[LLMProvider] = []
        self._factory = provider_factory or build_provider

        primary = settings.llm_provider or "mock"
        raw_fallbacks = getattr(settings, "llm_fallback_providers", None) or ""
        if isinstance(raw_fallbacks, list):
            raw_fallbacks = ",".join(str(p) for p in raw_fallbacks)
        fallbacks = [
            p.strip()
            for p in raw_fallbacks.split(",")
            if p.strip() and p.strip() != primary
        ]
        order = [primary, *fallbacks]
        for name in order:
            self.providers.append(self._factory(settings, name))
        logger.info(
            "LLM chain: %s",
            " -> ".join(p.name for p in self.providers),
        )

    @property
    def active_provider(self) -> str:
        return self.providers[0].name if self.providers else "none"

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
    ) -> dict:
        """Run one structured completion, failing over between providers.

        Each provider is retried with the configured backoff; after exhausting
        a provider's retries the chain moves to the next provider. Raises if
        every provider fails.
        """
        if not self.providers:
            raise RuntimeError("No LLM providers configured")

        last_exc: Exception | None = None
        for provider in self.providers:
            try:
                return self._run_with_retries(
                    provider,
                    system,
                    user,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - chain any provider failure
                last_exc = exc
                logger.warning("Provider %s failed: %s", provider.name, exc)
                if len(self.providers) == 1 or _is_non_retryable(exc):
                    # A single provider, or an auth/config error that a backup
                    # would hit too (e.g. bad API key) — stop early.
                    break
                logger.info("Failing over to next provider in chain")
        raise RuntimeError(
            f"All LLM providers failed ({self.active_provider} was primary): {last_exc}"
        ) from last_exc

    def _run_with_retries(
        self,
        provider: LLMProvider,
        system: str,
        user: str,
        *,
        max_tokens: int,
    ) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                return provider.complete_json(system, user, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last_exc = exc
                if attempt >= self.settings.llm_max_retries:
                    break
                if _is_non_retryable(exc):
                    break
                delay = self.settings.llm_backoff_base_seconds * (2**attempt)
                logger.warning(
                    "Provider %s call failed (attempt %s/%s): %s. Retrying in %.1fs.",
                    provider.name,
                    attempt + 1,
                    self.settings.llm_max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Provider {provider.name} failed after retries: {last_exc}"
        ) from last_exc


def _is_non_retryable(exc: Exception) -> bool:
    """True for errors that will not succeed on retry (auth, invalid key, 400)."""
    text = str(exc).lower()
    if any(
        marker in text
        for marker in (
            "401",
            "403",
            "invalid_api_key",
            "authentication",
            "permission",
            "unauthorized",
            "not_found",
            "404",
        )
    ):
        return True
    return False
