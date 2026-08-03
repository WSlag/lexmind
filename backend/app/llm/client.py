"""LLM client facade.

Thin wrapper over :class:`~app.llm.manager.LLMManager` exposing the historical
``LLMClient.complete_json`` API so pipeline nodes, services, and tests keep a
single stable entry point while the provider/fallback mechanics live behind
the abstraction layer.
"""
from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.llm.manager import LLMManager

# System prompt from our SYSTEM_PROMPT.md brain: the model must never invent
# clauses, legislation or facts, and every conclusion must cite contract text.
NON_HALLUCINATION_RULE = (
    "You are reviewing a legal contract. NEVER invent clauses, sections, "
    "legislation, facts or quotations. Every conclusion MUST be grounded in "
    "the contract text provided to you and MAY cite it verbatim."
)


class LLMClient:
    """Facade returning structured JSON from the configured chat model.

    Use :meth:`complete_json` for all pipeline stages. A parsing failure or an
    exhausted provider chain raises and the workflow node will surface it via
    its error path.
    """

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._manager = LLMManager(self.settings)

    @property
    def provider(self) -> str:
        return self._manager.active_provider

    def complete_json(self, system: str, user: str, *, max_tokens: int = 2000) -> dict:
        """Ask the model to produce a JSON object and return it as a dict."""
        return self._manager.complete_json(
            f"{NON_HALLUCINATION_RULE}\n\n{system}",
            user,
            max_tokens=max_tokens,
        )


def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())
