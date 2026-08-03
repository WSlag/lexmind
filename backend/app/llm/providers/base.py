"""Abstract provider interface for the LLM abstraction layer.

Every concrete provider (OpenAI-compatible, Anthropic, Ollama, mock, ...)
implements :class:`LLMProvider`. The manager in ``manager.py`` holds an ordered
list of providers and falls through on transient failure, so a primary provider
can fail over to a backup without the pipeline noticing.
"""
from __future__ import annotations

import abc
from typing import Any


class LLMProvider(abc.ABC):
    """A single LLM backend able to return structured JSON.

    Providers are responsible for constructing the message list, invoking the
    model, normalising the raw response, and parsing it via ``parse_ir``.
    """

    #: Stable identifier used in logs and error messages (e.g. "deepseek").
    name: str

    @abc.abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
    ) -> dict:
        """Run one completion and return a parsed JSON object."""
        raise NotImplementedError
