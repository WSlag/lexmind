"""Tests for the IR parser and the LLM abstraction layer (fallback chain)."""
from __future__ import annotations

import pytest

from app.llm.ir import parse_ir
from app.llm.manager import LLMManager


class _FakeSettings:
    llm_max_retries = 0
    llm_backoff_base_seconds = 0.0
    llm_fallback_providers = ""
    llm_provider = "mock"


class TestParseIR:
    def test_plain_json(self) -> None:
        assert parse_ir('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}

    def test_fenced_json(self) -> None:
        raw = 'Here is the result:\n```json\n{"answer": "yes"}\n```\nHope that helps.'
        assert parse_ir(raw) == {"answer": "yes"}

    def test_tagged_ir_block(self) -> None:
        raw = 'Below: <ir>{"citations": []}</ir> -- end'
        assert parse_ir(raw) == {"citations": []}

    def test_prose_surrounding_balanced_region(self) -> None:
        raw = 'Sure! The structured output is {"k": "v"} which I computed above.'
        assert parse_ir(raw) == {"k": "v"}

    def test_trailing_comma_repaired(self) -> None:
        assert parse_ir('{"a": 1,}') == {"a": 1}

    def test_python_literals_repaired(self) -> None:
        assert parse_ir('{"exists": True, "none": None}') == {
            "exists": True,
            "none": None,
        }

    def test_unquoted_keys_repaired(self) -> None:
        assert parse_ir('{title: "x", count: 3}') == {"title": "x", "count": 3}

    def test_raises_on_garbage(self) -> None:
        with pytest.raises(ValueError):
            parse_ir("this is not structured output at all")


class _FailingProvider:
    """Provider that always raises — used to exercise the fallback chain."""

    name = "failing"

    def complete_json(self, system: str, user: str, *, max_tokens: int = 2000) -> dict:
        raise RuntimeError("simulated outage")


class TestFallbackChain:
    def test_primary_provider_used(self) -> None:
        settings = _FakeSettings()
        manager = LLMManager(settings)
        assert manager.active_provider == "mock"
        out = manager.complete_json("sys", '{"citations": []}')
        assert "citations" in out

    def test_fallthrough_on_failure(self) -> None:
        class _Settings(_FakeSettings):
            llm_provider = "failing"
            llm_fallback_providers = "mock"

        def factory(settings, name):
            if name == "failing":
                return _FailingProvider()
            from app.llm.factory import build_provider

            return build_provider(settings, name)

        manager = LLMManager(_Settings(), provider_factory=factory)
        # The mock returns a dict even for prompts without JSON hints, so the
        # chain must have fallen through to it.
        out = manager.complete_json("sys", "anything")
        assert isinstance(out, dict)

    def test_raises_when_all_fail(self) -> None:
        class _Settings(_FakeSettings):
            llm_provider = "failing"
            llm_fallback_providers = "also-failing"

        def factory(settings, name):
            if name == "failing":
                return _FailingProvider()
            return _FailingProvider()

        manager = LLMManager(_Settings(), provider_factory=factory)
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            manager.complete_json("sys", "anything")
