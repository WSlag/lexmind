"""Deterministic offline provider used for tests, demos, and CI.

It inspects the JSON shape hints embedded in the user prompt and returns a
small valid payload that keeps the pipeline runnable end-to-end without any
network access or API keys. The mock is intentionally simple: it exercises the
schema and the parsing path, not the model's reasoning.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm.ir import parse_ir
from app.llm.providers.base import LLMProvider


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self) -> None:
        self._call_count = 0

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
    ) -> dict:
        self._call_count += 1
        prompt_text = user or ""

        payload: dict[str, Any] = {
            "summary": "Deterministic mock output (configure an LLM provider "
            "in .env to enable real analysis)."
        }
        if '"clauses"' in prompt_text:
            payload["clauses"] = [
                {
                    "title": "Pricing and Payment",
                    "number": "3.1",
                    "kind": "pricing",
                    "text": "Buyer shall pay Seller the Contract Price per MMBtu.",
                    "confidence": 0.9,
                }
            ]
        if '"risk_level"' in prompt_text:
            payload["risk_level"] = "medium"
            payload["risk_categories"] = ["pricing"]
            payload["recommendation"] = "Verify price index definition."
            payload["confidence_reason"] = "Standard wording detected."
            payload["authorities"] = ["AS 4000", "FIDIC Red Book"]
        if '"missing_clauses"' in prompt_text:
            payload["missing_clauses"] = [
                {
                    "clause": "Insurance",
                    "category": "Insurance",
                    "rationale": "No insurance clause found.",
                    "severity": "medium",
                    "exists": False,
                    "authorities": ["AS 4000"],
                    "benchmark": "Commonly included under AS 4000 practice.",
                }
            ]
        if '"conflicts"' in prompt_text:
            payload["conflicts"] = []
        if '"overall_risk"' in prompt_text:
            payload["overall_risk"] = "medium"
            payload["recommendation"] = "Resolve payment timing before signing."
            payload["key_issues"] = ["Payment timing"]
            payload["memo"] = (
                "MEMORANDUM. The agreement presents a medium overall risk. "
                "Payment timing and missing protections should be resolved "
                "before execution."
            )
        if '"citations"' in prompt_text:
            uid_match = re.search(r"\[(clause-[0-9a-f]+)\]", prompt_text)
            uid = uid_match.group(1) if uid_match else "clause-0001"
            payload["answer"] = (
                "The agreement addresses this in the clauses identified below. "
                "Confirm the exact position against the clause text before relying on it."
            )
            payload["grounded"] = True
            payload["citations"] = [
                {
                    "clause_uid": uid,
                    "clause_title": "Pricing and Payment",
                    "clause_number": "3.1",
                    "snippet": "Buyer shall pay Seller the Contract Price per MMBtu.",
                }
            ]

        return parse_ir(json.dumps(payload, ensure_ascii=False))
