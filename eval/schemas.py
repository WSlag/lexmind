"""Schemas for the GasMind evaluation harness (gold standard + metrics).

These are independent of the runtime review schemas: eval declares what the
*correct* answer is so we can objectively measure the pipeline's precision and
recall against it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExpectedClause(BaseModel):
    """A clause the model MUST identify in a given contract."""

    title: str
    number: str = ""
    kind: str = "other"  # must match a pipeline 'kind'


class GoldStandard(BaseModel):
    """The labelled 'correct answer' for one contract."""

    contract: str = Field(..., description="Filename in eval/test_contracts")
    expected_clauses: list[ExpectedClause] = Field(default_factory=list)
    # Subset of expected_clauses by title that is genuinely high risk.
    expected_high_risk: list[str] = Field(default_factory=list)
    # Clause KINDs that must be present for a complete gas supply agreement.
    required_kinds: list[str] = Field(default_factory=list)
    # Standard protections that are genuinely absent (so we should flag them).
    expected_missing: list[str] = Field(default_factory=list)
    # Protections that ARE present (so we must NOT flag them as missing).
    expected_present: list[str] = Field(default_factory=list)


class ClausesReport(BaseModel):
    contract: str
    precision: float
    recall: float
    f1: float
    matched: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    spurious: list[str] = Field(default_factory=list)


class MissingReport(BaseModel):
    contract: str
    true_positive: int
    false_positive: int
    false_negative: int
    flagged: list[str] = Field(default_factory=list)
    missed_missing: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    summary: dict = Field(default_factory=dict)  # aggregate metrics
    clause_reports: list[ClausesReport] = Field(default_factory=list)
    missing_reports: list[MissingReport] = Field(default_factory=list)