"""Pipeline nodes implementing the PRD workflow stages:

upload -> parse -> extract -> risk review -> missing-clause review ->
negotiation review -> executive summary.

Each node is a pure function on :class:`ReviewState` so it can be tested in
isolation and composed in any graph layout.
"""
from __future__ import annotations

import json
import logging

from app.core.config import get_settings
from app.llm.client import LLMClient, get_llm_client
from app.parsers.document import parse_file
from app.prompts import skills
from app.prompts.skills import fill
from app.schemas.review import (
    ActionPriority,
    ClauseReview,
    CrossClauseConflict,
    MissingClause,
    NegotiationPoint,
    ParsedClause,
    RiskLevel,
    SourceSpan,
)
from app.workflow.state import ReviewState

logger = logging.getLogger(__name__)


def node_parse(state: ReviewState) -> ReviewState:
    """Parse uploaded bytes into a normalized Document."""
    state["document"] = parse_file(state["filename"], state["file_bytes"])
    return state


def node_extract(state: ReviewState, llm: LLMClient) -> ReviewState:
    """Identify and inventory every clause in the document."""
    settings = get_settings()
    doc = state["document"]
    user = fill(
        skills.CLAUSE_EXTRACTOR_USER,
        doc_content=doc.content[: settings.max_context_chars],
        max_clauses=settings.max_clauses,
    )
    raw = llm.complete_json(skills.CLAUSE_EXTRACTOR_SYSTEM, user)
    clauses = [
        ParsedClause(
            uid=f"clause-{i:04d}",
            title=str(c.get("title", f"Clause {i + 1}")),
            number=str(c.get("number", "")),
            kind=str(c.get("kind", "other")),
            text=str(c.get("text", "")),
            start=int(c.get("start", 0)),
            end=int(c.get("end", 0)),
        )
        for i, c in enumerate(raw.get("clauses", []))
    ]
    state["clauses"] = _dedup_headings(clauses)
    return state


def _dedup_headings(clauses: list[ParsedClause]) -> list[ParsedClause]:
    """Drop bare section headings that carry no substantive body text.

    The extractor returns a top-level heading ("3. PRICING AND PAYMENT") as its
    own clause in addition to the sub-clauses beneath it (``3.1``, ``3.2``).
    We keep a clause only if it is either a numbered sub-clause (``X.Y``) or
    its text contains more than the bare heading itself.
    """
    import re

    kept: list[ParsedClause] = []
    for clause in clauses:
        text = clause.text.strip()
        title = clause.title.strip()
        number = clause.number.strip()
        is_sub = "." in number
        # Strip a leading number prefix like "3." or "3.1" from the text.
        bare = re.sub(r"^\s*\d+(?:\.\d+)*\s*\.?\s*", "", text).strip()
        if not is_sub and bare and title and bare.lower() == title.lower():
            continue  # pure section heading with no body content
        kept.append(clause)
    return kept


def node_risk_review(state: ReviewState, llm: LLMClient) -> ReviewState:
    """Score risk for each clause (batched to cut latency/cost)."""
    settings = get_settings()
    doc = state["document"]
    clauses = state.get("clauses", [])
    reviews: list[ClauseReview] = []

    batch_size = settings.clauses_per_batch
    for i in range(0, len(clauses), batch_size):
        batch = clauses[i : i + batch_size]
        clause_payload = [
            {
                "uid": c.uid,
                "title": c.title,
                "number": c.number,
                "text": c.text[: settings.evidence_snippet_chars],
            }
            for c in batch
        ]
        user = fill(
            skills.RISK_REVIEWER_BATCH_USER,
            context=doc.content[: settings.max_context_chars],
            clauses_json=json.dumps(clause_payload, ensure_ascii=False),
        )
        raw = llm.complete_json(
            skills.RISK_REVIEWER_BATCH_SYSTEM,
            user,
            max_tokens=settings.max_review_tokens_per_clause * min(batch_size, 4),
        )
        reviews.extend(_parse_batch_reviews(raw, batch))
    state["clause_reviews"] = reviews
    return state


def _parse_batch_reviews(raw: dict, batch: list[ParsedClause]) -> list[ClauseReview]:
    """Flatten a uid-keyed batch response into ClauseReview objects.

    Falls back defensively (or to the mock's single-"risk_level" shape) when
    the model does not return per-uid entries.
    """
    reviews: list[ClauseReview] = []
    for clause in batch:
        entry = raw.get(clause.uid) or {}
        if not entry and "risk_level" in raw:
            entry = raw  # mock / minimal responder shape
        if not entry:
            entry = {}
        spans = [
            SourceSpan(start=int(s.get("start", 0)), end=int(s.get("end", 0)), snippet=str(s.get("snippet", "")))
            for s in entry.get("evidence", [])
        ]
        reviews.append(
            ClauseReview(
                clause_uid=clause.uid,
                clause_title=clause.title,
                risk_level=_risk(entry.get("risk_level", "low")),
                risk_categories=entry.get("risk_categories", []),
                business_impact=str(entry.get("business_impact", "")),
                legal_impact=str(entry.get("legal_impact", "")),
                commercial_analysis=str(entry.get("commercial_analysis", "")),
                recommendation=str(entry.get("recommendation", "")),
                confidence=float(entry.get("confidence", 0.0)),
                confidence_reason=str(entry.get("confidence_reason", "")),
                authorities=entry.get("authorities", []),
                action_priority=_action_priority(entry.get("risk_level", "low")),
                source_spans=spans,
            )
        )
    return reviews


def _action_priority(risk_level: str) -> ActionPriority:
    """Deterministic mapping from risk level to a lawyer-facing action priority."""
    mapping = {
        "critical": ActionPriority.FIX,
        "high": ActionPriority.FIX,
        "medium": ActionPriority.NEGOTIATE,
        "low": ActionPriority.ACCEPTABLE,
        "none": ActionPriority.INFORMATIONAL,
    }
    return mapping.get(_risk(risk_level).value, ActionPriority.INFORMATIONAL)


def node_missing_clauses(state: ReviewState, llm: LLMClient) -> ReviewState:
    """Detect critical clauses that are absent from the agreement."""
    settings = get_settings()
    doc = state["document"]
    user = fill(skills.MISSING_CLAUSE_USER, document=doc.content[: settings.max_context_chars])
    raw = llm.complete_json(skills.MISSING_CLAUSE_SYSTEM, user)
    missing = []
    for m in raw.get("missing_clauses", []):
        if isinstance(m, str):
            missing.append(
                MissingClause(
                    clause=m, category="", severity=RiskLevel.MEDIUM,
                    rationale="", exists=False, benchmark="",
                )
            )
            continue
        if isinstance(m, dict) and not m.get("exists"):
            missing.append(
                MissingClause(
                    clause=str(m.get("clause", "")),
                    category=str(m.get("category", "")),
                    severity=_risk(m.get("severity", "medium")),
                    rationale=str(m.get("rationale", "")),
                    benchmark=str(m.get("benchmark", "")),
                    authorities=m.get("authorities", []),
                    exists=bool(m.get("exists", False)),
                )
            )
    state["missing_clauses"] = missing
    return state


def node_negotiation(state: ReviewState, llm: LLMClient) -> ReviewState:
    """Build negotiation advice for the highest-risk clause(s)."""
    settings = get_settings()
    reviews = state.get("clause_reviews", [])
    priorities = [
        r
        for r in sorted(reviews, key=lambda r: r.risk_level.value, reverse=True)
        if r.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    ][:3]
    points: list[NegotiationPoint] = []
    for r in priorities:
        clause = next((c for c in state.get("clauses", []) if c.uid == r.clause_uid), None)
        if clause is None:
            continue
        user = fill(
            skills.NEGOTIATION_USER,
            clause_title=r.clause_title,
            risk_level=r.risk_level.value,
            clause_text=clause.text[: settings.max_review_tokens_per_clause],
        )
        raw = llm.complete_json(skills.NEGOTIATION_SYSTEM, user)
        points.append(
            NegotiationPoint(
                clause_uid=r.clause_uid,
                topic=r.clause_title,
                suggested_wording=str(raw.get("suggested_wording", "")),
                fallback_wording=str(raw.get("fallback_wording", "")),
                strategy=str(raw.get("strategy", "")),
                commercial_reasoning=str(raw.get("commercial_reasoning", "")),
                owner_position=str(raw.get("owner_position", "")),
                contractor_position=str(raw.get("contractor_position", "")),
                priority=r.risk_level,
            )
        )
    state["negotiation_points"] = points
    return state


def node_cross_clause_check(state: ReviewState, llm: LLMClient) -> ReviewState:
    """Detect internal inconsistencies between clauses (feature #3)."""
    settings = get_settings()
    clauses = state.get("clauses", [])
    if len(clauses) < 2:
        state["cross_clause_conflicts"] = []
        return state
    payload = [
        {
            "uid": c.uid,
            "title": c.title,
            "text": c.text[: settings.evidence_snippet_chars],
        }
        for c in clauses
    ]
    user = fill(
        skills.CROSS_CLAUSE_USER,
        clauses_json=json.dumps(payload, ensure_ascii=False),
    )
    raw = llm.complete_json(
        skills.CROSS_CLAUSE_SYSTEM,
        user,
        max_tokens=settings.max_review_tokens_per_clause * 2,
    )
    conflicts = []
    for c in raw.get("conflicts", []):
        if not isinstance(c, dict):
            continue
        conflicts.append(
            CrossClauseConflict(
                clause_uid_a=str(c.get("clause_uid_a", "")),
                clause_title_a=str(c.get("clause_title_a", "")),
                clause_uid_b=str(c.get("clause_uid_b", "")),
                clause_title_b=str(c.get("clause_title_b", "")),
                conflict_type=str(c.get("conflict_type", "")),
                description=str(c.get("description", "")),
                recommendation=str(c.get("recommendation", "")),
                severity=_risk(c.get("severity", "medium")),
            )
        )
    state["cross_clause_conflicts"] = conflicts
    return state


def node_executive_summary(state: ReviewState, llm: LLMClient) -> ReviewState:
    """Produce the final executive report from aggregated review data."""
    from app.schemas.review import ExecutiveSummary

    data = {
        "overall_risk": _aggregate_risk(state.get("clause_reviews", [])),
        "clause_reviews": [r.model_dump(mode="json") for r in state.get("clause_reviews", [])],
        "missing_clauses": [m.model_dump(mode="json") for m in state.get("missing_clauses", [])],
        "cross_clause_conflicts": [
            c.model_dump(mode="json") for c in state.get("cross_clause_conflicts", [])
        ],
    }
    user = fill(skills.EXECUTIVE_SUMMARY_USER, summary_data=data)
    raw = llm.complete_json(skills.EXECUTIVE_SUMMARY_SYSTEM, user)
    state["executive_summary"] = ExecutiveSummary(
        overall_risk=_risk(raw.get("overall_risk", "low")),
        recommendation=str(raw.get("recommendation", "")),
        memo=str(raw.get("memo", "")),
        key_issues=[str(k) for k in raw.get("key_issues", [])],
        estimated_time_saved_minutes=_estimate_time_saved(state.get("document")),
        top_risks=[r for r in state.get("clause_reviews", []) if r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)][:5],
        missing_clauses=state.get("missing_clauses", []),
        negotiation_priorities=state.get("negotiation_points", []),
        cross_clause_conflicts=state.get("cross_clause_conflicts", []),
    )
    return state


def _estimate_time_saved(document) -> int:
    """Deterministic partner-facing estimate: manual review ~ 6 min per 100 words."""
    from app.schemas.review import Document

    if not isinstance(document, Document):
        return 0
    words = document.word_count
    return int(round((words / 100) * 6))


def node_fail(state: ReviewState, error: Exception) -> ReviewState:
    state.setdefault("errors", []).append(str(error))
    return state


def _aggregate_risk(reviews: list[ClauseReview]) -> RiskLevel:
    if not reviews:
        return RiskLevel.LOW
    worst = max(reviews, key=lambda r: r.risk_level.value)
    return worst.risk_level


def _risk(value: str) -> RiskLevel:
    value = (value or "low").strip().lower()
    for level in RiskLevel:
        if value == level.value:
            return level
    return RiskLevel.LOW