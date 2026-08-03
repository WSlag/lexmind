"""State schema shared across the LangGraph pipeline nodes.

Declared as a :class:`typing.TypedDict` because LangGraph reifies state into a
plain dict; subclasses of ``dict`` do not survive ``invoke``. Nodes read and
update the fields below using ``state[key] = ...`` / ``state["key"]``.
"""
from __future__ import annotations

from typing import NotRequired, TypedDict

from app.schemas.review import (
    ClauseReview,
    ContractReview,
    CrossClauseConflict,
    Document,
    ExecutiveSummary,
    MissingClause,
    NegotiationPoint,
    ParsedClause,
)


class ReviewState(TypedDict, total=False):
    review_id: str
    filename: str
    file_bytes: bytes
    document: Document | None
    clauses: list[ParsedClause]
    clause_reviews: list[ClauseReview]
    missing_clauses: list[MissingClause]
    negotiation_points: list[NegotiationPoint]
    cross_clause_conflicts: list[CrossClauseConflict]
    executive_summary: ExecutiveSummary | None
    errors: list[str]


def initial_review_state(
    review_id: str,
    filename: str = "",
    file_bytes: bytes | None = None,
) -> ReviewState:
    return ReviewState(
        review_id=review_id,
        filename=filename,
        file_bytes=file_bytes,
        document=None,
        clauses=[],
        clause_reviews=[],
        missing_clauses=[],
        negotiation_points=[],
        cross_clause_conflicts=[],
        executive_summary=None,
        errors=[],
    )


def state_to_contract_review(state: ReviewState) -> ContractReview:
    from app.schemas.review import ReviewStatus

    return ContractReview(
        review_id=state.get("review_id", ""),
        status=ReviewStatus.FAILED if state.get("errors") else ReviewStatus.COMPLETE,
        document=state.get("document"),
        clauses=state.get("clauses", []),
        clause_reviews=state.get("clause_reviews", []),
        missing_clauses=state.get("missing_clauses", []),
        negotiation_points=state.get("negotiation_points", []),
        cross_clause_conflicts=state.get("cross_clause_conflicts", []),
        executive_summary=state.get("executive_summary"),
    )