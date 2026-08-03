"""Pydantic models for every stage of the review pipeline.

These are the JSON output contracts produced by each skill and persisted to
the client. Every model follows the PRD principle of *traceable evidence*:
conclusions carry ``source_spans`` pointing back into the parsed contract text.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    REVIEWING = "reviewing"
    COMPLETE = "complete"
    FAILED = "failed"


class ActionPriority(str, Enum):
    FIX = "fix-before-signing"
    NEGOTIATE = "negotiate-if-possible"
    ACCEPTABLE = "acceptable"
    INFORMATIONAL = "informational"


class SourceSpan(BaseModel):
    """Pointer into the parsed document that a conclusion cites as evidence."""

    start: int = Field(..., description="Character offset (inclusive) in normalized text")
    end: int = Field(..., description="Character offset (exclusive) in normalized text")
    snippet: str = Field(..., description="Short quoted excerpt serving as evidence")
    clause_uid: str = Field(default="", description="Optional owning clause")


class Document(BaseModel):
    """Normalized representation of an uploaded file after parsing."""

    filename: str
    file_type: str  # pdf | docx | txt
    content: str = Field(..., description="Normalized full text")
    contract_type: str = Field(
        default="",
        description="Detected agreement type, e.g. 'Commercial Construction Agreement', 'Gas Supply Agreement'",
    )
    headings: list[str] = Field(default_factory=list)
    page_breaks: list[int] = Field(default_factory=list)
    char_count: int = 0
    word_count: int = 0


class ParsedClause(BaseModel):
    """A single contract clause located during extraction."""

    uid: str = Field(..., description="Stable id, e.g. clause-0001")
    title: str
    number: str = ""
    kind: str = "other"  # definitions|pricing|payment|...|other
    text: str
    start: int = 0
    end: int = 0


class ClauseReview(BaseModel):
    """Risk review output for one clause."""

    clause_uid: str
    clause_title: str
    risk_level: RiskLevel
    risk_categories: list[str] = Field(default_factory=list)
    business_impact: str = ""
    legal_impact: str = ""
    commercial_analysis: str = ""
    recommendation: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    confidence_reason: str = Field(
        default="",
        description="Plain-language explanation of why the confidence score is what it is",
    )
    action_priority: ActionPriority = ActionPriority.INFORMATIONAL
    authorities: list[str] = Field(
        default_factory=list,
        description="Market-standard / standard-form references that ground the review",
    )
    source_spans: list[SourceSpan] = Field(default_factory=list)


class MissingClause(BaseModel):
    """A standard clause expected in a gas supply agreement that was not found."""

    clause: str
    category: str
    rationale: str
    severity: RiskLevel = RiskLevel.MEDIUM
    exists: bool = False
    authorities: list[str] = Field(
        default_factory=list,
        description="Standard-form / market references grounding why this protection is expected",
    )
    benchmark: str = Field(
        default="",
        description="Market-standard context (e.g. AS 4000 / FIDIC practice) for why this clause matters",
    )


class CrossClauseConflict(BaseModel):
    """An internal inconsistency detected between two clauses of the agreement."""

    clause_uid_a: str
    clause_title_a: str
    clause_uid_b: str
    clause_title_b: str
    conflict_type: str = Field(
        default="",
        description="e.g. termination-vs-payment, notice-period, definition-conflict",
    )
    description: str = Field(
        default="",
        description="What the clauses say and why they conflict or interact ambiguously",
    )
    recommendation: str = Field(
        default="",
        description="Concrete fix to resolve the inconsistency",
    )
    severity: RiskLevel = RiskLevel.MEDIUM


class NegotiationPoint(BaseModel):
    """Negotiation advice attached to a high-risk clause."""

    clause_uid: str
    topic: str
    suggested_wording: str
    fallback_wording: str
    strategy: str
    commercial_reasoning: str
    owner_position: str = Field(
        default="",
        description="Recommended position if representing the owner",
    )
    contractor_position: str = Field(
        default="",
        description="Recommended position if representing the contractor",
    )
    priority: RiskLevel = RiskLevel.MEDIUM


class RiskMatrixCell(BaseModel):
    likelihood: RiskLevel = RiskLevel.LOW
    impact: RiskLevel = RiskLevel.LOW


class ExecutiveSummary(BaseModel):
    """Final report the user receives within the target SLA."""

    overall_risk: RiskLevel
    top_risks: list[ClauseReview] = Field(default_factory=list, max_length=5)
    risk_matrix: dict[str, RiskMatrixCell] = Field(default_factory=dict)
    missing_clauses: list[MissingClause] = Field(default_factory=list)
    negotiation_priorities: list[NegotiationPoint] = Field(default_factory=list)
    cross_clause_conflicts: list[CrossClauseConflict] = Field(default_factory=list)
    key_issues: list[str] = Field(default_factory=list)
    estimated_time_saved_minutes: int = Field(
        default=0,
        description="Partner-facing estimate of review minutes saved versus manual review",
    )
    recommendation: str = ""
    memo: str = Field(
        default="",
        description="Narrative partner-ready memo appended at the end of the report",
    )


class ChatCitation(BaseModel):
    """A verbatim excerpt from a clause cited in a chat answer."""

    clause_uid: str = ""
    clause_title: str = ""
    clause_number: str = ""
    snippet: str = ""


class ChatMessage(BaseModel):
    """One user/assistant turn in the Ask-Contract conversation."""

    role: str = "user"  # user | assistant
    content: str


class ChatResponse(BaseModel):
    """Grounded answer to a question about a reviewed contract."""

    answer: str = ""
    grounded: bool = True
    citations: list[ChatCitation] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Body for the Ask-Contract endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class ContractReview(BaseModel):
    """Complete output of the review pipeline."""

    review_id: str
    status: ReviewStatus = ReviewStatus.QUEUED
    document: Document | None = None
    clauses: list[ParsedClause] = Field(default_factory=list)
    clause_reviews: list[ClauseReview] = Field(default_factory=list)
    missing_clauses: list[MissingClause] = Field(default_factory=list)
    negotiation_points: list[NegotiationPoint] = Field(default_factory=list)
    cross_clause_conflicts: list[CrossClauseConflict] = Field(default_factory=list)
    executive_summary: ExecutiveSummary | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")