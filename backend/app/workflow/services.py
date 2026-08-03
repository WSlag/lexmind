"""Service layer: high-level handlers used by the API and CLI.

Keeps transport (FastAPI) separated from business logic so the same entrypoint
can be used from a REST endpoint, a CLI runner, or a job worker.
"""
from __future__ import annotations

import logging
import uuid

from app.llm.client import LLMClient, get_llm_client
from app.prompts.skills import ASK_CONTRACT_SYSTEM, ASK_CONTRACT_USER, fill
from app.schemas.review import ChatCitation, ChatMessage, ChatResponse, ContractReview
from app.workflow.graph import ReviewGraph
from app.workflow.state import initial_review_state, state_to_contract_review

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 8
MAX_QUESTION_CHARS = 2000


class ReviewService:
    """Orchestrates a single contract review run."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm_client()

    def run(
        self,
        filename: str,
        data: bytes,
        review_id: str | None = None,
        on_progress=None,
    ) -> ContractReview:
        rid = review_id or f"rev-{uuid.uuid4().hex[:12]}"
        state = initial_review_state(rid, filename=filename, file_bytes=bytes(data))
        graph = ReviewGraph(self.llm, on_progress=on_progress)
        final = graph.invoke(state)
        return state_to_contract_review(final)

    def ask_contract(
        self,
        review: ContractReview,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> ChatResponse:
        """Answer a free-text question about a completed review, grounded in
        the parsed clauses with verbatim citations.

        Only the parsed clause text and document are treated as fact; the
        automated review findings are provided as awareness context and the
        prompt forbids restating them as contract text.
        """
        settings = self.llm.settings
        history = (history or [])[-MAX_HISTORY_TURNS:]
        question = (question or "").strip()[:MAX_QUESTION_CHARS]

        clauses = review.clauses or []
        lines = []
        for c in clauses:
            head = f"[{c.uid}] {c.number} {c.title}".strip()
            lines.append(f"{head}\n{c.text}")
        clauses_json = "\n\n".join(lines) or "(no clauses extracted)"
        if len(clauses_json) > settings.max_context_chars:
            clauses_json = clauses_json[: settings.max_context_chars] + "\n[truncated]"

        findings = _summarise_findings(review)
        history_text = "\n".join(f"{m.role}: {m.content}" for m in history) or "(none)"
        user = fill(
            ASK_CONTRACT_USER,
            contract_type=review.document.contract_type if review.document else "",
            clauses_json=clauses_json,
            findings=findings,
            history=history_text,
            question=question,
        )
        raw = self.llm.complete_json(
            ASK_CONTRACT_SYSTEM, user, max_tokens=settings.max_review_tokens_per_clause
        )
        return _parse_chat_response(raw, clauses)


def _summarise_findings(review: ContractReview) -> str:
    """Compact awareness-only digest of the automated review."""
    parts: list[str] = []
    if review.executive_summary and review.executive_summary.overall_risk:
        parts.append(f"Overall risk: {review.executive_summary.overall_risk.value}")
    high = [
        r for r in (review.clause_reviews or [])
        if r.risk_level.value in ("high", "critical")
    ]
    if high:
        parts.append(
            "High/critical clause risks: "
            + ", ".join(f"{r.clause_title} ({r.risk_level.value})" for r in high[:6])
        )
    if review.missing_clauses:
        parts.append(
            "Missing protections: " + ", ".join(m.clause for m in review.missing_clauses[:8])
        )
    if review.cross_clause_conflicts:
        parts.append(
            "Cross-clause conflicts: "
            + ", ".join(
                f"{c.clause_title_a} vs {c.clause_title_b}" for c in review.cross_clause_conflicts[:5]
            )
        )
    if review.negotiation_points:
        parts.append(
            "Negotiation topics: "
            + ", ".join(n.topic for n in review.negotiation_points[:6])
        )
    return "\n".join(parts) or "(none)"


def _parse_chat_response(raw: dict, clauses: list) -> ChatResponse:
    """Build a ChatResponse, keeping only citations that match a real clause uid."""
    answer = str(raw.get("answer") or "").strip()
    if not answer:
        # Empty answer (rare but possible with low-quality providers). Surface
        # the limitation instead of returning a silent dead-end to the lawyer.
        return ChatResponse(
            answer="I could not produce a grounded answer for that question. "
            "The contract may not address it, or the review context was too "
            "large to answer precisely. Try rephrasing or asking about a "
            "specific clause.",
            grounded=False,
            citations=[],
        )
    citations: list[ChatCitation] = []
    known = {c.uid for c in clauses}
    for item in raw.get("citations") or []:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("clause_uid") or "")
        if uid and uid not in known:
            continue
        citations.append(
            ChatCitation(
                clause_uid=uid,
                clause_title=str(item.get("clause_title") or ""),
                clause_number=str(item.get("clause_number") or ""),
                snippet=str(item.get("snippet") or "")[:500],
            )
        )
    return ChatResponse(
        answer=answer or "(no answer produced)",
        grounded=bool(raw.get("grounded", True)),
        citations=citations,
    )