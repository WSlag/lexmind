"""LangGraph workflow orchestrating the GasMind review pipeline.

Composes the PRD stages into a directed graph. A leading ``run`` node forwards
the pipeline through parse -> extract -> risk review -> missing clauses ->
negotiation -> executive summary, with shared error handling.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.llm.client import LLMClient
from app.workflow.nodes import (
    node_cross_clause_check,
    node_executive_summary,
    node_extract,
    node_missing_clauses,
    node_negotiation,
    node_parse,
    node_risk_review,
)
from app.workflow.state import ReviewState

logger = logging.getLogger(__name__)

# Human-readable labels for each pipeline node, shown in the UI progress feed.
NODE_LABELS = {
    "parse": "Parsing document",
    "extract": "Extracting clauses",
    "risk_review": "Reviewing clause risk",
    "missing": "Checking missing protections",
    "negotiation": "Building negotiation playbook",
    "consistency": "Checking cross-clause consistency",
    "summary": "Drafting executive summary",
}


class ReviewGraph:
    """Compiled LangGraph ready to ``.invoke(state)``."""

    def __init__(self, llm: LLMClient, on_progress=None) -> None:
        self.llm = llm
        self.on_progress = on_progress
        builder = StateGraph(ReviewState)

        builder.add_node("parse", self._wrap("parse", node_parse))
        builder.add_node("extract", self._wrap("extract", node_extract, llm))
        builder.add_node("risk_review", self._wrap("risk_review", node_risk_review, llm))
        builder.add_node("missing", self._wrap("missing", node_missing_clauses, llm))
        builder.add_node("negotiation", self._wrap("negotiation", node_negotiation, llm))
        builder.add_node("consistency", self._wrap("consistency", node_cross_clause_check, llm))
        builder.add_node("summary", self._wrap("summary", node_executive_summary, llm))

        builder.add_edge(START, "parse")
        builder.add_edge("parse", "extract")
        builder.add_edge("extract", "risk_review")
        builder.add_edge("risk_review", "missing")
        builder.add_edge("missing", "negotiation")
        builder.add_edge("negotiation", "consistency")
        builder.add_edge("consistency", "summary")
        builder.add_edge("summary", END)

        self.graph = builder.compile()

    def _wrap(self, name: str, fn, llm: LLMClient | None = None):
        """Bind the LLM client into a pure state->state node with progress hook."""

        def wrapped(state: ReviewState) -> ReviewState:
            try:
                self._emit(name, state)
                if llm is not None:
                    return fn(state, llm)
                return fn(state)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Node %s failed", fn.__name__)
                state["errors"] = state.get("errors", []) + [
                    f"{getattr(fn, '__name__', 'node')}: {exc}"
                ]
                return state

        return wrapped

    def _emit(self, name: str, state: ReviewState) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(name, NODE_LABELS.get(name, name), state)
        except Exception:  # pragma: no cover - progress hooks must never break the pipeline
            logger.exception("Progress hook failed")

    def invoke(self, state: ReviewState) -> ReviewState:
        return self.graph.invoke(state)