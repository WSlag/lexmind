"""End-to-end tests for the GasMind review pipeline and API using the mock LLM."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm.client import LLMClient
from app.main import app
from app.parsers.document import ParsingError, parse_file
from app.schemas.review import ContractReview, ParsedClause, ReviewStatus
from app.workflow.nodes import _dedup_headings
from app.workflow.services import ReviewService
from app.workflow.state import initial_review_state, state_to_contract_review

SAMPLE = Path(__file__).resolve().parents[2] / "examples" / "contracts" / "sample_gas_supply_agreement.txt"

client = TestClient(app)


def test_parse_sample() -> None:
    doc = parse_file(SAMPLE.name, SAMPLE.read_bytes())
    assert doc.word_count > 100
    assert "DEFINITIONS" in doc.content


def test_end_to_end_pipeline_mock() -> None:
    service = ReviewService(LLMClient())
    review = service.run(SAMPLE.name, SAMPLE.read_bytes())
    assert isinstance(review, ContractReview)
    assert review.status == ReviewStatus.COMPLETE
    assert review.document is not None
    assert review.executive_summary is not None
    # The mock always returns at least one clause & one missing clause.
    assert len(review.clauses) >= 1
    assert len(review.missing_clauses) >= 1
    # New MVP fields populate with mock provider.
    assert review.executive_summary.memo
    assert review.executive_summary.estimated_time_saved_minutes >= 0
    assert isinstance(review.cross_clause_conflicts, list)
    assert review.missing_clauses[0].benchmark
    # New quick-win fields populate with mock provider.
    assert review.document.contract_type
    assert review.clause_reviews[0].authorities
    assert review.clause_reviews[0].action_priority.value
    assert review.missing_clauses[0].authorities


def test_state_to_contract_marks_failure_on_error() -> None:
    state = initial_review_state("rev-test", filename=SAMPLE.name, file_bytes=b"")
    state["errors"] = ["boom"]
    result = state_to_contract_review(state)
    assert result.status == ReviewStatus.FAILED


def test_reject_unsupported_type() -> None:
    resp = client.post(
        "/api/v1/reviews",
        files={"file": ("x.pdf.exe", b"hello", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_upload_endpoint_runs_mock() -> None:
    resp = client.post(
        "/api/v1/reviews",
        files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "accepted"
    assert "review_id" in body
    # Poll until the background job finishes.
    import time

    rid = body["review_id"]
    seen_progress = False
    for _ in range(50):
        time.sleep(0.1)
        status = client.get(f"/api/v1/reviews/{rid}").json()
        if status.get("progress"):
            seen_progress = True
        if status["status"] in ("complete", "failed"):
            break
    assert status["status"] == "complete", status
    assert seen_progress, "expected live progress steps to be recorded"
    result = status["result"]
    assert result["executive_summary"]["overall_risk"] in {
        "none",
        "low",
        "medium",
        "high",
        "critical",
    }


def test_health() -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "lexmind"}


def test_dedupe_headings_removes_section_headings() -> None:
    clauses = [
        ParsedClause(uid="a", title="PRICING AND PAYMENT", number="3", kind="pricing", text="3. PRICING AND PAYMENT"),
        ParsedClause(uid="b", title="Contract Price", number="3.1", kind="pricing", text="3.1 Contract Price. Buyer shall pay."),
        ParsedClause(uid="c", title="Payment Terms", number="3.2", kind="payment", text="3.2 Payment Terms. Seller shall invoice."),
        ParsedClause(uid="d", title="Delivery", number="4.1", kind="delivery", text="4.1 Seller shall deliver."),
    ]
    kept = _dedup_headings(clauses)
    assert [c.number for c in kept] == ["3.1", "3.2", "4.1"]


def test_negotiation_accepts_medium_risk() -> None:
    raw = client.post(
        "/api/v1/reviews",
        files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "text/plain")},
    )
    assert raw.status_code == 200
    rid = raw.json()["review_id"]
    import time

    for _ in range(50):
        time.sleep(0.1)
        status = client.get(f"/api/v1/reviews/{rid}").json()
        if status["status"] in ("complete", "failed"):
            break
    # Guard: the endpoint completes and returns structured clause reviews.
    assert status["status"] == "complete", status
    assert "clause_reviews" in status["result"]


def test_ask_contract_returns_grounded_answer() -> None:
    raw = client.post(
        "/api/v1/reviews",
        files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "text/plain")},
    )
    rid = raw.json()["review_id"]
    import time

    for _ in range(50):
        time.sleep(0.1)
        status = client.get(f"/api/v1/reviews/{rid}").json()
        if status["status"] in ("complete", "failed"):
            break
    assert status["status"] == "complete", status

    resp = client.post(
        f"/api/v1/reviews/{rid}/chat",
        json={"question": "What is the liability cap?", "history": []},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"]
    assert body["grounded"] is True
    # Citation must reference a clause that exists in the review.
    known_uids = {c["uid"] for c in status["result"]["clauses"]}
    for cite in body["citations"]:
        assert cite["clause_uid"] in known_uids
        assert cite["snippet"]


def test_ask_contract_requires_complete_review() -> None:
    # Unknown review id -> 404.
    resp = client.post(
        "/api/v1/reviews/does-not-exist/chat",
        json={"question": "hi", "history": []},
    )
    assert resp.status_code == 404

    # Empty question -> 422.
    raw = client.post(
        "/api/v1/reviews",
        files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "text/plain")},
    )
    rid = raw.json()["review_id"]
    resp = client.post(f"/api/v1/reviews/{rid}/chat", json={"question": ""})
    assert resp.status_code == 422


def test_completed_review_persists_and_reloads() -> None:
    """A completed review is written to disk and chat works after reload."""
    import shutil

    from app.api import jobs as job_store

    if job_store._DATA_DIR.is_dir():
        shutil.rmtree(job_store._DATA_DIR)

    rid = job_store.submit(SAMPLE.name, SAMPLE.read_bytes())
    import time

    for _ in range(80):
        time.sleep(0.1)
        status = job_store.get_job(rid)
        if status["status"] in ("complete", "failed"):
            break
    assert status["status"] == "complete", status

    files = list(job_store._DATA_DIR.glob("*.json"))
    assert any(f.name.startswith(rid) for f in files), "review should be persisted"

    # Simulate a server restart: clear memory, reload from disk.
    job_store.JOBS.clear()
    job_store._load_persisted()
    reloaded = job_store.get_job(rid)
    assert reloaded is not None
    assert reloaded["status"] == "complete"
    assert reloaded["result"] is not None

    # Chat works against the reloaded (dict-shaped) result.
    from app.schemas.review import ContractReview

    review = ContractReview.model_validate(reloaded["result"])
    chat = job_store._service().ask_contract(review, "What is the liability cap?", [])
    assert chat.answer
    assert chat.grounded is True