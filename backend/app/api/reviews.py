"""REST endpoints exposing the review pipeline to the web client.

Endpoints
* ``POST /api/v1/reviews``   - upload a contract; kicks off a background review
                               and returns a job id immediately.
* ``GET  /api/v1/reviews/{id}`` - poll for job status / completed result.
* ``POST /api/v1/reviews/{id}/chat`` - ask a grounded question about a
                               completed review.
* ``GET  /api/v1/health``    - liveness probe.

The async job flow keeps the browser responsive even when a slow/free LLM
provider is in use. A real deployment would add persistence and auth.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api import jobs as job_store
from app.parsers.document import ParsingError
from app.schemas.review import ChatRequest, ChatResponse, ContractReview

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reviews"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "lexmind"}


@router.post("/reviews")
async def create_review(file: Annotated[UploadFile, File(...)]) -> dict:
    name = file.filename or "contract"
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed: .pdf, .docx, .txt, .md, .csv",
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes")

    review_id = job_store.submit(name, data)
    return {"review_id": review_id, "status": "accepted"}


@router.get("/reviews/{review_id}")
def get_review(review_id: str) -> dict:
    job = job_store.get_job(review_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return job


@router.post("/reviews/{review_id}/chat", response_model=ChatResponse)
def chat_review(review_id: str, body: ChatRequest) -> ChatResponse:
    """Grounded Q&A over a completed review's parsed clauses."""
    job = job_store.get_job(review_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if job.get("status") != "complete" or not job.get("result"):
        raise HTTPException(status_code=409, detail="Review is not complete yet")
    review = ContractReview.model_validate(job["result"])
    return job_store._service().ask_contract(review, body.question, body.history)