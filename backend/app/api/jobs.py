"""Background job store for async contract reviews.

Holds review jobs in a thread-safe in-memory map so the API can return quickly
with a job id and the browser can poll ``GET /api/v1/reviews/{id}`` for
progress. Completed jobs are also persisted to disk (``data/reviews/*.json``)
and reloaded at startup so a server restart does not orphan running reviews —
the Ask-Contract chat keeps working after a restart.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from app.workflow.services import ReviewService

logger = logging.getLogger(__name__)

# review_id -> dict(job)
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_SERVICE: ReviewService | None = None

# Persist completed reviews here. Overridable via DATA_DIR so cloud hosts can
# mount a durable volume (e.g. /data on Fly.io) instead of the ephemeral disk.
_DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "reviews")))


def _service() -> ReviewService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ReviewService()
    return _SERVICE


def submit(filename: str, data: bytes) -> str:
    review_id = f"rev-{uuid.uuid4().hex[:12]}"
    with _LOCK:
        JOBS[review_id] = {
            "review_id": review_id,
            "stage": "queued",
            "status": "queued",
            "progress": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
        }
    thread = threading.Thread(
        target=_run,
        args=(review_id, filename, data),
        daemon=True,
    )
    thread.start()
    return review_id


def _run(review_id: str, filename: str, data: bytes) -> None:
    def update(stage: str, status: str) -> None:
        with _LOCK:
            job = JOBS.get(review_id)
            if job is None:
                return
            job["stage"] = stage
            job["status"] = status
            job["updated_at"] = time.time()

    def on_progress(name: str, label: str, state: dict) -> None:
        """Record each pipeline stage so the UI can show live progress."""
        detail = ""
        clauses = state.get("clauses") or []
        if name == "risk_review":
            detail = f"{len(clauses)} clauses detected"
        elif name == "summary":
            detail = f"{len(clauses)} clauses reviewed"
        with _LOCK:
            job = JOBS.get(review_id)
            if job is None:
                return
            steps = list(job.get("progress", []))
            steps.append(
                {
                    "stage": name,
                    "label": label,
                    "detail": detail,
                    "at": time.time(),
                }
            )
            job["progress"] = steps
            job["stage"] = name
            job["status"] = "running"
            job["updated_at"] = time.time()

    try:
        update("parsing", "running")
        review = _service().run(
            filename, data, review_id=review_id, on_progress=on_progress
        )
        with _LOCK:
            JOBS[review_id]["result"] = review
            JOBS[review_id]["stage"] = "complete"
            JOBS[review_id]["status"] = review.status.value
            JOBS[review_id]["updated_at"] = time.time()
        _persist(review_id)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        with _LOCK:
            JOBS[review_id]["status"] = "failed"
            JOBS[review_id]["stage"] = "failed"
            JOBS[review_id]["error"] = str(exc)
            JOBS[review_id]["updated_at"] = time.time()
        _persist(review_id)


def get_job(review_id: str) -> dict | None:
    with _LOCK:
        job = JOBS.get(review_id)
        if job is None:
            return None
        return {
            "review_id": job["review_id"],
            "stage": job["stage"],
            "status": job["status"],
            "progress": job.get("progress", []),
            "updated_at": job["updated_at"],
            "result": _to_result_dict(job.get("result")),
            "error": job["error"],
        }


def _to_result_dict(result) -> dict | None:
    """Normalize a stored result (Pydantic object or plain dict) to a dict."""
    if result is None:
        return None
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported result type: {type(result)}")


def _persist(review_id: str) -> None:
    """Snapshot a job to disk so it survives a server restart."""
    job = get_job(review_id)
    if job is None:
        return
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": job["stage"],
            "status": job["status"],
            "progress": job["progress"],
            "updated_at": job["updated_at"],
            "result": job["result"],
            "error": job["error"],
        }
        target = _DATA_DIR / f"{review_id}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    except Exception:  # noqa: BLE001 - persistence is best-effort, never fatal
        logger.exception("Failed to persist review %s", review_id)


def _load_persisted() -> None:
    """Reload completed/failed jobs from disk at startup."""
    if not _DATA_DIR.is_dir():
        return
    for path in sorted(_DATA_DIR.glob("rev-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rid = path.stem
            if rid in JOBS:
                continue
            JOBS[rid] = {
                "review_id": rid,
                "stage": payload.get("stage", "complete"),
                "status": payload.get("status", "complete"),
                "progress": payload.get("progress", []),
                "created_at": payload.get("updated_at", time.time()),
                "updated_at": payload.get("updated_at", time.time()),
                "result": payload.get("result"),
                "error": payload.get("error"),
            }
        except Exception:  # noqa: BLE001 - skip corrupt files
            logger.exception("Failed to load persisted review %s", path.name)


_load_persisted()