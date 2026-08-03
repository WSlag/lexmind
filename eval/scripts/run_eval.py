"""Eval harness runner.

Runs the GasMind pipeline over every contract in ``eval/test_contracts``,
compares against the gold standard in ``eval/gold``, and prints precision /
recall / F1 for clause detection and missing-clause detection.

Usage (from repo root):
    python eval/scripts/run_eval.py

Uses the configured LLM provider (mock by default in a clean checkout).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.workflow.services import ReviewService  # noqa: E402
from eval import metrics  # noqa: E402
from eval.schemas import (  # noqa: E402
    ClausesReport,
    EvaluationResult,
    GoldStandard,
    MissingReport,
)

TEST_DIR = ROOT / "eval" / "test_contracts"
GOLD_DIR = ROOT / "eval" / "gold"
OUT_DIR = ROOT / "eval" / "reports"


def load_gold(path: Path) -> GoldStandard:
    return GoldStandard.model_validate_json(path.read_text(encoding="utf-8"))


def clauses_report(contract: str, predicted: list[str], expected: list[str]) -> ClausesReport:
    m = metrics.clause_metrics(predicted, expected)
    return ClausesReport(
        contract=contract,
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        matched=m["matched"],
        missed=m["missed"],
        spurious=m["spurious"],
    )


def evaluate_contract(
    service: ReviewService, gold: GoldStandard
) -> tuple[ClausesReport, MissingReport, str]:
    contract_path = TEST_DIR / gold.contract
    if not contract_path.exists():
        raise FileNotFoundError(f"Missing test contract: {contract_path}")
    review = service.run(gold.contract, contract_path.read_bytes())

    predicted = [c.title for c in review.clauses]
    expected = [c.title for c in gold.expected_clauses]
    clause_rep = clauses_report(gold.contract, predicted, expected)

    flagged = [m.clause for m in review.missing_clauses]
    mm = metrics.missing_metrics(flagged, gold.expected_missing, gold.expected_present)
    missing_rep = MissingReport(
        contract=gold.contract,
        true_positive=mm["true_positive"],
        false_positive=mm["false_positive"],
        false_negative=mm["false_negative"],
        flagged=mm["flagged"],
        missed_missing=mm["missed_missing"],
    )
    return clause_rep, missing_rep, review.status.value


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def main() -> None:
    service = ReviewService()
    gold_files = sorted(GOLD_DIR.glob("*.json"))
    if not gold_files:
        print(f"No gold standard found in {GOLD_DIR}")
        sys.exit(1)

    results: list[EvaluationResult] = []
    for gold_path in gold_files:
        gold = load_gold(gold_path)
        start = time.perf_counter()
        clause_rep, missing_rep, status = evaluate_contract(service, gold)
        elapsed = time.perf_counter() - start
        results.append(
            EvaluationResult(
                summary={
                    "contract": gold.contract,
                    "status": status,
                    "elapsed_s": round(elapsed, 1),
                },
                clause_reports=[clause_rep],
                missing_reports=[missing_rep],
            )
        )

    all_clauses = [r.clause_reports[0] for r in results]
    all_missing = [r.missing_reports[0] for r in results]
    aggregated = {
        "clauses": {
            "avg_precision": _avg([c.precision for c in all_clauses]),
            "avg_recall": _avg([c.recall for c in all_clauses]),
            "avg_f1": _avg([c.f1 for c in all_clauses]),
        },
        "missing": {
            "total_tp": sum(m.true_positive for m in all_missing),
            "total_fp": sum(m.false_positive for m in all_missing),
            "total_fn": sum(m.false_negative for m in all_missing),
        },
        "contracts": len(results),
    }

    print("=== GasMind Eval Report ===")
    print(f"Contracts: {aggregated['contracts']}")
    c = aggregated["clauses"]
    print(
        f"Clause detection  P={c['avg_precision']}  R={c['avg_recall']}  "
        f"F1={c['avg_f1']}"
    )
    m = aggregated["missing"]
    print(
        f"Missing-clause    TP={m['total_tp']}  FP={m['total_fp']}  "
        f"FN={m['total_fn']}"
    )
    for r in results:
        cr, mr = r.clause_reports[0], r.missing_reports[0]
        print(
            f"  - {r.summary['contract']} [{r.summary['status']}] "
            f"clauses(P={cr.precision} R={cr.recall} F1={cr.f1}) "
            f"missing(TP={mr.true_positive} FP={mr.false_positive} "
            f"FN={mr.false_negative})"
        )

    # Write machine-readable report.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = EvaluationResult(summary=aggregated, clause_reports=all_clauses, missing_reports=all_missing)
    target = OUT_DIR / "report.json"
    target.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"\nReport written to {target}")


if __name__ == "__main__":
    main()