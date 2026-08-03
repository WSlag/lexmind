"""Metrics for evaluating the GasMind pipeline against a gold standard."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable


def normalize(value: str) -> str:
    """Lowercase, strip punctuation and whitespace for fuzzy matching."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def match_titles(predicted: Iterable[str], expected: Iterable[str], threshold: float = 0.6) -> set:
    """Return the set of predicted titles that fuzzy-match an expected title."""
    def scores(p, e):
        return SequenceMatcher(None, normalize(p), normalize(e)).ratio()

    expected_norm = {normalize(e): e for e in expected}
    matched = set()
    for pred in predicted:
        pnorm = normalize(pred)
        for enorm, etitle in expected_norm.items():
            if not pnorm or not enorm:
                continue
            if scores(pred, etitle) >= threshold:
                matched.add(pred)
                break
    return matched


def clause_metrics(predicted_titles: list[str], expected: list[str]) -> dict:
    """Precision / recall / F1 for clause detection as a set task.

    precision = matched unique predicted / total predicted uniqueness
    recall    = matched unique predicted / total expected
    """
    matched = match_titles(predicted_titles, expected)
    tp = len(matched)
    precision = tp / len(set(predicted_titles)) if predicted_titles else 0.0
    recall = tp / len(expected) if expected else 0.0
    missed = [e for e in expected if not any(
        SequenceMatcher(None, normalize(m), normalize(e)).ratio() >= 0.75 for m in matched
    )]
    spurious = [p for p in set(predicted_titles) if normalize(p) not in {normalize(m) for m in matched}]
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1(precision, recall), 3),
        "matched": sorted(matched),
        "missed": sorted(missed),
        "spurious": sorted(spurious),
    }


def missing_metrics(flagged: Iterable[str], expected_missing: Iterable[str], expected_present: Iterable[str]) -> dict:
    """TP = flagged & genuinely absent; FP = flagged but actually present;
    FN = genuinely absent but not flagged."""
    flagged_set = {normalize(f) for f in flagged}
    absent = {normalize(x) for x in expected_missing}
    present = {normalize(x) for x in expected_present}
    tp = len(flagged_set & absent)
    fp = len(flagged_set & present)
    fn = len(absent - flagged_set)
    missed = [x for x in expected_missing if normalize(x) not in flagged_set]
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "flagged": list(flagged_set),
        "missed_missing": sorted(missed),
    }