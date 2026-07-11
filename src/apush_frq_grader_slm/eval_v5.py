"""Calibration diagnostics for final v5 two-pass evaluation."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence

from apush_frq_grader_slm.eval import _quadratic_weighted_kappa
from apush_frq_grader_slm.rubric import CRITERIA, SCORE_RANGES
from apush_frq_grader_slm.schemas import FRQCase


def v5_diagnostics(
    cases: Sequence[FRQCase],
    predictions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    if len(cases) != len(predictions) or not cases:
        raise ValueError("V5 diagnostics require one prediction per non-empty case list")
    rows = []
    confusions = {
        criterion: [[0 for _ in range(high + 1)] for _ in range(high + 1)]
        for criterion, (_, high) in SCORE_RANGES.items()
    }
    for case, prediction in zip(cases, predictions, strict=True):
        scores = prediction["scores"]
        ref = case.reference_scores.model_dump()
        predicted_total = sum(int(scores[key]) for key in CRITERIA)
        row = {
            "reference_total": case.reference_scores.total,
            "predicted_total": predicted_total,
            "word_count": len(case.student_response.split()),
            "reference_scores": ref,
            "predicted_scores": {key: int(scores[key]) for key in CRITERIA},
        }
        rows.append(row)
        for criterion in CRITERIA:
            confusions[criterion][int(ref[criterion])][int(scores[criterion])] += 1

    return {
        "count": len(rows),
        "criterion_confusion_matrices": confusions,
        "total_distributions": {
            "reference": _distribution(row["reference_total"] for row in rows),
            "predicted": _distribution(row["predicted_total"] for row in rows),
        },
        "calibration_by_length": _calibration_groups(rows, _length_band),
        "calibration_by_reference_total": _calibration_groups(
            rows, lambda row: str(row["reference_total"])
        ),
        "bootstrap_confidence_intervals": _bootstrap(rows, bootstrap_samples, seed),
    }


def _distribution(values: Any) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(total): counts[total] for total in range(7)}


def _length_band(row: Mapping[str, Any]) -> str:
    words = int(row["word_count"])
    if words < 150:
        return "000-149"
    if words < 300:
        return "150-299"
    if words < 500:
        return "300-499"
    return "500+"


def _calibration_groups(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return {key: _aggregate(group) for key, group in sorted(groups.items())}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    refs = [int(row["reference_total"]) for row in rows]
    preds = [int(row["predicted_total"]) for row in rows]
    n = len(rows)
    return {
        "count": n,
        "reference_mean": round(sum(refs) / n, 4),
        "predicted_mean": round(sum(preds) / n, 4),
        "mae": round(sum(abs(a - b) for a, b in zip(refs, preds, strict=True)) / n, 4),
        "within_one_rate": round(
            sum(abs(a - b) <= 1 for a, b in zip(refs, preds, strict=True)) / n, 4
        ),
        "qwk": (
            round(value, 4)
            if n >= 5 and (value := _quadratic_weighted_kappa(refs, preds)) is not None
            else None
        ),
    }


def _bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    if samples <= 0:
        return {}
    rng = random.Random(seed)
    collected: dict[str, list[float]] = {"qwk": [], "total_mae": [], "within_one_rate": []}
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        refs = [int(row["reference_total"]) for row in sample]
        preds = [int(row["predicted_total"]) for row in sample]
        qwk = _quadratic_weighted_kappa(refs, preds)
        if qwk is not None:
            collected["qwk"].append(qwk)
        collected["total_mae"].append(
            sum(abs(a - b) for a, b in zip(refs, preds, strict=True)) / len(sample)
        )
        collected["within_one_rate"].append(
            sum(abs(a - b) <= 1 for a, b in zip(refs, preds, strict=True)) / len(sample)
        )
    return {key: _percentile_interval(values) for key, values in collected.items() if values}


def _percentile_interval(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return {"low_95": round(low, 4), "high_95": round(high, 4)}
