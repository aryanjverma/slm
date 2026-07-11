"""Release-gate checks for the final v5 grader evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


V4_CRITERION_EXACT = {
    "thesis": 0.5283,
    "contextualization": 0.3962,
    "evidence": 0.1698,
    "analysis_reasoning": 0.3208,
}


@dataclass(frozen=True)
class V5ReleaseThresholds:
    qwk: float = 0.40
    total_mae: float = 1.50
    total_within_one_rate: float = 0.60
    structured_output_valid_rate: float = 0.98
    evidence_grounding_rate: float = 0.85
    reference_total_mean: float = 4.0377
    maximum_mean_bias: float = 0.50


def evaluate_v5_release(
    summary: Mapping[str, Any],
    *,
    thresholds: V5ReleaseThresholds = V5ReleaseThresholds(),
    criterion_baseline: Mapping[str, float] = V4_CRITERION_EXACT,
) -> dict[str, Any]:
    """Return a machine-readable decision without relaxing failed gates."""

    checks = {
        "qwk": _at_least(summary.get("qwk"), thresholds.qwk),
        "total_mae": _at_most(summary.get("total_mae"), thresholds.total_mae),
        "total_within_one_rate": _at_least(
            summary.get("total_within_one_rate"), thresholds.total_within_one_rate
        ),
        "structured_output_valid_rate": _at_least(
            summary.get("structured_output_valid_rate"),
            thresholds.structured_output_valid_rate,
        ),
        "evidence_grounding_rate": _at_least(
            summary.get("evidence_grounding_rate"), thresholds.evidence_grounding_rate
        ),
        "predicted_total_mean_bias": _mean_bias_ok(
            summary.get("total_score_mean"), thresholds
        ),
    }
    actual_criteria = summary.get("criterion_exact_rates")
    if not isinstance(actual_criteria, Mapping):
        actual_criteria = {}
    criterion_checks = {
        criterion: _strictly_greater(actual_criteria.get(criterion), baseline)
        for criterion, baseline in criterion_baseline.items()
    }
    checks["all_criteria_improve_over_v4"] = all(criterion_checks.values())
    passed = all(checks.values())
    return {
        "release_ready": passed,
        "decision": "release_ready" if passed else "non_production_ready",
        "checks": checks,
        "criterion_checks": criterion_checks,
        "thresholds": {
            **thresholds.__dict__,
            "criterion_exact_strictly_above": dict(criterion_baseline),
        },
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _at_least(value: object, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number >= threshold


def _at_most(value: object, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number <= threshold


def _strictly_greater(value: object, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number > threshold


def _mean_bias_ok(value: object, thresholds: V5ReleaseThresholds) -> bool:
    number = _number(value)
    return number is not None and abs(number - thresholds.reference_total_mean) <= thresholds.maximum_mean_bias
