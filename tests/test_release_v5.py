from __future__ import annotations

from apush_frq_grader_slm.release_v5 import evaluate_v5_release


def _passing_summary() -> dict:
    return {
        "qwk": 0.41,
        "total_mae": 1.49,
        "total_within_one_rate": 0.61,
        "structured_output_valid_rate": 0.99,
        "evidence_grounding_rate": 0.86,
        "total_score_mean": 4.1,
        "criterion_exact_rates": {
            "thesis": 0.54,
            "contextualization": 0.41,
            "evidence": 0.18,
            "analysis_reasoning": 0.33,
        },
    }


def test_release_requires_every_locked_gate() -> None:
    decision = evaluate_v5_release(_passing_summary())
    assert decision["release_ready"] is True
    assert decision["decision"] == "release_ready"


def test_equal_v4_criterion_rate_is_not_an_improvement() -> None:
    summary = _passing_summary()
    summary["criterion_exact_rates"]["evidence"] = 0.1698
    decision = evaluate_v5_release(summary)
    assert decision["release_ready"] is False
    assert decision["criterion_checks"]["evidence"] is False


def test_missing_or_biased_metrics_fail_closed() -> None:
    summary = _passing_summary()
    summary.pop("qwk")
    summary["total_score_mean"] = 0.9
    decision = evaluate_v5_release(summary)
    assert decision["checks"]["qwk"] is False
    assert decision["checks"]["predicted_total_mean_bias"] is False
