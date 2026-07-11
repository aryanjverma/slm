from __future__ import annotations

from pathlib import Path

from apush_frq_grader_slm.failure_analysis_v3 import analyze_v2_failures


def test_saved_v2_failure_analysis_reproduces_known_counts() -> None:
    root = Path("apush-frq-grader-v2-eval/apush-frq-grader-v2")
    if not root.exists():
        return
    report = analyze_v2_failures(
        results_path=root / "apush_frq_grader_v2_cb_eval_real_results.jsonl",
        cases_path=Path("artifacts/data/eval_cb_cases.jsonl"),
        diagnostics_path=root / "apush_frq_grader_v2_cb_eval_real_results_diagnostics.json",
    )
    assert report["overall"]["raw_schema_valid"] == 19
    assert report["overall"]["parseable_json"] == 44
    assert report["overall"]["normalized_usable"] == 36
    assert report["overall"]["normalized_total_mae"] == 1.8889
