"""Compare base and tuned model grading on the nine-essay LEQ practice set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.eval import summarize_real_eval
from apush_frq_grader_slm.filters import parse_grade_json
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.rubric import CRITERIA
from apush_frq_grader_slm.schemas import EvalResult, FRQCase


def compare(
    cases: list[FRQCase], base_results: list[EvalResult], tuned_results: list[EvalResult]
) -> dict[str, Any]:
    _validate_results(cases, base_results, "base")
    _validate_results(cases, tuned_results, "tuned")
    base_by_id = {row.case_id: row for row in base_results}
    tuned_by_id = {row.case_id: row for row in tuned_results}
    rows = []
    for case in cases:
        base_total = _predicted_total(base_by_id[case.id])
        tuned_total = _predicted_total(tuned_by_id[case.id])
        reference = case.reference_scores.total
        rows.append(
            {
                "case_id": case.id,
                "reference_total": reference,
                "base_total": base_total,
                "tuned_total": tuned_total,
                "base_absolute_error": _absolute_error(base_total, reference),
                "tuned_absolute_error": _absolute_error(tuned_total, reference),
            }
        )
    base = summarize_real_eval(base_results, cases).model_dump(mode="json")
    tuned = summarize_real_eval(tuned_results, cases).model_dump(mode="json")
    base.update(_bias_metrics(rows, "base"))
    tuned.update(_bias_metrics(rows, "tuned"))
    delta = {
        "total_mae_reduction": round(base["total_mae"] - tuned["total_mae"], 4),
        "total_exact_match_rate": round(
            tuned["total_exact_match_rate"] - base["total_exact_match_rate"], 4
        ),
        "total_within_one_rate": round(
            tuned["total_within_one_rate"] - base["total_within_one_rate"], 4
        ),
        "rubric_row_exact_match_rate": round(
            tuned["exact_match_rate"] - base["exact_match_rate"], 4
        ),
        "structured_output_valid_rate": round(
            tuned["structured_output_valid_rate"] - base["structured_output_valid_rate"], 4
        ),
        "qwk": _optional_delta(tuned["qwk"], base["qwk"]),
    }
    for criterion in CRITERIA:
        delta[f"{criterion}_exact_rate"] = round(
            tuned["criterion_exact_rates"][criterion]
            - base["criterion_exact_rates"][criterion],
            4,
        )
    return {
        "evaluation": "leq_grading_practice_v1",
        "count": len(cases),
        "interpretation": (
            "Directional external-validity check only; nine essays are too few for a definitive "
            "performance claim and were not used for tuning."
        ),
        "base": base,
        "tuned": tuned,
        "improvement": delta,
        "cases": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    base, tuned, delta = report["base"], report["tuned"], report["improvement"]
    lines = [
        "# Qwen Base vs APUSH Grader v5 — LEQ Practice Comparison",
        "",
        report["interpretation"],
        "",
        "| Metric | Base | Tuned | Improvement |",
        "|---|---:|---:|---:|",
        f"| Total MAE | {base['total_mae']:.3f} | {tuned['total_mae']:.3f} | {delta['total_mae_reduction']:+.3f} reduction |",
        f"| Exact total | {base['total_exact_match_rate']:.1%} | {tuned['total_exact_match_rate']:.1%} | {delta['total_exact_match_rate']:+.1%} |",
        f"| Within one | {base['total_within_one_rate']:.1%} | {tuned['total_within_one_rate']:.1%} | {delta['total_within_one_rate']:+.1%} |",
        f"| Exact rubric rows | {base['exact_match_rate']:.1%} | {tuned['exact_match_rate']:.1%} | {delta['rubric_row_exact_match_rate']:+.1%} |",
        f"| QWK | {_fmt(base['qwk'])} | {_fmt(tuned['qwk'])} | {_fmt(delta['qwk'], signed=True)} |",
        f"| Structured output | {base['structured_output_valid_rate']:.1%} | {tuned['structured_output_valid_rate']:.1%} | {delta['structured_output_valid_rate']:+.1%} |",
        f"| Mean signed error | {base['mean_signed_error']:+.3f} | {tuned['mean_signed_error']:+.3f} | — |",
        "",
        "## Per-essay totals",
        "",
        "| Essay | Reference | Base | Tuned | Base error | Tuned error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['reference_total']} | {_fmt(row['base_total'])} | "
            f"{_fmt(row['tuned_total'])} | {row['base_absolute_error']} | "
            f"{row['tuned_absolute_error']} |"
        )
    return "\n".join(lines) + "\n"


def _validate_results(cases: list[FRQCase], results: list[EvalResult], label: str) -> None:
    wanted = {case.id for case in cases}
    observed = [row.case_id for row in results]
    if len(observed) != len(set(observed)) or set(observed) != wanted:
        raise ValueError(f"{label} results do not contain exactly the {len(cases)} practice cases")


def _predicted_total(result: EvalResult) -> int | None:
    payload, _ = parse_grade_json(result.response)
    if payload is None:
        return None
    value = payload.get("total")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _absolute_error(predicted: int | None, reference: int) -> int:
    return 6 if predicted is None else abs(predicted - reference)


def _bias_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    pairs = [
        (row[f"{prefix}_total"], row["reference_total"])
        for row in rows
        if row[f"{prefix}_total"] is not None
    ]
    if not pairs:
        return {"mean_signed_error": 0.0, "overgrade_rate": 0.0, "undergrade_rate": 0.0}
    errors = [predicted - reference for predicted, reference in pairs]
    return {
        "mean_signed_error": round(sum(errors) / len(errors), 4),
        "overgrade_rate": round(sum(error > 0 for error in errors) / len(errors), 4),
        "undergrade_rate": round(sum(error < 0 for error in errors) / len(errors), 4),
    }


def _optional_delta(tuned: float | None, base: float | None) -> float | None:
    return None if tuned is None or base is None else round(tuned - base, 4)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.3f}" if signed else f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--tuned-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.eval_path)]
    base = [EvalResult.model_validate(row) for row in read_jsonl(args.base_results)]
    tuned = [EvalResult.model_validate(row) for row in read_jsonl(args.tuned_results)]
    report = compare(cases, base, tuned)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
