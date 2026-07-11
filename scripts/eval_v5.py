"""Evaluate a packaged v5 two-pass bundle and write calibration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.eval import score_response, summarize_real_eval
from apush_frq_grader_slm.eval_v5 import v5_diagnostics
from apush_frq_grader_slm.inference_v5 import V5BundleGrader
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import EvalResult, FRQCase


def main() -> None:
    args = parse_args()
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.eval_path)]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError(f"No evaluation cases found in {args.eval_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.model_name}_real_results.jsonl"
    if not args.resume and results_path.exists():
        results_path.unlink()
    wanted_ids = {case.id for case in cases}
    existing = []
    if args.resume and results_path.exists():
        existing = [EvalResult.model_validate(row) for row in read_jsonl(results_path)]
        if len({row.case_id for row in existing}) != len(existing):
            raise ValueError(f"Duplicate case IDs in resumable results: {results_path}")
        if any(row.case_id not in wanted_ids or row.model_name != args.model_name for row in existing):
            raise ValueError("Resumable results do not match this evaluation run")
    results_by_id = {row.case_id: row for row in existing}
    pending = [case for case in cases if case.id not in results_by_id]
    grader = V5BundleGrader(args.bundle, verify_hashes=args.verify_hashes) if pending else None
    with results_path.open("a", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(pending, start=1):
            assert grader is not None
            prediction = grader.grade(case.prompt, case.student_response)
            response = json.dumps(prediction, ensure_ascii=True, separators=(",", ":"))
            result = score_response(case, response, args.model_name)
            results_by_id[case.id] = result
            stream.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=True) + "\n")
            stream.flush()
            print(f"v5 evaluation {len(existing) + index}/{len(cases)}", flush=True)
    results = [results_by_id[case.id] for case in cases]
    predictions = [json.loads(result.response) for result in results]
    summary = summarize_real_eval(results, cases).model_dump(mode="json")
    summary.update(
        {
            "reference_total_mean": round(
                sum(case.reference_scores.total for case in cases) / len(cases), 4
            ),
            "predicted_total_mean": summary["total_score_mean"],
            "deterministic_total_rate": round(
                sum(
                    prediction.get("total") == sum(prediction.get("scores", {}).values())
                    for prediction in predictions
                )
                / len(predictions),
                4,
            ),
            "feedback_fallback_rate": round(
                sum(
                    any("model feedback was unavailable" in str(value) for value in prediction.get("feedback", {}).values())
                    for prediction in predictions
                )
                / len(predictions),
                4,
            ),
            "development_informed": args.development_informed,
        }
    )
    diagnostics = v5_diagnostics(
        cases, predictions, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    (args.output_dir / f"{args.model_name}_real_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / f"{args.model_name}_v5_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/v5"))
    parser.add_argument("--model-name", default="apush-frq-grader-v5")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--development-informed", action="store_true")
    parser.add_argument("--verify-hashes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
