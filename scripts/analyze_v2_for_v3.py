"""Generate the reproducible v2 failure-analysis inputs for v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.failure_analysis_v3 import analyze_v2_failures, render_failure_report


def main() -> None:
    args = parse_args()
    report = analyze_v2_failures(
        results_path=args.results,
        cases_path=args.cases,
        diagnostics_path=args.diagnostics,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_failure_report(report), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_markdown}")


def parse_args() -> argparse.Namespace:
    root = Path("apush-frq-grader-v2-eval/apush-frq-grader-v2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=root / "apush_frq_grader_v2_cb_eval_real_results.jsonl",
    )
    parser.add_argument("--cases", type=Path, default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=root / "apush_frq_grader_v2_cb_eval_real_results_diagnostics.json",
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("artifacts/audits/v2_failure_analysis_v3.json")
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("docs/v2_failure_analysis_for_v3.md")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
