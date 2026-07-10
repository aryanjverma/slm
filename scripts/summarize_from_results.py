"""Recompute an eval summary from an already-saved per-case results file.

`eval_hf_model.py` writes per-case results incrementally as it generates, then
computes the summary at the end. If summarizing fails (or you simply want to
re-derive the numbers after a code fix), this script rebuilds the summary from
the saved `*_results.jsonl` without re-running the model.

    python scripts/summarize_from_results.py \
      --results artifacts/eval/apush_frq_grader_v1_real_results.jsonl \
      --eval-path artifacts/data/eval_real_cases.jsonl \
      --model-name apush_frq_grader_v1 \
      --output-dir artifacts/eval \
      --real-eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.eval import (
    summarize,
    summarize_by_dimensions,
    summarize_real_eval,
    summarize_real_eval_by_rubric,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import EvalResult, FRQCase


def main() -> None:
    args = parse_args()
    results = [EvalResult.model_validate(row) for row in read_jsonl(args.results)]
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.eval_path)]
    if not results:
        raise SystemExit(f"No results found in {args.results}")

    model_name = args.model_name or results[0].model_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.real_eval:
        summary = summarize_real_eval(results, cases)
        out_path = args.output_dir / f"{model_name}_real_summary.jsonl"
        write_jsonl(
            args.output_dir / f"{model_name}_real_by_rubric.jsonl",
            [
                {"rubric_version": version, **item.model_dump(mode="json")}
                for version, item in summarize_real_eval_by_rubric(results, cases).items()
            ],
        )
    else:
        summary = summarize(results, model_name)
        out_path = args.output_dir / f"{model_name}_summary.jsonl"

    write_jsonl(out_path, [summary])
    (args.output_dir / f"{model_name}_dimensions.json").write_text(
        json.dumps(summarize_by_dimensions(results, cases), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Recomputed summary for {len(results)} results -> {out_path}")
    print(summary.model_dump())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, required=True, help="Path to a saved *_results.jsonl file."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        required=True,
        help="The eval cases file the results were generated from.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the model name; defaults to the name stored in the results.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument(
        "--real-eval",
        action="store_true",
        help="Use real CB eval metrics (row agreement, QWK) instead of the litmus summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
