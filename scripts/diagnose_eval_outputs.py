"""Classify invalid model outputs as truncated, incomplete, or malformed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.eval_diagnostics import classify_response, diagnose_rows
from apush_frq_grader_slm.io import read_jsonl

__all__ = ["classify_response", "diagnose_rows"]


def main() -> None:
    args = parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report = diagnose_rows(
        read_jsonl(args.results),
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        token_margin=args.token_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--token-margin", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/eval/v2/output_diagnostics.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
