"""Verify v3 context length and assistant-only label masking with the real tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.training_v3 import (
    assert_assistant_only_example,
    tokenize_assistant_only,
)


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths: list[int] = []
    target_lengths: list[int] = []
    for row in read_jsonl(args.data):
        example = tokenize_assistant_only(
            row["messages"], tokenizer, max_length=args.max_length
        )
        assert_assistant_only_example(example)
        lengths.append(len(example["input_ids"]))
        target_lengths.append(sum(value != -100 for value in example["labels"]))
    report = {
        "rows": len(lengths),
        "tokenizer": args.tokenizer,
        "max_context_tokens": args.max_length,
        "minimum_row_tokens": min(lengths) if lengths else 0,
        "maximum_row_tokens": max(lengths) if lengths else 0,
        "minimum_assistant_tokens": min(target_lengths) if target_lengths else 0,
        "maximum_assistant_tokens": max(target_lengths) if target_lengths else 0,
        "all_assistant_targets_visible": bool(lengths) and all(target_lengths),
        "only_assistant_tokens_supervised": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/v3/train_chat_v3.jsonl"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audits/v3_training_token_audit.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
