"""Classify invalid model outputs as truncated, incomplete, or malformed JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from apush_frq_grader_slm.filters import parse_grade_json
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.rubric import validate_grade_payload


def classify_response(
    response: str,
    *,
    structured_output_valid: bool,
    token_count: int | None,
    max_new_tokens: int,
    token_margin: int = 2,
) -> tuple[str, list[str]]:
    payload, parse_reasons = parse_grade_json(response)
    if payload is not None:
        payload_valid, schema_reasons = validate_grade_payload(payload)
        if structured_output_valid and payload_valid:
            return "valid_structured_json", []
        return "json_schema_invalid", schema_reasons or parse_reasons

    if token_count is not None and token_count >= max_new_tokens - token_margin:
        return "likely_max_token_truncation", parse_reasons

    stripped = response.strip()
    if stripped.startswith("{") and (
        not stripped.endswith("}") or stripped.count("{") != stripped.count("}")
    ):
        return "incomplete_json_below_limit", parse_reasons
    return "malformed_or_non_json", parse_reasons


def diagnose_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any | None,
    max_new_tokens: int,
    token_margin: int,
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    token_counts: list[int] = []

    for row in rows:
        response = str(row.get("response", ""))
        token_count = None
        if tokenizer is not None:
            token_count = len(tokenizer.encode(response, add_special_tokens=False))
            token_counts.append(token_count)
        category, _ = classify_response(
            response,
            structured_output_valid=bool(row.get("structured_output_valid")),
            token_count=token_count,
            max_new_tokens=max_new_tokens,
            token_margin=token_margin,
        )
        categories[category] += 1
        if len(examples[category]) < 10:
            examples[category].append(str(row.get("case_id", "unknown")))

    total = len(rows)
    return {
        "total": total,
        "max_new_tokens": max_new_tokens,
        "tokenizer_used": tokenizer is not None,
        "categories": dict(sorted(categories.items())),
        "rates": {
            key: round(value / total, 4) if total else 0.0
            for key, value in sorted(categories.items())
        },
        "response_tokens": {
            "minimum": min(token_counts) if token_counts else None,
            "median": median(token_counts) if token_counts else None,
            "maximum": max(token_counts) if token_counts else None,
        },
        "example_case_ids": dict(sorted(examples.items())),
    }


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
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--token-margin", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/eval/v2/output_diagnostics.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
