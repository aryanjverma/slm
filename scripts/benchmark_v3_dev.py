"""Assemble the required pre-training set1 benchmark from base and saved v2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.eval_v3 import (
    build_run_identity,
    make_record,
    select_official_split,
    summarize_v3,
)
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.structured_output_v3 import has_repetition


def main() -> None:
    args = parse_args()
    all_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.cases)]
    cases = select_official_split(all_cases)
    case_ids = {case.id for case in cases}
    v2_rows = [row for row in read_jsonl(args.v2_results) if row["case_id"] in case_ids]
    truncated_ids: set[str] = set()
    if args.v2_diagnostics.exists():
        diagnostics = json.loads(args.v2_diagnostics.read_text(encoding="utf-8"))
        truncated_ids = set(
            diagnostics.get("example_case_ids", {}).get("likely_max_token_truncation", [])
        )
    identity = build_run_identity(
        model_name="apush_frq_grader_v2_saved",
        model_hash=args.v2_model_hash,
        data_hash=args.data_hash,
        split="set1",
        decoding_settings={"max_new_tokens": 512, "source": "saved_v2"},
    )
    records = []
    for row in v2_rows:
        response = str(row.get("response", ""))
        record = make_record(
            case_id=row["case_id"],
            identity=identity,
            raw_response=response,
            prompt_tokens=0,
            completion_tokens=512 if row["case_id"] in truncated_ids else 0,
            finish_reason="length" if row["case_id"] in truncated_ids else "unknown",
            repetition_detected=has_repetition(response),
        )
        # The v2 raw target included total, so retain its original strict-schema result.
        record.raw_schema_valid = bool(row.get("structured_output_valid"))
        records.append(record)
    v2_summary = summarize_v3(records, cases, identity).model_dump(mode="json")
    benchmark = {
        "split": "set1",
        "rows": len(cases),
        "configurations": {
            "base_qwen_0_5b_layered": _load_optional(args.base_summary),
            "v2_raw_generation": v2_summary["raw_model"],
            "v2_with_v3_structured_layer": v2_summary["layered_system"],
        },
        "note": (
            "All three pretraining configurations are populated."
            if args.base_summary
            else "Base metrics are pending a Qwen 0.5B set1 generation run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(benchmark, indent=2, sort_keys=True))


def _load_optional(path: Path | None):
    if path is None:
        return {"status": "pending_base_generation"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["layered_system"]


def parse_args() -> argparse.Namespace:
    root = Path("apush-frq-grader-v2-eval/apush-frq-grader-v2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument(
        "--v2-results",
        type=Path,
        default=root / "apush_frq_grader_v2_cb_eval_real_results.jsonl",
    )
    parser.add_argument(
        "--v2-diagnostics",
        type=Path,
        default=root / "apush_frq_grader_v2_cb_eval_real_results_diagnostics.json",
    )
    parser.add_argument("--base-summary", type=Path)
    parser.add_argument("--v2-model-hash", default="saved-v2-adapter")
    parser.add_argument("--data-hash", default="eval_cb_cases_local_unverified")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/eval/v3/pretraining_dev_benchmark.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
