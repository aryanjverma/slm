"""Audit generated candidates before independent grading and write accepted rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.candidate_audit import audit_candidates
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.synth_realistic import GenTask


def _load_cases(paths: list[Path]) -> list[FRQCase]:
    cases: list[FRQCase] = []
    for path in paths:
        if path.exists():
            cases.extend(FRQCase.model_validate(row) for row in read_jsonl(path))
    return cases


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    accepted, audit = audit_candidates(
        tasks,
        read_jsonl(args.raw),
        leakage_sources=_load_cases(args.leakage_sources),
    )
    write_jsonl(args.output, [candidate.to_row() for candidate in accepted])
    write_jsonl(args.rejects, audit.rejected)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Audited {audit.total_rows} candidates: accepted={audit.accepted_rows}, "
        f"rejected={len(audit.rejected)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("artifacts/data/synth_realistic_raw_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/synth_realistic_validated_v2.jsonl")
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_candidate_rejects_v2.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/audits/synth_candidates_v2.json")
    )
    parser.add_argument(
        "--leakage-sources",
        type=Path,
        action="append",
        default=[
            Path("artifacts/data/eval_cb_golden_v2.jsonl"),
            Path("artifacts/data/eval_external_v2.jsonl"),
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
