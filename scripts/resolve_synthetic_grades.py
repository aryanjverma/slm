"""Resolve precomputed anonymous reader outputs without calling an external API."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.independent_grading import (
    GradeDecision,
    parse_grader_output,
    resolve_independent_grades,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.synth_realistic import GenTask, parse_candidate_row


def _by_task(path: Path) -> dict[str, dict]:
    return {str(row["task_id"]): row for row in read_jsonl(path)} if path.exists() else {}


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    candidates = _by_task(args.raw)
    reader_a_rows = _by_task(args.reader_a)
    reader_b_rows = _by_task(args.reader_b)
    adjudicator_rows = _by_task(args.adjudicator) if args.adjudicator else {}
    decisions: list[dict] = []
    rejects: list[dict] = []

    for task_id, task in tasks.items():
        candidate_row = candidates.get(task_id)
        if candidate_row is None:
            continue
        try:
            candidate = parse_candidate_row(candidate_row, task)
            reader_a = parse_grader_output(reader_a_rows[task_id], candidate, "reader_a:offline")
            reader_b = parse_grader_output(reader_b_rows[task_id], candidate, "reader_b:offline")
            adjudicator = (
                parse_grader_output(adjudicator_rows[task_id], candidate, "adjudicator:offline")
                if task_id in adjudicator_rows
                else None
            )
            decision = resolve_independent_grades(
                task_id,
                reader_a,
                reader_b,
                adjudicator,
                minimum_confidence=args.minimum_confidence,
            )
        except Exception as exc:
            decision = GradeDecision(
                task_id,
                "rejected",
                None,
                None,
                None,
                {"resolution": "offline_parse_error"},
                (f"offline_parse_error:{exc}",),
            )
        decisions.append(decision.to_row())
        if decision.status == "rejected":
            rejects.append({"task_id": task_id, "reasons": list(decision.reasons)})

    write_jsonl(args.output, decisions)
    write_jsonl(args.rejects, rejects)
    accepted = sum(row["status"] == "accepted" for row in decisions)
    print(f"Resolved {len(decisions)} candidates: accepted={accepted}, rejected={len(rejects)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("artifacts/data/synth_realistic_validated_v2.jsonl")
    )
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path, required=True)
    parser.add_argument("--adjudicator", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/synth_realistic_grades_v2.jsonl")
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_grading_rejects_v2.jsonl")
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
