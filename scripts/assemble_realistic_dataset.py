"""Assemble independently graded realistic candidates into training cases."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.independent_grading import GradeDecision, assemble_consensus_case
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.synth_realistic import (
    GenTask,
    parse_candidate_row,
    validate_generated_candidate,
)


def load_cases(path: Path) -> list[FRQCase]:
    if not path.exists():
        return []
    return [FRQCase.model_validate(row) for row in read_jsonl(path)]


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    decisions = {
        row["task_id"]: GradeDecision.from_row(row) for row in read_jsonl(args.grades)
    }
    leakage_sources = (
        load_cases(args.seeds)
        + load_cases(args.golden_eval)
        + load_cases(args.external_eval)
    )
    if not leakage_sources:
        print("WARNING: no seed/frozen-eval essays loaded; anti-leakage check is weak.")

    accepted_rows: list[dict] = []
    accepted_cases: list[FRQCase] = []
    rejects: list[dict] = []
    seen_task_ids: set[str] = set()
    for row in read_jsonl(args.raw):
        task_id = row.get("task_id")
        if task_id in seen_task_ids:
            rejects.append({"task_id": task_id, "reasons": ["duplicate_candidate_task_id"]})
            continue
        seen_task_ids.add(task_id)
        task = tasks.get(task_id)
        if task is None:
            rejects.append({"task_id": task_id, "reasons": ["unknown_task_id"]})
            continue
        try:
            candidate = parse_candidate_row(row, task)
        except Exception as exc:
            rejects.append({"task_id": task_id, "reasons": [f"parse_error:{exc}"]})
            continue
        candidate_ok, candidate_reasons = validate_generated_candidate(
            candidate, task, leakage_sources
        )
        if not candidate_ok:
            rejects.append({"task_id": task_id, "reasons": candidate_reasons})
            continue

        decision = decisions.get(task_id)
        if decision is None:
            rejects.append({"task_id": task_id, "reasons": ["missing_independent_grade"]})
            continue
        case, metadata, reasons = assemble_consensus_case(
            candidate,
            task,
            decision,
            max_target_distance=args.max_target_distance,
        )
        if case is None:
            rejects.append(
                {"task_id": task_id, "reasons": reasons, "labeling_metadata": metadata}
            )
            continue
        output_row = case.model_dump(mode="json")
        output_row["consensus_audit"] = metadata
        accepted_rows.append(output_row)
        accepted_cases.append(case)

    write_jsonl(args.output, accepted_rows)
    write_jsonl(args.rejects, rejects)
    _report(accepted_cases, accepted_rows, rejects)


def _report(accepted: list[FRQCase], rows: list[dict], rejects: list[dict]) -> None:
    print(f"Accepted {len(accepted)} independently graded cases; rejected {len(rejects)}.")
    if rejects:
        reason_counts: Counter[str] = Counter()
        for row in rejects:
            reason_counts.update(row["reasons"])
        print("  reject reasons: " + ", ".join(f"{r}:{c}" for r, c in reason_counts.most_common()))
    if not accepted:
        return
    totals = Counter(case.reference_scores.total for case in accepted)
    resolutions = Counter(row["consensus_audit"]["resolution"] for row in rows)
    print("  total distribution: " + ", ".join(f"{total}:{totals[total]}" for total in range(7)))
    print("  resolutions: " + ", ".join(f"{key}:{value}" for key, value in resolutions.items()))
    words = sorted(len(case.student_response.split()) for case in accepted)
    print(f"  essay words: min={words[0]} median={words[len(words) // 2]} max={words[-1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble independently graded realistic training cases."
    )
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("artifacts/data/synth_realistic_validated_v2.jsonl")
    )
    parser.add_argument(
        "--grades", type=Path, default=Path("artifacts/data/synth_realistic_grades_v2.jsonl")
    )
    parser.add_argument("--seeds", type=Path, default=Path("artifacts/data/seed_real_cases.jsonl"))
    parser.add_argument(
        "--golden-eval", type=Path, default=Path("artifacts/data/v2/eval_cb_golden_v2.jsonl")
    )
    parser.add_argument(
        "--external-eval", type=Path, default=Path("artifacts/data/eval_external_v2.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/train_realistic_v2_unreviewed.jsonl"),
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_realistic_rejects_v2.jsonl")
    )
    parser.add_argument("--max-target-distance", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
