"""Run deterministic hard gates on pilot essays before human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.authenticity_gates_v5 import hard_gate_reasons
from apush_frq_grader_slm.dataset_v5 import (
    V5GenerationTask,
    attach_style_reference,
    load_style_reference_essays,
    select_v5_pilot_tasks,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl


def task_from_row(row: dict) -> V5GenerationTask:
    return V5GenerationTask(
        task_id=row["task_id"],
        shard_id=row["shard_id"],
        prompt=row["prompt"],
        prompt_family_id=row["prompt_family_id"],
        style_seed_id=row["style_seed_id"],
        style_excerpt=row.get("style_excerpt", ""),
        period=row.get("period"),
        reasoning_skill=row.get("reasoning_skill", ""),
        capability_profile=dict(row["capability_profile"]),
        composition_profile=dict(row["composition_profile"]),
        amsco_chapter_ids=tuple(row.get("amsco_chapter_ids") or ()),
        coverage_class=row.get("coverage_class", "golden_matched"),
        boundary_type=row.get("boundary_type", ""),
        contrast_pair_id=row.get("contrast_pair_id", ""),
        contrast_side=row.get("contrast_side", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path,
                        default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"))
    parser.add_argument("--essays", type=Path, required=True)
    parser.add_argument("--seed-profiles", type=Path,
                        default=Path("artifacts/data/v5/planning/cb_seed_profiles_v5.jsonl"))
    parser.add_argument("--golden-cases", type=Path,
                        default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    references = load_style_reference_essays(args.seed_profiles, args.golden_cases)
    pilot_tasks = {
        task.task_id: attach_style_reference(task, references)
        for task in select_v5_pilot_tasks(
            [task_from_row(row) for row in read_jsonl(args.tasks)], seed=args.seed
        )
    }
    essays = {str(row["task_id"]): row for row in read_jsonl(args.essays)}
    missing = sorted(set(pilot_tasks) - set(essays))
    extra = sorted(set(essays) - set(pilot_tasks))
    rejected: dict[str, list[str]] = {}
    accepted: list[dict] = []
    for task_id, task in sorted(pilot_tasks.items()):
        row = essays.get(task_id)
        if row is None:
            rejected[task_id] = ["missing_pilot_essay"]
            continue
        essay = str(row.get("student_response") or "")
        reasons = hard_gate_reasons(
            essay,
            style_reference_essay=task.style_reference_essay,
            reference_word_count=task.reference_word_count,
        )
        if reasons:
            rejected[task_id] = reasons
            continue
        accepted.append({"task_id": task_id, "student_response": essay})

    audit = {
        "pilot_count": len(pilot_tasks),
        "returned": len(essays),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "missing_task_ids": missing,
        "extra_task_ids": extra,
        "rejection_reasons": {
            task_id: reasons for task_id, reasons in sorted(rejected.items())
        },
        "ready_for_human_review": len(accepted) == len(pilot_tasks) and not missing and not extra,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.accepted_output is not None:
        write_jsonl(args.accepted_output, accepted)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["ready_for_human_review"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
