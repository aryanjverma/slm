"""Hard-gate production essay returns (meta/process, copy limits, length)."""

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
    parser.add_argument("--essays", type=Path, required=True,
                        help="JSONL or directory of JSONL essay shards")
    parser.add_argument("--seed-profiles", type=Path,
                        default=Path("artifacts/data/v5/planning/cb_seed_profiles_v5.jsonl"))
    parser.add_argument("--golden-cases", type=Path,
                        default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, default=None)
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--exclude-pilot", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    references = load_style_reference_essays(args.seed_profiles, args.golden_cases)
    all_tasks = [task_from_row(row) for row in read_jsonl(args.tasks)]
    if args.exclude_pilot:
        pilot_ids = {t.task_id for t in select_v5_pilot_tasks(all_tasks, seed=args.seed)}
        tasks = {
            t.task_id: attach_style_reference(t, references)
            for t in all_tasks
            if t.task_id not in pilot_ids
        }
    else:
        tasks = {t.task_id: attach_style_reference(t, references) for t in all_tasks}

    essay_rows: list[dict] = []
    essay_path = args.essays
    if essay_path.is_dir():
        for path in sorted(essay_path.glob("*.jsonl")):
            essay_rows.extend(read_jsonl(path))
    else:
        essay_rows = read_jsonl(essay_path)
    essays = {str(row["task_id"]): row for row in essay_rows if row.get("task_id")}

    missing = sorted(set(tasks) - set(essays))
    extra = sorted(set(essays) - set(tasks))
    rejected: dict[str, list[str]] = {}
    accepted: list[dict] = []
    for task_id, task in sorted(tasks.items()):
        row = essays.get(task_id)
        if row is None:
            rejected[task_id] = ["missing_essay"]
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
        accepted.append(
            {
                "task_id": task_id,
                "student_response": essay,
                "shard_id": task.shard_id,
                "generation_attempt": row.get("generation_attempt", 1),
            }
        )

    audit = {
        "expected": len(tasks),
        "returned": len(set(essays) & set(tasks)),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "missing_task_ids_count": len(missing),
        "extra_task_ids_count": len(extra),
        "missing_task_ids_sample": missing[:20],
        "rejection_reason_counts": {},
        "ready_for_judging": len(accepted) == len(tasks) and not missing and not extra,
    }
    reason_counts: dict[str, int] = {}
    for reasons in rejected.values():
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    audit["rejection_reason_counts"] = dict(sorted(reason_counts.items()))
    if args.rejected_output is not None:
        write_jsonl(
            args.rejected_output,
            [
                {"task_id": task_id, "reasons": reasons}
                for task_id, reasons in sorted(rejected.items())
            ],
        )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.accepted_output is not None:
        write_jsonl(args.accepted_output, accepted)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["ready_for_judging"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
