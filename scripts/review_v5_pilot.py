"""Terminal reviewer for the 30-essay v5 pilot (hash-bound approval)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from apush_frq_grader_slm.dataset_v5 import (
    V5GenerationTask,
    attach_style_reference,
    build_pilot_approval,
    load_style_reference_essays,
    select_v5_pilot_tasks,
)
from apush_frq_grader_slm.io import read_jsonl


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


def _prompt(msg: str) -> str:
    return input(msg).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--essays", type=Path,
                        default=Path("artifacts/data/v5/private/pilot_essays_v5.jsonl"))
    parser.add_argument("--tasks", type=Path,
                        default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"))
    parser.add_argument("--seed-profiles", type=Path,
                        default=Path("artifacts/data/v5/planning/cb_seed_profiles_v5.jsonl"))
    parser.add_argument("--golden-cases", type=Path,
                        default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument("--approval", type=Path,
                        default=Path("artifacts/data/v5/private/pilot_approval_v5.json"))
    parser.add_argument("--decisions", type=Path,
                        default=Path("artifacts/data/v5/private/pilot_review_decisions_v5.json"))
    parser.add_argument("--reviewer", type=str, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument(
        "--write-approval-from-decisions",
        action="store_true",
        help="Non-interactive: build approval from an existing decisions JSON.",
    )
    args = parser.parse_args()

    references = load_style_reference_essays(args.seed_profiles, args.golden_cases)
    tasks = {
        task.task_id: attach_style_reference(task, references)
        for task in select_v5_pilot_tasks(
            [task_from_row(row) for row in read_jsonl(args.tasks)], seed=args.seed
        )
    }
    essays = {str(row["task_id"]): row for row in read_jsonl(args.essays)}
    if set(essays) != set(tasks):
        raise SystemExit(
            f"pilot essays must exactly match pilot tasks; "
            f"missing={sorted(set(tasks)-set(essays))[:5]} "
            f"extra={sorted(set(essays)-set(tasks))[:5]}"
        )

    if args.write_approval_from_decisions:
        decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
        approval = build_pilot_approval(
            reviewer=args.reviewer,
            approved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            pilot_essays_path=args.essays,
            decisions=decisions,
        )
        args.approval.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote pilot approval -> {args.approval}")
        return

    decisions: dict[str, str] = {}
    if args.decisions.exists():
        decisions = {
            str(k): str(v)
            for k, v in json.loads(args.decisions.read_text(encoding="utf-8")).items()
        }

    for index, task_id in enumerate(sorted(tasks), start=1):
        task = tasks[task_id]
        essay = str(essays[task_id].get("student_response") or "")
        print("\n" + "=" * 72)
        print(f"[{index}/30] {task_id}  class={task.coverage_class} "
              f"boundary={task.boundary_type or '-'} side={task.contrast_side or '-'}")
        print(f"period={task.period} capability={task.capability_profile}")
        print("- prompt -")
        print(task.prompt)
        print("- essay -")
        print(essay)
        print("- end -")
        current = decisions.get(task_id, "pending")
        print(f"current decision: {current}")
        while True:
            choice = _prompt("Decision [a]ccept / [r]eject / [s]kip / [q]uit: ")
            if choice in {"a", "accept"}:
                decisions[task_id] = "accept"
                break
            if choice in {"r", "reject"}:
                decisions[task_id] = "reject"
                break
            if choice in {"s", "skip"}:
                break
            if choice in {"q", "quit"}:
                args.decisions.write_text(
                    json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                print(f"Saved partial decisions -> {args.decisions}")
                return
            print("Enter a, r, s, or q.")
        args.decisions.write_text(
            json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if len(decisions) != 30 or any(v not in {"accept", "corrected"} for v in decisions.values()):
        rejected = [tid for tid, dec in decisions.items() if dec == "reject"]
        print(
            f"Pilot not fully accepted yet. decisions={len(decisions)}/30 "
            f"rejected={rejected}. Regenerate rejects, then re-run."
        )
        return

    approval = build_pilot_approval(
        reviewer=args.reviewer,
        approved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pilot_essays_path=args.essays,
        decisions=decisions,
    )
    args.approval.parent.mkdir(parents=True, exist_ok=True)
    args.approval.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"All 30 accepted. Wrote {args.approval}")


if __name__ == "__main__":
    main()
