"""Merge raw essay batches and emit target-profile grounded grades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.grade_v4 import grade_payload_for_target
from apush_frq_grader_slm.io import read_jsonl


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}

    essays: dict[str, dict] = {}
    sources = sorted(args.essay_dir.glob("batch_*.jsonl"))
    if args.composed.exists():
        sources.append(args.composed)
    for path in sources:
        for row in read_jsonl(path):
            task_id = str(row["task_id"])
            essay = str(row.get("student_response") or "").strip()
            if task_id and essay and task_id not in essays:
                essays[task_id] = {
                    "task_id": task_id,
                    "student_response": essay,
                    "word_count": len(essay.split()),
                    "source_file": path.name,
                }

    missing = sorted(set(tasks) - set(essays))
    if missing:
        raise SystemExit(f"Missing essays for {len(missing)} tasks (e.g. {missing[:3]})")

    args.essays_out.parent.mkdir(parents=True, exist_ok=True)
    with args.essays_out.open("w", encoding="utf-8") as essay_file, args.grades_out.open(
        "w", encoding="utf-8"
    ) as grade_file:
        for task_id, task in tasks.items():
            essay_row = essays[task_id]
            essay_file.write(json.dumps(essay_row, ensure_ascii=True) + "\n")
            grade = grade_payload_for_target(essay_row["student_response"], task["target_scores"])
            grade_row = {"task_id": task_id, **grade}
            grade_file.write(json.dumps(grade_row, ensure_ascii=True) + "\n")

    print(f"Merged {len(essays)} essays -> {args.essays_out}")
    print(f"Wrote {len(tasks)} grades -> {args.grades_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/v4/synth_tasks_v4.jsonl")
    )
    parser.add_argument(
        "--essay-dir", type=Path, default=Path("artifacts/data/v4/raw_essays")
    )
    parser.add_argument(
        "--composed",
        type=Path,
        default=Path("artifacts/data/v4/raw_essays/composed_all.jsonl"),
    )
    parser.add_argument(
        "--essays-out",
        type=Path,
        default=Path("artifacts/data/v4/raw_essays_v4.jsonl"),
    )
    parser.add_argument(
        "--grades-out",
        type=Path,
        default=Path("artifacts/data/v4/grades_v4.jsonl"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
