"""Compose AMSCO-grounded v4 essays (full set or fill missing task_ids)."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.compose_v4 import compose_essay, rng_for_task
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.knowledge.amsco import facts_for_prompt, load_kb


def _existing_task_ids(existing_dir: Path) -> set[str]:
    found: set[str] = set()
    if not existing_dir.is_dir():
        return found
    for path in sorted(existing_dir.glob("batch_*.jsonl")):
        for row in read_jsonl(path):
            task_id = row.get("task_id")
            if task_id:
                found.add(str(task_id))
    return found


def compose_rows(
    tasks: list[dict],
    kb: list[dict],
    *,
    only_missing: bool,
    existing_dir: Path | None,
) -> list[dict]:
    skip = _existing_task_ids(existing_dir) if only_missing and existing_dir else set()
    rows: list[dict] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in skip:
            continue
        bundle = facts_for_prompt(kb, task["prompt"], max_facts=30)
        # Prefer chapter ids already attached to the task when present.
        if task.get("amsco_chapter_ids"):
            bundle = {**bundle, "chapter_ids": list(task["amsco_chapter_ids"])}
        rng = rng_for_task(task_id)
        essay = compose_essay(task, bundle, rng)
        rows.append(
            {
                "task_id": task_id,
                "student_response": essay,
                "word_count": len(essay.split()),
                "composer": "amsco_v4",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks)
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]
    kb = load_kb(args.kb)
    rows = compose_rows(
        tasks,
        kb,
        only_missing=args.only_missing,
        existing_dir=args.existing_dir if args.only_missing else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} composed essays -> {args.output}")
    if args.smoke and rows:
        for row in rows[:5]:
            text = row["student_response"]
            preview = text[:100].replace("\n", " ")
            print(f"{row['task_id']}: words={row['word_count']} | {preview!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/data/v4/synth_tasks_v4.jsonl"),
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=Path("artifacts/knowledge/amsco_2016_kb.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/v4/raw_essays/composed_all.jsonl"),
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only compose task_ids not already present in batch_*.jsonl under --existing-dir.",
    )
    parser.add_argument(
        "--existing-dir",
        type=Path,
        default=Path("artifacts/data/v4/raw_essays"),
        help="Directory scanned for batch_*.jsonl when --only-missing is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of tasks read (for smoke tests).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Print word counts and first 100 chars for up to 5 essays.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
