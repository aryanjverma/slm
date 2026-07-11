"""Blind-judge raw v5 essays into external candidate JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.judge_v5 import judge_essay


def _load_tasks(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    return {str(row["task_id"]): row for row in read_jsonl(path) if row.get("task_id")}


def _essay_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    out: list[dict] = []
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        essay = str(row.get("student_response") or row.get("essay") or "").strip()
        if task_id and essay:
            out.append(row)
    return out


def main() -> None:
    args = parse_args()
    tasks = _load_tasks(args.tasks)
    essays = _essay_rows(args.essays)

    if args.shard:
        shard = str(args.shard)
        filtered: list[dict] = []
        for row in essays:
            task_id = str(row["task_id"])
            row_shard = str(row.get("shard_id") or "")
            task = tasks.get(task_id) or {}
            task_shard = str(task.get("shard_id") or "")
            if row_shard == shard or task_shard == shard or shard in task_id:
                filtered.append(row)
        essays = filtered

    judged: list[dict] = []
    missing_prompt: list[str] = []
    for row in essays:
        task_id = str(row["task_id"])
        task = tasks.get(task_id) or {}
        prompt = str(row.get("prompt") or task.get("prompt") or "").strip()
        essay = str(row.get("student_response") or row.get("essay") or "").strip()
        if not prompt:
            missing_prompt.append(task_id)
            continue
        judged.append(judge_essay(prompt, essay, task_id=task_id))

    if missing_prompt:
        raise SystemExit(
            f"Missing prompts for {len(missing_prompt)} essays "
            f"(e.g. {missing_prompt[:3]}); pass --tasks or include prompt on rows"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard:
        out_name = f"judged_{args.shard}.jsonl"
    else:
        out_name = args.output_name
    out_path = args.output_dir / out_name
    write_jsonl(out_path, judged)
    summary = {
        "essays_in": len(essays),
        "judged": len(judged),
        "shard": args.shard,
        "output": str(out_path),
        "adjudicated": sum(1 for row in judged if row["resolved_grade"]["adjudicated"]),
        "fact_check_failed": sum(1 for row in judged if not row["fact_check"]["passed"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--essays",
        type=Path,
        required=True,
        help="JSONL with task_id + student_response (prompt optional if --tasks given)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"),
        help="Task JSONL used to supply prompts / shard_id when missing on essay rows",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v5/private/judged"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="judged_candidates.jsonl",
        help="Filename when --shard is not set",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Optional shard id (e.g. v5-shard-00) for parallel-friendly outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
