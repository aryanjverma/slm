"""Create recalibrated retry tasks from target-distance or quality rejects."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.synth_realistic import GenTask, recalibrate_task


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    retry_ids = {str(row["task_id"]) for row in read_jsonl(args.rejects)}
    retries = [recalibrate_task(tasks[task_id]) for task_id in sorted(retry_ids) if task_id in tasks]
    write_jsonl(args.output, [task.to_row() for task in retries])
    print(f"Wrote {len(retries)} recalibrated retry tasks to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_realistic_rejects_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/pilot/retry_tasks_v2.jsonl")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
