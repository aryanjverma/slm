"""Join validated candidates to prompts while excluding all writer controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.synth_realistic import GenTask, parse_candidate_row


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    packets: list[dict[str, str]] = []
    for row in read_jsonl(args.raw):
        task_id = str(row.get("task_id", ""))
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")
        candidate = parse_candidate_row(row, task)
        packets.append(
            {
                "task_id": task_id,
                "prompt": candidate.prompt,
                "student_response": candidate.student_response,
                "rubric_version": candidate.rubric_version,
            }
        )
    write_jsonl(args.output, packets)
    print(f"Wrote {len(packets)} anonymous grading packets to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
