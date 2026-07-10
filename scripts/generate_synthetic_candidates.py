"""Generate realistic, unlabeled student essays from deterministic persona tasks."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.synth_realistic import (
    GenTask,
    parse_writer_response,
    render_generation_prompt,
)


def _request(client: Any, model: str, task: GenTask) -> tuple[str, str]:
    response = client.responses.create(model=model, input=render_generation_prompt(task))
    candidate = parse_writer_response(response.output_text, task)
    return task.task_id, json.dumps(candidate.to_row(), ensure_ascii=True)


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the 'judge' extra to generate candidates.") from exc

    tasks = [GenTask.from_row(row) for row in read_jsonl(args.tasks)]
    existing_ids: set[str] = set()
    if args.resume and args.output.exists():
        existing_ids = {str(row.get("task_id")) for row in read_jsonl(args.output)}
    pending = [task for task in tasks if task.task_id not in existing_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejects: list[dict[str, str]] = []
    mode = "a" if args.resume else "w"
    client = OpenAI()
    with args.output.open(mode, encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_request, client, args.model, task): task for task in pending
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    _, row = future.result()
                    output_file.write(row + "\n")
                    output_file.flush()
                except Exception as exc:
                    rejects.append({"task_id": task.task_id, "reason": f"generation_error:{exc}"})

    if rejects:
        args.rejects.parent.mkdir(parents=True, exist_ok=True)
        args.rejects.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rejects),
            encoding="utf-8",
        )
    print(
        f"Generated {len(pending) - len(rejects)} candidates; "
        f"rejected {len(rejects)}; output={args.output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/synth_realistic_raw_v2.jsonl")
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_generation_rejects_v2.jsonl")
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
