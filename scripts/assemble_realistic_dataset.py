"""Validate agent-generated raw essays into the realistic training slice.

Joins agent output (synth_realistic_raw.jsonl) to the planned tasks
(synth_tasks.jsonl), applies label discipline + rubric validity + the quality
gate + anti-leakage checks, and writes accepted cases to
artifacts/data/train_realistic_cases.jsonl (rejects -> a sidecar log).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.synth_realistic import (
    GenTask,
    parse_agent_row,
    validate_generated_case,
)


def load_cases(path: Path) -> list[FRQCase]:
    if not path.exists():
        return []
    return [FRQCase.model_validate(row) for row in read_jsonl(path)]


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    leakage_sources = load_cases(args.seeds) + load_cases(args.frozen_eval)
    if not leakage_sources:
        print("WARNING: no seed/frozen-eval essays loaded; anti-leakage check is weak.")

    accepted: list[FRQCase] = []
    rejects: list[dict] = []
    for row in read_jsonl(args.raw):
        task_id = row.get("task_id")
        task = tasks.get(task_id)
        if task is None:
            rejects.append({"task_id": task_id, "reasons": ["unknown_task_id"]})
            continue
        try:
            case = parse_agent_row(row, task)
        except Exception as exc:  # malformed structure from the agent
            rejects.append({"task_id": task_id, "reasons": [f"parse_error:{exc}"]})
            continue
        ok, reasons = validate_generated_case(case, task, row, leakage_sources)
        if ok:
            accepted.append(case)
        else:
            rejects.append({"task_id": task_id, "reasons": reasons})

    write_jsonl(args.output, accepted)
    if rejects:
        write_jsonl(args.rejects, rejects)

    _report(accepted, rejects)


def _report(accepted: list[FRQCase], rejects: list[dict]) -> None:
    print(f"Accepted {len(accepted)} realistic cases; rejected {len(rejects)}.")
    if rejects:
        reason_counts: Counter[str] = Counter()
        for row in rejects:
            reason_counts.update(row["reasons"])
        print("  reject reasons: " + ", ".join(f"{r}:{c}" for r, c in reason_counts.most_common()))
    if not accepted:
        return
    totals = Counter(c.reference_scores.total for c in accepted)
    print("  total distribution: " + ", ".join(f"{t}:{totals[t]}" for t in range(7)))
    words = sorted(len(c.student_response.split()) for c in accepted)
    mid = words[len(words) // 2]
    print(f"  essay words: min={words[0]} median={mid} max={words[-1]}")
    prompts = {c.prompt for c in accepted}
    degenerate = sum(
        1
        for p in prompts
        if len({c.reference_scores.total for c in accepted if c.prompt == p}) <= 1
    )
    print(f"  prompts: {len(prompts)} distinct; {degenerate} with a single total (degenerate)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble realistic training cases from agent output.")
    parser.add_argument("--tasks", type=Path, default=Path("artifacts/data/synth_tasks.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("artifacts/data/synth_realistic_raw.jsonl"))
    parser.add_argument("--seeds", type=Path, default=Path("artifacts/data/seed_real_cases.jsonl"))
    parser.add_argument(
        "--frozen-eval", type=Path, default=Path("artifacts/data/eval_real_cases.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/train_realistic_cases.jsonl")
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_realistic_rejects.jsonl")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
