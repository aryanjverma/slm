"""Plan 250 CB-seeded v4 synthetic generation tasks."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.dataset_v4 import (
    load_prompt_families,
    load_seed_profiles,
    plan_v4_tasks,
)
from apush_frq_grader_slm.io import write_jsonl


def main() -> None:
    args = parse_args()
    if not args.seeds.exists():
        print(
            f"ERROR: CB seed profiles not found at {args.seeds}. "
            "Generate artifacts/data/v4/cb_seed_profiles.jsonl first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    profiles = load_seed_profiles(args.seeds)
    families = load_prompt_families(args.prompt_families)
    tasks = plan_v4_tasks(
        profiles,
        families,
        target_count=args.target_count,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, [task.to_row() for task in tasks])

    by_total = Counter(task.target_total for task in tasks)
    by_seed = Counter(task.seed_id for task in tasks)
    by_time = Counter(task.persona.time_budget_minutes for task in tasks)
    by_knowledge = Counter(task.persona.historical_knowledge for task in tasks)
    print(f"Wrote {len(tasks)} v4 tasks to {args.output}")
    print(f"  seeds used: {len(by_seed)}")
    print("  target-total distribution: " + ", ".join(f"{t}:{by_total[t]}" for t in range(7)))
    print("  time budgets: " + ", ".join(f"{k}:{by_time[k]}" for k in sorted(by_time)))
    print(
        "  knowledge: "
        + ", ".join(f"{k}:{by_knowledge[k]}" for k in sorted(by_knowledge))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("artifacts/data/v4/cb_seed_profiles.jsonl"),
        help="CB seed profiles JSONL (required).",
    )
    parser.add_argument(
        "--prompt-families",
        type=Path,
        default=Path("artifacts/data/v4/prompt_families.jsonl"),
        help="Optional prompt-family catalog with adapted prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/v4/synth_tasks_v4.jsonl"),
    )
    parser.add_argument("--target-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
