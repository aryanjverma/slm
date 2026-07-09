"""Plan the realistic-essay generation worklist (deterministic).

Reads real seed essays (if present) + the canonical LEQ prompts, fans out
seeds x score-profiles x length-bands, and writes one task per row to
artifacts/data/synth_tasks.jsonl for the generation agent to consume.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.data import LEQ_PROMPTS
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.synth_realistic import plan_generation_tasks


def load_seeds(path: Path) -> list[FRQCase]:
    if not path.exists():
        return []
    return [FRQCase.model_validate(row) for row in read_jsonl(path)]


def main() -> None:
    args = parse_args()
    seeds = load_seeds(args.seeds)
    prompts = [entry["prompt"] for entry in LEQ_PROMPTS]
    tasks = plan_generation_tasks(seeds, prompts, variants_per_seed=args.variants)

    write_jsonl(args.output, [task.to_row() for task in tasks])

    by_total = Counter(task.target_total for task in tasks)
    by_ref = len({task.seed_id for task in tasks})
    real_refs = sum(1 for task in tasks if task.seed_essay_excerpt)
    print(f"Wrote {len(tasks)} generation tasks to {args.output}")
    print(f"  refs: {by_ref} ({'has real excerpts' if real_refs else 'prompt-only, no seeds'})")
    print(f"  seed essays loaded: {len(seeds)}")
    print("  target-total distribution: " + ", ".join(f"{t}:{by_total[t]}" for t in range(7)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan realistic-essay generation tasks.")
    parser.add_argument("--seeds", type=Path, default=Path("artifacts/data/seed_real_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/synth_tasks.jsonl"))
    parser.add_argument(
        "--variants",
        type=int,
        default=24,
        help="Tasks per ref (prompt/seed); total tasks = refs x variants.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
