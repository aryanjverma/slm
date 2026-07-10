"""Plan deterministic writer-only tasks for realistic, unlabeled LEQ candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.prompt_catalog import (
    PromptCatalogEntry,
    PromptSplit,
    build_default_prompt_catalog,
)
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.synth_realistic import plan_generation_tasks, select_balanced_task_pilot


def load_seeds(path: Path | None) -> list[FRQCase]:
    if path is None or not path.exists():
        return []
    return [FRQCase.model_validate(row) for row in read_jsonl(path)]


def main() -> None:
    args = parse_args()
    if args.seeds is not None and not args.allow_seed_excerpts:
        raise SystemExit(
            "Real essay excerpts require explicit permission; pass --allow-seed-excerpts only "
            "when your written permission record authorizes generation use."
        )
    seeds = load_seeds(args.seeds)
    if args.prompt_catalog.exists():
        catalog = [PromptCatalogEntry.model_validate(row) for row in read_jsonl(args.prompt_catalog)]
    else:
        catalog = build_default_prompt_catalog()
    selected = [entry for entry in catalog if entry.split == PromptSplit(args.prompt_split)]
    prompts = [entry.prompt_text for entry in selected]
    prompt_metadata = [
        {
            "prompt_family_id": entry.prompt_family_id,
            "period": entry.period,
            "reasoning_skill": entry.reasoning_skill.value,
        }
        for entry in selected
    ]
    tasks = plan_generation_tasks(
        seeds,
        prompts,
        variants_per_seed=args.variants,
        prompt_metadata=prompt_metadata,
        case_split=(
            "train"
            if args.prompt_split == PromptSplit.TRAIN.value
            else (
                "eval"
                if args.prompt_split == PromptSplit.SYNTHETIC_DEV.value
                else "adversarial"
            )
        ),
        prompt_split=args.prompt_split,
    )
    if args.limit is not None:
        tasks = select_balanced_task_pilot(tasks, args.limit)

    write_jsonl(args.output, [task.to_row() for task in tasks])

    by_total = Counter(task.target_total for task in tasks)
    by_time = Counter(task.persona.time_budget_minutes for task in tasks)
    by_knowledge = Counter(task.persona.historical_knowledge for task in tasks)
    by_task_type = Counter(task.task_type for task in tasks)
    by_ref = len({task.seed_id for task in tasks})
    real_refs = sum(1 for task in tasks if task.seed_essay_excerpt)
    print(f"Wrote {len(tasks)} generation tasks to {args.output}")
    print(f"  refs: {by_ref} ({'has real excerpts' if real_refs else 'prompt-only, no seeds'})")
    print(f"  seed essays loaded: {len(seeds)}")
    print("  target-total distribution: " + ", ".join(f"{t}:{by_total[t]}" for t in range(7)))
    print("  time budgets: " + ", ".join(f"{key}:{by_time[key]}" for key in sorted(by_time)))
    print(
        "  knowledge profiles: "
        + ", ".join(f"{key}:{by_knowledge[key]}" for key in sorted(by_knowledge))
    )
    print("  task types: " + ", ".join(f"{key}:{by_task_type[key]}" for key in sorted(by_task_type)))
    audit_path = args.audit_output or args.output.with_name(f"{args.output.stem}_audit.json")
    payload = args.output.read_bytes()
    audit = {
        "task_count": len(tasks),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "prompt_split": args.prompt_split,
        "prompt_family_count": len({task.prompt_family_id for task in tasks}),
        "target_totals": {str(total): by_total[total] for total in range(7)},
        "time_budgets": {str(key): by_time[key] for key in sorted(by_time)},
        "knowledge_profiles": {key: by_knowledge[key] for key in sorted(by_knowledge)},
        "task_types": {key: by_task_type[key] for key in sorted(by_task_type)},
        "seed_essay_excerpt_count": real_refs,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  audit: {audit_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan realistic-essay generation tasks.")
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--allow-seed-excerpts", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--prompt-catalog",
        type=Path,
        default=Path("artifacts/data/prompt_catalog_v1.jsonl"),
    )
    parser.add_argument(
        "--prompt-split",
        choices=[split.value for split in PromptSplit],
        default=PromptSplit.TRAIN.value,
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=24,
        help="Tasks per ref (prompt/seed); total tasks = refs x variants.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
