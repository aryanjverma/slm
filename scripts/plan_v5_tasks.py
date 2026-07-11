"""Plan the fixed 1,500-task, 30-shard v5 cloud generation campaign."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.dataset_v4 import load_seed_profiles
from apush_frq_grader_slm.dataset_v5 import plan_v5_tasks
from apush_frq_grader_slm.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=Path("artifacts/data/v4/cb_seed_profiles.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/v5/planning"))
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    tasks = plan_v5_tasks(load_seed_profiles(args.seeds), seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "generation_tasks_v5.jsonl", [task.to_row() for task in tasks])
    manifest = {
        "campaign": "v5-final", "candidate_count": 1500, "shard_count": 30,
        "shard_size": 50, "score_targets_visible_to_writer": False,
        "planned_coverage": dict(sorted(Counter(task.coverage_class for task in tasks).items())),
        "private_use": True, "seed": args.seed,
        "shards": dict(sorted(Counter(task.shard_id for task in tasks).items())),
    }
    (args.output_dir / "generation_manifest_v5.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Planned 1,500 tasks in 30 x 50 shards at {args.output_dir}")


if __name__ == "__main__":
    main()
