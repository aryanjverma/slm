"""Build adapted LEQ prompt families and v5 seed profiles with adapted_prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.adapted_prompts_v5 import (
    attach_adapted_prompts_to_seeds,
    build_adapted_prompt_family_row,
)
from apush_frq_grader_slm.fact_cards_v5 import attach_amsco_chapter_ids_to_seeds
from apush_frq_grader_slm.io import read_jsonl, write_jsonl


def main() -> None:
    args = parse_args()
    families = read_jsonl(args.families)
    seeds = read_jsonl(args.seeds)
    adapted_families = [
        build_adapted_prompt_family_row(family, count=args.count) for family in families
    ]
    adapted_seeds = attach_adapted_prompts_to_seeds(seeds, adapted_families)
    adapted_seeds = attach_amsco_chapter_ids_to_seeds(adapted_seeds, kb_path=args.kb)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families_path = args.output_dir / "prompt_families_v5.jsonl"
    seeds_path = args.output_dir / "cb_seed_profiles_v5.jsonl"
    write_jsonl(families_path, adapted_families)
    write_jsonl(seeds_path, adapted_seeds)
    summary = {
        "n_families": len(adapted_families),
        "n_seeds": len(adapted_seeds),
        "adapted_per_family": args.count,
        "seeds_with_adapted_prompts": sum(
            1 for row in adapted_seeds if row.get("adapted_prompts")
        ),
        "seeds_with_amsco_chapter_ids": sum(
            1 for row in adapted_seeds if row.get("amsco_chapter_ids")
        ),
        "mean_amsco_chapters_per_seed": (
            sum(len(row.get("amsco_chapter_ids") or ()) for row in adapted_seeds)
            / max(1, len(adapted_seeds))
        ),
        "families_path": str(families_path),
        "seeds_path": str(seeds_path),
        "contains_essays": False,
    }
    (args.output_dir / "adapted_prompts_v5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("artifacts/data/v4/cb_seed_profiles.jsonl"),
        help="v4 CB seed profiles JSONL",
    )
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("artifacts/data/v4/prompt_families_v4.jsonl"),
        help="v4 prompt-family catalog JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v5/planning"),
        help="Directory for prompt_families_v5.jsonl and cb_seed_profiles_v5.jsonl",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Adapted prompts per family (2–3; default 3)",
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=None,
        help="Optional AMSCO KB JSONL path (default: artifacts/knowledge/amsco_2016_kb.jsonl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
