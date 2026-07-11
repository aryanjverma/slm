"""Build v4 College Board seed profiles from eval_cb_cases.jsonl.

Writes structural/score/style metadata only — never full CB essays into
training artifacts. Style excerpts are short cleaned prefixes when usable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.dataset_v4_seeds import write_v4_seed_artifacts


def main() -> None:
    args = parse_args()
    result = write_v4_seed_artifacts(
        cases_path=args.input,
        output_dir=args.output_dir,
    )
    summary = result["summary"]
    print(f"Wrote {result['n_profiles']} seed profiles → {result['profiles_path']}")
    print(f"Wrote {result['n_prompt_families']} prompt families → {result['families_path']}")
    print(f"Summary → {result['summary_path']}")
    print(json.dumps(
        {
            "n_profiles": summary["n_profiles"],
            "n_prompt_families": summary["n_prompt_families"],
            "by_total": summary["by_total"],
            "by_period": summary["by_period"],
            "by_reasoning_skill": summary["by_reasoning_skill"],
            "style_usable": summary["style_usable"],
            "example_seed_ids": summary["seed_ids"][:5],
        },
        indent=2,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/data/eval_cb_cases.jsonl"),
        help="CB eval cases JSONL (default: artifacts/data/eval_cb_cases.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v4"),
        help="Output directory for seed profiles (default: artifacts/data/v4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
