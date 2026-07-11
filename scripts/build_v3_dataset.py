"""Build the immutable, balanced, strictly audited v3 training artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from apush_frq_grader_slm.dataset_v3 import (
    assign_compatible_rubric_versions,
    audit_v3_training_cases,
    select_balanced_v3_cases,
    write_v3_dataset,
)
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def main() -> None:
    args = parse_args()
    candidates: list[FRQCase] = []
    for path in args.sources:
        for row in read_jsonl(path):
            case = FRQCase.model_validate(row)
            if (
                path.as_posix() == "artifacts/data/train_cases.jsonl"
                and case.provenance.source_type == "unknown"
            ):
                case.provenance.source_type = "synthetic"
                case.provenance.source_id = case.id
                case.provenance.generator_name = "legacy_template_v1"
                case.provenance.generator_config = {
                    **case.provenance.generator_config,
                    "v3_legacy_source": path.as_posix(),
                }
                case.provenance.review_status = "machine_checked"
                case.labeling.method = "rule_based"
                case.labeling.confidence = 1.0
            candidates.append(case)
    candidates = [
        case
        for case in candidates
        if not audit_v3_training_cases([case], min_long_essay_rate=0).rejected
    ]
    selected = select_balanced_v3_cases(candidates, target_count=args.target_count, seed=args.seed)
    selected = assign_compatible_rubric_versions(selected)
    audit = audit_v3_training_cases(selected, min_long_essay_rate=args.min_long_essay_rate)
    settings = {
        "seed": args.seed,
        "target_count": args.target_count,
        "min_long_essay_rate": args.min_long_essay_rate,
        "sources": [
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.sources
        ],
    }
    manifest = write_v3_dataset(args.output_dir, selected, audit, settings=settings)
    print(f"Wrote {manifest['rows']} audited rows to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-long-essay-rate", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/v3"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
