"""Create or apply the required human-review sample for synthetic v2 labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.dataset_v2 import (
    SyntheticReviewEntry,
    apply_human_reviews,
    select_human_review_sample,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def main() -> None:
    args = parse_args()
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.input)]
    if args.create_template:
        selected = select_human_review_sample(cases, rate=args.rate, seed=args.seed)
        write_jsonl(args.review_log, [SyntheticReviewEntry(case_id=case.id) for case in selected])
        print(f"Wrote {len(selected)} review rows to {args.review_log}")
        return

    reviews = [SyntheticReviewEntry.model_validate(row) for row in read_jsonl(args.review_log)]
    reviewed = apply_human_reviews(cases, reviews)
    write_jsonl(args.output, reviewed)
    accepted = sum(case.labeling.human_reviewed for case in reviewed)
    print(f"Applied {accepted} accepted human reviews; wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/data/train_realistic_v2_unreviewed.jsonl")
    )
    parser.add_argument(
        "--review-log", type=Path, default=Path("artifacts/reviews/synthetic_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/train_realistic_v2_reviewed.jsonl")
    )
    parser.add_argument("--create-template", action="store_true")
    parser.add_argument("--rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    main()
