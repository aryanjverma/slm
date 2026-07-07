"""Create a targeted v2 dataset by oversampling known LEQ failure modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.data import generate_cases, to_chat_rows
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.schemas import FailureType


TARGET_FAILURES = {
    FailureType.GRADE_INFLATION_REQUEST,
    FailureType.PROMPT_INJECTION,
    FailureType.WEAK_THESIS,
    FailureType.WRONG_PERIOD,
}


def main() -> None:
    args = parse_args()
    pool = generate_cases(count=args.pool_size, split="train", seed=args.seed, adversarial_ratio=0.4)
    targeted = []
    for case in pool:
        if case.failure_type in TARGET_FAILURES:
            ok, _ = passes_quality_gate(case)
            if ok:
                case.id = f"v2-{len(targeted):05d}"
                targeted.append(case)
        if len(targeted) >= args.count:
            break

    write_jsonl(args.output_dir / "train_cases_v2.jsonl", targeted)
    write_jsonl(args.output_dir / "train_chat_v2.jsonl", to_chat_rows(targeted))
    print(f"Wrote {len(targeted)} targeted v2 cases to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


if __name__ == "__main__":
    main()
