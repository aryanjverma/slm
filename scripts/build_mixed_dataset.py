"""Build synthetic training dataset; real AP essays stay eval-only."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.data import generate_cases, to_chat_rows
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


REAL_SOURCE_PREFIXES = ("ap_central_", "real_eval")


def build_train_cases(
    *,
    train_count: int,
    seed: int,
    adversarial_ratio: float,
) -> list[FRQCase]:
    pool_size = max(train_count * 3, train_count + 200)
    pool = generate_cases(
        count=pool_size,
        split="train",
        seed=seed,
        adversarial_ratio=adversarial_ratio,
    )
    accepted: list[FRQCase] = []
    for case in pool:
        ok, _ = passes_quality_gate(case)
        if not ok:
            continue
        case.id = f"train-{len(accepted):05d}"
        accepted.append(case)
        if len(accepted) >= train_count:
            break
    if len(accepted) < train_count:
        raise RuntimeError(f"Only generated {len(accepted)} quality, need {train_count}")
    return accepted


def assert_no_real_essays(cases: list[FRQCase]) -> None:
    for case in cases:
        if case.split != "train":
            raise ValueError(f"Training case {case.id} has split={case.split}")
        if "ap_central" in case.tags or "real_eval" in case.tags:
            raise ValueError(f"Real essay leaked into train set: {case.id}")
        if any(case.id.startswith(prefix) for prefix in REAL_SOURCE_PREFIXES):
            raise ValueError(f"Real essay id in train set: {case.id}")


def main() -> None:
    args = parse_args()
    train_cases = build_train_cases(
        train_count=args.train_count,
        seed=args.seed,
        adversarial_ratio=args.adversarial_ratio,
    )
    assert_no_real_essays(train_cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_cases.jsonl", train_cases)
    write_jsonl(args.output_dir / "train_chat.jsonl", to_chat_rows(train_cases))

    real_path = args.real_eval_path
    if real_path.exists():
        real_cases = [FRQCase.model_validate(row) for row in read_jsonl(real_path)]
        for case in real_cases:
            if case.split != "eval":
                raise ValueError(f"Real eval case {case.id} must have split=eval")
        write_jsonl(args.output_dir / "eval_real_cases.jsonl", real_cases)
        write_jsonl(args.output_dir / "eval_real_chat.jsonl", to_chat_rows(real_cases))
        print(f"Copied {len(real_cases)} real eval cases from {real_path}")
    else:
        print(f"No real eval file at {real_path}; run scripts/ingest_ap_essays.py first")

    print(f"Wrote {len(train_cases)} synthetic train cases to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed dataset (synthetic train, real eval).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--adversarial-ratio", type=float, default=0.25)
    parser.add_argument(
        "--real-eval-path",
        type=Path,
        default=Path("artifacts/data/eval_real_cases.jsonl"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
