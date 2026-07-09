"""Build synthetic training dataset; real AP essays stay eval-only."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.data import generate_cases, to_chat_rows
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


REAL_EVAL_TAGS = ("ap_central", "real_eval", "tom_richey", "quizlet", "seed_real")


def build_train_cases(
    *,
    train_count: int,
    seed: int,
    adversarial_ratio: float,
    id_prefix: str = "train",
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
        case.id = f"{id_prefix}-{len(accepted):05d}"
        accepted.append(case)
        if len(accepted) >= train_count:
            break
    if len(accepted) < train_count:
        raise RuntimeError(f"Only generated {len(accepted)} quality, need {train_count}")
    return accepted


def build_mixed_train_cases(
    *,
    train_count: int,
    realistic_cases: list[FRQCase],
    seed: int,
    realistic_ratio: float = 0.65,
    adversarial_ratio: float = 0.15,
) -> list[FRQCase]:
    """Mix agent-generated realistic cases with template synthetic cases.

    Realistic essays teach the model the real input distribution; the template
    portion keeps the adversarial slices (prompt injection / grade-inflation)
    so litmus robustness does not regress. Template gets `adversarial_ratio` of
    the total; the remainder of the non-realistic budget is standard slices.
    """
    realistic_target = round(train_count * realistic_ratio)
    realistic_used = [_gate_pass(c) for c in realistic_cases[:realistic_target]]
    realistic_used = [c for c in realistic_used if c is not None]

    template_count = train_count - len(realistic_used)
    if template_count <= 0:
        return realistic_used

    # Within the template budget, hit `adversarial_ratio` of the *total* count.
    adv_within_template = min(1.0, (adversarial_ratio * train_count) / template_count)
    template_cases = build_train_cases(
        train_count=template_count,
        seed=seed,
        adversarial_ratio=adv_within_template,
        id_prefix="train-tmpl",
    )
    return realistic_used + template_cases


def _gate_pass(case: FRQCase) -> FRQCase | None:
    ok, _ = passes_quality_gate(case)
    return case if ok else None


def assert_no_real_essays(cases: list[FRQCase]) -> None:
    for case in cases:
        if case.split != "train":
            raise ValueError(f"Training case {case.id} has split={case.split}")
        if any(tag in case.tags for tag in REAL_EVAL_TAGS):
            raise ValueError(f"Real essay leaked into train set: {case.id}")
        if case.id.startswith(("ap_central_", "tom_richey_", "quizlet_", "real_eval")):
            raise ValueError(f"Real essay id in train set: {case.id}")


def main() -> None:
    args = parse_args()
    realistic_cases: list[FRQCase] = []
    if args.realistic_path.exists():
        realistic_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.realistic_path)]

    if realistic_cases:
        train_cases = build_mixed_train_cases(
            train_count=args.train_count,
            realistic_cases=realistic_cases,
            seed=args.seed,
            realistic_ratio=args.realistic_ratio,
            adversarial_ratio=args.adversarial_ratio,
        )
        realistic_n = sum(1 for c in train_cases if "synth_realistic" in c.tags)
        print(f"Mixed: {realistic_n} realistic + {len(train_cases) - realistic_n} template")
    else:
        print(f"No realistic cases at {args.realistic_path}; building synthetic-only train set")
        train_cases = build_train_cases(
            train_count=args.train_count,
            seed=args.seed,
            adversarial_ratio=args.adversarial_ratio,
        )
    assert_no_real_essays(train_cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_name = args.output_name.replace("chat", "cases")
    write_jsonl(args.output_dir / cases_name, train_cases)
    write_jsonl(args.output_dir / args.output_name, to_chat_rows(train_cases))

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
    parser.add_argument("--output-name", type=str, default="train_chat.jsonl")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--adversarial-ratio", type=float, default=0.25)
    parser.add_argument(
        "--realistic-path",
        type=Path,
        default=Path("artifacts/data/train_realistic_cases.jsonl"),
        help="Agent-generated realistic cases to mix in (synthetic-only if absent).",
    )
    parser.add_argument(
        "--realistic-ratio",
        type=float,
        default=0.65,
        help="Fraction of the train set drawn from realistic cases.",
    )
    parser.add_argument(
        "--real-eval-path",
        type=Path,
        default=Path("artifacts/data/eval_real_cases.jsonl"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
