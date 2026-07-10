"""Build immutable, audited v2 SFT artifacts from independently labeled candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.dataset_v2 import (
    ADVERSARIAL_TYPES,
    assemble_training_mix,
    audit_training_cases,
    write_training_artifacts,
)
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def _load(path: Path) -> list[FRQCase]:
    if not path.exists():
        return []
    return [FRQCase.model_validate(row) for row in read_jsonl(path)]


def main() -> None:
    args = parse_args()
    realistic = _load(args.realistic)
    adversarial = [
        case for case in _load(args.adversarial) if case.failure_type.value in ADVERSARIAL_TYPES
    ]
    forbidden = _load(args.golden) + _load(args.external)
    dev = _load(args.dev)
    challenge = _load(args.challenge)

    mix = assemble_training_mix(
        realistic,
        adversarial,
        target_count=args.target_count,
        adversarial_ratio=args.adversarial_ratio,
        seed=args.seed,
    )
    if len(mix) != args.target_count:
        raise RuntimeError(
            f"Only {len(mix)} eligible rows are available; target_count={args.target_count}"
        )
    audit = audit_training_cases(mix, forbidden_cases=forbidden)
    if not audit.clean:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(
            json.dumps(audit.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Dataset audit failed; see {args.audit_report}")

    manifest = write_training_artifacts(
        args.output_dir,
        mix,
        audit,
        seed=args.seed,
        settings={
            "target_count": args.target_count,
            "adversarial_ratio": args.adversarial_ratio,
            "realistic_input": str(args.realistic),
            "adversarial_input": str(args.adversarial),
        },
        additional_case_sets={
            **({"dev_synthetic_v2.jsonl": dev} if dev else {}),
            **({"eval_challenge_v2.jsonl": challenge} if challenge else {}),
            **({"eval_cb_golden_v2.jsonl": _load(args.golden)} if args.golden.exists() else {}),
            **({"eval_external_v2.jsonl": _load(args.external)} if args.external.exists() else {}),
        },
        force=args.force,
    )
    print(f"Wrote {len(mix)} cases and manifest with {len(manifest.artifacts)} artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realistic",
        type=Path,
        default=Path("artifacts/data/train_realistic_v2_reviewed.jsonl"),
    )
    parser.add_argument(
        "--adversarial", type=Path, default=Path("artifacts/data/train_cases.jsonl")
    )
    parser.add_argument(
        "--golden", type=Path, default=Path("artifacts/data/eval_cb_golden_v2.jsonl")
    )
    parser.add_argument(
        "--external", type=Path, default=Path("artifacts/data/eval_external_v2.jsonl")
    )
    parser.add_argument(
        "--dev", type=Path, default=Path("artifacts/data/dev_synthetic_v2.jsonl")
    )
    parser.add_argument(
        "--challenge", type=Path, default=Path("artifacts/data/eval_challenge_v2.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/v2"))
    parser.add_argument(
        "--audit-report", type=Path, default=Path("artifacts/audits/train_v2.json")
    )
    parser.add_argument("--target-count", type=int, default=1200)
    parser.add_argument("--adversarial-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
