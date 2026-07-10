"""Summarize accepted v2 pilot cases and remaining release gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.dataset_v2 import audit_training_cases
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def _load_cases(path: Path) -> list[FRQCase]:
    return [FRQCase.model_validate(row) for row in read_jsonl(path)] if path.exists() else []


def main() -> None:
    args = parse_args()
    cases = _load_cases(args.cases)
    forbidden = _load_cases(args.golden) + _load_cases(args.external)
    audit = audit_training_cases(cases, forbidden_cases=forbidden)
    target_matrix: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    for case in cases:
        target = case.labeling.generation_target_total
        target_matrix[f"{target}->{case.reference_scores.total}"] += 1
        resolutions[case.labeling.resolution or "unknown"] += 1
    words = sorted(len(case.student_response.split()) for case in cases)
    remaining_gates = list(audit.global_reasons)
    if not args.golden.exists():
        remaining_gates.append("college_board_permission_unresolved")
    report = {
        "accepted_case_count": len(cases),
        "ready_for_training_artifact": audit.clean,
        "audit": audit.model_dump(mode="json"),
        "target_to_consensus": dict(sorted(target_matrix.items())),
        "resolutions": dict(sorted(resolutions.items())),
        "essay_words": {
            "minimum": words[0] if words else 0,
            "median": words[len(words) // 2] if words else 0,
            "maximum": words[-1] if words else 0,
        },
        "remaining_gates": remaining_gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Pilot cases={len(cases)}, audit_rejects={len(audit.rejected)}, "
        f"human_review_rate={audit.human_review_rate:.2f}, ready={audit.clean}"
    )
    print(f"Report: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("artifacts/data/train_realistic_v2_unreviewed.jsonl")
    )
    parser.add_argument(
        "--golden", type=Path, default=Path("artifacts/data/v2/eval_cb_golden_v2.jsonl")
    )
    parser.add_argument(
        "--external", type=Path, default=Path("artifacts/data/eval_external_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audits/pilot_v2.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
