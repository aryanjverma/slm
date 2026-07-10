"""Ingest AP Central PDFs into a real-essay SEED pool for generation.

Seeds are real CB essays used only as reference context for realistic synthetic
generation -- NEVER trained on directly and NEVER evaluated on. Every candidate
is deduped against the frozen 72-essay eval set, so the held-out essays (and
their near-duplicates) can never leak in. Survivors are tagged `seed_real`,
which `build_mixed_dataset.assert_no_real_essays` treats as real (defense in depth).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.golden import load_permission_record, require_permission
from apush_frq_grader_slm.ingest.apc_parser import parse_apc_pdf
from apush_frq_grader_slm.ingest.dedup import is_duplicate_essay
from apush_frq_grader_slm.ingest.distill import raw_sample_to_frq_case
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def build_seed_cases(
    input_dir: Path,
    frozen_eval_path: Path,
    *,
    seed: int = 13,
) -> tuple[list[FRQCase], list[dict]]:
    frozen = [FRQCase.model_validate(row) for row in read_jsonl(frozen_eval_path)]
    accepted: list[FRQCase] = []
    dropped: list[dict] = []
    pdfs = sorted(input_dir.glob("ap*-apc-us-history-leq*.pdf"))
    for pdf_path in pdfs:
        try:
            samples = parse_apc_pdf(pdf_path)
        except Exception as exc:
            dropped.append({"file": pdf_path.name, "reason": f"parse_error:{exc}"})
            continue
        for sample in samples:
            case = raw_sample_to_frq_case(
                sample, case_id=f"seed_real_{len(accepted):04d}", distill=False, seed=seed
            )
            # Dedup against the frozen eval set AND already-accepted seeds.
            if is_duplicate_essay(
                case.student_response, frozen + accepted, prompt=case.prompt
            ):
                dropped.append({"file": pdf_path.name, "sample_id": sample.sample_id,
                                "reason": "duplicate_of_frozen_or_seed"})
                continue
            ok, reasons = passes_quality_gate(case)
            if not ok:
                dropped.append({"file": pdf_path.name, "sample_id": sample.sample_id,
                                "reasons": reasons})
                continue
            if "seed_real" not in case.tags:
                case.tags.append("seed_real")
            accepted.append(case)
    return accepted, dropped


def main() -> None:
    args = parse_args()
    require_permission(load_permission_record(args.permission_record), "training")
    seeds, dropped = build_seed_cases(args.input, args.frozen_eval, seed=args.seed)
    write_jsonl(args.output, seeds)
    print(f"Wrote {len(seeds)} net-new seed essays to {args.output}")
    print(f"Dropped {len(dropped)} samples (duplicate of frozen eval / gate / parse errors)")
    if args.verbose:
        for row in dropped[:15]:
            print("  ", row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-essay seed pool (deduped vs frozen eval).")
    parser.add_argument("--input", type=Path, default=Path("artifacts/raw/ap_central"))
    parser.add_argument(
        "--frozen-eval", type=Path, default=Path("artifacts/data/eval_real_cases.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/seed_real_cases.jsonl")
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--permission-record",
        type=Path,
        default=Path("config/college_board_permission.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
