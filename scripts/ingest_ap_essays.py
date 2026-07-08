"""Ingest College Board AP essays into eval-only FRQCase JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.apc_parser import parse_apc_pdf
from apush_frq_grader_slm.ingest.distill import raw_sample_to_frq_case
from apush_frq_grader_slm.io import write_jsonl


def ingest_directory(
    input_dir: Path,
    *,
    distill: bool = False,
    seed: int = 13,
) -> tuple[list, list[dict]]:
    cases = []
    rejected: list[dict] = []
    pdfs = sorted(input_dir.glob("ap*-apc-us-history-leq*.pdf"))
    for pdf_path in pdfs:
        try:
            samples = parse_apc_pdf(pdf_path)
        except Exception as exc:
            rejected.append({"file": pdf_path.name, "reason": f"parse_error:{exc}"})
            continue
        for sample in samples:
            case = raw_sample_to_frq_case(sample, distill=distill, seed=seed)
            ok, reasons = passes_quality_gate(case)
            if ok:
                cases.append(case)
            else:
                rejected.append(
                    {
                        "file": pdf_path.name,
                        "sample_id": sample.sample_id,
                        "reasons": reasons,
                    }
                )
    return cases, rejected


def main() -> None:
    args = parse_args()
    cases, rejected = ingest_directory(args.input, distill=args.distill, seed=args.seed)
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} eval cases to {args.output}")
    if rejected:
        print(f"Rejected {len(rejected)} samples (quality gate or parse errors)")
        if args.verbose:
            for row in rejected[:10]:
                print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest AP Central LEQ PDFs for real eval.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/raw/ap_central"),
        help="Directory containing downloaded APC PDFs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/eval_real_cases.jsonl"),
        help="Eval-only JSONL output (never used for training)",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Use LLM to rewrite CB commentary (requires OPENAI_API_KEY)",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
