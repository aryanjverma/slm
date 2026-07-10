"""Ingest CB, Tom Richey, and Quizlet eval sources with deduplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.golden import load_permission_record, require_permission
from apush_frq_grader_slm.ingest.apc_parser import parse_apc_pdf
from apush_frq_grader_slm.ingest.dedup import is_duplicate_essay
from apush_frq_grader_slm.ingest.distill import raw_sample_to_frq_case
from apush_frq_grader_slm.ingest.quizlet_parser import load_quizlet_json
from apush_frq_grader_slm.ingest.tomrichey_parser import parse_tomrichey_pdf
from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def _sample_to_case(sample, *, distill: bool, seed: int) -> FRQCase:
    source = sample.metadata.get("source", sample.sample_id)
    return raw_sample_to_frq_case(sample, case_id=str(source), distill=distill, seed=seed)


def ingest_cb(
    input_dir: Path,
    *,
    distill: bool = False,
    seed: int = 13,
) -> tuple[list[FRQCase], list[dict]]:
    cases: list[FRQCase] = []
    rejected: list[dict] = []
    pdfs = sorted(input_dir.glob("ap*-apc-us-history-leq*.pdf"))
    for pdf_path in pdfs:
        try:
            samples = parse_apc_pdf(pdf_path)
        except Exception as exc:
            rejected.append({"file": pdf_path.name, "reason": f"parse_error:{exc}"})
            continue
        for sample in samples:
            sample.metadata.setdefault("provider", "ap_central")
            case = _sample_to_case(sample, distill=distill, seed=seed)
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


def ingest_tomrichey(
    input_dir: Path,
    *,
    existing: list[FRQCase],
    distill: bool = False,
    seed: int = 13,
) -> tuple[list[FRQCase], list[dict]]:
    cases: list[FRQCase] = []
    rejected: list[dict] = []
    manifest_path = input_dir / "manifest.json"
    entries: list[dict] = []
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdfs = sorted(input_dir.glob("tomrichey_*.pdf"))
    if not pdfs and not entries:
        pdfs = sorted(input_dir.glob("*.pdf"))

    meta_by_file = {Path(row["local_path"]).name: row for row in entries if row.get("local_path")}

    for pdf_path in pdfs:
        meta = {
            k: meta_by_file.get(pdf_path.name, {}).get(k)
            for k in ("year", "leq_num", "set", "prompt")
            if meta_by_file.get(pdf_path.name, {}).get(k) is not None
        }
        try:
            samples = parse_tomrichey_pdf(pdf_path, metadata=meta or None)
        except Exception as exc:
            rejected.append({"file": pdf_path.name, "reason": f"parse_error:{exc}"})
            continue
        for sample in samples:
            if is_duplicate_essay(sample.essay, existing + cases, prompt=sample.prompt):
                rejected.append(
                    {
                        "file": pdf_path.name,
                        "sample_id": sample.sample_id,
                        "reason": "duplicate_of_cb",
                    }
                )
                continue
            case = _sample_to_case(sample, distill=distill, seed=seed)
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


def ingest_quizlet(
    input_dir: Path,
    *,
    existing: list[FRQCase],
    distill: bool = False,
    seed: int = 13,
) -> tuple[list[FRQCase], list[dict]]:
    cases: list[FRQCase] = []
    rejected: list[dict] = []
    json_files = sorted(input_dir.glob("*.json"))
    json_files = [path for path in json_files if path.name != "manifest.json"]
    for json_path in json_files:
        try:
            samples = load_quizlet_json(json_path)
        except Exception as exc:
            rejected.append({"file": json_path.name, "reason": f"parse_error:{exc}"})
            continue
        for sample in samples:
            if is_duplicate_essay(sample.essay, existing + cases, prompt=sample.prompt):
                rejected.append(
                    {
                        "file": json_path.name,
                        "sample_id": sample.sample_id,
                        "reason": "duplicate",
                    }
                )
                continue
            case = _sample_to_case(sample, distill=distill, seed=seed)
            ok, reasons = passes_quality_gate(case)
            if ok:
                cases.append(case)
            else:
                rejected.append(
                    {
                        "file": json_path.name,
                        "sample_id": sample.sample_id,
                        "reasons": reasons,
                    }
                )
    return cases, rejected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_cb:
        cb_cases, cb_rejected = [], []
    else:
        require_permission(load_permission_record(args.permission_record), "evaluation")
        cb_cases, cb_rejected = ingest_cb(args.cb_dir, distill=args.distill, seed=args.seed)
    tr_cases, tr_rejected = ingest_tomrichey(
        args.tomrichey_dir,
        existing=cb_cases,
        distill=args.distill,
        seed=args.seed,
    )
    qz_cases, qz_rejected = ingest_quizlet(
        args.quizlet_dir,
        existing=cb_cases + tr_cases,
        distill=args.distill,
        seed=args.seed,
    )

    external = tr_cases + qz_cases
    all_cases = cb_cases + external

    write_jsonl(args.output_dir / "eval_cb_cases.jsonl", cb_cases)
    write_jsonl(args.output_dir / "eval_tomrichey_cases.jsonl", tr_cases)
    write_jsonl(args.output_dir / "eval_quizlet_cases.jsonl", qz_cases)
    write_jsonl(args.output_dir / "eval_external_cases.jsonl", external)
    write_jsonl(args.output_dir / "eval_real_cases.jsonl", all_cases)

    print(f"CB: {len(cb_cases)} cases ({len(cb_rejected)} rejected)")
    print(f"Tom Richey: {len(tr_cases)} cases ({len(tr_rejected)} rejected/deduped)")
    print(f"Quizlet: {len(qz_cases)} cases ({len(qz_rejected)} rejected/deduped)")
    print(f"Combined eval_real_cases.jsonl: {len(all_cases)} cases")
    if args.verbose:
        for label, rejected in (
            ("cb", cb_rejected),
            ("tomrichey", tr_rejected),
            ("quizlet", qz_rejected),
        ):
            if rejected:
                print(f"\n{label} rejections (first 5):")
                for row in rejected[:5]:
                    print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest CB, Tom Richey, and Quizlet eval sources."
    )
    parser.add_argument(
        "--cb-dir",
        type=Path,
        default=Path("artifacts/raw/ap_central"),
    )
    parser.add_argument(
        "--tomrichey-dir",
        type=Path,
        default=Path("artifacts/raw/tomrichey"),
    )
    parser.add_argument(
        "--quizlet-dir",
        type=Path,
        default=Path("artifacts/raw/quizlet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Use LLM to rewrite commentary (requires OPENAI_API_KEY)",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-cb", action="store_true")
    parser.add_argument(
        "--permission-record",
        type=Path,
        default=Path("config/college_board_permission.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
