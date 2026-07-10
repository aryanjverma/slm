"""Audit local AP Central parser coverage without writing a derived essay corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.ingest.apc_parser import (
    detect_essay_contamination,
    parse_apc_pdf,
)


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    for path in sorted(args.input.glob("ap*-apc-us-history-leq*.pdf")):
        try:
            samples = parse_apc_pdf(path)
            records.append(
                {
                    "filename": path.name,
                    "status": "parsed",
                    "sample_count": len(samples),
                    "sample_ids": [sample.sample_id for sample in samples],
                    "minimum_confidence": min(
                        (sample.parser_confidence for sample in samples), default=0.0
                    ),
                    "contaminated_samples": [
                        sample.sample_id
                        for sample in samples
                        if detect_essay_contamination(sample.essay)
                    ],
                }
            )
        except Exception as exc:
            records.append(
                {"filename": path.name, "status": "rejected", "error": str(exc)}
            )

    summary = {
        "pdf_count": len(records),
        "parsed_pdfs": sum(record["status"] == "parsed" for record in records),
        "rejected_pdfs": sum(record["status"] == "rejected" for record in records),
        "sample_count": sum(record.get("sample_count", 0) for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Audited {summary['pdf_count']} PDFs: parsed={summary['parsed_pdfs']}, "
        f"rejected={summary['rejected_pdfs']}, samples={summary['sample_count']}"
    )
    print(f"Report: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/raw/ap_central"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audits/apc_parser_v2.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
