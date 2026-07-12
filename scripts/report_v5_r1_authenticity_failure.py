"""Produce the aggregate v5_r1 authenticity failure report (no private essay text)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.authenticity_gates_v5 import aggregate_artifact_audit
from apush_frq_grader_slm.io import read_jsonl


def _load_essays(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                rows.extend(read_jsonl(child))
        elif path.is_file():
            rows.extend(read_jsonl(path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-essays-dir",
        type=Path,
        default=Path("artifacts/data/v5/private/raw_essays"),
    )
    parser.add_argument(
        "--selected",
        type=Path,
        action="append",
        default=[],
        help="Optional selected/train JSONL paths to audit separately.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/v5/planning/v5_r1_authenticity_failure.json"),
    )
    args = parser.parse_args()

    raw_rows = _load_essays([args.raw_essays_dir])
    raw_audit = aggregate_artifact_audit(raw_rows)
    selected_paths = args.selected or [
        Path("artifacts/data/v5/private/train_cases_v5.jsonl"),
        Path("artifacts/data/v5/private/dev_cases_v5.jsonl"),
        Path("artifacts/data/v5/private/selected_cases_v5_provisional.jsonl"),
    ]
    selected_audits = {}
    for path in selected_paths:
        if path.exists():
            selected_audits[path.name] = aggregate_artifact_audit(read_jsonl(path))

    report = {
        **raw_audit,
        "selected_split_audits": selected_audits,
        "corpus_status": "discarded_pending_regeneration",
        "deterministic_composer_retired_from_production": True,
        "evaluation_53_case_note": (
            "The final 53-case College Board evaluation is development-informed / "
            "contaminated because all 53 full essays are used as writer style references. "
            "Do not claim it is an independent generalization test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "essays_scanned": report["essays_scanned"],
        "contamination_rate": report["contamination_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
