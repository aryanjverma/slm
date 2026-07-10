"""Validate one anonymous reader JSONL against its grading packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.independent_grading import parse_grader_output
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.synth_realistic import SyntheticCandidate


def main() -> None:
    args = parse_args()
    packets = {str(row["task_id"]): row for row in read_jsonl(args.packets)}
    invalid: list[dict[str, str]] = []
    rows = read_jsonl(args.reader)
    for row in rows:
        task_id = str(row.get("task_id", ""))
        packet = packets.get(task_id)
        if packet is None:
            invalid.append({"task_id": task_id, "reason": "unknown_task_id"})
            continue
        candidate = SyntheticCandidate(
            task_id,
            str(packet["prompt"]),
            str(packet["student_response"]),
            str(packet["rubric_version"]),
        )
        try:
            parse_grader_output(row, candidate, args.reader_id)
        except Exception as exc:
            invalid.append({"task_id": task_id, "reason": str(exc)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(invalid, indent=2), encoding="utf-8")
    print(f"Validated {len(rows)} rows: valid={len(rows) - len(invalid)}, invalid={len(invalid)}")
    if invalid:
        for row in invalid:
            print(f"  {row['task_id']}: {row['reason']}")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--reader-id", default="offline_reader")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
