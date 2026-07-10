"""Create anonymous third-reader packets for substantive reader disagreements."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.independent_grading import (
    parse_grader_output,
    resolve_independent_grades,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.synth_realistic import SyntheticCandidate


def main() -> None:
    args = parse_args()
    packets = {str(row["task_id"]): row for row in read_jsonl(args.packets)}
    reader_a = {str(row["task_id"]): row for row in read_jsonl(args.reader_a)}
    reader_b = {str(row["task_id"]): row for row in read_jsonl(args.reader_b)}
    output: list[dict] = []
    for task_id, packet in packets.items():
        candidate = SyntheticCandidate(
            task_id,
            str(packet["prompt"]),
            str(packet["student_response"]),
            str(packet["rubric_version"]),
        )
        grade_a = parse_grader_output(reader_a[task_id], candidate, "reader_a:offline")
        grade_b = parse_grader_output(reader_b[task_id], candidate, "reader_b:offline")
        decision = resolve_independent_grades(task_id, grade_a, grade_b)
        if "grader_disagreement_requires_adjudication" not in decision.reasons:
            continue
        output.append(
            {
                **packet,
                "reader_a": grade_a.to_payload(),
                "reader_b": grade_b.to_payload(),
            }
        )
    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} adjudication packets to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
