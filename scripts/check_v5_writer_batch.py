#!/usr/bin/env python3
"""Validate one production writer batch against hard gates (helper for cloud writers)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.authenticity_gates_v5 import hard_gate_reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-essays", type=Path, required=True)
    parser.add_argument("--packets-dir", type=Path,
                        default=Path("artifacts/data/v5/packets_r2/by_task"))
    args = parser.parse_args()
    essays = [
        json.loads(line)
        for line in args.batch_essays.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = []
    for row in essays:
        task_id = row["task_id"]
        packet = json.loads((args.packets_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        reasons = hard_gate_reasons(
            row["student_response"],
            style_reference_essay=packet["style_reference_essay"],
            reference_word_count=packet["reference_word_count"],
        )
        status = "PASS" if not reasons else "FAIL"
        print(f"{task_id} words={len(row['student_response'].split())} {status} {reasons}")
        if reasons:
            failed.append(task_id)
    if failed:
        raise SystemExit(f"{len(failed)} failures: {failed}")
    print(f"OK {len(essays)} essays")


if __name__ == "__main__":
    main()
