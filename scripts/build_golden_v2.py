"""Audit and, when authorized, build the immutable College Board v2 golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.golden import (
    GoldenReviewEntry,
    audit_golden_cases,
    load_permission_record,
    require_permission,
    write_official_artifacts,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def main() -> None:
    args = parse_args()
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.input)]
    reviews = (
        [GoldenReviewEntry.model_validate(row) for row in read_jsonl(args.review_log)]
        if args.review_log.exists()
        else []
    )
    if args.create_review_template:
        template = [
            GoldenReviewEntry(case_id=case.id, reviewer="").model_dump(mode="json")
            for case in cases
        ]
        write_jsonl(args.review_log, template)
        print(f"Wrote manual review template to {args.review_log}")
        return
    audit = audit_golden_cases(cases, reviews)

    report = audit.model_dump(mode="json") | {"rejection_counts": audit.rejection_counts}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited {audit.total} cases: {len(audit.accepted_ids)} accepted")
    print(f"Report: {args.report}")

    if args.audit_only:
        return

    permission = load_permission_record(args.permission_record)
    require_permission(permission, "evaluation")
    if not audit.clean:
        raise RuntimeError("Golden audit is not clean; refusing to write official eval artifacts")

    by_id = {case.id: case for case in cases}
    accepted = []
    for case_id in audit.accepted_ids:
        case = by_id[case_id]
        case.provenance.review_status = "human_verified"
        case.labeling.human_reviewed = True
        accepted.append(case)
    manifest = write_official_artifacts(args.output_dir, accepted)
    counts = {name: item["count"] for name, item in manifest["artifacts"].items()}
    print(f"Wrote verified official artifacts to {args.output_dir}: {counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument(
        "--review-log", type=Path, default=Path("artifacts/reviews/cb_golden_v2.jsonl")
    )
    parser.add_argument(
        "--permission-record", type=Path, default=Path("config/college_board_permission.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/v2"))
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/audits/cb_golden_v2.json")
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--create-review-template", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
