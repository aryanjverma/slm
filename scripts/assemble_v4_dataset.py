"""Assemble v4 train cases + chat rows from tasks, raw essays, and grades."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.dataset_v4 import (
    V4Task,
    assert_no_real_essays,
    assemble_v4_case,
    audit_v4_cases,
    dedup_against_eval,
    v4_chat_row,
)
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase, RubricFeedback, RubricScores


def _load_cases(path: Path) -> list[FRQCase]:
    if not path.exists():
        return []
    return [FRQCase.model_validate(row) for row in read_jsonl(path)]


def _index_by_task_id(rows: list[dict[str, Any]], key: str = "task_id") -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        task_id = str(row.get(key) or row.get("id") or "")
        if task_id:
            indexed[task_id] = row
    return indexed


def _parse_scores(row: dict[str, Any]) -> RubricScores:
    if "scores" in row:
        return RubricScores.model_validate(row["scores"])
    if "reference_scores" in row:
        return RubricScores.model_validate(row["reference_scores"])
    return RubricScores.model_validate(row)


def _parse_feedback(row: dict[str, Any]) -> RubricFeedback:
    if "feedback" in row:
        return RubricFeedback.model_validate(row["feedback"])
    if "reference_feedback" in row:
        return RubricFeedback.model_validate(row["reference_feedback"])
    raise KeyError("feedback")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {row["task_id"]: V4Task.from_row(row) for row in read_jsonl(args.tasks)}
    essays = _index_by_task_id(read_jsonl(args.essays))
    grades = _index_by_task_id(read_jsonl(args.grades))

    eval_cases = _load_cases(args.eval_cb) + _load_cases(args.eval_real)

    accepted: list[FRQCase] = []
    rejects: list[dict[str, Any]] = []
    seen_essays: set[str] = set()

    for task_id, task in tasks.items():
        essay_row = essays.get(task_id)
        grade_row = grades.get(task_id)
        if essay_row is None:
            rejects.append({"task_id": task_id, "reasons": ["missing_essay"]})
            continue
        if grade_row is None:
            rejects.append({"task_id": task_id, "reasons": ["missing_grade"]})
            continue
        essay = str(
            essay_row.get("student_response")
            or essay_row.get("essay")
            or essay_row.get("text")
            or ""
        ).strip()
        if not essay:
            rejects.append({"task_id": task_id, "reasons": ["empty_essay"]})
            continue
        essay_key = " ".join(essay.lower().split())
        if essay_key in seen_essays:
            rejects.append({"task_id": task_id, "reasons": ["duplicate_essay_in_batch"]})
            continue
        try:
            scores = _parse_scores(grade_row)
            feedback = _parse_feedback(grade_row)
        except Exception as exc:
            rejects.append({"task_id": task_id, "reasons": [f"grade_parse_error:{exc}"]})
            continue

        labeling_method = str(grade_row.get("labeling_method") or grade_row.get("method") or "rule_based")
        grader_ids = grade_row.get("grader_ids") or []
        case = assemble_v4_case(
            task,
            essay,
            scores,
            feedback,
            labeling_method=labeling_method,
            grader_ids=grader_ids,
        )

        reasons: list[str] = []
        try:
            assert_no_real_essays([case])
        except ValueError as exc:
            reasons.append(str(exc))
        ok, gate_reasons = passes_quality_gate(case)
        if not ok:
            reasons.extend(gate_reasons)
        reasons.extend(dedup_against_eval(case, eval_cases))
        if reasons:
            rejects.append({"task_id": task_id, "reasons": reasons})
            continue

        seen_essays.add(essay_key)
        accepted.append(case)

    assert_no_real_essays(accepted)
    audit = audit_v4_cases(accepted)
    chat_rows = [v4_chat_row(case) for case in accepted]

    write_jsonl(args.output_dir / "train_cases_v4.jsonl", accepted)
    write_jsonl(args.output_dir / "train_chat_v4.jsonl", chat_rows)
    write_jsonl(args.output_dir / "rejects_v4.jsonl", rejects)

    manifest = {
        "generator_name": "v4_amsco_cb_seeded",
        "task_count": len(tasks),
        "accepted": len(accepted),
        "rejected": len(rejects),
        "totals": dict(Counter(case.reference_scores.total for case in accepted)),
        "paths": {
            "tasks": str(args.tasks),
            "essays": str(args.essays),
            "grades": str(args.grades),
            "train_cases": str(args.output_dir / "train_cases_v4.jsonl"),
            "train_chat": str(args.output_dir / "train_chat_v4.jsonl"),
            "rejects": str(args.output_dir / "rejects_v4.jsonl"),
            "audit": str(args.output_dir / "dataset_audit_v4.json"),
        },
    }
    (args.output_dir / "dataset_manifest_v4.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dataset_audit_v4.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Accepted {len(accepted)} v4 cases; rejected {len(rejects)}.")
    if rejects:
        reason_counts: Counter[str] = Counter()
        for row in rejects:
            reason_counts.update(row["reasons"])
        print(
            "  reject reasons: "
            + ", ".join(f"{reason}:{count}" for reason, count in reason_counts.most_common(8))
        )
    print(f"  wrote {args.output_dir / 'train_cases_v4.jsonl'}")
    print(f"  wrote {args.output_dir / 'train_chat_v4.jsonl'}")
    print(f"  audit clean={audit.get('clean')} median_words={audit.get('median_word_count')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/data/v4/synth_tasks_v4.jsonl"),
    )
    parser.add_argument(
        "--essays",
        type=Path,
        default=Path("artifacts/data/v4/raw_essays_v4.jsonl"),
        help="JSONL with task_id + student_response/essay.",
    )
    parser.add_argument(
        "--grades",
        type=Path,
        default=Path("artifacts/data/v4/grades_v4.jsonl"),
        help="JSONL with task_id + scores + feedback.",
    )
    parser.add_argument(
        "--eval-cb",
        type=Path,
        default=Path("artifacts/data/eval_cb_cases.jsonl"),
        help="Held-out CB eval cases for dedup (optional if missing).",
    )
    parser.add_argument(
        "--eval-real",
        type=Path,
        default=Path("artifacts/data/eval_real_cases.jsonl"),
        help="Real eval cases for dedup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v4"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
