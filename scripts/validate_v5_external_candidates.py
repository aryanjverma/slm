"""Validate and normalize essay/review records returned by an external generation tool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.dataset_v5 import (
    V5_CANDIDATE_COUNT,
    OverlapIndex,
    V5GenerationTask,
    annotate_distribution_match,
    candidate_gate_reasons,
    normalize_external_candidate,
)
from apush_frq_grader_slm.compose_v5 import COMPOSER_STOCK_EXEMPTIONS
from apush_frq_grader_slm.fact_cards_v5 import (
    default_allowed_overlap_phrases,
    load_allowed_phrases_file,
    merge_allowed_phrases,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.knowledge.amsco import load_kb
from apush_frq_grader_slm.schemas import FRQCase


def task_from_row(row: dict) -> V5GenerationTask:
    return V5GenerationTask(
        task_id=row["task_id"],
        shard_id=row["shard_id"],
        prompt=row["prompt"],
        prompt_family_id=row["prompt_family_id"],
        style_seed_id=row["style_seed_id"],
        style_excerpt=row.get("style_excerpt", ""),
        period=row.get("period"),
        reasoning_skill=row.get("reasoning_skill", ""),
        capability_profile=dict(row["capability_profile"]),
        composition_profile=dict(row["composition_profile"]),
        amsco_chapter_ids=tuple(row.get("amsco_chapter_ids") or ()),
        coverage_class=row.get("coverage_class", "golden_matched"),
        boundary_type=row.get("boundary_type", ""),
        contrast_pair_id=row.get("contrast_pair_id", ""),
        contrast_side=row.get("contrast_side", ""),
    )


def collect_allowed_phrases(args: argparse.Namespace) -> list[str]:
    groups: list[list[str]] = []
    if args.allowed_phrases:
        groups.append(load_allowed_phrases_file(args.allowed_phrases))
    kb_path = args.amsco_kb if args.amsco_kb and Path(args.amsco_kb).exists() else None
    kb_rows = load_kb(kb_path) if kb_path else None
    fact_cards = read_jsonl(args.fact_cards) if args.fact_cards else None
    if kb_path or args.fact_cards or args.include_default_phrases:
        groups.append(
            default_allowed_overlap_phrases(
                kb=kb_rows,
                fact_cards=fact_cards,
                kb_path=kb_path,
            )
        )
    # Residual short composer fragments (<8 words preferred) after template diversification.
    groups.append(list(COMPOSER_STOCK_EXEMPTIONS))
    return merge_allowed_phrases(*groups)


def main() -> None:
    args = parse_args()
    tasks = {row["task_id"]: task_from_row(row) for row in read_jsonl(args.tasks)}
    returned_rows = read_jsonl(args.candidates)
    returned: dict[str, dict] = {}
    duplicate_ids: list[str] = []
    for row in returned_rows:
        task_id = str(row.get("task_id") or "")
        if task_id in returned:
            duplicate_ids.append(task_id)
        returned[task_id] = row
    unknown_ids = sorted(set(returned) - set(tasks))
    missing_ids = sorted(set(tasks) - set(returned))
    if duplicate_ids or unknown_ids:
        raise ValueError(
            f"External return has duplicate task IDs {sorted(set(duplicate_ids))[:10]} "
            f"or unknown task IDs {unknown_ids[:10]}"
        )
    if args.require_complete and (len(tasks) != V5_CANDIDATE_COUNT or missing_ids):
        raise ValueError(
            f"Complete v5 validation requires {V5_CANDIDATE_COUNT} planned and returned tasks; "
            f"planned={len(tasks)} returned={len(returned)} missing={len(missing_ids)}"
        )

    overlap_texts: list[str] = []
    for path in args.overlap_corpus:
        overlap_texts.extend(
            str(row.get("student_response") or row.get("essay") or "")
            for row in read_jsonl(path)
        )
    allowed_phrases = collect_allowed_phrases(args)
    golden_cases: list[FRQCase] = []
    if args.golden_cases and Path(args.golden_cases).exists():
        golden_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.golden_cases)]
    accepted: list[dict] = []
    rejected: dict[str, list[str]] = {}
    index = OverlapIndex.build(overlap_texts, allowed_phrases=allowed_phrases)
    for task_id in sorted(set(returned) & set(tasks)):
        row = normalize_external_candidate(tasks[task_id], returned[task_id])
        if golden_cases:
            row = annotate_distribution_match([row], golden_cases)[0]
        reasons = candidate_gate_reasons(
            row, allowed_phrases=allowed_phrases, overlap_index=index
        )
        if reasons:
            rejected[task_id] = sorted(set(reasons))
            continue
        accepted.append(row)
        index.add(str(row.get("student_response") or row.get("essay") or ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, accepted)
    reason_counts = Counter(reason for reasons in rejected.values() for reason in reasons)
    audit = {
        "planned": len(tasks),
        "returned": len(returned),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "missing_task_ids": missing_ids,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "coverage": dict(sorted(Counter(row["selection_class"] for row in accepted).items())),
        "allowed_phrase_count": len(allowed_phrases),
        "contains_private_rows": True,
        "redistribution_authorized": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--overlap-corpus", type=Path, action="append", default=[])
    parser.add_argument(
        "--golden-cases",
        type=Path,
        default=Path("artifacts/data/eval_cb_cases.jsonl"),
        help="Golden FRQCase JSONL used to recompute distribution_match before gating",
    )
    parser.add_argument(
        "--allowed-phrases",
        type=Path,
        default=None,
        help="JSONL ({phrase|text|term}) or plain-text file of overlap exemptions",
    )
    parser.add_argument(
        "--fact-cards",
        type=Path,
        default=None,
        help="Semantic fact cards JSONL; evidence terms auto-join allowed phrases",
    )
    parser.add_argument(
        "--amsco-kb",
        type=Path,
        default=Path("artifacts/knowledge/amsco_2016_kb.jsonl"),
        help="AMSCO KB JSONL; evidence_bank terms/years auto-join allowed phrases",
    )
    parser.add_argument(
        "--include-default-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include built-in historical name/date exemptions (default: on)",
    )
    parser.add_argument(
        "--require-complete", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
