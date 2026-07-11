"""Construction and strict auditing for the balanced v3 training dataset."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from apush_frq_grader_slm.filters import feedback_references_essay
from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.rubric import CRITERIA, RubricVersion
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.structured_output_v3 import (
    V3_SYSTEM_PROMPT,
    sentence_count,
    validate_model_payload,
)

OFFICIAL_SOURCE_TAGS = {"ap_central", "real_eval", "college_board", "tom_richey", "quizlet"}


class V3DatasetAudit(BaseModel):
    total: int
    accepted: int
    rejected: dict[str, list[str]] = Field(default_factory=dict)
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)
    global_reasons: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.total > 102 and self.accepted == self.total and not self.global_reasons


def format_v3_user_message(case: FRQCase) -> str:
    return (
        f"Rubric version: {case.provenance.rubric_version.value}\n\n"
        f"LEQ prompt:\n{case.prompt}\n\nStudent essay:\n{case.student_response}"
    )


def v3_target_payload(case: FRQCase) -> dict[str, Any]:
    return {
        "scores": case.reference_scores.model_dump(),
        "feedback": case.reference_feedback.model_dump(),
    }


def v3_chat_row(case: FRQCase) -> dict[str, Any]:
    target = json.dumps(v3_target_payload(case), ensure_ascii=True, separators=(",", ":"))
    return {
        "id": case.id,
        "messages": [
            {"role": "system", "content": V3_SYSTEM_PROMPT},
            {"role": "user", "content": format_v3_user_message(case)},
            {"role": "assistant", "content": target},
        ],
    }


def select_balanced_v3_cases(
    cases: list[FRQCase], *, target_count: int, seed: int = 13
) -> list[FRQCase]:
    """Select totals round-robin, using every scarce score band before filling extras."""
    if target_count <= 102:
        raise ValueError("v3 must contain substantially more than the 102-row v2 dataset")
    buckets: dict[int, list[FRQCase]] = {total: [] for total in range(7)}
    seen_ids: set[str] = set()
    seen_essays: set[str] = set()
    for case in cases:
        essay_key = " ".join(case.student_response.lower().split())
        if case.id not in seen_ids and essay_key not in seen_essays:
            buckets[case.reference_scores.total].append(case)
            seen_ids.add(case.id)
            seen_essays.add(essay_key)
    missing = [str(total) for total, bucket in buckets.items() if not bucket]
    if missing:
        raise ValueError(f"Source candidates do not cover totals: {', '.join(missing)}")
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        bucket.sort(key=lambda case: len(case.student_response.split()) < 400)

    selected: list[FRQCase] = []
    cursor = {total: 0 for total in buckets}
    while len(selected) < target_count:
        progressed = False
        for total in range(7):
            index = cursor[total]
            if index < len(buckets[total]) and len(selected) < target_count:
                selected.append(buckets[total][index])
                cursor[total] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != target_count:
        raise ValueError(
            f"Only {len(selected)} unique audited candidates for target {target_count}"
        )
    return selected


def assign_compatible_rubric_versions(
    cases: list[FRQCase],
) -> list[FRQCase]:
    """Add 2023 coverage only where the 0/1 reasoning label is version-invariant."""
    converted: list[FRQCase] = []
    eligible_index = 0
    for original in cases:
        case = original.model_copy(deep=True)
        if case.reference_scores.analysis_reasoning < 2:
            if eligible_index % 2 == 0:
                case.provenance.rubric_version = RubricVersion.LEQ_2023
                case.provenance.generator_config = {
                    **case.provenance.generator_config,
                    "v3_rubric_assignment": "2023_version_invariant_score",
                }
            eligible_index += 1
        case.assistant_response = json.dumps(
            v3_target_payload(case), ensure_ascii=True, separators=(",", ":")
        )
        converted.append(case)
    return converted


def audit_v3_training_cases(
    cases: list[FRQCase], *, min_long_essay_rate: float = 0.10
) -> V3DatasetAudit:
    rejected: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    seen_essays: set[str] = set()
    totals: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    labeling_methods: Counter[str] = Counter()
    for case in cases:
        reasons: list[str] = []
        if case.id in seen_ids:
            reasons.append("duplicate_id")
        seen_ids.add(case.id)
        essay_key = " ".join(case.student_response.lower().split())
        if essay_key in seen_essays:
            reasons.append("duplicate_essay")
        seen_essays.add(essay_key)
        if case.split != "train":
            reasons.append("not_train_split")
        if case.provenance.source_type != "synthetic":
            reasons.append("non_synthetic_or_unknown_source")
        if OFFICIAL_SOURCE_TAGS.intersection(case.tags):
            reasons.append("official_source_tag")

        target = v3_target_payload(case)
        reasons.extend(validate_model_payload(target, strict_keys=True))
        serialized = json.dumps(target, ensure_ascii=True)
        try:
            if json.loads(serialized) != target:
                reasons.append("target_json_roundtrip_failed")
        except json.JSONDecodeError:
            reasons.append("target_json_invalid")
        for criterion in CRITERIA:
            feedback = target["feedback"][criterion]
            if not feedback_references_essay(feedback, case.student_response):
                reasons.append(f"feedback_not_grounded:{criterion}")
            if sentence_count(feedback) != 1:
                reasons.append(f"feedback_not_one_sentence:{criterion}")

        if reasons:
            rejected[case.id] = sorted(set(reasons))
            continue
        totals[str(case.reference_scores.total)] += 1
        versions[case.provenance.rubric_version.value] += 1
        lengths["400_plus" if len(case.student_response.split()) >= 400 else "under_400"] += 1
        source_types[case.provenance.source_type] += 1
        labeling_methods[case.labeling.method] += 1

    accepted = len(cases) - len(rejected)
    global_reasons: list[str] = []
    if len(cases) <= 102:
        global_reasons.append("not_substantially_larger_than_v2")
    if set(totals) != {str(total) for total in range(7)}:
        global_reasons.append("missing_total_band")
    if set(versions) != {version.value for version in RubricVersion}:
        global_reasons.append("missing_rubric_version")
    long_rate = lengths["400_plus"] / accepted if accepted else 0
    if long_rate < min_long_essay_rate:
        global_reasons.append("insufficient_long_essays")
    return V3DatasetAudit(
        total=len(cases),
        accepted=accepted,
        rejected=rejected,
        global_reasons=global_reasons,
        distributions={
            "total": dict(sorted(totals.items())),
            "rubric_version": dict(sorted(versions.items())),
            "essay_length": dict(sorted(lengths.items())),
            "source_type": dict(sorted(source_types.items())),
            "labeling_method": dict(sorted(labeling_methods.items())),
        },
    )


def write_v3_dataset(
    output_dir: Path,
    cases: list[FRQCase],
    audit: V3DatasetAudit,
    *,
    settings: dict[str, Any] | None = None,
) -> dict:
    if not audit.clean:
        raise ValueError(f"Refusing to write v3 dataset with failed audit: {audit.global_reasons}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": output_dir / "train_cases_v3.jsonl",
        "chat": output_dir / "train_chat_v3.jsonl",
        "audit": output_dir / "dataset_audit_v3.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("v3 dataset outputs are immutable; choose an empty output directory")
    write_jsonl(paths["cases"], cases)
    write_jsonl(paths["chat"], [v3_chat_row(case) for case in cases])
    paths["audit"].write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "version": "v3",
        "rows": len(cases),
        "settings": settings or {},
        "files": {
            name: {"path": path.as_posix(), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "dataset_manifest_v3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
