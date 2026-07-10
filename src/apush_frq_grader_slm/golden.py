"""Permission-aware validation for immutable College Board evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.dedup import normalize_essay
from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.rubric import rubric_version_for_year
from apush_frq_grader_slm.schemas import FRQCase


OFFICIAL_COMBINED_FILENAME = "eval_cb_golden_v2.jsonl"
OFFICIAL_DEV_FILENAME = "eval_cb_dev_v2.jsonl"
OFFICIAL_FINAL_FILENAME = "eval_cb_final_v2.jsonl"
OFFICIAL_MANIFEST_FILENAME = "eval_cb_manifest_v2.json"
EXPECTED_OFFICIAL_DEV_COUNT = 27
EXPECTED_OFFICIAL_FINAL_COUNT = 26


class PermissionRecord(BaseModel):
    status: Literal["unresolved", "granted", "denied"] = "unresolved"
    allowed_uses: list[Literal["storage", "evaluation", "training", "redistribution"]] = Field(
        default_factory=list
    )
    approved_by: str = ""
    effective_date: date | None = None
    reference: str = ""
    notes: str = ""

    def permits(self, use: str) -> bool:
        return self.status == "granted" and use in self.allowed_uses


class GoldenReviewEntry(BaseModel):
    case_id: str
    reviewer: str
    essay_verified: bool = False
    scores_verified: bool = False
    provenance_verified: bool = False
    notes: str = ""

    @property
    def accepted(self) -> bool:
        return bool(
            self.reviewer
            and self.essay_verified
            and self.scores_verified
            and self.provenance_verified
        )


class GoldenAudit(BaseModel):
    total: int
    accepted_ids: list[str]
    rejected: dict[str, list[str]]

    @property
    def clean(self) -> bool:
        return self.total > 0 and not self.rejected and len(self.accepted_ids) == self.total

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for reasons in self.rejected.values():
            counts.update(reasons)
        return dict(counts)


def audit_golden_cases(
    cases: list[FRQCase],
    reviews: list[GoldenReviewEntry],
) -> GoldenAudit:
    review_by_id = {entry.case_id: entry for entry in reviews}
    case_id_counts = Counter(case.id for case in cases)
    review_id_counts = Counter(entry.case_id for entry in reviews)
    seen_essays: set[str] = set()
    accepted_ids: list[str] = []
    rejected: dict[str, list[str]] = {}

    for case in cases:
        reasons: list[str] = []
        provenance = case.provenance
        if case_id_counts[case.id] > 1:
            reasons.append("duplicate_case_id")
        if provenance.source_type != "college_board":
            reasons.append("not_official_college_board")
        if (
            not provenance.source_id
            or not provenance.source_url
            or not provenance.file_sha256
            or not provenance.sample_id
            or provenance.year is None
            or provenance.set_number not in {1, 2}
            or provenance.leq_number is None
        ):
            reasons.append("incomplete_provenance")
        if provenance.source_url:
            hostname = (urlparse(provenance.source_url).hostname or "").lower()
            if hostname != "apcentral.collegeboard.org":
                reasons.append("unverified_source_url")
        if provenance.extraction_method != "pdf_text":
            reasons.append("untrusted_extraction_method")
        if provenance.extraction_confidence is None or provenance.extraction_confidence < 0.9:
            reasons.append("low_extraction_confidence")
        if provenance.rubric_version != rubric_version_for_year(provenance.year):
            reasons.append("rubric_version_mismatch")

        gate_ok, gate_reasons = passes_quality_gate(case, strict=True)
        if not gate_ok:
            reasons.extend(gate_reasons)

        normalized = normalize_essay(case.student_response)
        if normalized in seen_essays:
            reasons.append("duplicate_essay")
        seen_essays.add(normalized)

        review = review_by_id.get(case.id)
        if review is None:
            reasons.append("missing_manual_review")
        elif review_id_counts[case.id] > 1:
            reasons.append("duplicate_manual_review")
        elif not review.accepted:
            reasons.append("incomplete_manual_review")

        if reasons:
            rejected[case.id] = sorted(set(reasons))
        else:
            accepted_ids.append(case.id)

    extra_review_ids = set(review_by_id) - {case.id for case in cases}
    for case_id in sorted(extra_review_ids):
        rejected[f"review:{case_id}"] = ["review_for_unknown_case"]

    return GoldenAudit(total=len(cases), accepted_ids=accepted_ids, rejected=rejected)


def require_permission(record: PermissionRecord, use: str) -> None:
    if not record.permits(use):
        raise PermissionError(
            f"College Board permission does not authorize {use!r}; "
            "record written permission before producing this artifact"
        )


def load_permission_record(path: Path) -> PermissionRecord:
    if not path.exists():
        raise PermissionError(
            f"Missing private permission record: {path}. Start from "
            "config/college_board_permission.example.json after written authorization."
        )
    return PermissionRecord.model_validate_json(path.read_text(encoding="utf-8"))


def split_official_cases(
    cases: list[FRQCase],
    *,
    expected_dev_count: int | None = EXPECTED_OFFICIAL_DEV_COUNT,
    expected_final_count: int | None = EXPECTED_OFFICIAL_FINAL_COUNT,
) -> tuple[list[FRQCase], list[FRQCase]]:
    """Split official rows by released set while enforcing holdout isolation."""
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Official cases contain duplicate case IDs")

    dev: list[FRQCase] = []
    final: list[FRQCase] = []
    group_samples: dict[tuple[int | None, int | None, int, str], set[str]] = {}
    for case in cases:
        set_number = case.provenance.set_number
        if set_number not in {1, 2}:
            raise ValueError(f"{case.id}: official set_number must be 1 or 2")
        prompt_key = _normalize_prompt(case.prompt)
        group_key = (
            case.provenance.year,
            case.provenance.leq_number,
            set_number,
            prompt_key,
        )
        samples = group_samples.setdefault(group_key, set())
        sample_id = case.provenance.sample_id.upper()
        if sample_id in samples:
            raise ValueError(f"{case.id}: duplicate sample {sample_id} in released prompt")
        samples.add(sample_id)
        (dev if set_number == 1 else final).append(case)

    dev_prompts = {_normalize_prompt(case.prompt) for case in dev}
    final_prompts = {_normalize_prompt(case.prompt) for case in final}
    overlap = dev_prompts & final_prompts
    if overlap:
        raise ValueError(f"Official dev/final prompt overlap detected ({len(overlap)} prompts)")
    if expected_dev_count is not None and len(dev) != expected_dev_count:
        raise ValueError(f"Official dev must contain {expected_dev_count} rows, found {len(dev)}")
    if expected_final_count is not None and len(final) != expected_final_count:
        raise ValueError(
            f"Official final must contain {expected_final_count} rows, found {len(final)}"
        )
    return dev, final


def write_official_artifacts(output_dir: Path, cases: list[FRQCase]) -> dict:
    """Write combined/dev/final official JSONL files and their portable manifest."""
    unverified = [
        case.id
        for case in cases
        if case.provenance.review_status != "human_verified"
        or not case.labeling.human_reviewed
    ]
    if unverified:
        raise ValueError(
            f"Refusing to write official artifacts with {len(unverified)} unverified rows"
        )
    dev, final = split_official_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": output_dir / OFFICIAL_COMBINED_FILENAME,
        "dev": output_dir / OFFICIAL_DEV_FILENAME,
        "final": output_dir / OFFICIAL_FINAL_FILENAME,
    }
    rows = {"combined": cases, "dev": dev, "final": final}
    for name, path in paths.items():
        write_jsonl(path, rows[name])

    manifest = {
        "schema_version": 1,
        "dataset": "college_board_official_eval_v2",
        "split_policy": {
            "dev": "set1",
            "final": "set2",
            "prompt_overlap_allowed": False,
        },
        "artifacts": {
            name: {
                "path": path.name,
                "count": len(rows[name]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / OFFICIAL_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())
