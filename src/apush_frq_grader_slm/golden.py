"""Permission-aware validation for immutable College Board evaluation artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.dedup import normalize_essay
from apush_frq_grader_slm.rubric import rubric_version_for_year
from apush_frq_grader_slm.schemas import FRQCase


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
    seen_essays: set[str] = set()
    accepted_ids: list[str] = []
    rejected: dict[str, list[str]] = {}

    for case in cases:
        reasons: list[str] = []
        provenance = case.provenance
        if provenance.source_type != "college_board":
            reasons.append("not_official_college_board")
        if not provenance.source_url or not provenance.sample_id or provenance.year is None:
            reasons.append("incomplete_provenance")
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
