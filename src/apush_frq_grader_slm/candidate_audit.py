"""Pre-grading validation for realistic synthetic candidate essays."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from apush_frq_grader_slm.filters import (
    contains_generation_leakage,
    contains_source_contamination,
)
from apush_frq_grader_slm.ingest.dedup import essay_fingerprint, normalize_essay
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.synth_realistic import (
    GenTask,
    SyntheticCandidate,
    parse_candidate_row,
    validate_generated_candidate,
)


class CandidateAudit(BaseModel):
    total_rows: int
    accepted_rows: int
    rejected: list[dict]
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)


def audit_candidates(
    tasks: dict[str, GenTask],
    rows: list[dict],
    *,
    leakage_sources: list[FRQCase] | None = None,
) -> tuple[list[SyntheticCandidate], CandidateAudit]:
    accepted: list[SyntheticCandidate] = []
    rejected: list[dict] = []
    seen_task_ids: set[str] = set()
    by_total: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_time: Counter[str] = Counter()

    for row in rows:
        task_id = str(row.get("task_id", ""))
        reasons: list[str] = []
        if task_id in seen_task_ids:
            reasons.append("duplicate_task_id")
        seen_task_ids.add(task_id)
        task = tasks.get(task_id)
        if task is None:
            rejected.append({"task_id": task_id, "reasons": ["unknown_task_id"]})
            continue
        try:
            candidate = parse_candidate_row(row, task)
        except Exception as exc:
            rejected.append({"task_id": task_id, "reasons": [f"parse_error:{exc}"]})
            continue

        valid, validation_reasons = validate_generated_candidate(
            candidate, task, leakage_sources or []
        )
        if not valid:
            reasons.extend(validation_reasons)
        if contains_source_contamination(candidate.student_response):
            reasons.append("source_text_contamination")
        if contains_generation_leakage(candidate.student_response):
            reasons.append("generation_prompt_leakage")
        duplicate_id = _near_duplicate_id(candidate, accepted)
        if duplicate_id is not None:
            reasons.append(f"near_duplicate_of:{duplicate_id}")

        if reasons:
            rejected.append({"task_id": task_id, "reasons": sorted(set(reasons))})
            continue
        accepted.append(candidate)
        by_total[str(task.target_total)] += 1
        by_family[task.prompt_family_id or task.seed_id] += 1
        by_time[str(task.persona.time_budget_minutes)] += 1

    audit = CandidateAudit(
        total_rows=len(rows),
        accepted_rows=len(accepted),
        rejected=rejected,
        distributions={
            "target_total": dict(sorted(by_total.items())),
            "prompt_family": dict(sorted(by_family.items())),
            "time_budget": dict(sorted(by_time.items())),
        },
    )
    return accepted, audit


def _near_duplicate_id(
    candidate: SyntheticCandidate, accepted: list[SyntheticCandidate]
) -> str | None:
    normalized = normalize_essay(candidate.student_response)
    fingerprint = essay_fingerprint(candidate.student_response)
    for prior in accepted:
        prior_normalized = normalize_essay(prior.student_response)
        if normalized == prior_normalized:
            return prior.task_id
        prior_fingerprint = essay_fingerprint(prior.student_response)
        overlap = fingerprint & prior_fingerprint
        if candidate.prompt == prior.prompt:
            union = fingerprint | prior_fingerprint
            if union and len(overlap) / len(union) >= 0.82:
                return prior.task_id
        elif _word_spans(candidate.student_response, 12) & _word_spans(
            prior.student_response, 12
        ):
            return prior.task_id
    return None


def _word_spans(text: str, size: int) -> set[str]:
    words = normalize_essay(text).split()
    return {
        " ".join(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    }
