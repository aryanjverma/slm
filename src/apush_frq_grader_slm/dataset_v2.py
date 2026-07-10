"""Dataset-level quality checks and immutable v2 artifact manifests."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from apush_frq_grader_slm.data import to_chat_rows
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.dedup import (
    contains_verbatim_span,
    is_duplicate_essay,
    normalize_essay,
)
from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


REAL_SOURCE_TAGS = {"ap_central", "real_eval", "tom_richey", "quizlet", "seed_real"}
ADVERSARIAL_TYPES = {"grade_inflation_request", "prompt_injection"}


class DatasetAudit(BaseModel):
    total: int
    accepted_ids: list[str]
    rejected: dict[str, list[str]]
    exact_duplicate_count: int = 0
    repeated_feedback_count: int = 0
    human_review_rate: float = 0
    global_reasons: list[str] = Field(default_factory=list)
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return bool(
            self.total
            and not self.rejected
            and not self.global_reasons
            and len(self.accepted_ids) == self.total
        )


class ArtifactRecord(BaseModel):
    path: str
    sha256: str
    bytes: int
    rows: int | None = None


class DatasetManifest(BaseModel):
    version: str = "v2"
    seed: int
    split_policy: str
    artifacts: list[ArtifactRecord]
    settings: dict = Field(default_factory=dict)
    audit: DatasetAudit
    track_audits: dict[str, DatasetAudit] = Field(default_factory=dict)


class SyntheticReviewEntry(BaseModel):
    case_id: str
    reviewer: str = ""
    scores_verified: bool = False
    feedback_verified: bool = False
    historical_accuracy_verified: bool = False
    notes: str = ""

    @property
    def accepted(self) -> bool:
        return bool(
            self.reviewer
            and self.scores_verified
            and self.feedback_verified
            and self.historical_accuracy_verified
        )


def select_human_review_sample(
    cases: list[FRQCase], *, rate: float = 0.10, seed: int = 13
) -> list[FRQCase]:
    if not 0 < rate <= 1:
        raise ValueError("review rate must be in (0, 1]")
    target = min(len(cases), max(1, math.ceil(len(cases) * rate))) if cases else 0
    rng = random.Random(seed)
    randomized = list(cases)
    rng.shuffle(randomized)

    def priority(case: FRQCase) -> tuple[int, int, str]:
        disagreement = case.labeling.agreement is not None and case.labeling.agreement < 1.0
        edge_total = case.reference_scores.total in {0, 1, 5, 6}
        complexity = case.reference_scores.analysis_reasoning == 2
        return (-(int(disagreement) * 4 + int(edge_total) * 2 + int(complexity)), 0, case.id)

    return sorted(randomized, key=priority)[:target]


def apply_human_reviews(
    cases: list[FRQCase], reviews: list[SyntheticReviewEntry]
) -> list[FRQCase]:
    review_by_id = {review.case_id: review for review in reviews}
    for case in cases:
        review = review_by_id.get(case.id)
        if review is not None and review.accepted:
            case.labeling.human_reviewed = True
    return cases


def audit_training_cases(
    cases: list[FRQCase],
    *,
    forbidden_cases: list[FRQCase] | None = None,
) -> DatasetAudit:
    forbidden = forbidden_cases or []
    rejected: dict[str, list[str]] = {}
    accepted_ids: list[str] = []
    seen_essays: dict[str, str] = {}
    seen_feedback: dict[str, str] = {}
    exact_duplicates = 0
    repeated_feedback = 0

    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    prompt_families: Counter[str] = Counter()
    time_budgets: Counter[str] = Counter()
    knowledge_profiles: Counter[str] = Counter()
    periods: Counter[str] = Counter()
    reasoning_skills: Counter[str] = Counter()
    independently_labeled = 0
    human_reviewed = 0

    for case in cases:
        reasons: list[str] = []
        if case.split != "train":
            reasons.append("not_train_split")
        if case.provenance.source_type in {"college_board", "external"}:
            reasons.append("real_source_in_training")
        if REAL_SOURCE_TAGS.intersection(case.tags):
            reasons.append("real_source_tag_in_training")
        is_adversarial = case.failure_type.value in ADVERSARIAL_TYPES
        if not is_adversarial and case.labeling.method not in {
            "independent_consensus",
            "adjudicated",
        }:
            reasons.append("ordinary_case_not_independently_labeled")
        if not is_adversarial and not case.provenance.prompt_family_id:
            reasons.append("missing_prompt_family_id")

        gate_ok, gate_reasons = passes_quality_gate(case, strict=True)
        if not gate_ok:
            reasons.extend(gate_reasons)

        normalized = normalize_essay(case.student_response)
        prior = seen_essays.get(normalized)
        if prior is not None:
            reasons.append(f"exact_duplicate_of:{prior}")
            exact_duplicates += 1
        else:
            seen_essays[normalized] = case.id

        feedback_text = " ".join(case.reference_feedback.model_dump().values())
        feedback_norm = normalize_essay(feedback_text)
        prior_feedback = seen_feedback.get(feedback_norm)
        if prior_feedback is not None:
            repeated_feedback += 1
            reasons.append(f"repeated_feedback_of:{prior_feedback}")
        else:
            seen_feedback[feedback_norm] = case.id

        if forbidden and (
            is_duplicate_essay(case.student_response, forbidden, prompt=case.prompt)
            or contains_verbatim_span(case.student_response, forbidden, ignore=case.prompt)
        ):
            reasons.append("forbidden_eval_leakage")

        if reasons:
            rejected[case.id] = sorted(set(reasons))
            continue

        accepted_ids.append(case.id)
        if case.labeling.method in {"independent_consensus", "adjudicated"}:
            independently_labeled += 1
            if case.labeling.human_reviewed:
                human_reviewed += 1
        totals[str(case.reference_scores.total)] += 1
        failures[case.failure_type.value] += 1
        family = case.provenance.prompt_family_id or "unknown"
        prompt_families[family] += 1
        config = case.provenance.generator_config
        persona = config.get("persona", {}) if isinstance(config, dict) else {}
        time_budgets[str(persona.get("time_budget_minutes", "unknown"))] += 1
        knowledge_profiles[str(persona.get("historical_knowledge", "unknown"))] += 1
        periods[str(config.get("period", "unknown"))] += 1
        reasoning_skills[str(config.get("reasoning_skill", "unknown"))] += 1

    review_rate = human_reviewed / independently_labeled if independently_labeled else 0.0
    global_reasons = []
    if independently_labeled and review_rate < 0.10:
        global_reasons.append("human_review_rate_below_10_percent")

    return DatasetAudit(
        total=len(cases),
        accepted_ids=accepted_ids,
        rejected=rejected,
        exact_duplicate_count=exact_duplicates,
        repeated_feedback_count=repeated_feedback,
        human_review_rate=round(review_rate, 4),
        global_reasons=global_reasons,
        distributions={
            "total": dict(sorted(totals.items())),
            "failure_type": dict(sorted(failures.items())),
            "prompt_family": dict(sorted(prompt_families.items())),
            "time_budget": dict(sorted(time_budgets.items())),
            "knowledge_profile": dict(sorted(knowledge_profiles.items())),
            "period": dict(sorted(periods.items())),
            "reasoning_skill": dict(sorted(reasoning_skills.items())),
        },
    )


def audit_case_collection(
    cases: list[FRQCase], *, allowed_splits: set[str]
) -> DatasetAudit:
    rejected: dict[str, list[str]] = {}
    accepted_ids: list[str] = []
    seen_essays: dict[str, str] = {}
    seen_feedback: dict[str, str] = {}
    exact_duplicates = 0
    repeated_feedback = 0
    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    families: Counter[str] = Counter()
    periods: Counter[str] = Counter()
    reasoning_skills: Counter[str] = Counter()

    for case in cases:
        reasons: list[str] = []
        if case.split not in allowed_splits:
            reasons.append("unexpected_split")
        gate_ok, gate_reasons = passes_quality_gate(case, strict=True)
        if not gate_ok:
            reasons.extend(gate_reasons)

        essay_norm = normalize_essay(case.student_response)
        if essay_norm in seen_essays:
            reasons.append(f"exact_duplicate_of:{seen_essays[essay_norm]}")
            exact_duplicates += 1
        else:
            seen_essays[essay_norm] = case.id

        feedback_norm = normalize_essay(" ".join(case.reference_feedback.model_dump().values()))
        if feedback_norm in seen_feedback:
            reasons.append(f"repeated_feedback_of:{seen_feedback[feedback_norm]}")
            repeated_feedback += 1
        else:
            seen_feedback[feedback_norm] = case.id

        if reasons:
            rejected[case.id] = sorted(set(reasons))
            continue
        accepted_ids.append(case.id)
        totals[str(case.reference_scores.total)] += 1
        failures[case.failure_type.value] += 1
        families[case.provenance.prompt_family_id or "unknown"] += 1
        config = case.provenance.generator_config
        periods[str(config.get("period", "unknown"))] += 1
        reasoning_skills[str(config.get("reasoning_skill", "unknown"))] += 1

    return DatasetAudit(
        total=len(cases),
        accepted_ids=accepted_ids,
        rejected=rejected,
        exact_duplicate_count=exact_duplicates,
        repeated_feedback_count=repeated_feedback,
        distributions={
            "total": dict(sorted(totals.items())),
            "failure_type": dict(sorted(failures.items())),
            "prompt_family": dict(sorted(families.items())),
            "period": dict(sorted(periods.items())),
            "reasoning_skill": dict(sorted(reasoning_skills.items())),
        },
    )


def assemble_training_mix(
    realistic: list[FRQCase],
    adversarial: list[FRQCase],
    *,
    target_count: int,
    adversarial_ratio: float = 0.15,
    seed: int = 13,
) -> list[FRQCase]:
    if not 0 <= adversarial_ratio <= 1:
        raise ValueError("adversarial_ratio must be between 0 and 1")
    adversarial_target = round(target_count * adversarial_ratio)
    realistic_target = target_count - adversarial_target
    selected: list[FRQCase] = []
    seen_ids: set[str] = set()
    seen_essays: set[str] = set()
    seen_feedback: set[str] = set()

    def append_unique(candidates: list[FRQCase], limit: int) -> None:
        added = 0
        for case in candidates:
            if added >= limit:
                break
            essay = normalize_essay(case.student_response)
            feedback = normalize_essay(" ".join(case.reference_feedback.model_dump().values()))
            if case.id in seen_ids or essay in seen_essays or feedback in seen_feedback:
                continue
            selected.append(case)
            seen_ids.add(case.id)
            seen_essays.add(essay)
            seen_feedback.add(feedback)
            added += 1

    append_unique(realistic, realistic_target)
    append_unique(adversarial, adversarial_target)
    if len(selected) < target_count:
        append_unique(realistic + adversarial, target_count - len(selected))
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def write_training_artifacts(
    output_dir: Path,
    cases: list[FRQCase],
    audit: DatasetAudit,
    *,
    seed: int,
    settings: dict,
    additional_case_sets: dict[str, list[FRQCase]] | None = None,
    force: bool = False,
) -> DatasetManifest:
    realistic_path = output_dir / "train_realistic_v2.jsonl"
    adversarial_path = output_dir / "train_adversarial_v2.jsonl"
    chat_path = output_dir / "train_chat_v2.jsonl"
    manifest_path = output_dir / "dataset_manifest_v2.json"
    additional = additional_case_sets or {}
    additional_paths = [output_dir / name for name in additional]
    for path in (realistic_path, adversarial_path, chat_path, manifest_path, *additional_paths):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")

    if not audit.clean:
        raise ValueError("Dataset audit is not clean; refusing to write v2 training artifacts")

    realistic = [case for case in cases if case.failure_type.value not in ADVERSARIAL_TYPES]
    adversarial = [case for case in cases if case.failure_type.value in ADVERSARIAL_TYPES]
    write_jsonl(realistic_path, realistic)
    write_jsonl(adversarial_path, adversarial)
    write_jsonl(chat_path, to_chat_rows(cases))
    records = [
        _artifact_record(realistic_path, rows=len(realistic)),
        _artifact_record(adversarial_path, rows=len(adversarial)),
        _artifact_record(chat_path, rows=len(cases)),
    ]
    track_audits: dict[str, DatasetAudit] = {}
    for name, artifact_cases in sorted(additional.items()):
        allowed_splits = {"adversarial"} if "challenge" in name else {"eval"}
        track_audit = audit_case_collection(artifact_cases, allowed_splits=allowed_splits)
        if not track_audit.clean:
            raise ValueError(f"Additional track audit failed for {name}")
        track_audits[name] = track_audit
        path = output_dir / name
        write_jsonl(path, artifact_cases)
        records.append(_artifact_record(path, rows=len(artifact_cases)))
    manifest = DatasetManifest(
        seed=seed,
        split_policy="prompt-family split; official and external essays excluded from training",
        artifacts=records,
        settings=settings,
        audit=audit,
        track_audits=track_audits,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _artifact_record(path: Path, *, rows: int | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        rows=rows,
    )
