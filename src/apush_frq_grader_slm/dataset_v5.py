"""Auditable, privacy-aware data preparation primitives for the final v5 run.

This module deliberately does not call a model.  It plans blinded cloud work,
validates returned reviews, and assembles data only after the human approval
gate has been satisfied.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from apush_frq_grader_slm.dataset_v4 import CBSeedProfile
from apush_frq_grader_slm.ingest.dedup import normalize_essay
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.dataset_v4_seeds import infer_failure_type
from apush_frq_grader_slm.schemas import FRQCase, RubricFeedback, RubricScores

V5_CANDIDATE_COUNT = 1_500
V5_SHARD_COUNT = 30
V5_SHARD_SIZE = 50
V5_ACCEPTED_COUNT = 600
V5_GOLDEN_MATCHED_COUNT = 420
V5_BOUNDARY_COUNT = 180
V5_DEV_COUNT = 60
V5_REPLAY_COUNT = 75
V5_MANUAL_REVIEW_COUNT = 60
BOUNDARY_TYPES = (
    "thesis_0_1", "contextualization_0_1", "evidence_0_1",
    "evidence_1_2", "analysis_reasoning_0_1", "analysis_reasoning_1_2",
)
PRIVATE_USE_NOTICE = (
    "Private training use only. Do not redistribute essays, style excerpts, "
    "per-case labels, or review records. Aggregate audits may be shared."
)


@dataclass(frozen=True)
class V5GenerationTask:
    task_id: str
    shard_id: str
    prompt: str
    prompt_family_id: str
    style_seed_id: str
    style_excerpt: str
    period: int | None
    reasoning_skill: str
    capability_profile: dict[str, Any]
    composition_profile: dict[str, Any]
    amsco_chapter_ids: tuple[str, ...]
    coverage_class: str = "golden_matched"
    boundary_type: str = ""
    contrast_pair_id: str = ""
    contrast_side: str = ""
    private_use_notice: str = PRIVATE_USE_NOTICE

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["amsco_chapter_ids"] = list(self.amsco_chapter_ids)
        return row


_CAPABILITIES = (
    {"historical_knowledge": "limited", "argument_control": "emerging"},
    {"historical_knowledge": "uneven", "argument_control": "partial"},
    {"historical_knowledge": "competent", "argument_control": "consistent"},
    {"historical_knowledge": "strong", "argument_control": "nuanced"},
)
_COMPOSITIONS = (
    {"time_pressure": "severe", "mechanics": "frequent_natural_errors", "organization": "rough"},
    {"time_pressure": "moderate", "mechanics": "occasional_errors", "organization": "uneven"},
    {"time_pressure": "normal", "mechanics": "minor_errors", "organization": "clear"},
    {"time_pressure": "moderate", "mechanics": "fragments_and_runons", "organization": "repetitive"},
)

_BOUNDARY_BEHAVIORS = {
    "thesis_0_1": {
        "lower": "discusses the topic but never settles on one overall answer",
        "upper": "states one overall answer and previews why it is defensible",
    },
    "contextualization_0_1": {
        "lower": "starts directly with the period being asked about",
        "upper": "briefly explains an earlier or broader development before the argument",
    },
    "evidence_0_1": {
        "lower": "relies on broad trends and at most one concrete historical example",
        "upper": "recalls at least two concrete relevant examples but connects them unevenly",
    },
    "evidence_1_2": {
        "lower": "mentions concrete examples without consistently using them to prove the argument",
        "upper": "explains how concrete examples support the argument",
    },
    "analysis_reasoning_0_1": {
        "lower": "lists developments without organizing them through a historical relationship",
        "upper": "organizes the explanation around cause, comparison, or change over time",
    },
    "analysis_reasoning_1_2": {
        "lower": "uses a clear historical relationship but stays straightforward",
        "upper": "qualifies the argument or sustains multiple connected perspectives",
    },
}


def plan_v5_tasks(
    seeds: Sequence[CBSeedProfile], *, count: int = V5_CANDIDATE_COUNT, seed: int = 51
) -> list[V5GenerationTask]:
    """Plan score-blind tasks, evenly assigned to exactly 30 cloud shards."""
    if count != V5_CANDIDATE_COUNT:
        raise ValueError(f"v5 final plan requires exactly {V5_CANDIDATE_COUNT} candidates")
    if not seeds:
        raise ValueError("at least one CB-derived seed profile is required")
    rng = random.Random(seed)
    specifications: list[tuple[str, str, str, str]] = []
    # Generate surplus for every boundary while keeping 70%+ of the campaign distribution-like.
    for boundary_type in BOUNDARY_TYPES:
        for pair_index in range(36):
            pair_id = f"{boundary_type}-{pair_index:02d}"
            specifications.extend(
                ("boundary", boundary_type, pair_id, side) for side in ("lower", "upper")
            )
    specifications.extend(("golden_matched", "", "", "") for _ in range(count - len(specifications)))
    rng.shuffle(specifications)
    shard_slots = list(range(count))
    rng.shuffle(shard_slots)
    tasks: list[V5GenerationTask] = []
    for index, (coverage, boundary_type, pair_id, side) in enumerate(specifications):
        if coverage == "boundary":
            pair_number = int(pair_id.rsplit("-", 1)[-1])
            seed_index = (BOUNDARY_TYPES.index(boundary_type) * 36 + pair_number) % len(seeds)
        else:
            seed_index = index % len(seeds)
        profile = seeds[seed_index]
        prompt_options = profile.adapted_prompts or (profile.prompt,)
        prompt_selector = pair_number if coverage == "boundary" else index // len(seeds)
        prompt = prompt_options[prompt_selector % len(prompt_options)]
        style_excerpt = profile.style_excerpt.strip()[:400]
        capability = dict(_CAPABILITIES[(index + index // 7) % len(_CAPABILITIES)])
        if coverage == "boundary":
            capability["observable_writing_behavior"] = _BOUNDARY_BEHAVIORS[boundary_type][side]
        composition = dict(_COMPOSITIONS[(index * 3 + index // 11) % len(_COMPOSITIONS)])
        shard_number = shard_slots[index] // V5_SHARD_SIZE
        tasks.append(V5GenerationTask(
            task_id=f"v5-{index:04d}", shard_id=f"v5-shard-{shard_number:02d}",
            prompt=prompt, prompt_family_id=profile.prompt_family_id,
            style_seed_id=profile.seed_id, style_excerpt=style_excerpt,
            period=profile.period, reasoning_skill=profile.reasoning_skill,
            capability_profile=capability, composition_profile=composition,
            amsco_chapter_ids=profile.amsco_chapter_ids,
            coverage_class=coverage,
            boundary_type=boundary_type,
            contrast_pair_id=pair_id,
            contrast_side=side,
        ))
    counts = Counter(task.shard_id for task in tasks)
    if len(counts) != V5_SHARD_COUNT or set(counts.values()) != {V5_SHARD_SIZE}:
        raise AssertionError("planner failed the 30 x 50 shard invariant")
    return tasks


def normalize_external_candidate(
    task: V5GenerationTask, external_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore trusted planner metadata without exposing it to the writer."""

    if str(external_row.get("task_id")) != task.task_id:
        raise ValueError("External candidate task_id does not match its planned task")
    result = dict(external_row)
    result.update(
        {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "prompt_family_id": task.prompt_family_id,
            "style_seed_id": task.style_seed_id,
            "selection_class": task.coverage_class,
            "boundary_type": task.boundary_type,
            "contrast_pair_id": task.contrast_pair_id,
            "contrast_side": task.contrast_side,
        }
    )
    return result


def generator_packet(task: V5GenerationTask, fact_cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the score-blind packet shown to a cloud essay writer."""
    cards = []
    for item in fact_cards:
        concept = str(item.get("concept") or item.get("fact") or "").strip()
        if concept:
            cards.append({"concept": concept, "use": "paraphrase from memory; do not copy wording"})
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "student_capability": task.capability_profile,
        "timed_composition_style": task.composition_profile,
        "style_reference": task.style_excerpt,
        "semantic_fact_cards": cards,
        "instructions": (
            "Write one authentic timed APUSH student essay. Follow the student's capability and "
            "composition profile. Paraphrase remembered evidence; do not quote fact cards or imitate "
            "the reference verbatim. Let grammar, spelling, fragments, repetition, and uncertainty "
            "arise naturally rather than mechanically corrupting polished prose. Return only the essay."
        ),
        "private_use_notice": task.private_use_notice,
    }


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = normalize_essay(text).split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def overlap_reasons(
    essay: str, source_texts: Iterable[str], *, allowed_phrases: Iterable[str] = ()
) -> list[str]:
    """Detect exact/near duplicates and normalized eight-word source copying."""
    norm = normalize_essay(essay)
    if len(norm) < 80:
        return ["essay_too_short_for_overlap_audit"]
    allowed = set().union(*(_ngrams(item) for item in allowed_phrases)) if allowed_phrases else set()
    essay_grams = _ngrams(essay) - allowed
    essay_words = set(norm.split())
    for source in source_texts:
        source_norm = normalize_essay(source)
        if norm == source_norm:
            return ["exact_duplicate"]
        source_words = set(source_norm.split())
        union = essay_words | source_words
        if union and len(essay_words & source_words) / len(union) >= 0.82:
            return ["near_duplicate"]
        if essay_grams & (_ngrams(source) - allowed):
            return ["verbatim_eight_word_overlap"]
    return []


def candidate_gate_reasons(
    row: Mapping[str, Any], *, source_texts: Iterable[str] = (), allowed_phrases: Iterable[str] = ()
) -> list[str]:
    """Apply blind authenticity, rubric-consensus, fact, and copying gates."""
    reasons: list[str] = []
    essay = str(row.get("student_response") or row.get("essay") or "").strip()
    reasons.extend(overlap_reasons(essay, source_texts, allowed_phrases=allowed_phrases))

    auth = list(row.get("authenticity_reviews") or [])
    if len(auth) < 2:
        reasons.append("missing_two_authenticity_reviews")
    else:
        if len({str(r.get("reviewer_id") or "") for r in auth}) != len(auth):
            reasons.append("authenticity_reviewers_not_independent")
        decisions = [bool(r.get("student_like")) and bool(r.get("timed_ap_consistent")) for r in auth]
        if decisions[0] != decisions[1] and len(auth) < 3:
            reasons.append("authenticity_disagreement_requires_third_reader")
        elif sum(decisions) < 2:
            reasons.append("authenticity_gate_failed")

    reviews = list(row.get("rubric_reviews") or [])
    if len(reviews) < 3:
        reasons.append("missing_three_blind_rubric_reviews")
    else:
        if len({str(r.get("reader_id") or "") for r in reviews[:3]}) != 3:
            reasons.append("rubric_readers_not_independent")
        for review in reviews[:3]:
            try:
                RubricScores.model_validate(review.get("scores"))
            except Exception:
                reasons.append("invalid_reader_scores")
        score_rows = [r.get("scores") for r in reviews[:3]]
        low_confidence = any(float(r.get("confidence", 0)) < 0.85 for r in reviews[:3])
        disagreement = len({json.dumps(s, sort_keys=True) for s in score_rows}) > 1
        resolution = row.get("resolved_grade") or {}
        if (low_confidence or disagreement) and not bool(resolution.get("adjudicated")):
            reasons.append("rubric_adjudication_required")
        if not resolution.get("scores"):
            reasons.append("missing_resolved_scores")
        else:
            try:
                RubricScores.model_validate(resolution["scores"])
            except Exception:
                reasons.append("invalid_resolved_scores")
        if not resolution.get("feedback"):
            reasons.append("missing_resolved_feedback")
        else:
            try:
                RubricFeedback.model_validate(resolution["feedback"])
            except Exception:
                reasons.append("invalid_resolved_feedback")

    fact = row.get("fact_check") or {}
    if not bool(fact.get("passed")):
        reasons.append("historical_fact_check_failed")
    if row.get("selection_class") not in {"golden_matched", "boundary"}:
        reasons.append("invalid_selection_class")
    elif row.get("selection_class") == "golden_matched" and not bool(
        (row.get("distribution_match") or {}).get("passed")
    ):
        reasons.append("golden_distribution_match_not_verified")
    elif row.get("selection_class") == "boundary":
        if row.get("boundary_type") not in BOUNDARY_TYPES:
            reasons.append("invalid_boundary_type")
        if not row.get("contrast_pair_id") or row.get("contrast_side") not in {"lower", "upper"}:
            reasons.append("incomplete_boundary_pair_metadata")
    if not row.get("prompt"):
        reasons.append("missing_prompt")
    if not row.get("prompt_family_id") or not row.get("style_seed_id"):
        reasons.append("missing_split_group_metadata")
    return reasons


def apply_manual_correction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a reviewer's score/feedback correction without accepting arbitrary edits."""

    result = dict(row)
    review = result.get("manual_review") or {}
    if review.get("decision") != "corrected":
        return result
    corrections = review.get("corrections") or {}
    allowed = {"scores", "feedback"}
    unknown = set(corrections) - allowed
    if unknown:
        raise ValueError(f"Unsupported manual correction fields: {sorted(unknown)}")
    if not corrections:
        raise ValueError("A corrected manual-review decision requires score or feedback changes")
    resolution = dict(result.get("resolved_grade") or {})
    resolution.update(corrections)
    resolution["human_corrected"] = True
    result["resolved_grade"] = resolution
    return result


def candidate_to_case(row: Mapping[str, Any], *, split: str) -> FRQCase:
    """Convert an approved external review record into the repository training schema."""

    if split not in {"train", "dev"}:
        raise ValueError(f"Unsupported v5 split: {split}")
    row = apply_manual_correction(row)
    reasons = candidate_gate_reasons(row)
    # Copying is checked during selection with the full private overlap corpus. Rechecking
    # here without that corpus still catches every structural/review failure.
    reasons = [reason for reason in reasons if not reason.startswith("essay_too_short")]
    if reasons:
        raise ValueError(f"Candidate {row.get('task_id')} is not finalizable: {reasons}")

    resolution = dict(row["resolved_grade"])
    scores = RubricScores.model_validate(resolution["scores"])
    feedback = RubricFeedback.model_validate(resolution["feedback"])
    payload = {
        "scores": scores.model_dump(),
        "total": scores.total,
        "feedback": feedback.model_dump(),
    }
    reviews = list(row.get("rubric_reviews") or [])
    score_signatures = [json.dumps(review.get("scores"), sort_keys=True) for review in reviews]
    agreement = max(Counter(score_signatures).values()) / len(score_signatures)
    reader_confidence = [float(review.get("confidence", 0)) for review in reviews]
    review = row.get("manual_review") or {}
    human_reviewed = review.get("decision") in {"accept", "corrected"}
    failure_type = infer_failure_type(scores)
    difficulty = "weak" if scores.total <= 2 else "strong" if scores.total >= 5 else "borderline"
    case = FRQCase.model_validate(
        {
            "id": str(row["task_id"]),
            "split": "train" if split == "train" else "eval",
            "prompt": str(row["prompt"]),
            "student_response": str(row.get("student_response") or row.get("essay")),
            "reference_scores": scores,
            "reference_feedback": feedback,
            "failure_type": failure_type,
            "difficulty": difficulty,
            "assistant_response": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "tags": [
                "synth_v5",
                "cloud_generated",
                "triple_blind_reviewed",
                str(row["selection_class"]),
            ],
            "provenance": {
                "source_type": "synthetic",
                "source_id": str(row["task_id"]),
                "prompt_family_id": str(row["prompt_family_id"]),
                "generator_name": "external_v5_score_blind",
                "generator_config": {
                    "style_seed_id": str(row["style_seed_id"]),
                    "selection_class": str(row["selection_class"]),
                    "boundary_type": str(row.get("boundary_type") or ""),
                    "contrast_pair_id": str(row.get("contrast_pair_id") or ""),
                },
                "review_status": "human_verified" if human_reviewed else "machine_checked",
            },
            "labeling": {
                "method": "adjudicated" if resolution.get("adjudicated") else "independent_consensus",
                "grader_ids": [str(item.get("reader_id") or "") for item in reviews],
                "agreement": agreement,
                "confidence": min(reader_confidence),
                "adjudicated": bool(resolution.get("adjudicated")),
                "human_reviewed": human_reviewed,
                "protocol_version": "v5_triple_blind_authenticity_v1",
                "resolution": "human_corrected" if resolution.get("human_corrected") else "cloud_consensus",
            },
        }
    )
    passed, quality_reasons = passes_quality_gate(case)
    if not passed:
        raise ValueError(f"Candidate {case.id} failed final training quality gates: {quality_reasons}")
    return case


def _grouped_dev_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["prompt_family_id"]), str(row["style_seed_id"]))].append(row)
    # Exact stratified subset: 42 golden-matched + 18 boundary = 60.
    states: dict[tuple[int, int], tuple[tuple[str, str], ...]] = {(0, 0): ()}
    for key in sorted(groups):
        members = groups[key]
        g = sum(r["selection_class"] == "golden_matched" for r in members)
        b = len(members) - g
        for state, chosen in list(states.items())[::-1]:
            nxt = (state[0] + g, state[1] + b)
            if nxt[0] <= 42 and nxt[1] <= 18 and nxt not in states:
                states[nxt] = chosen + (key,)
    chosen = states.get((42, 18))
    if chosen is None:
        raise ValueError("cannot create grouped 42/18 development split; add or regroup candidates")
    keys = set(chosen)
    return {str(r["task_id"]) for key in keys for r in groups[key]}


def _score_signature(scores: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(scores[key]) for key in ("thesis", "contextualization", "evidence", "analysis_reasoning"))


def _matched_score_sample(
    rows: Sequence[dict], golden_cases: Sequence[FRQCase], count: int
) -> list[dict]:
    """Scale the golden joint score-vector distribution with largest-remainder quotas."""

    golden_counts = Counter(_score_signature(case.reference_scores.model_dump()) for case in golden_cases)
    exact = {key: value * count / len(golden_cases) for key, value in golden_counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(exact, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[:remaining]:
        quotas[key] += 1
    buckets: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        scores = (row.get("resolved_grade") or {}).get("scores") or {}
        try:
            buckets[_score_signature(scores)].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    selected: list[dict] = []
    for signature, quota in sorted(quotas.items()):
        bucket = sorted(buckets[signature], key=lambda row: str(row["task_id"]))
        if len(bucket) < quota:
            raise ValueError(
                f"golden-matched pool needs {quota} cases with score vector {signature}; "
                f"only {len(bucket)} passed"
            )
        selected.extend(bucket[:quota])
    if len(selected) != count:
        raise AssertionError("golden score-vector quota selection failed")
    return selected


def style_features(text: str) -> dict[str, float]:
    """Extract transparent timed-writing features without retaining golden wording."""

    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    sentence_lengths = [len(re.findall(r"[A-Za-z]+", part)) for part in sentences] or [0]
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    denominator = max(len(words), 1) / 100
    informal = re.findall(r"\b(?:wasnt|didnt|couldnt|wouldnt|dont|cant|alot|goverment)\b", text.lower())
    lowercase_i = re.findall(r"(?<![A-Za-z])i(?![A-Za-z])", text)
    return {
        "word_count": float(len(words)),
        "paragraph_count": float(len(paragraphs) or bool(text.strip())),
        "sentence_word_mean": float(sum(sentence_lengths) / len(sentence_lengths)),
        "sentence_word_std": float(statistics.pstdev(sentence_lengths)),
        "informal_marker_per_100": float((len(informal) + len(lowercase_i)) / denominator),
        "punctuation_per_100": float(len(re.findall(r"[,;:!?]", text)) / denominator),
    }


def style_distribution_audit(
    rows: Sequence[Mapping[str, Any]], golden_cases: Sequence[FRQCase]
) -> dict[str, Any]:
    """Compare aggregate candidate mechanics to golden cases without exposing their text."""

    candidate_features = [
        style_features(str(row.get("student_response") or row.get("essay") or "")) for row in rows
    ]
    golden_features = [style_features(case.student_response) for case in golden_cases]
    tolerances = {
        "word_count": 0.20,
        "paragraph_count": 0.30,
        "sentence_word_mean": 0.25,
        "sentence_word_std": 0.35,
        "informal_marker_per_100": 1.00,
        "punctuation_per_100": 0.40,
    }
    metrics: dict[str, Any] = {}
    passed = True
    for key, relative_tolerance in tolerances.items():
        candidate_mean = sum(item[key] for item in candidate_features) / len(candidate_features)
        golden_mean = sum(item[key] for item in golden_features) / len(golden_features)
        floor = 0.5 if key in {"paragraph_count", "informal_marker_per_100"} else 0.1
        allowed_delta = max(abs(golden_mean) * relative_tolerance, floor)
        metric_passed = abs(candidate_mean - golden_mean) <= allowed_delta
        passed = passed and metric_passed
        metrics[key] = {
            "candidate_mean": round(candidate_mean, 4),
            "golden_mean": round(golden_mean, 4),
            "allowed_delta": round(allowed_delta, 4),
            "passed": metric_passed,
        }
    return {"passed": passed, "metrics": metrics, "golden_text_retained": False}


def assemble_v5_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_texts: Iterable[str] = (),
    golden_cases: Sequence[FRQCase] = (),
) -> tuple[list[dict], list[dict]]:
    """Select the final 420/180 corpus and make a leakage-safe 540/60 split."""
    sources = list(source_texts)
    clean: list[dict] = []
    for row in sorted(rows, key=lambda r: str(r.get("task_id", ""))):
        if candidate_gate_reasons(row, source_texts=sources):
            continue
        clean.append(dict(row))
        sources.append(str(row.get("student_response") or row.get("essay") or ""))
    by_class = {
        name: sorted((r for r in clean if r["selection_class"] == name), key=lambda r: str(r["task_id"]))
        for name in ("golden_matched", "boundary")
    }
    if len(by_class["golden_matched"]) < V5_GOLDEN_MATCHED_COUNT:
        raise ValueError("fewer than 420 accepted golden-matched candidates")
    golden_selected = (
        _matched_score_sample(by_class["golden_matched"], golden_cases, V5_GOLDEN_MATCHED_COUNT)
        if golden_cases
        else by_class["golden_matched"][:V5_GOLDEN_MATCHED_COUNT]
    )
    boundary_selected: list[dict] = []
    for boundary_type in BOUNDARY_TYPES:
        pairs: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in by_class["boundary"]:
            if row.get("boundary_type") == boundary_type:
                pairs[str(row["contrast_pair_id"])][str(row["contrast_side"])] = row
        complete = [pairs[key] for key in sorted(pairs) if set(pairs[key]) == {"lower", "upper"}]
        if len(complete) < 15:
            raise ValueError(f"fewer than 15 complete contrast pairs for {boundary_type}")
        for pair in complete[:15]:
            boundary_selected.extend((pair["lower"], pair["upper"]))
    selected = golden_selected + boundary_selected
    if golden_cases:
        style_audit = style_distribution_audit(golden_selected, golden_cases)
        if not style_audit["passed"]:
            failed = [key for key, value in style_audit["metrics"].items() if not value["passed"]]
            raise ValueError(f"golden-matched style distribution failed: {failed}")
    dev_ids = _grouped_dev_ids(selected)
    train, dev = [], []
    for original in selected:
        row = dict(original)
        row["split"] = "dev" if str(row["task_id"]) in dev_ids else "train"
        row["private_use_notice"] = PRIVATE_USE_NOTICE
        (dev if row["split"] == "dev" else train).append(row)
    if len(train) != 540 or len(dev) != 60:
        raise AssertionError("v5 split invariant failed")
    return train, dev


def select_v4_replay(cases: Sequence[FRQCase], *, count: int = V5_REPLAY_COUNT) -> list[FRQCase]:
    """Select deterministic high-agreement replay data, round-robin by observed total."""
    eligible = [c for c in cases if (c.labeling.agreement or 0) >= 0.85 or c.labeling.human_reviewed]
    buckets: dict[int, list[FRQCase]] = defaultdict(list)
    for case in sorted(eligible, key=lambda c: c.id):
        buckets[case.reference_scores.total].append(case)
    chosen: list[FRQCase] = []
    while len(chosen) < count and any(buckets.values()):
        for total in range(7):
            if buckets[total] and len(chosen) < count:
                chosen.append(buckets[total].pop(0))
    if len(chosen) != count:
        raise ValueError(f"only {len(chosen)} of {count} required high-agreement v4 replay cases exist")
    return chosen


def manual_review_packet(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose a deterministic 10% packet, stratified by class and observed total."""
    buckets: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: str(r["task_id"])):
        scores = (row.get("resolved_grade") or {}).get("scores") or {}
        buckets[(str(row["selection_class"]), sum(int(v) for v in scores.values()))].append(row)
    packet: list[dict[str, Any]] = []
    while len(packet) < V5_MANUAL_REVIEW_COUNT and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(packet) < V5_MANUAL_REVIEW_COUNT:
                row = dict(buckets[key].pop(0))
                row["manual_review"] = {"decision": "pending", "corrections": {}}
                row["private_use_notice"] = PRIVATE_USE_NOTICE
                packet.append(row)
    if len(packet) != V5_MANUAL_REVIEW_COUNT:
        raise ValueError("need 60 accepted cases for the manual review packet")
    return packet


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_manual_approval(packet_path: Path, approval_path: Path) -> dict[str, Any]:
    """Block final assembly unless the reviewed packet and signed approval match."""
    if not approval_path.exists():
        raise PermissionError("manual review approval is missing")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not approval.get("approved") or not approval.get("reviewer") or not approval.get("approved_at"):
        raise PermissionError("manual review approval is incomplete")
    if approval.get("packet_sha256") != file_sha256(packet_path):
        raise PermissionError("manual review packet changed after approval")
    rows = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 60 or any((r.get("manual_review") or {}).get("decision") not in {"accept", "corrected"} for r in rows):
        raise PermissionError("all 60 manual review decisions must be accept or corrected")
    return approval
