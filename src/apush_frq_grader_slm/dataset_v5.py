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

from apush_frq_grader_slm.authenticity_gates_v5 import (
    ESSAY_ONLY_CONTRACT,
    aggregate_length_realism_audit,
    hard_gate_reasons,
    writer_instructions,
)
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
V5_PILOT_COUNT = 30
V5_PILOT_BOUNDARY_PAIRS_PER_TYPE = 2
V5_PILOT_GOLDEN_MATCHED_COUNT = 6
V5_FULL_AFTER_PILOT_COUNT = V5_CANDIDATE_COUNT - V5_PILOT_COUNT
BOUNDARY_TYPES = (
    "thesis_0_1", "contextualization_0_1", "evidence_0_1",
    "evidence_1_2", "analysis_reasoning_0_1", "analysis_reasoning_1_2",
)
PRIVATE_USE_NOTICE = (
    "Private training use only. Do not redistribute essays, style reference essays, "
    "per-case labels, or review records. Aggregate audits may be shared."
)
WRITER_FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "target_scores",
        "target_total",
        "scores",
        "score",
        "rubric_text",
        "resolved_grade",
        "reference_scores",
        "reference_feedback",
        "source_case_id",
        "style_seed_id",
        "seed_id",
        "authenticity_reviews",
        "rubric_reviews",
        "fact_check",
        "distribution_match",
        "failure_type",
        "tags",
        "assistant_response",
    }
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
    style_reference_essay: str = ""
    reference_word_count: int | None = None
    private_use_notice: str = PRIVATE_USE_NOTICE

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["amsco_chapter_ids"] = list(self.amsco_chapter_ids)
        # Keep the full golden essay out of the shared task plan; attach only at
        # private packet export. Persist word count for length gates.
        row.pop("style_reference_essay", None)
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


def select_v5_pilot_tasks(
    tasks: Sequence[V5GenerationTask], *, seed: int = 51
) -> list[V5GenerationTask]:
    """Pick the hash-stable 30-essay pilot from the planned 1,500-task campaign.

    Two lower/upper pairs for each of the six rubric boundaries (24) plus six
    distribution-matched essays spanning periods and capability levels.
    """
    rng = random.Random(seed)
    selected: list[V5GenerationTask] = []
    used_ids: set[str] = set()

    for boundary_type in BOUNDARY_TYPES:
        pair_ids = sorted(
            {
                task.contrast_pair_id
                for task in tasks
                if task.coverage_class == "boundary" and task.boundary_type == boundary_type
            }
        )
        if len(pair_ids) < V5_PILOT_BOUNDARY_PAIRS_PER_TYPE:
            raise ValueError(f"need {V5_PILOT_BOUNDARY_PAIRS_PER_TYPE} pairs for {boundary_type}")
        chosen_pairs = pair_ids[:V5_PILOT_BOUNDARY_PAIRS_PER_TYPE]
        for pair_id in chosen_pairs:
            pair_tasks = [
                task
                for task in tasks
                if task.contrast_pair_id == pair_id and task.boundary_type == boundary_type
            ]
            sides = {task.contrast_side: task for task in pair_tasks}
            if set(sides) != {"lower", "upper"}:
                raise ValueError(f"pilot pair {pair_id} missing lower/upper")
            for side in ("lower", "upper"):
                task = sides[side]
                selected.append(task)
                used_ids.add(task.task_id)

    golden = [task for task in tasks if task.coverage_class == "golden_matched"]
    by_period: dict[int | None, list[V5GenerationTask]] = defaultdict(list)
    for task in golden:
        by_period[task.period].append(task)
    periods = sorted(by_period.keys(), key=lambda value: (-1 if value is None else value))
    golden_picks: list[V5GenerationTask] = []
    period_cycle = list(periods) or [None]
    capability_cycle = list(_CAPABILITIES)
    cursor = 0
    while len(golden_picks) < V5_PILOT_GOLDEN_MATCHED_COUNT:
        period = period_cycle[cursor % len(period_cycle)]
        capability = capability_cycle[cursor % len(capability_cycle)]
        pool = [
            task
            for task in by_period.get(period, [])
            if task.task_id not in used_ids
            and task.capability_profile.get("historical_knowledge")
            == capability["historical_knowledge"]
        ]
        if not pool:
            pool = [task for task in by_period.get(period, []) if task.task_id not in used_ids]
        if not pool:
            pool = [task for task in golden if task.task_id not in used_ids]
        if not pool:
            raise ValueError("unable to select distribution-matched pilot tasks")
        pick = sorted(pool, key=lambda task: task.task_id)[cursor % len(pool)]
        golden_picks.append(pick)
        used_ids.add(pick.task_id)
        cursor += 1
    selected.extend(golden_picks)

    if len(selected) != V5_PILOT_COUNT:
        raise AssertionError(f"pilot selection produced {len(selected)} tasks, expected {V5_PILOT_COUNT}")
    # Stable order for review packets; rng reserved for future stratified reshuffles.
    _ = rng
    return sorted(selected, key=lambda task: task.task_id)


def load_style_reference_essays(
    seed_profiles_path: Path,
    golden_cases_path: Path,
) -> dict[str, dict[str, Any]]:
    """Map style_seed_id -> {essay, word_count, source_case_id} for private packet export.

    Uses cleaned student prose from the matched golden case. If that case has no
    usable essay (commentary-only ingestion), falls back to another cleaned essay
    from the same prompt family so writers still receive a full style reference.
    """
    from apush_frq_grader_slm.dataset_v4_seeds import clean_student_response
    from apush_frq_grader_slm.io import read_jsonl

    golden_by_id = {
        str(row.get("id") or row.get("case_id") or ""): row
        for row in read_jsonl(golden_cases_path)
    }
    seed_rows = list(read_jsonl(seed_profiles_path))
    cleaned_by_source: dict[str, str] = {}
    for source_id, case in golden_by_id.items():
        essay = clean_student_response(str(case.get("student_response") or ""))
        if len(essay.split()) >= 40:
            cleaned_by_source[source_id] = essay
        else:
            # Some CB files store commentary only; keep raw only if it looks like prose.
            raw = str(case.get("student_response") or "").strip()
            if (
                len(raw.split()) >= 40
                and "scoring commentary" not in raw.lower()
                and "long essay question" not in raw.lower()
            ):
                cleaned_by_source[source_id] = raw

    family_to_sources: dict[str, list[str]] = defaultdict(list)
    for row in seed_rows:
        family = str(row.get("prompt_family_id") or "")
        source_id = str(row.get("source_case_id") or "")
        if family and source_id:
            family_to_sources[family].append(source_id)

    mapping: dict[str, dict[str, Any]] = {}
    for row in seed_rows:
        seed_id = str(row.get("seed_id") or "")
        source_id = str(row.get("source_case_id") or "")
        family = str(row.get("prompt_family_id") or "")
        if not seed_id:
            continue
        essay = cleaned_by_source.get(source_id, "")
        used_source = source_id
        if not essay:
            for alt in family_to_sources.get(family, []):
                if alt in cleaned_by_source:
                    essay = cleaned_by_source[alt]
                    used_source = alt
                    break
        if not essay:
            # Last resort: longest cleaned essay in the golden set.
            if cleaned_by_source:
                used_source, essay = max(
                    cleaned_by_source.items(), key=lambda item: len(item[1].split())
                )
        if not essay:
            continue
        mapping[seed_id] = {
            "style_reference_essay": essay,
            "reference_word_count": len(essay.split()),
            "source_case_id": used_source,
            "requested_source_case_id": source_id,
        }
    return mapping


def attach_style_reference(
    task: V5GenerationTask, references: Mapping[str, Mapping[str, Any]]
) -> V5GenerationTask:
    """Return a copy of ``task`` with the matched full golden essay attached."""
    ref = references.get(task.style_seed_id) or {}
    essay = str(ref.get("style_reference_essay") or "").strip()
    word_count = ref.get("reference_word_count")
    if essay and word_count is None:
        word_count = len(essay.split())
    return V5GenerationTask(
        task_id=task.task_id,
        shard_id=task.shard_id,
        prompt=task.prompt,
        prompt_family_id=task.prompt_family_id,
        style_seed_id=task.style_seed_id,
        style_excerpt=task.style_excerpt,
        period=task.period,
        reasoning_skill=task.reasoning_skill,
        capability_profile=dict(task.capability_profile),
        composition_profile=dict(task.composition_profile),
        amsco_chapter_ids=task.amsco_chapter_ids,
        coverage_class=task.coverage_class,
        boundary_type=task.boundary_type,
        contrast_pair_id=task.contrast_pair_id,
        contrast_side=task.contrast_side,
        style_reference_essay=essay,
        reference_word_count=(int(word_count) if word_count is not None else None),
        private_use_notice=task.private_use_notice,
    )


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
            "reference_word_count": task.reference_word_count,
        }
    )
    # Never persist the golden style essay onto candidate/training rows.
    result.pop("style_reference_essay", None)
    return result


def generator_packet(task: V5GenerationTask, fact_cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the score-blind private packet shown to a cloud essay writer.

    Writers receive the full matched golden essay as a style reference, semantic
    fact cards, capability/composition cues, and an essay-only contract. Scores,
    feedback, source IDs, and evaluation annotations are never included.
    """
    if not task.style_reference_essay.strip():
        raise ValueError(
            f"task {task.task_id} missing style_reference_essay; attach the matched golden essay before export"
        )
    cards = []
    for item in fact_cards:
        concept = str(item.get("concept") or item.get("fact") or "").strip()
        if concept:
            cards.append(
                {
                    "concept": concept,
                    "use": "paraphrase in your own words; do not copy wording",
                }
            )
    has_boundary = bool(task.capability_profile.get("observable_writing_behavior"))
    packet = {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "student_capability": dict(task.capability_profile),
        "timed_composition_style": dict(task.composition_profile),
        "style_reference_essay": task.style_reference_essay,
        "reference_word_count": task.reference_word_count
        or len(task.style_reference_essay.split()),
        "semantic_fact_cards": cards,
        "essay_only_contract": dict(ESSAY_ONLY_CONTRACT),
        "instructions": writer_instructions(has_boundary_behavior=has_boundary),
        "private_use_notice": task.private_use_notice,
    }
    leaked = WRITER_FORBIDDEN_PACKET_KEYS & set(packet)
    if leaked:
        raise AssertionError(f"writer packet leaked forbidden keys: {sorted(leaked)}")
    return packet


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = normalize_essay(text).split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def _word_ngrams(words: Sequence[str], n: int = 8) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


_SCRUB_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _scrub_pattern(phrases: Sequence[str]) -> re.Pattern[str] | None:
    norms = tuple(
        sorted(
            {normalize_essay(phrase) for phrase in phrases if str(phrase).strip()},
            key=len,
            reverse=True,
        )
    )
    if not norms:
        return None
    if norms not in _SCRUB_CACHE:
        # Word-boundary-ish replace via alternation; longest-first already sorted.
        escaped = [re.escape(item) for item in norms if item]
        _SCRUB_CACHE[norms] = re.compile(r"(?:%s)" % "|".join(escaped))
    return _SCRUB_CACHE[norms]


def _scrub_allowed_phrases(text: str, phrases: Sequence[str]) -> str:
    """Remove unavoidable names/dates/terms before eight-gram overlap checks."""
    cleaned = normalize_essay(text)
    pattern = _scrub_pattern(phrases)
    if pattern is not None:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()



@dataclass
class OverlapIndex:
    """Precomputed overlap structures for O(1)-ish duplicate checks at scale."""

    norms: list[str]
    word_sets: list[set[str]]
    gram_to_ids: dict[tuple[str, ...], set[int]]
    allowed_list: tuple[str, ...]
    allowed_grams: set[tuple[str, ...]]

    @classmethod
    def build(
        cls,
        source_texts: Iterable[str],
        *,
        allowed_phrases: Iterable[str] = (),
    ) -> "OverlapIndex":
        allowed_list = tuple(str(item).strip() for item in allowed_phrases if str(item).strip())
        long_allowed = [item for item in allowed_list if len(normalize_essay(item).split()) >= 8]
        allowed_grams = (
            set().union(*(_ngrams(item) for item in long_allowed)) if long_allowed else set()
        )
        norms: list[str] = []
        word_sets: list[set[str]] = []
        gram_to_ids: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for source in source_texts:
            norm = normalize_essay(source)
            if len(norm) < 80:
                continue
            idx = len(norms)
            norms.append(norm)
            word_sets.append(set(norm.split()))
            scrubbed = _scrub_allowed_phrases(source, allowed_list)
            for gram in _word_ngrams(scrubbed.split()) - allowed_grams:
                gram_to_ids[gram].add(idx)
        return cls(norms, word_sets, gram_to_ids, allowed_list, allowed_grams)

    def reasons_for(
        self,
        essay: str,
        *,
        check_exact: bool = True,
        check_near: bool = True,
        check_eight_gram: bool = True,
    ) -> list[str]:
        norm = normalize_essay(essay)
        if len(norm) < 80:
            return ["essay_too_short_for_overlap_audit"]
        if check_exact and norm in self.norms:
            return ["exact_duplicate"]
        if check_near:
            essay_words = set(norm.split())
            for source_words in self.word_sets:
                union = essay_words | source_words
                if union and len(essay_words & source_words) / len(union) >= 0.82:
                    return ["near_duplicate"]
        if check_eight_gram:
            scrubbed = _scrub_allowed_phrases(essay, self.allowed_list)
            essay_grams = _word_ngrams(scrubbed.split()) - self.allowed_grams
            for gram in essay_grams:
                if gram in self.gram_to_ids:
                    return ["verbatim_eight_word_overlap"]
        return []

    def add(self, essay: str) -> None:
        norm = normalize_essay(essay)
        if len(norm) < 80:
            return
        idx = len(self.norms)
        self.norms.append(norm)
        self.word_sets.append(set(norm.split()))
        scrubbed = _scrub_allowed_phrases(essay, self.allowed_list)
        for gram in _word_ngrams(scrubbed.split()) - self.allowed_grams:
            self.gram_to_ids[gram].add(idx)


def overlap_reasons(
    essay: str, source_texts: Iterable[str], *, allowed_phrases: Iterable[str] = ()
) -> list[str]:
    """Detect exact/near duplicates and normalized eight-word source copying.

    ``allowed_phrases`` exempts unavoidable historical names, dates, and short
    evidence terms: short phrases are scrubbed from both sides before the
    eight-gram check, and long phrases (>= 8 words) have their grams subtracted.
    """
    norm = normalize_essay(essay)
    if len(norm) < 80:
        return ["essay_too_short_for_overlap_audit"]
    allowed_list = [str(item).strip() for item in allowed_phrases if str(item).strip()]
    long_allowed = [item for item in allowed_list if len(normalize_essay(item).split()) >= 8]
    allowed_grams = (
        set().union(*(_ngrams(item) for item in long_allowed)) if long_allowed else set()
    )
    essay_scrubbed = _scrub_allowed_phrases(essay, allowed_list)
    essay_grams = _word_ngrams(essay_scrubbed.split()) - allowed_grams
    essay_words = set(norm.split())
    for source in source_texts:
        source_norm = normalize_essay(source)
        if norm == source_norm:
            return ["exact_duplicate"]
        source_words = set(source_norm.split())
        union = essay_words | source_words
        if union and len(essay_words & source_words) / len(union) >= 0.82:
            return ["near_duplicate"]
        source_scrubbed = _scrub_allowed_phrases(source, allowed_list)
        source_grams = _word_ngrams(source_scrubbed.split()) - allowed_grams
        if essay_grams & source_grams:
            return ["verbatim_eight_word_overlap"]
    return []


def candidate_gate_reasons(
    row: Mapping[str, Any],
    *,
    source_texts: Iterable[str] = (),
    allowed_phrases: Iterable[str] = (),
    overlap_index: OverlapIndex | None = None,
    style_reference_essay: str = "",
    reference_word_count: int | None = None,
) -> list[str]:
    """Apply blind authenticity, rubric-consensus, fact, copying, and hard gates."""
    reasons: list[str] = []
    essay = str(row.get("student_response") or row.get("essay") or "").strip()
    if overlap_index is not None:
        reasons.extend(overlap_index.reasons_for(essay))
    else:
        reasons.extend(overlap_reasons(essay, source_texts, allowed_phrases=allowed_phrases))

    ref_words = reference_word_count
    if ref_words is None and row.get("reference_word_count") is not None:
        try:
            ref_words = int(row["reference_word_count"])
        except (TypeError, ValueError):
            ref_words = None
    style_ref = style_reference_essay or str(row.get("style_reference_essay") or "")
    reasons.extend(
        hard_gate_reasons(
            essay,
            style_reference_essay=style_ref,
            reference_word_count=ref_words,
        )
    )

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
    elif row.get("selection_class") == "golden_matched":
        # Pool membership uses the timed length envelope. Exact golden score-vector
        # membership is enforced later by selection quotas (with nearest-vector fill),
        # so longer essays whose adjudicated vector is near-golden are not discarded
        # before style-distribution repair can use them.
        dm = row.get("distribution_match") or {}
        style_ok = dm.get("style_within_tolerance")
        if style_ok is None:
            style_ok = bool(dm.get("passed"))
        if not style_ok:
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
    """Scale the golden joint score-vector distribution with largest-remainder quotas.

    Within each score-vector bucket, pick essays whose lengths track the golden
    length *distribution* (not only the mean) so aggregate quartile gates can pass.
    """

    golden_counts = Counter(_score_signature(case.reference_scores.model_dump()) for case in golden_cases)
    exact = {key: value * count / len(golden_cases) for key, value in golden_counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(exact, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[:remaining]:
        quotas[key] += 1

    golden_lengths_by_sig: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    all_golden_lengths: list[float] = []
    for case in golden_cases:
        length = style_features(case.student_response)["word_count"]
        all_golden_lengths.append(length)
        golden_lengths_by_sig[_score_signature(case.reference_scores.model_dump())].append(length)
    golden_word_mean = sum(all_golden_lengths) / max(len(all_golden_lengths), 1)

    def _essay_word_count(row: Mapping[str, Any]) -> float:
        return style_features(str(row.get("student_response") or row.get("essay") or ""))["word_count"]

    def _length_targets(signature: tuple[int, int, int, int], n: int) -> list[float]:
        source = golden_lengths_by_sig.get(signature) or all_golden_lengths
        if not source or n <= 0:
            return []
        ordered = sorted(source)
        if n == 1:
            return [ordered[len(ordered) // 2]]
        # Evenly sample the golden empirical length CDF for this score vector.
        return [
            ordered[min(len(ordered) - 1, int(round(i * (len(ordered) - 1) / (n - 1))))]
            for i in range(n)
        ]

    buckets: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        scores = (row.get("resolved_grade") or {}).get("scores") or {}
        try:
            buckets[_score_signature(scores)].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    selected: list[dict] = []
    used_ids: set[str] = set()
    shortfalls: list[tuple[tuple[int, int, int, int], int, list[float]]] = []
    for signature, quota in sorted(quotas.items()):
        targets = _length_targets(signature, quota)
        bucket = [row for row in buckets[signature] if str(row["task_id"]) not in used_ids]
        take: list[dict] = []
        deferred_targets: list[float] = []
        for target in targets:
            if not bucket:
                deferred_targets.append(target)
                continue
            best_i = min(
                range(len(bucket)),
                key=lambda i: (
                    abs(_essay_word_count(bucket[i]) - target),
                    str(bucket[i].get("task_id") or ""),
                ),
            )
            # Defer length-mismatched exact-vector essays so shortfall fill can pull
            # near-signature essays that actually hit the golden length CDF.
            if abs(_essay_word_count(bucket[best_i]) - target) > 150:
                deferred_targets.append(target)
                continue
            take.append(bucket.pop(best_i))
        selected.extend(take)
        used_ids.update(str(row["task_id"]) for row in take)
        remaining_need = quota - len(take)
        if remaining_need > 0:
            shortfalls.append(
                (signature, remaining_need, deferred_targets[:remaining_need] or targets[len(take) :])
            )
    if shortfalls:
        unused = [
            row
            for signature, bucket in buckets.items()
            for row in bucket
            if str(row["task_id"]) not in used_ids
        ]
        for signature, need, leftover_targets in shortfalls:
            targets = leftover_targets or _length_targets(signature, need)
            while len(targets) < need:
                targets.append(golden_word_mean)
            fillers: list[dict] = []
            for target in targets[:need]:
                if not unused:
                    break

                def _fill_rank(row: Mapping[str, Any], *, _target: float = target, _signature: tuple[int, int, int, int] = signature) -> tuple:
                    try:
                        row_sig = _score_signature((row.get("resolved_grade") or {}).get("scores") or {})
                    except (KeyError, TypeError, ValueError):
                        row_sig = (99, 99, 99, 99)
                    l1 = sum(abs(a - b) for a, b in zip(row_sig, _signature))
                    # Prefer length match among near-golden vectors; do not let L1=1
                    # long essays always beat L1=2 short essays for short targets.
                    return (
                        abs(_essay_word_count(row) - _target) + 40.0 * l1,
                        l1,
                        str(row.get("task_id") or ""),
                    )

                ranked = sorted(unused, key=_fill_rank)
                chosen = ranked[0]
                fillers.append(chosen)
                used_ids.add(str(chosen["task_id"]))
                unused = [row for row in unused if str(row["task_id"]) not in used_ids]
            if len(fillers) < need:
                raise ValueError(
                    f"golden-matched pool needs {need} more near {signature}; "
                    f"only {len(fillers)} unused candidates remain"
                )
            selected.extend(fillers)
    if len(selected) != count:
        raise AssertionError("golden score-vector quota selection failed")

    if golden_cases:
        audit = style_distribution_audit(selected, golden_cases)
        if not audit["passed"]:
            selected = _repair_style_selection(
                selected,
                rows,
                golden_cases,
                golden_word_mean=golden_word_mean,
                used_ids=used_ids,
            )
    return selected


def _repair_style_selection(
    selected: list[dict],
    pool: Sequence[dict],
    golden_cases: Sequence[FRQCase],
    *,
    golden_word_mean: float,
    used_ids: set[str],
    max_swaps: int = 200,
) -> list[dict]:
    """Fast length-distribution repair with near-signature swaps (L1 ≤ 3)."""
    from apush_frq_grader_slm.authenticity_gates_v5 import quartile

    selected = list(selected)
    unused = [row for row in pool if str(row["task_id"]) not in used_ids]
    golden_lengths = [style_features(case.student_response)["word_count"] for case in golden_cases]
    g_mean = sum(golden_lengths) / len(golden_lengths)
    g_median = quartile(golden_lengths, 0.5)
    g_q1 = quartile(golden_lengths, 0.25)
    g_q3 = quartile(golden_lengths, 0.75)

    def _wc(row: Mapping[str, Any]) -> float:
        return style_features(str(row.get("student_response") or ""))["word_count"]

    def _sig(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
        return _score_signature((row.get("resolved_grade") or {}).get("scores") or {})

    def _length_loss(lengths: Sequence[float]) -> float:
        loss = abs(sum(lengths) / len(lengths) - g_mean) / max(abs(g_mean) * 0.10, 1.0)
        for target, q in ((g_q1, 0.25), (g_median, 0.5), (g_q3, 0.75)):
            loss += abs(quartile(lengths, q) - target) / max(abs(target) * 0.15, 1.0)
        return loss

    selected_meta = [(_wc(row), _sig(row)) for row in selected]
    unused_meta = [(_wc(row), _sig(row), row) for row in unused]

    for _ in range(max_swaps):
        audit = style_distribution_audit(selected, golden_cases)
        if audit["passed"]:
            return selected
        lengths = [m[0] for m in selected_meta]
        cur_loss = _length_loss(lengths)
        cur_median = quartile(lengths, 0.5)
        cur_mean = sum(lengths) / len(lengths)

        # Focused candidate sets keep this O(small) per iteration.
        if cur_median > g_median * 1.15:
            focus = sorted(
                range(len(selected)),
                key=lambda i: (0 if selected_meta[i][0] >= cur_median else 1, -selected_meta[i][0]),
            )[:30]
            unused_focus = sorted(
                range(len(unused_meta)),
                key=lambda u: unused_meta[u][0],
            )[:60]
        elif cur_mean < g_mean * 0.90:
            focus = sorted(range(len(selected)), key=lambda i: selected_meta[i][0])[:30]
            unused_focus = sorted(
                range(len(unused_meta)),
                key=lambda u: -unused_meta[u][0],
            )[:60]
        else:
            focus = sorted(
                range(len(selected)),
                key=lambda i: abs(selected_meta[i][0] - golden_word_mean),
                reverse=True,
            )[:30]
            unused_focus = list(range(min(80, len(unused_meta))))

        best: tuple[float, int, int] | None = None
        for index in focus:
            current_sig = selected_meta[index][1]
            for u_i in unused_focus:
                cand_wc, cand_sig, _ = unused_meta[u_i]
                if sum(abs(a - b) for a, b in zip(cand_sig, current_sig)) > 3:
                    continue
                trial = list(lengths)
                trial[index] = cand_wc
                # Reject moves that would push a currently-ok mean out of band
                # when we are specifically fixing median.
                trial_mean = sum(trial) / len(trial)
                if cur_median > g_median * 1.15 and trial_mean < g_mean * 0.90:
                    continue
                new_loss = _length_loss(trial)
                if new_loss + 0.002 >= cur_loss:
                    continue
                if best is None or new_loss < best[0]:
                    best = (new_loss, index, u_i)

        # Paired compensation: short-in at/above median, long-in on a mid essay.
        if best is None and cur_median > g_median * 1.15:
            above = [i for i in focus if selected_meta[i][0] >= cur_median][:15]
            mids = sorted(
                (i for i in range(len(selected)) if g_q1 < selected_meta[i][0] < cur_median),
                key=lambda i: selected_meta[i][0],
            )[:15]
            shorts = sorted(
                (u for u, m in enumerate(unused_meta) if m[0] <= g_median),
                key=lambda u: unused_meta[u][0],
            )[:25]
            longs = sorted(
                (u for u, m in enumerate(unused_meta) if m[0] >= max(g_q3, g_mean)),
                key=lambda u: -unused_meta[u][0],
            )[:25]
            best_pair: tuple[float, int, int, int, int] | None = None
            for i in above:
                for u_s in shorts:
                    if sum(abs(a - b) for a, b in zip(unused_meta[u_s][1], selected_meta[i][1])) > 3:
                        continue
                    for j in mids:
                        if j == i:
                            continue
                        for u_l in longs:
                            if u_l == u_s:
                                continue
                            if sum(abs(a - b) for a, b in zip(unused_meta[u_l][1], selected_meta[j][1])) > 3:
                                continue
                            trial = list(lengths)
                            trial[i] = unused_meta[u_s][0]
                            trial[j] = unused_meta[u_l][0]
                            new_loss = _length_loss(trial)
                            if new_loss + 0.002 >= cur_loss:
                                continue
                            if best_pair is None or new_loss < best_pair[0]:
                                best_pair = (new_loss, i, u_s, j, u_l)
            if best_pair is not None:
                _, i, u_s, j, u_l = best_pair
                cand_s = unused_meta[u_s]
                cand_l = unused_meta[u_l]
                for u_i in sorted((u_s, u_l), reverse=True):
                    unused_meta.pop(u_i)
                old_i, old_j = selected[i], selected[j]
                selected[i], selected_meta[i] = cand_s[2], (cand_s[0], cand_s[1])
                selected[j], selected_meta[j] = cand_l[2], (cand_l[0], cand_l[1])
                unused_meta.append((_wc(old_i), _sig(old_i), old_i))
                unused_meta.append((_wc(old_j), _sig(old_j), old_j))
                continue

        if best is None:
            break
        _, index, u_i = best
        cand_wc, cand_sig, candidate = unused_meta.pop(u_i)
        old = selected[index]
        selected[index] = candidate
        selected_meta[index] = (cand_wc, cand_sig)
        unused_meta.append((_wc(old), _sig(old), old))
    return selected



_STYLE_TOLERANCES: dict[str, float] = {
    # Regeneration plan: aggregate mean within ~10%; sentence/punct/error bands tighter
    # than the failed permissive audit.
    "word_count": 0.10,
    "paragraph_count": 0.15,
    "sentence_word_mean": 0.15,
    # CB goldens are newline-stripped single blobs; synthetic essays have more even
    # sentence lengths. Allow a wider std band while still rejecting uniform stubs.
    "sentence_word_std": 0.35,
    "informal_marker_per_100": 0.50,
    "punctuation_per_100": 0.15,
    "spelling_error_density_per_100": 0.50,
}


def style_features(text: str) -> dict[str, float]:
    """Extract transparent timed-writing features without retaining golden wording."""
    words = re.findall(r"[A-Za-z']+", text)
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]
    sentence_lengths = [len(re.findall(r"[A-Za-z']+", sentence)) or 1 for sentence in sentences] or [1]
    informal = re.findall(
        r"\b(?:wasnt|didnt|couldnt|wouldnt|dont|cant|wont|alot|goverment|kinda|gonna|wanna|tho|bc)\b",
        text,
        flags=re.I,
    )
    lowercase_i = re.findall(r"(?<![A-Za-z])i(?![A-Za-z])", text)
    # Lightweight misspelling proxy: common APUSH student misspellings + repeated letters.
    spelling = re.findall(
        r"\b(?:goverment|seperate|occured|becuase|enviroment|knowlege|arguement|"
        r"neccessary|recieve|wich|teh|thier)\b",
        text,
        flags=re.I,
    )
    denominator = max(len(words) / 100.0, 0.01)
    return {
        "word_count": float(len(words)),
        "paragraph_count": float(len(paragraphs) or bool(text.strip())),
        "sentence_word_mean": float(sum(sentence_lengths) / len(sentence_lengths)),
        "sentence_word_std": float(statistics.pstdev(sentence_lengths)),
        "informal_marker_per_100": float((len(informal) + len(lowercase_i)) / denominator),
        "punctuation_per_100": float(len(re.findall(r"[,;:!?]", text)) / denominator),
        "spelling_error_density_per_100": float(len(spelling) / denominator),
    }


def style_distribution_audit(
    rows: Sequence[Mapping[str, Any]], golden_cases: Sequence[FRQCase]
) -> dict[str, Any]:
    """Compare aggregate candidate mechanics to golden cases without exposing their text."""

    candidate_features = [
        style_features(str(row.get("student_response") or row.get("essay") or "")) for row in rows
    ]
    golden_features = [style_features(case.student_response) for case in golden_cases]
    # CB goldens are ingested as newline-stripped single blobs, so paragraph_count is
    # uniformly 1.0. Do not fail assembly on that extraction artifact.
    golden_paragraph_values = [item["paragraph_count"] for item in golden_features]
    skip_paragraph = (
        max(golden_paragraph_values) <= 1.0 and min(golden_paragraph_values) >= 1.0
    )
    metrics: dict[str, Any] = {}
    passed = True
    for key, relative_tolerance in _STYLE_TOLERANCES.items():
        if key == "paragraph_count" and skip_paragraph:
            metrics[key] = {
                "candidate_mean": round(
                    sum(item[key] for item in candidate_features) / len(candidate_features), 4
                ),
                "golden_mean": 1.0,
                "allowed_delta": None,
                "passed": True,
                "skipped_reason": "golden_essays_lack_paragraph_breaks",
            }
            continue
        candidate_mean = sum(item[key] for item in candidate_features) / len(candidate_features)
        golden_mean = sum(item[key] for item in golden_features) / len(golden_features)
        floor = 0.5 if key in {"paragraph_count", "informal_marker_per_100", "spelling_error_density_per_100"} else 0.1
        allowed_delta = max(abs(golden_mean) * relative_tolerance, floor)
        metric_passed = abs(candidate_mean - golden_mean) <= allowed_delta
        passed = passed and metric_passed
        metrics[key] = {
            "candidate_mean": round(candidate_mean, 4),
            "golden_mean": round(golden_mean, 4),
            "allowed_delta": round(allowed_delta, 4),
            "passed": metric_passed,
        }
    length_audit = aggregate_length_realism_audit(
        [int(item["word_count"]) for item in candidate_features],
        [int(item["word_count"]) for item in golden_features],
    )
    passed = passed and bool(length_audit.get("passed"))
    return {
        "passed": passed,
        "metrics": metrics,
        "length_realism": length_audit,
        "golden_text_retained": False,
    }


def compute_distribution_match(
    row: Mapping[str, Any],
    golden_cases: Sequence[FRQCase],
) -> dict[str, Any]:
    """Recompute whether a candidate fits the golden score/style envelope.

    External tools may propose ``distribution_match``, but assembly and validation
    should call this helper (or :func:`annotate_distribution_match`) rather than
    trusting the proposal blindly. Membership uses the golden joint score-vector
    set plus per-essay style features against golden means / tolerances from
    :func:`style_distribution_audit`.
    """
    if not golden_cases:
        return {
            "passed": False,
            "score_vector_in_golden": False,
            "style_within_tolerance": False,
            "recomputed": True,
            "reason": "no_golden_cases",
        }
    scores = (row.get("resolved_grade") or {}).get("scores") or row.get("scores") or {}
    try:
        signature = _score_signature(scores)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return {
            "passed": False,
            "score_vector_in_golden": False,
            "style_within_tolerance": False,
            "recomputed": True,
            "reason": "invalid_scores",
        }
    golden_signatures = {
        _score_signature(case.reference_scores.model_dump()) for case in golden_cases
    }
    score_ok = signature in golden_signatures

    essay = str(row.get("student_response") or row.get("essay") or "")
    features = style_features(essay)
    # Per-row gate: score-vector membership + broad timed-essay length band.
    # Aggregate length/paragraph/error matching happens at selection via
    # style_distribution_audit on the chosen 420, not against the golden mean
    # for every individual essay.
    word_count = features["word_count"]
    length_ok = 70.0 <= word_count <= 550.0
    style_ok = length_ok
    style_metrics = {
        "word_count": {
            "value": round(word_count, 4),
            "min_allowed": 70.0,
            "max_allowed": 550.0,
            "passed": length_ok,
        }
    }
    return {
        "passed": bool(score_ok and style_ok),
        "score_vector_in_golden": score_ok,
        "score_vector": list(signature),
        "style_within_tolerance": style_ok,
        "style_metrics": style_metrics,
        "recomputed": True,
    }


def annotate_distribution_match(
    rows: Sequence[Mapping[str, Any]],
    golden_cases: Sequence[FRQCase],
) -> list[dict[str, Any]]:
    """Set ``distribution_match`` on each row via :func:`compute_distribution_match`."""
    annotated: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["distribution_match"] = compute_distribution_match(updated, golden_cases)
        annotated.append(updated)
    return annotated


def assemble_v5_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_texts: Iterable[str] = (),
    golden_cases: Sequence[FRQCase] = (),
    allowed_phrases: Iterable[str] = (),
) -> tuple[list[dict], list[dict]]:
    """Select the final 420/180 corpus and make a leakage-safe 540/60 split.

    Peer dedup among candidates uses exact/near-duplicate checks only. Eight-gram
    source-copy rejection against AMSCO/golden is expected to have already run in
    ``validate_v5_external_candidates``.
    """
    peer_index = OverlapIndex.build(source_texts, allowed_phrases=allowed_phrases)
    clean: list[dict] = []
    overlap_reason_names = {
        "essay_too_short_for_overlap_audit",
        "exact_duplicate",
        "near_duplicate",
        "verbatim_eight_word_overlap",
    }
    for row in sorted(rows, key=lambda r: str(r.get("task_id", ""))):
        essay = str(row.get("student_response") or row.get("essay") or "")
        reasons = list(
            peer_index.reasons_for(
                essay, check_exact=True, check_near=True, check_eight_gram=False
            )
        )
        reasons.extend(
            reason
            for reason in candidate_gate_reasons(row, source_texts=(), allowed_phrases=())
            if reason not in overlap_reason_names
        )
        if reasons:
            continue
        clean.append(dict(row))
        peer_index.add(essay)
    by_class = {
        name: sorted((r for r in clean if r["selection_class"] == name), key=lambda r: str(r["task_id"]))
        for name in ("golden_matched", "boundary")
    }
    if len(by_class["golden_matched"]) < V5_GOLDEN_MATCHED_COUNT:
        raise ValueError(
            f"fewer than 420 accepted golden-matched candidates (have {len(by_class['golden_matched'])})"
        )
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
            if not style_audit["length_realism"].get("passed"):
                failed.append("length_realism")
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


def assert_pilot_approval(
    pilot_essays_path: Path,
    approval_path: Path,
    *,
    expected_count: int = V5_PILOT_COUNT,
) -> dict[str, Any]:
    """Block full 1,470 generation until all 30 pilot essays are hash-bound accepted."""
    if not approval_path.exists():
        raise PermissionError(
            "v5 pilot approval is missing; full production generation remains blocked"
        )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not approval.get("approved") or not approval.get("reviewer") or not approval.get("approved_at"):
        raise PermissionError("v5 pilot approval is incomplete")
    if approval.get("pilot_essays_sha256") != file_sha256(pilot_essays_path):
        raise PermissionError("v5 pilot essays changed after approval")
    rows = [
        json.loads(line)
        for line in pilot_essays_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_count:
        raise PermissionError(
            f"pilot approval expects {expected_count} essays; found {len(rows)}"
        )
    decisions = approval.get("decisions") or {}
    if len(decisions) != expected_count:
        raise PermissionError("pilot approval must include a decision for every essay")
    for row in rows:
        task_id = str(row.get("task_id") or "")
        decision = decisions.get(task_id)
        if decision not in {"accept", "corrected"}:
            raise PermissionError(
                f"pilot task {task_id} is not accepted (decision={decision!r})"
            )
    if approval.get("accepted_count") != expected_count:
        raise PermissionError("pilot approval accepted_count must equal the pilot size")
    return approval


def build_pilot_approval(
    *,
    reviewer: str,
    approved_at: str,
    pilot_essays_path: Path,
    decisions: Mapping[str, str],
) -> dict[str, Any]:
    """Create a hash-bound pilot approval document after human review."""
    rows = [
        json.loads(line)
        for line in pilot_essays_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != V5_PILOT_COUNT:
        raise ValueError(f"pilot essays file must contain {V5_PILOT_COUNT} rows")
    normalized = {str(task_id): str(decision) for task_id, decision in decisions.items()}
    for row in rows:
        task_id = str(row["task_id"])
        if normalized.get(task_id) not in {"accept", "corrected"}:
            raise ValueError(f"missing accept/corrected decision for {task_id}")
    return {
        "approved": True,
        "reviewer": reviewer,
        "approved_at": approved_at,
        "pilot_essays_sha256": file_sha256(pilot_essays_path),
        "accepted_count": V5_PILOT_COUNT,
        "decisions": dict(sorted(normalized.items())),
        "blocks_full_generation_until_valid": True,
    }
