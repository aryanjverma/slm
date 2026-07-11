"""V4 AMSCO + College Board seed-driven synthetic dataset planning and assembly."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.dedup import is_duplicate_essay
from apush_frq_grader_slm.prompts_v4 import V4_TRAIN_SYSTEM_PROMPT
from apush_frq_grader_slm.rubric import DEFAULT_RUBRIC_VERSION, compute_total
from apush_frq_grader_slm.schemas import (
    CaseProvenance,
    FailureType,
    FRQCase,
    LabelingMetadata,
    RubricFeedback,
    RubricScores,
)
from apush_frq_grader_slm.synth_realistic import (
    KNOWLEDGE_LEVELS,
    MECHANICS_PROFILES,
    MISCONCEPTION_PROFILES,
    PLANNING_STYLES,
    TIME_BUDGETS,
    StudentPersona,
    length_band_for_persona,
    persona_for_variant,
)

OFFICIAL_SOURCE_TAGS = frozenset(
    {"ap_central", "real_eval", "college_board", "tom_richey", "quizlet", "seed_real"}
)

GENERATOR_NAME = "v4_amsco_cb_seeded"

# Rough CB-like band counts for a 250-row set (sums to 250).
DEFAULT_TOTAL_BAND_TARGETS: dict[int, int] = {
    0: 15,
    1: 25,
    2: 40,
    3: 35,
    4: 50,
    5: 35,
    6: 50,
}

# Score tuples (thesis, contextualization, evidence, analysis_reasoning) per total.
_SCORE_OPTIONS_BY_TOTAL: dict[int, list[tuple[int, int, int, int]]] = {
    0: [(0, 0, 0, 0)],
    1: [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)],
    2: [(1, 1, 0, 0), (1, 0, 1, 0), (0, 1, 1, 0), (0, 0, 2, 0), (1, 0, 0, 1)],
    3: [(1, 1, 1, 0), (1, 0, 2, 0), (1, 1, 0, 1), (0, 1, 2, 0)],
    4: [(1, 1, 2, 0), (1, 1, 1, 1), (1, 0, 2, 1)],
    5: [(1, 1, 2, 1), (1, 1, 1, 2)],
    6: [(1, 1, 2, 2)],
}


@dataclass(frozen=True)
class CBSeedProfile:
    """Normalized College Board seed profile consumed by v4 task planning."""

    seed_id: str
    prompt: str
    prompt_family_id: str = ""
    period: int | None = None
    reasoning_skill: str = ""
    scores: dict[str, int] | None = None
    total: int | None = None
    style_excerpt: str = ""
    amsco_chapter_ids: tuple[str, ...] = ()
    adapted_prompts: tuple[str, ...] = ()

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> CBSeedProfile:
        scores = row.get("scores") or row.get("reference_scores")
        score_dict: dict[str, int] | None = None
        if isinstance(scores, Mapping):
            score_dict = {str(key): int(value) for key, value in scores.items()}
        chapters = row.get("amsco_chapter_ids") or row.get("chapter_ids") or ()
        adapted = row.get("adapted_prompts") or ()
        seed_id = str(row.get("seed_id") or row.get("id") or "")
        if not seed_id:
            raise ValueError("seed profile missing seed_id")
        prompt = str(row.get("prompt") or row.get("prompt_text") or "").strip()
        if not prompt:
            raise ValueError(f"seed profile {seed_id} missing prompt")
        total = row.get("total")
        if total is None and score_dict is not None:
            total = sum(score_dict.get(key, 0) for key in (
                "thesis", "contextualization", "evidence", "analysis_reasoning"
            ))
        return cls(
            seed_id=seed_id,
            prompt=prompt,
            prompt_family_id=str(
                row.get("prompt_family_id") or row.get("family_id") or seed_id
            ),
            period=(int(row["period"]) if row.get("period") is not None else None),
            reasoning_skill=str(row.get("reasoning_skill") or ""),
            scores=score_dict,
            total=(int(total) if total is not None else None),
            style_excerpt=str(row.get("style_excerpt") or row.get("excerpt") or ""),
            amsco_chapter_ids=tuple(str(item) for item in chapters),
            adapted_prompts=tuple(str(item).strip() for item in adapted if str(item).strip()),
        )


@dataclass(frozen=True)
class PromptFamilyRef:
    """Optional prompt-family catalog entry for adapted close prompts."""

    prompt_family_id: str
    prompt_text: str
    period: int | None = None
    reasoning_skill: str = ""
    adapted_prompts: tuple[str, ...] = ()

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> PromptFamilyRef:
        family_id = str(row.get("prompt_family_id") or row.get("prompt_id") or "")
        text = str(row.get("prompt_text") or row.get("prompt") or "").strip()
        if not family_id or not text:
            raise ValueError("prompt family requires prompt_family_id and prompt_text")
        adapted = row.get("adapted_prompts") or ()
        return cls(
            prompt_family_id=family_id,
            prompt_text=text,
            period=(int(row["period"]) if row.get("period") is not None else None),
            reasoning_skill=str(row.get("reasoning_skill") or ""),
            adapted_prompts=tuple(str(item).strip() for item in adapted if str(item).strip()),
        )


@dataclass(frozen=True)
class V4Task:
    task_id: str
    seed_id: str
    prompt: str
    prompt_family_id: str
    period: int | None
    reasoning_skill: str
    target_scores: dict[str, int]
    target_total: int
    failure_type: str
    persona: StudentPersona
    length_band: tuple[int, int]
    amsco_chapter_ids: tuple[str, ...]
    style_excerpt: str
    rubric_version: str
    split: str = "train"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["length_band"] = list(self.length_band)
        row["amsco_chapter_ids"] = list(self.amsco_chapter_ids)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> V4Task:
        band = row["length_band"]
        chapters = row.get("amsco_chapter_ids") or ()
        return cls(
            task_id=str(row["task_id"]),
            seed_id=str(row["seed_id"]),
            prompt=str(row["prompt"]),
            prompt_family_id=str(row.get("prompt_family_id") or row["seed_id"]),
            period=(int(row["period"]) if row.get("period") is not None else None),
            reasoning_skill=str(row.get("reasoning_skill") or ""),
            target_scores={str(key): int(value) for key, value in row["target_scores"].items()},
            target_total=int(row["target_total"]),
            failure_type=str(row["failure_type"]),
            persona=StudentPersona.from_row(row.get("persona")),
            length_band=(int(band[0]), int(band[1])),
            amsco_chapter_ids=tuple(str(item) for item in chapters),
            style_excerpt=str(row.get("style_excerpt") or ""),
            rubric_version=str(row.get("rubric_version", DEFAULT_RUBRIC_VERSION.value)),
            split=str(row.get("split", "train")),
        )


def failure_type_for_scores(scores: Mapping[str, int]) -> FailureType:
    """Map a target score vector onto the nearest FailureType slice."""
    thesis = int(scores.get("thesis", 0))
    context = int(scores.get("contextualization", 0))
    evidence = int(scores.get("evidence", 0))
    analysis = int(scores.get("analysis_reasoning", 0))
    total = thesis + context + evidence + analysis
    if thesis == 0:
        return FailureType.WEAK_THESIS
    if context == 0 and total <= 3:
        return FailureType.MISSING_CONTEXT
    if evidence <= 1 and analysis == 0:
        return FailureType.EVIDENCE_LIST
    if analysis == 0 and evidence == 2:
        return FailureType.BORDERLINE_COMPLEXITY
    if analysis == 1 and total <= 5:
        return FailureType.BORDERLINE_COMPLEXITY
    if total >= 5:
        return FailureType.STRONG
    if evidence == 0:
        return FailureType.WRONG_PERIOD
    return FailureType.BORDERLINE_COMPLEXITY


def _difficulty_for_total(total: int) -> str:
    if total <= 2:
        return "weak"
    if total <= 4:
        return "borderline"
    return "strong"


def _allocate_band_counts(target_count: int) -> dict[int, int]:
    """Scale DEFAULT_TOTAL_BAND_TARGETS to target_count while keeping all bands present."""
    if target_count < 7:
        raise ValueError("target_count must be at least 7 so every total band is present")
    base_sum = sum(DEFAULT_TOTAL_BAND_TARGETS.values())
    exact = {
        total: target_count * DEFAULT_TOTAL_BAND_TARGETS[total] / base_sum
        for total in range(7)
    }
    counts = {total: max(1, int(exact[total])) for total in range(7)}
    assigned = sum(counts.values())
    # Largest-remainder fill/trim to hit target_count exactly.
    remainders = sorted(
        range(7),
        key=lambda total: (exact[total] - int(exact[total]), -total),
        reverse=True,
    )
    while assigned < target_count:
        for total in remainders:
            if assigned >= target_count:
                break
            counts[total] += 1
            assigned += 1
    while assigned > target_count:
        for total in reversed(remainders):
            if assigned <= target_count:
                break
            if counts[total] > 1:
                counts[total] -= 1
                assigned -= 1
    return counts


def _scores_for_total(total: int, rng: random.Random) -> dict[str, int]:
    options = _SCORE_OPTIONS_BY_TOTAL[total]
    thesis, context, evidence, analysis = rng.choice(options)
    return {
        "thesis": thesis,
        "contextualization": context,
        "evidence": evidence,
        "analysis_reasoning": analysis,
    }


def _family_index(
    seed_profiles: Sequence[CBSeedProfile],
    prompt_families: Sequence[PromptFamilyRef],
) -> dict[str, PromptFamilyRef]:
    index: dict[str, PromptFamilyRef] = {}
    for family in prompt_families:
        index[family.prompt_family_id] = family
    for seed in seed_profiles:
        if seed.prompt_family_id not in index:
            index[seed.prompt_family_id] = PromptFamilyRef(
                prompt_family_id=seed.prompt_family_id,
                prompt_text=seed.prompt,
                period=seed.period,
                reasoning_skill=seed.reasoning_skill,
                adapted_prompts=seed.adapted_prompts,
            )
    return index


def _choose_prompt(
    seed: CBSeedProfile,
    family: PromptFamilyRef | None,
    rng: random.Random,
    *,
    prefer_exact: bool = True,
) -> str:
    """Prefer the exact CB seed prompt; occasionally use a same-family adapted prompt."""
    adapted: list[str] = []
    if family is not None:
        adapted.extend(family.adapted_prompts)
    adapted.extend(seed.adapted_prompts)
    adapted = [text for text in adapted if text and text.strip() != seed.prompt.strip()]
    if prefer_exact or not adapted:
        # ~85% exact CB prompt when adaptations exist.
        if adapted and rng.random() < 0.15:
            return rng.choice(adapted)
        return seed.prompt
    return rng.choice(adapted)


def plan_v4_tasks(
    seed_profiles: Sequence[CBSeedProfile | Mapping[str, Any]],
    prompt_families: Sequence[PromptFamilyRef | Mapping[str, Any]] | None = None,
    *,
    target_count: int = 250,
    seed: int = 42,
) -> list[V4Task]:
    """Plan CB-seeded synthetic tasks covering totals 0-6 with varied personas."""
    profiles = [
        profile if isinstance(profile, CBSeedProfile) else CBSeedProfile.from_row(profile)
        for profile in seed_profiles
    ]
    if not profiles:
        raise ValueError("plan_v4_tasks requires at least one CB seed profile")

    families_raw = list(prompt_families or [])
    families = [
        family if isinstance(family, PromptFamilyRef) else PromptFamilyRef.from_row(family)
        for family in families_raw
    ]
    family_by_id = _family_index(profiles, families)
    band_counts = _allocate_band_counts(target_count)
    rng = random.Random(seed)

    # Round-robin seeds so every task is CB-seeded.
    seed_cycle = list(profiles)
    rng.shuffle(seed_cycle)

    tasks: list[V4Task] = []
    per_seed_idx: Counter[str] = Counter()
    seed_cursor = 0
    for total in range(7):
        for _ in range(band_counts[total]):
            profile = seed_cycle[seed_cursor % len(seed_cycle)]
            seed_cursor += 1
            family = family_by_id.get(profile.prompt_family_id)
            prompt = _choose_prompt(profile, family, rng, prefer_exact=True)
            target_scores = _scores_for_total(total, rng)
            failure = failure_type_for_scores(target_scores)
            variant = per_seed_idx[profile.seed_id]
            per_seed_idx[profile.seed_id] += 1
            persona = persona_for_variant(seed_cursor, variant, total)
            # Ensure misconception variety even for high totals.
            if total <= 2 and persona.misconception == "none":
                persona = StudentPersona(
                    time_budget_minutes=persona.time_budget_minutes,
                    historical_knowledge=persona.historical_knowledge,
                    planning_style=persona.planning_style,
                    mechanics=persona.mechanics,
                    misconception=MISCONCEPTION_PROFILES[
                        (variant + seed_cursor) % (len(MISCONCEPTION_PROFILES) - 1) + 1
                    ],
                )
            length_band = length_band_for_persona(persona, total)
            task_id = f"v4-{profile.seed_id}-t{total}-n{variant:03d}"
            tasks.append(
                V4Task(
                    task_id=task_id,
                    seed_id=profile.seed_id,
                    prompt=prompt,
                    prompt_family_id=profile.prompt_family_id or profile.seed_id,
                    period=profile.period if profile.period is not None else (
                        family.period if family is not None else None
                    ),
                    reasoning_skill=profile.reasoning_skill or (
                        family.reasoning_skill if family is not None else ""
                    ),
                    target_scores=target_scores,
                    target_total=total,
                    failure_type=failure.value,
                    persona=persona,
                    length_band=length_band,
                    amsco_chapter_ids=profile.amsco_chapter_ids,
                    style_excerpt=profile.style_excerpt,
                    rubric_version=DEFAULT_RUBRIC_VERSION.value,
                    split="train",
                )
            )

    rng.shuffle(tasks)
    # Re-number n-index per (seed_id, total) for stable unique ids after shuffle.
    renumber: Counter[tuple[str, int]] = Counter()
    renumbered: list[V4Task] = []
    for task in tasks:
        key = (task.seed_id, task.target_total)
        idx = renumber[key]
        renumber[key] += 1
        new_id = f"v4-{task.seed_id}-t{task.target_total}-n{idx:03d}"
        renumbered.append(
            V4Task(
                task_id=new_id,
                seed_id=task.seed_id,
                prompt=task.prompt,
                prompt_family_id=task.prompt_family_id,
                period=task.period,
                reasoning_skill=task.reasoning_skill,
                target_scores=task.target_scores,
                target_total=task.target_total,
                failure_type=task.failure_type,
                persona=task.persona,
                length_band=task.length_band,
                amsco_chapter_ids=task.amsco_chapter_ids,
                style_excerpt=task.style_excerpt,
                rubric_version=task.rubric_version,
                split=task.split,
            )
        )
    return renumbered


def format_v4_user_message(case: FRQCase) -> str:
    return f"LEQ Prompt:\n{case.prompt}\n\nStudent Essay:\n{case.student_response}"


def v4_target_payload(case: FRQCase) -> dict[str, Any]:
    return {
        "scores": case.reference_scores.model_dump(),
        "total": case.reference_scores.total,
        "feedback": case.reference_feedback.model_dump(),
    }


def v4_chat_row(case: FRQCase) -> dict[str, Any]:
    """SFT chat row using the full-rubric v4 train system prompt."""
    target = case.assistant_response
    try:
        parsed = json.loads(target)
        if "total" not in parsed:
            parsed["total"] = case.reference_scores.total
            target = json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        target = json.dumps(v4_target_payload(case), ensure_ascii=True, separators=(",", ":"))
    return {
        "id": case.id,
        "messages": [
            {"role": "system", "content": V4_TRAIN_SYSTEM_PROMPT},
            {"role": "user", "content": format_v4_user_message(case)},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "prompt": case.prompt,
            "failure_type": case.failure_type.value,
            "reference_total": case.reference_scores.total,
            "tags": case.tags,
            "generator_name": case.provenance.generator_name,
        },
    }


def assemble_v4_case(
    task: V4Task | Mapping[str, Any],
    essay: str,
    scores: RubricScores | Mapping[str, int],
    feedback: RubricFeedback | Mapping[str, str],
    *,
    labeling_method: str = "rule_based",
    grader_ids: Sequence[str] | None = None,
) -> FRQCase:
    """Assemble a synthetic FRQCase from a planned task + essay + grades."""
    resolved = task if isinstance(task, V4Task) else V4Task.from_row(task)
    rubric_scores = (
        scores if isinstance(scores, RubricScores) else RubricScores.model_validate(scores)
    )
    rubric_feedback = (
        feedback
        if isinstance(feedback, RubricFeedback)
        else RubricFeedback.model_validate(feedback)
    )
    payload = {
        "scores": rubric_scores.model_dump(),
        "total": compute_total(rubric_scores),
        "feedback": rubric_feedback.model_dump(),
    }
    assistant_response = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    total = rubric_scores.total
    try:
        failure = FailureType(resolved.failure_type)
    except ValueError:
        failure = failure_type_for_scores(rubric_scores.model_dump())
    method = labeling_method
    if method not in {"rule_based", "independent_consensus", "source_scores", "adjudicated", "unknown"}:
        method = "rule_based"
    tags = [
        "synth_v4",
        "amsco_kb",
        resolved.seed_id,
        f"seed:{resolved.seed_id}",
        failure.value,
        _difficulty_for_total(total),
    ]
    if resolved.prompt_family_id:
        tags.append(f"family:{resolved.prompt_family_id}")
    return FRQCase(
        id=resolved.task_id,
        split=resolved.split,  # type: ignore[arg-type]
        prompt=resolved.prompt,
        student_response=essay.strip(),
        reference_scores=rubric_scores,
        reference_feedback=rubric_feedback,
        failure_type=failure,
        difficulty=_difficulty_for_total(total),  # type: ignore[arg-type]
        assistant_response=assistant_response,
        tags=tags,
        provenance=CaseProvenance(
            source_type="synthetic",
            source_id=resolved.task_id,
            prompt_family_id=resolved.prompt_family_id,
            generator_name=GENERATOR_NAME,
            generator_config={
                "seed_id": resolved.seed_id,
                "target_scores": resolved.target_scores,
                "target_total": resolved.target_total,
                "amsco_chapter_ids": list(resolved.amsco_chapter_ids),
                "length_band": list(resolved.length_band),
                "persona": asdict(resolved.persona),
            },
            rubric_version=resolved.rubric_version,  # type: ignore[arg-type]
            review_status="machine_checked",
        ),
        labeling=LabelingMetadata(
            method=method,  # type: ignore[arg-type]
            grader_ids=list(grader_ids or []),
            generation_target_total=resolved.target_total,
            target_distance=abs(total - resolved.target_total),
            protocol_version="v4",
        ),
    )


def assert_no_real_essays(cases: Sequence[FRQCase]) -> None:
    """Raise if any official/real eval tags or id prefixes leak into the set."""
    for case in cases:
        if OFFICIAL_SOURCE_TAGS.intersection(case.tags):
            raise ValueError(f"Real/official tag leaked into v4 set: {case.id} tags={case.tags}")
        if case.id.startswith(("ap_central_", "tom_richey_", "quizlet_", "real_eval")):
            raise ValueError(f"Real essay id in v4 set: {case.id}")
        if case.provenance.source_type in {"college_board"}:
            raise ValueError(f"college_board provenance in v4 train set: {case.id}")
        if case.provenance.generator_name in {"ap_central", "tom_richey", "quizlet"}:
            raise ValueError(f"official generator_name in v4 set: {case.id}")


def audit_v4_cases(cases: Sequence[FRQCase]) -> dict[str, Any]:
    """Audit coverage, uniqueness, quality-gate fields, and length vs CB norms."""
    warnings: list[str] = []
    errors: list[str] = []
    rejected: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    seen_essays: set[str] = set()
    totals: Counter[int] = Counter()
    word_counts: list[int] = []

    for case in cases:
        reasons: list[str] = []
        if case.id in seen_ids:
            reasons.append("duplicate_id")
        seen_ids.add(case.id)
        essay_key = " ".join(case.student_response.lower().split())
        if essay_key in seen_essays:
            reasons.append("duplicate_essay")
        seen_essays.add(essay_key)
        if OFFICIAL_SOURCE_TAGS.intersection(case.tags):
            reasons.append("official_source_tag")
        if case.provenance.source_type != "synthetic":
            reasons.append("non_synthetic_source")
        if "synth_v4" not in case.tags:
            reasons.append("missing_synth_v4_tag")
        ok, gate_reasons = passes_quality_gate(case)
        if not ok:
            reasons.extend(gate_reasons)
        if reasons:
            rejected[case.id] = sorted(set(reasons))
            continue
        totals[case.reference_scores.total] += 1
        word_counts.append(len(case.student_response.split()))

    n = len(cases)
    accepted = n - len(rejected)
    if n < 250:
        warnings.append(f"n={n} below ideal 250")
    missing_bands = [total for total in range(7) if totals[total] == 0]
    if missing_bands:
        errors.append(f"missing_total_bands:{missing_bands}")
    median_words = 0
    if word_counts:
        ordered = sorted(word_counts)
        median_words = ordered[len(ordered) // 2]
        if median_words < 150:
            warnings.append(f"median_word_count={median_words} below CB-like 150")
    try:
        assert_no_real_essays(cases)
    except ValueError as exc:
        errors.append(str(exc))

    return {
        "total": n,
        "accepted": accepted,
        "rejected_count": len(rejected),
        "rejected": rejected,
        "totals": {str(total): totals[total] for total in range(7)},
        "median_word_count": median_words,
        "min_word_count": min(word_counts) if word_counts else 0,
        "max_word_count": max(word_counts) if word_counts else 0,
        "warnings": warnings,
        "errors": errors,
        "clean": accepted == n and not errors and not missing_bands,
        "persona_coverage": {
            "time_budgets": sorted(TIME_BUDGETS),
            "knowledge_levels": list(KNOWLEDGE_LEVELS),
            "planning_styles": list(PLANNING_STYLES),
            "mechanics": list(MECHANICS_PROFILES),
            "misconceptions": list(MISCONCEPTION_PROFILES),
        },
    }


def load_seed_profiles(path: Path) -> list[CBSeedProfile]:
    from apush_frq_grader_slm.io import read_jsonl

    return [CBSeedProfile.from_row(row) for row in read_jsonl(path)]


def load_prompt_families(path: Path | None) -> list[PromptFamilyRef]:
    if path is None or not path.exists():
        return []
    from apush_frq_grader_slm.io import read_jsonl

    return [PromptFamilyRef.from_row(row) for row in read_jsonl(path)]


def dedup_against_eval(
    case: FRQCase,
    eval_cases: Sequence[FRQCase],
) -> list[str]:
    """Reject only real leakage: same-prompt near-duplicates or long verbatim spans.

    The generic ``is_duplicate_essay`` cross-prompt branch flags shared APUSH
    vocabulary (federal, society, labor, …) as duplicates; that is too aggressive
    for AMSCO-grounded synthetic essays on related topics.

    Verbatim checks also ignore spans that mostly restate the LEQ prompt (students
    routinely echo the prompt's date range and topic in a thesis sentence).
    """
    from apush_frq_grader_slm.ingest.dedup import _word_spans, normalize_essay

    reasons: list[str] = []
    same_prompt = [
        eval_case
        for eval_case in eval_cases
        if normalize_essay(eval_case.prompt) == normalize_essay(case.prompt)
    ]
    if same_prompt and is_duplicate_essay(
        case.student_response, same_prompt, prompt=case.prompt, jaccard_threshold=0.82
    ):
        reasons.append("duplicate_of_eval_same_prompt")

    # Prompt-echo filter: drop essay spans that heavily overlap the prompt itself.
    prompt_spans = _word_spans(case.prompt, 8)
    essay_spans = _word_spans(case.student_response, 14)
    essay_spans -= _word_spans(case.prompt, 14)
    filtered: set[str] = set()
    prompt_words = set(normalize_essay(case.prompt).split())
    for span in essay_spans:
        span_words = span.split()
        overlap = sum(1 for word in span_words if word in prompt_words)
        if overlap / max(len(span_words), 1) >= 0.65:
            continue
        # Also drop if an 8-gram of the span sits inside the prompt.
        grams = {
            " ".join(span_words[i : i + 8])
            for i in range(max(0, len(span_words) - 7))
        }
        if grams & prompt_spans:
            continue
        filtered.add(span)
    leaked = False
    if filtered:
        for eval_case in eval_cases:
            if filtered & _word_spans(eval_case.student_response, 14):
                leaked = True
                break
    if leaked:
        reasons.append("verbatim_span_of_eval")
    return reasons
