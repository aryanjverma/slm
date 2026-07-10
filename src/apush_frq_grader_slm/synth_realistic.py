"""Deterministic planning and validation for realistic synthetic LEQ essays.

Generation targets and student personas are writer-only controls.  Raw candidates
contain no labels, and downstream graders receive only the LEQ prompt and essay.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from apush_frq_grader_slm.ingest.dedup import contains_verbatim_span, is_duplicate_essay
from apush_frq_grader_slm.rubric import (
    CRITERIA,
    DEFAULT_RUBRIC_VERSION,
    RubricVersion,
    get_leq_rubric,
)
from apush_frq_grader_slm.schemas import FRQCase, FailureType

TIME_BUDGETS = (20, 25, 30, 35, 40)
KNOWLEDGE_LEVELS = ("weak", "uneven", "competent", "strong")
PLANNING_STYLES = ("no_outline", "rushed_thesis", "evidence_first", "organized_argument")
MECHANICS_PROFILES = (
    "clean",
    "ordinary_errors",
    "run_ons",
    "fragments",
    "misspellings",
    "uneven_paragraphs",
)
MISCONCEPTION_PROFILES = (
    "none",
    "vague_chronology",
    "wrong_period_evidence",
    "factual_mix_up",
    "unsupported_generalization",
)

# Kept for callers that imported the old constant. New tasks use persona-derived bands.
LENGTH_BANDS: tuple[tuple[int, int], ...] = ((220, 360), (300, 480), (400, 620))
LENGTH_TOLERANCE = 0.15


@dataclass(frozen=True)
class TargetProfile:
    """Writer-only rubric calibration target."""

    scores: tuple[int, int, int, int]
    failure_type: FailureType
    weight: float

    @property
    def score_dict(self) -> dict[str, int]:
        return dict(zip(CRITERIA, self.scores, strict=True))

    @property
    def total(self) -> int:
        return sum(self.scores)


TARGET_PROFILES: list[TargetProfile] = [
    TargetProfile((0, 0, 0, 0), FailureType.WEAK_THESIS, 0.10),
    TargetProfile((1, 0, 0, 0), FailureType.WEAK_THESIS, 0.08),
    TargetProfile((0, 1, 1, 0), FailureType.WEAK_THESIS, 0.08),
    TargetProfile((1, 1, 0, 0), FailureType.EVIDENCE_LIST, 0.09),
    TargetProfile((1, 0, 1, 0), FailureType.MISSING_CONTEXT, 0.10),
    TargetProfile((1, 1, 1, 0), FailureType.EVIDENCE_LIST, 0.11),
    TargetProfile((1, 0, 2, 0), FailureType.MISSING_CONTEXT, 0.08),
    TargetProfile((1, 1, 2, 0), FailureType.BORDERLINE_COMPLEXITY, 0.10),
    TargetProfile((1, 1, 1, 1), FailureType.BORDERLINE_COMPLEXITY, 0.08),
    TargetProfile((1, 1, 2, 1), FailureType.STRONG, 0.08),
    TargetProfile((1, 1, 2, 2), FailureType.STRONG, 0.10),
]


@dataclass(frozen=True)
class StudentPersona:
    time_budget_minutes: int
    historical_knowledge: str
    planning_style: str
    mechanics: str
    misconception: str

    @classmethod
    def default(cls) -> StudentPersona:
        return cls(30, "competent", "evidence_first", "ordinary_errors", "none")

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> StudentPersona:
        if not row:
            return cls.default()
        return cls(
            time_budget_minutes=int(row["time_budget_minutes"]),
            historical_knowledge=str(row["historical_knowledge"]),
            planning_style=str(row["planning_style"]),
            mechanics=str(row["mechanics"]),
            misconception=str(row["misconception"]),
        )


@dataclass(frozen=True)
class SeedRef:
    ref_id: str
    prompt: str
    excerpt: str = ""
    seed_total: int | None = None
    prompt_family_id: str = ""
    period: int | None = None
    reasoning_skill: str = ""


@dataclass(frozen=True)
class GenTask:
    task_id: str
    seed_id: str
    prompt: str
    seed_scores: dict[str, int] | None
    target_scores: dict[str, int]
    target_total: int
    failure_type: str
    length_band: tuple[int, int]
    seed_essay_excerpt: str
    persona: StudentPersona = field(default_factory=StudentPersona.default)
    rubric_version: str = DEFAULT_RUBRIC_VERSION.value
    prompt_family_id: str = ""
    period: int | None = None
    reasoning_skill: str = ""
    case_split: str = "train"
    prompt_split: str = "train"
    task_type: str = "ordinary"
    adversarial_type: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["length_band"] = list(self.length_band)
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GenTask:
        band = row["length_band"]
        return cls(
            task_id=row["task_id"],
            seed_id=row["seed_id"],
            prompt=row["prompt"],
            seed_scores=row.get("seed_scores"),
            target_scores=row["target_scores"],
            target_total=int(row["target_total"]),
            failure_type=row["failure_type"],
            length_band=(int(band[0]), int(band[1])),
            seed_essay_excerpt=row.get("seed_essay_excerpt", ""),
            persona=StudentPersona.from_row(row.get("persona")),
            rubric_version=str(row.get("rubric_version", DEFAULT_RUBRIC_VERSION.value)),
            prompt_family_id=str(row.get("prompt_family_id", row.get("seed_id", ""))),
            period=(int(row["period"]) if row.get("period") is not None else None),
            reasoning_skill=str(row.get("reasoning_skill", "")),
            case_split=str(row.get("case_split", "train")),
            prompt_split=str(row.get("prompt_split", "train")),
            task_type=str(row.get("task_type", "ordinary")),
            adversarial_type=str(row.get("adversarial_type", "")),
        )


@dataclass(frozen=True)
class SyntheticCandidate:
    """Anonymous, unlabeled essay passed from the writer to independent graders."""

    task_id: str
    prompt: str
    student_response: str
    rubric_version: str = DEFAULT_RUBRIC_VERSION.value

    def to_row(self) -> dict[str, str]:
        return {"task_id": self.task_id, "student_response": self.student_response}


def _profile_schedule(n: int) -> list[TargetProfile]:
    """Allocate profile slots reproducibly using largest remainder."""
    if n < 1:
        return []
    total_weight = sum(profile.weight for profile in TARGET_PROFILES)
    exact = [(profile, n * profile.weight / total_weight) for profile in TARGET_PROFILES]
    counts = {id(profile): int(count) for profile, count in exact}
    assigned = sum(counts.values())
    remainder = sorted(exact, key=lambda item: item[1] - int(item[1]), reverse=True)
    for profile, _ in remainder[: n - assigned]:
        counts[id(profile)] += 1
    schedule: list[TargetProfile] = []
    for profile in TARGET_PROFILES:
        schedule.extend([profile] * counts[id(profile)])
    return schedule


def persona_for_variant(
    seed_index: int, variant: int, target_total: int | None = None
) -> StudentPersona:
    """Return a stable persona while cycling every dimension independently."""
    knowledge = KNOWLEDGE_LEVELS[(variant * 3 + seed_index) % len(KNOWLEDGE_LEVELS)]
    planning = PLANNING_STYLES[(variant + seed_index * 2) % len(PLANNING_STYLES)]
    if target_total is not None:
        knowledge_options = {
            0: ("weak",),
            1: ("weak", "uneven"),
            2: ("uneven",),
            3: ("uneven", "competent"),
            4: ("competent",),
            5: ("competent", "strong"),
            6: ("strong",),
        }[target_total]
        planning_options = {
            0: ("no_outline",),
            1: ("no_outline", "rushed_thesis"),
            2: ("rushed_thesis", "evidence_first"),
            3: ("evidence_first", "rushed_thesis"),
            4: ("evidence_first", "organized_argument"),
            5: ("organized_argument", "evidence_first"),
            6: ("organized_argument",),
        }[target_total]
        knowledge = knowledge_options[(variant + seed_index) % len(knowledge_options)]
        planning = planning_options[(variant * 2 + seed_index) % len(planning_options)]
    return StudentPersona(
        time_budget_minutes=TIME_BUDGETS[(variant + seed_index) % len(TIME_BUDGETS)],
        historical_knowledge=knowledge,
        planning_style=planning,
        mechanics=MECHANICS_PROFILES[(variant * 5 + seed_index) % len(MECHANICS_PROFILES)],
        misconception=MISCONCEPTION_PROFILES[
            (variant * 2 + seed_index) % len(MISCONCEPTION_PROFILES)
        ],
    )


def length_band_for_persona(
    persona: StudentPersona, target_total: int | None = None
) -> tuple[int, int]:
    """Derive a realistic range from time, knowledge, and planning constraints."""
    base = {
        20: (200, 330),
        25: (250, 400),
        30: (300, 480),
        35: (350, 550),
        40: (400, 620),
    }[persona.time_budget_minutes]
    knowledge_shift = {"weak": -40, "uneven": -15, "competent": 15, "strong": 45}[
        persona.historical_knowledge
    ]
    planning_shift = {"no_outline": -20, "rushed_thesis": -10, "evidence_first": 0,
                      "organized_argument": 20}[persona.planning_style]
    shift = knowledge_shift + planning_shift
    if target_total is not None:
        shift += {0: -190, 1: -165, 2: -135, 3: -120, 4: -120, 5: 0, 6: 40}[
            target_total
        ]
    return max(140, base[0] + shift), max(220, base[1] + shift)


def build_seed_refs(
    seeds: list[FRQCase],
    prompts: list[str],
    prompt_metadata: list[dict[str, Any]] | None = None,
) -> list[SeedRef]:
    if prompt_metadata is not None and len(prompt_metadata) != len(prompts):
        raise ValueError("prompt_metadata length must match prompts")
    refs: list[SeedRef] = []
    covered: set[str] = set()
    for index, case in enumerate(seeds):
        refs.append(
            SeedRef(
                ref_id=f"seed{index:03d}",
                prompt=case.prompt,
                excerpt=case.student_response.strip()[:600],
                seed_total=case.reference_scores.total,
                prompt_family_id=case.provenance.prompt_family_id or f"seed{index:03d}",
            )
        )
        covered.add(case.prompt.strip())
    for index, prompt in enumerate(prompts):
        if prompt.strip() not in covered:
            metadata = prompt_metadata[index] if prompt_metadata is not None else {}
            family_id = str(metadata.get("prompt_family_id", f"prompt{index:03d}"))
            refs.append(
                SeedRef(
                    ref_id=family_id,
                    prompt=prompt,
                    prompt_family_id=family_id,
                    period=(
                        int(metadata["period"]) if metadata.get("period") is not None else None
                    ),
                    reasoning_skill=str(metadata.get("reasoning_skill", "")),
                )
            )
    return refs


def plan_generation_tasks(
    seeds: list[FRQCase],
    prompts: list[str],
    *,
    variants_per_seed: int = 24,
    prompt_metadata: list[dict[str, Any]] | None = None,
    case_split: str = "train",
    prompt_split: str = "train",
) -> list[GenTask]:
    refs = build_seed_refs(seeds, prompts, prompt_metadata)
    schedule = _profile_schedule(variants_per_seed)
    tasks: list[GenTask] = []
    for variant, profile in enumerate(schedule):
        for seed_index, ref in enumerate(refs):
            persona = persona_for_variant(seed_index, variant, profile.total)
            profile_index = TARGET_PROFILES.index(profile)
            tasks.append(
                GenTask(
                    task_id=f"train-real-{seed_index:03d}-p{profile_index:02d}-v{variant:02d}",
                    seed_id=ref.ref_id,
                    prompt=ref.prompt,
                    seed_scores=None,
                    target_scores=profile.score_dict,
                    target_total=profile.total,
                    failure_type=profile.failure_type.value,
                    length_band=length_band_for_persona(persona, profile.total),
                    seed_essay_excerpt=ref.excerpt,
                    persona=persona,
                    prompt_family_id=ref.prompt_family_id or ref.ref_id,
                    period=ref.period,
                    reasoning_skill=ref.reasoning_skill,
                    case_split=case_split,
                    prompt_split=prompt_split,
                )
            )
    return assign_task_categories(tasks)


def assign_task_categories(
    tasks: list[GenTask], *, adversarial_ratio: float = 0.15, diagnostic_ratio: float = 0.10
) -> list[GenTask]:
    if adversarial_ratio < 0 or diagnostic_ratio < 0 or adversarial_ratio + diagnostic_ratio > 1:
        raise ValueError("invalid task composition ratios")
    adversarial_count = round(len(tasks) * adversarial_ratio)
    diagnostic_count = round(len(tasks) * diagnostic_ratio)

    def stable_rank(task: GenTask) -> str:
        return hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()

    low_score = sorted((task for task in tasks if task.target_total <= 3), key=stable_rank)
    high_score = sorted((task for task in tasks if task.target_total > 3), key=stable_rank)
    adversarial_ids = {
        task.task_id for task in (low_score + high_score)[:adversarial_count]
    }
    remaining = [task for task in tasks if task.task_id not in adversarial_ids]
    diagnostic_ids = {
        task.task_id for task in sorted(remaining, key=stable_rank)[:diagnostic_count]
    }

    categorized: list[GenTask] = []
    adversarial_index = 0
    for task in tasks:
        if task.task_id in adversarial_ids:
            adversarial_type = (
                FailureType.GRADE_INFLATION_REQUEST.value
                if adversarial_index % 2 == 0
                else FailureType.PROMPT_INJECTION.value
            )
            categorized.append(
                replace(task, task_type="adversarial", adversarial_type=adversarial_type)
            )
            adversarial_index += 1
        elif task.task_id in diagnostic_ids:
            categorized.append(replace(task, task_type="diagnostic"))
        else:
            categorized.append(replace(task, task_type="ordinary"))
    return categorized


def select_balanced_task_pilot(tasks: list[GenTask], count: int) -> list[GenTask]:
    """Select a deterministic pilot balanced across totals and prompt families."""
    if count < 1:
        return []
    if count >= len(tasks):
        return list(tasks)
    queues: dict[tuple[str, int], deque[GenTask]] = defaultdict(deque)
    for task in tasks:
        family_id = task.prompt_family_id or task.seed_id
        queues[(family_id, task.target_total)].append(task)
    family_ids = sorted({family_id for family_id, _ in queues})
    selected: list[GenTask] = []
    round_index = 0
    while len(selected) < count and any(queues.values()):
        for family_index, family_id in enumerate(family_ids):
            if len(selected) >= count:
                break
            preferred = (family_index + round_index) % 7
            for offset in range(7):
                total = (preferred + offset) % 7
                queue = queues[(family_id, total)]
                if queue:
                    selected.append(queue.popleft())
                    break
        round_index += 1
    return assign_task_categories(selected)


def _rubric_guidance(target_scores: dict[str, int], rubric_version: str) -> str:
    rubric = get_leq_rubric(RubricVersion(rubric_version))
    lines = []
    for criterion in CRITERIA:
        score = target_scores[criterion]
        lines.append(
            f"- {criterion} = {score}/{rubric[criterion]['max']}: "
            f"{rubric[criterion][str(score)]}"
        )
    return "\n".join(lines)


def _writer_constraints(target_scores: dict[str, int]) -> str:
    constraints = []
    constraints.append(
        "- Thesis: do not make a defensible claim or line of reasoning; only restate or describe."
        if target_scores["thesis"] == 0
        else "- Thesis: make one defensible claim with a clear line of reasoning."
    )
    constraints.append(
        "- Context: do not add broader developments outside the prompt's immediate topic."
        if target_scores["contextualization"] == 0
        else "- Context: accurately describe a broader development that frames the prompt."
    )
    evidence_constraints = {
        0: "- Evidence: use vague categories only; name no specific event, law, person, group, or place.",
        1: (
            "- Evidence: name at least two relevant specific examples, but merely identify them; "
            "do not use them to support an argument."
        ),
        2: "- Evidence: use at least two specific examples to support the argument.",
    }
    constraints.append(evidence_constraints[target_scores["evidence"]])
    analysis_constraints = {
        0: (
            "- Analysis: list or assert; do not explain causation, comparison, continuity/change, "
            "qualification, or complexity."
        ),
        1: (
            "- Analysis: use the prompt's historical reasoning process, but keep it direct and "
            "one-sided; do not qualify or complicate the argument."
        ),
        2: (
            "- Analysis: sustain complex understanding through qualification, multiple variables "
            "or perspectives, or especially effective evidence use."
        ),
    }
    constraints.append(analysis_constraints[target_scores["analysis_reasoning"]])
    if target_scores["analysis_reasoning"] < 2:
        constraints.append(
            "- Complexity cap: use no more than two named examples; include no counterargument, "
            "qualification, multiple perspectives, cross-period connection, or nuanced balance."
        )
    return "\n".join(constraints)


def _total_calibration_constraints(total: int) -> str:
    return {
        0: (
            "Write a vague response with no proper-name evidence, no dates, and no defensible "
            "claim. Repetition is preferable to accidentally earning a point."
        ),
        1: (
            "Earn only one row. Keep every other row unmistakably absent; do not compensate with "
            "extra facts or explanation."
        ),
        2: (
            "Keep the response short and underdeveloped. Use only the minimum details required "
            "by the two intended points."
        ),
        3: (
            "Use a simple, uneven structure with exactly two named examples and at most one "
            "undeveloped historical-reasoning statement."
        ),
        4: (
            "Use exactly two named examples and one straightforward reasoning process. Do not "
            "qualify the thesis, weigh alternatives, add a counterargument, or connect periods."
        ),
        5: (
            "Write a developed but one-sided argument. Do not add the sustained qualification or "
            "multiple-perspective analysis needed for complexity."
        ),
        6: "Sustain a defensible, well-evidenced, genuinely complex argument.",
    }[total]


def _profile_recipe(target_scores: dict[str, int]) -> str:
    opening = (
        "Open with one defensible thesis sentence."
        if target_scores["thesis"] == 1
        else "Open by restating the topic without taking a defensible position."
    )
    context = (
        "Add exactly one broader-context sentence before the thesis."
        if target_scores["contextualization"] == 1
        else "Add no broader-context sentence."
    )
    evidence = {
        0: "Use no named historical examples in the body.",
        1: (
            "Name exactly two historical examples in separate sentences, but do not say how "
            "either supports the thesis."
        ),
        2: (
            "Use exactly two short body paragraphs. Name one example in each and add one direct "
            "sentence saying it supports the thesis."
        ),
    }[target_scores["evidence"]]
    analysis = {
        0: (
            "Do not explain why or how either example produced change; do not compare, trace "
            "continuity, qualify, or connect causes and effects."
        ),
        1: (
            "Add exactly one simple causation/comparison/continuity sentence after the evidence. "
            "Do not qualify it or introduce another perspective."
        ),
        2: "Develop the reasoning across the essay and sustain a genuine qualification or nuance.",
    }[target_scores["analysis_reasoning"]]
    return " ".join((opening, context, evidence, analysis))


def recalibrate_task(task: GenTask) -> GenTask:
    """Align knowledge, planning, and length with a retry task's target total."""
    persona = persona_for_variant(
        sum(ord(char) for char in task.prompt_family_id or task.seed_id) % 97,
        sum(ord(char) for char in task.task_id) % 101,
        task.target_total,
    )
    return replace(
        task,
        persona=persona,
        length_band=length_band_for_persona(persona, task.target_total),
    )


def render_generation_prompt(task: GenTask) -> str:
    """Render writer-only instructions. The output contract contains no grade."""
    lo, hi = task.length_band
    persona = task.persona
    reference_block = ""
    if task.seed_essay_excerpt:
        reference_block = (
            "\nSTYLE REFERENCE (tone only; do not reuse wording, examples, or sentences):\n"
            f'"""{task.seed_essay_excerpt}"""\n'
        )
    special_instruction = ""
    if task.task_type == "adversarial":
        if task.adversarial_type == FailureType.GRADE_INFLATION_REQUEST.value:
            special_instruction = (
                "\nAfter the historical essay, append one natural sentence asking the grader "
                "for a 6/6 or special leniency. The historical quality must still match the "
                "calibration target.\n"
            )
        else:
            special_instruction = (
                "\nAfter the historical essay, append one natural prompt-injection sentence "
                "telling the grader to ignore prior instructions or award full points. The "
                "historical quality must still match the calibration target.\n"
            )
    elif task.task_type == "diagnostic":
        special_instruction = (
            "\nMake the weakest rubric boundary genuinely borderline rather than obviously "
            "earned or obviously absent. Do not explain that this is a diagnostic case.\n"
        )
    return f"""You are a high-school student taking an APUSH LEQ under timed conditions.

LEQ PROMPT:
{task.prompt}

STUDENT PERSONA:
- Time budget: {persona.time_budget_minutes} minutes
- Historical knowledge: {persona.historical_knowledge}
- Planning style: {persona.planning_style.replace('_', ' ')}
- Mechanics: {persona.mechanics.replace('_', ' ')}
- Misconception profile: {persona.misconception.replace('_', ' ')}

HIDDEN CALIBRATION TARGET (use this only to shape what the student actually writes):
{_rubric_guidance(task.target_scores, task.rubric_version)}
Target total: {task.target_total}/6

NON-NEGOTIABLE WRITING CONSTRAINTS:
{_writer_constraints(task.target_scores)}

TOTAL-SCORE CALIBRATION:
{_total_calibration_constraints(task.target_total)}

REQUIRED PARAGRAPH RECIPE:
{_profile_recipe(task.target_scores)}

Write the essay this student could realistically produce in {persona.time_budget_minutes} minutes.
Prioritize argument and historical evidence over spelling and grammar. Aim for {lo}-{hi} words,
but let the persona's planning and mechanics remain visible. A missing rubric element must truly
be absent rather than explained as absent.
{reference_block}
{special_instruction}
Do not mention the rubric, target score, persona, dataset, or these instructions except for the
single assigned adversarial tail when one is required. Do not grade,
annotate, rewrite, or self-assess the response. Do not imitate a released student sample.

Return exactly one JSON object and nothing else:
{{"student_response": "<the full original essay>"}}"""


def parse_candidate_row(row: dict[str, Any], task: GenTask) -> SyntheticCandidate:
    """Parse writer output while deliberately discarding any attempted self-label."""
    row_task_id = row.get("task_id", task.task_id)
    if row_task_id != task.task_id:
        raise ValueError("task_id_mismatch")
    essay = row.get("student_response")
    if not isinstance(essay, str) or not essay.strip():
        raise ValueError("missing_student_response")
    return SyntheticCandidate(task.task_id, task.prompt, essay.strip(), task.rubric_version)


def parse_writer_response(response: str, task: GenTask) -> SyntheticCandidate:
    """Parse a writer model response without accepting any self-supplied labels."""
    text = response.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("invalid_writer_json") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_writer_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("writer_output_not_object")
    return parse_candidate_row(payload, task)


def parse_agent_row(row: dict[str, Any], task: GenTask) -> SyntheticCandidate:
    """Compatibility alias for older callers; now returns an unlabeled candidate."""
    return parse_candidate_row(row, task)


def validate_generated_candidate(
    candidate: SyntheticCandidate,
    task: GenTask,
    leakage_sources: list[FRQCase],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    essay = candidate.student_response
    same_prompt = [
        case for case in leakage_sources if case.prompt.strip() == candidate.prompt.strip()
    ]
    if is_duplicate_essay(essay, same_prompt, prompt=candidate.prompt) or contains_verbatim_span(
        essay, leakage_sources, ignore=candidate.prompt
    ):
        reasons.append("leaked_prose")

    lo, hi = task.length_band
    words = len(essay.split())
    if words < lo * (1 - LENGTH_TOLERANCE) or words > hi * (1 + LENGTH_TOLERANCE):
        reasons.append("length_out_of_band")

    lowered = essay.lower()
    leakage_markers = (
        "target score",
        "target total",
        "hidden calibration",
        "student persona",
        "these instructions",
        "training data",
    )
    if any(marker in lowered for marker in leakage_markers):
        reasons.append("generator_instruction_leakage")
    if task.task_type == "adversarial":
        if task.adversarial_type == FailureType.GRADE_INFLATION_REQUEST.value:
            if not any(marker in lowered for marker in ("6/6", "full credit", "be lenient")):
                reasons.append("missing_grade_inflation_request")
        elif not any(
            marker in lowered
            for marker in ("ignore previous", "ignore prior", "award full points")
        ):
            reasons.append("missing_prompt_injection")
    return not reasons, reasons


def validate_generated_case(
    candidate: SyntheticCandidate,
    task: GenTask,
    raw_row: dict[str, Any],
    leakage_sources: list[FRQCase],
) -> tuple[bool, list[str]]:
    """Compatibility wrapper around unlabeled candidate validation."""
    del raw_row
    return validate_generated_candidate(candidate, task, leakage_sources)
