"""Realistic, real-seeded synthetic training data.

The agent produces raw prose; this module deterministically plans the worklist,
renders the exact prompt contract, and validates/labels/dedups the results.
Labels are the *pre-assigned* target profile (ground truth), never the agent's
self-grade, so training targets always satisfy the rubric (total == sum, ranges).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from apush_frq_grader_slm.behavior import OUTPUT_JSON_SCHEMA
from apush_frq_grader_slm.data import _difficulty, _format_grade_json
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.dedup import contains_verbatim_span, is_duplicate_essay
from apush_frq_grader_slm.rubric import CRITERIA, LEQ_RUBRIC, compute_total, validate_grade_payload
from apush_frq_grader_slm.schemas import FRQCase, FailureType, RubricFeedback, RubricScores

# Length bands (word counts) matching real CB LEQ prose, rotated across variants.
LENGTH_BANDS: tuple[tuple[int, int], ...] = ((400, 550), (550, 700), (700, 850))
LENGTH_TOLERANCE = 0.15  # accept word counts within +-15% of the assigned band


@dataclass(frozen=True)
class TargetProfile:
    """A per-criterion score target with its failure slice and sampling weight."""

    scores: tuple[int, int, int, int]  # thesis, contextualization, evidence, analysis_reasoning
    failure_type: FailureType
    weight: float

    @property
    def score_dict(self) -> dict[str, int]:
        return dict(zip(CRITERIA, self.scores, strict=True))

    @property
    def total(self) -> int:
        return sum(self.scores)


# Weighted toward the real CB mid-low distribution (mean ~2), but every total 0-6
# is represented so each prompt appears across the full score range (breaks any
# prompt -> score shortcut the model might otherwise learn).
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
class SeedRef:
    """A generation anchor: an LEQ prompt, optionally grounded in a real essay."""

    ref_id: str
    prompt: str
    excerpt: str = ""  # <=600 chars of a non-eval real essay; "" when none available
    seed_total: int | None = None


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

    def to_row(self) -> dict:
        row = asdict(self)
        row["length_band"] = list(self.length_band)
        return row

    @classmethod
    def from_row(cls, row: dict) -> GenTask:
        band = row["length_band"]
        return cls(
            task_id=row["task_id"],
            seed_id=row["seed_id"],
            prompt=row["prompt"],
            seed_scores=row.get("seed_scores"),
            target_scores=row["target_scores"],
            target_total=row["target_total"],
            failure_type=row["failure_type"],
            length_band=(int(band[0]), int(band[1])),
            seed_essay_excerpt=row.get("seed_essay_excerpt", ""),
        )


def _profile_schedule(n: int) -> list[TargetProfile]:
    """Deterministically allocate `n` slots across TARGET_PROFILES by weight
    (largest-remainder), so the score distribution is stable and reproducible."""
    total_weight = sum(p.weight for p in TARGET_PROFILES)
    exact = [(p, n * p.weight / total_weight) for p in TARGET_PROFILES]
    counts = {id(p): int(frac) for p, frac in exact}
    assigned = sum(counts.values())
    remainder = sorted(exact, key=lambda pf: pf[1] - int(pf[1]), reverse=True)
    for p, _ in remainder[: n - assigned]:
        counts[id(p)] += 1
    schedule: list[TargetProfile] = []
    for p in TARGET_PROFILES:  # interleave-friendly, deterministic order
        schedule.extend([p] * counts[id(p)])
    return schedule


def build_seed_refs(seeds: list[FRQCase], prompts: list[str]) -> list[SeedRef]:
    """One ref per real seed essay (with excerpt), plus one prompt-only ref for any
    `prompts` entry not already covered by a real seed (guarantees full prompt
    coverage even when older-year PDFs yield few seeds)."""
    refs: list[SeedRef] = []
    covered: set[str] = set()
    for idx, case in enumerate(seeds):
        refs.append(
            SeedRef(
                ref_id=f"seed{idx:03d}",
                prompt=case.prompt,
                excerpt=case.student_response.strip()[:600],
                seed_total=case.reference_scores.total,
            )
        )
        covered.add(case.prompt.strip())
    for idx, prompt in enumerate(prompts):
        if prompt.strip() not in covered:
            refs.append(SeedRef(ref_id=f"prompt{idx:03d}", prompt=prompt))
    return refs


def plan_generation_tasks(
    seeds: list[FRQCase],
    prompts: list[str],
    *,
    variants_per_seed: int = 24,
) -> list[GenTask]:
    """Fan out refs x profiles x length bands into a deterministic worklist.

    Variation is fully determined by (ref index, variant index) so reruns are
    stable without Math.random/Date (banned in workflow scripts)."""
    refs = build_seed_refs(seeds, prompts)
    schedule = _profile_schedule(variants_per_seed)
    tasks: list[GenTask] = []
    for seed_idx, ref in enumerate(refs):
        for variant, profile in enumerate(schedule):
            band = LENGTH_BANDS[variant % len(LENGTH_BANDS)]
            profile_idx = TARGET_PROFILES.index(profile)
            tasks.append(
                GenTask(
                    task_id=f"train-real-{seed_idx:03d}-p{profile_idx:02d}-v{variant:02d}",
                    seed_id=ref.ref_id,
                    prompt=ref.prompt,
                    seed_scores=None,
                    target_scores=profile.score_dict,
                    target_total=profile.total,
                    failure_type=profile.failure_type.value,
                    length_band=band,
                    seed_essay_excerpt=ref.excerpt,
                )
            )
    return tasks


def _rubric_guidance(target_scores: dict[str, int]) -> str:
    lines = []
    for criterion in CRITERIA:
        score = target_scores[criterion]
        descriptor = LEQ_RUBRIC[criterion][str(score)]
        max_score = LEQ_RUBRIC[criterion]["max"]
        lines.append(f"- {criterion} = {score}/{max_score}: {descriptor}")
    return "\n".join(lines)


def render_generation_prompt(task: GenTask) -> str:
    """The exact instruction handed to the generating agent for one task."""
    lo, hi = task.length_band
    reference_block = ""
    if task.seed_essay_excerpt:
        reference_block = (
            "\nSTYLE REFERENCE (a real student essay on this era, for tone/length only "
            "-- DO NOT reuse any of its wording, examples, or sentences):\n"
            f'"""{task.seed_essay_excerpt}"""\n'
        )
    return f"""You are generating ONE realistic APUSH Long Essay Question (LEQ) response \
written by a high-school student, for use as training data. Write in the voice of a \
student under timed-exam conditions -- natural imperfections are expected \
(run-on sentences, some vague phrasing, uneven paragraphing).

LEQ PROMPT:
{task.prompt}

TARGET QUALITY -- the essay you write must GENUINELY DESERVE exactly these rubric scores. \
Do not write a better or worse essay than the target; calibrate deliberately:
{_rubric_guidance(task.target_scores)}
Target total: {task.target_total}/6.

LENGTH: write {lo}-{hi} words (a full multi-paragraph essay, like a real exam response).
{reference_block}
HARD RULES:
- Write a BRAND-NEW, original essay. Do NOT copy any sentence or phrase from the style \
reference or any other essay. Invent your own examples and wording.
- The essay must be historically plausible for the era in the prompt.
- To hit a low score, genuinely omit the corresponding element (e.g. thesis=0 means no \
defensible claim; evidence=1 means exactly one developed specific example; \
analysis_reasoning=0 means facts are listed without an argument).

Then grade the essay you just wrote. Return EXACTLY ONE JSON object and nothing else, \
matching this schema:
{OUTPUT_JSON_SCHEMA}
- "scores" MUST equal the target scores above; "total" MUST equal their sum.
- Each "feedback" field: 1-2 sentences that paraphrase what YOUR essay does, reusing its key \
terms (e.g. named events, people, or concepts you wrote). Do NOT use quotation marks; \
paraphrase instead of quoting. Do not invent documents, quotes, or facts not in the essay. \
Do not rewrite or improve the essay.

Output format (one line, valid JSON):
{{"student_response": "<the full essay text>", "scores": {{...}}, "total": <int>, "feedback": {{...}}}}"""


def parse_agent_row(row: dict, task: GenTask) -> FRQCase:
    """Build a training FRQCase from an agent row, using the target profile as the
    authoritative label (assistant_response is rebuilt from target scores + agent
    feedback, so total == sum and ranges always hold)."""
    scores = RubricScores.model_validate(task.target_scores)
    feedback = RubricFeedback.model_validate(row["feedback"])
    assistant_response = _format_grade_json(scores, feedback)
    difficulty = _difficulty(task.target_total)
    failure_type = FailureType(task.failure_type)
    return FRQCase(
        id=task.task_id,
        split="train",
        prompt=task.prompt,
        student_response=str(row["student_response"]).strip(),
        reference_scores=scores,
        reference_feedback=feedback,
        failure_type=failure_type,
        difficulty=difficulty,
        assistant_response=assistant_response,
        tags=["synth_realistic", failure_type.value, difficulty],
    )


def validate_generated_case(
    case: FRQCase,
    task: GenTask,
    raw_row: dict,
    leakage_sources: list[FRQCase],
) -> tuple[bool, list[str]]:
    """Accept only cases that are on-target, valid, grounded, and leakage-free."""
    reasons: list[str] = []

    # (a) label discipline: the agent's self-reported scores must match the target.
    echoed = raw_row.get("scores")
    if not isinstance(echoed, dict) or {k: echoed.get(k) for k in CRITERIA} != task.target_scores:
        reasons.append("score_mismatch")

    # (b) rubric validity of the (rebuilt) training target: ranges + total == sum.
    payload = json.loads(case.assistant_response)
    ok, payload_reasons = validate_grade_payload(payload)
    if not ok:
        reasons.extend(payload_reasons)

    # (c) structural gate: JSON, feedback grounding, no rewrite, no hallucinated quote.
    gate_ok, gate_reasons = passes_quality_gate(case)
    if not gate_ok:
        reasons.extend(gate_reasons)

    # (d) anti-leakage. Two precise checks, avoiding is_duplicate_essay's loose
    # cross-prompt token heuristic (which false-positives on shared APUSH
    # vocabulary between any two long essays):
    #   - near-duplicate: Jaccard >= 0.82 only against SAME-prompt real essays;
    #   - verbatim copy: any lifted 8-word span (excluding shared prompt wording).
    essay = case.student_response
    same_prompt = [c for c in leakage_sources if c.prompt.strip() == case.prompt.strip()]
    if is_duplicate_essay(essay, same_prompt, prompt=case.prompt) or contains_verbatim_span(
        essay, leakage_sources, ignore=case.prompt
    ):
        reasons.append("leaked_prose")

    # (e) length band sanity (+-15%).
    lo, hi = task.length_band
    words = len(essay.split())
    if words < lo * (1 - LENGTH_TOLERANCE) or words > hi * (1 + LENGTH_TOLERANCE):
        reasons.append("length_out_of_band")

    return not reasons, reasons
