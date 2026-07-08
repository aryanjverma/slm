"""Hybrid commentary distillation: CB scores + essay-anchored feedback."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from apush_frq_grader_slm.data import _essay_anchor, _format_grade_json
from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.apc_parser import RawAPCSample
from apush_frq_grader_slm.rubric import compute_total
from apush_frq_grader_slm.schemas import FRQCase, FailureType, RubricFeedback, RubricScores


def raw_sample_to_frq_case(
    sample: RawAPCSample,
    *,
    case_id: str | None = None,
    distill: bool = False,
    seed: int = 13,
) -> FRQCase:
    """Convert a parsed AP Central sample into an eval-only FRQCase."""
    scores = RubricScores.model_validate(sample.scores)
    failure_type = infer_failure_type(sample)
    feedback = build_feedback(
        sample,
        scores,
        distill=distill,
        seed=seed,
    )
    assistant_response = _format_grade_json(scores, feedback)
    difficulty = _difficulty(scores.total)
    source = sample.metadata.get("source", sample.sample_id)
    resolved_id = case_id or source

    case = FRQCase(
        id=resolved_id,
        split="eval",
        prompt=sample.prompt,
        student_response=sample.essay,
        reference_scores=scores,
        reference_feedback=feedback,
        failure_type=failure_type,
        difficulty=difficulty,
        assistant_response=assistant_response,
        tags=[
            "ap_central",
            "real_eval",
            str(sample.metadata.get("year", "")),
            f"leq{sample.metadata.get('leq_num', '')}",
            failure_type.value,
            difficulty,
        ],
    )
    return case


def infer_failure_type(sample: RawAPCSample) -> FailureType:
    """Infer failure slice from CB row scores and commentary cues."""
    scores = sample.scores
    total = sample.total_score
    commentary = " ".join(sample.commentary_by_row.values()).lower()

    if total >= 5:
        return FailureType.STRONG
    if total <= 2:
        if scores.get("thesis", 1) == 0:
            return FailureType.WEAK_THESIS
        if "list" in commentary or scores.get("analysis_reasoning", 0) == 0:
            return FailureType.EVIDENCE_LIST
        return FailureType.WEAK_THESIS
    if scores.get("contextualization", 1) == 0:
        return FailureType.MISSING_CONTEXT
    if "complex" in commentary or "qualif" in commentary or "nuanc" in commentary:
        return FailureType.BORDERLINE_COMPLEXITY
    if scores.get("analysis_reasoning", 0) <= 1:
        return FailureType.BORDERLINE_COMPLEXITY
    return FailureType.MISSING_CONTEXT


def build_feedback(
    sample: RawAPCSample,
    scores: RubricScores,
    *,
    distill: bool = False,
    seed: int = 13,
) -> RubricFeedback:
    """Build essay-anchored feedback using CB commentary or rule templates."""
    if distill:
        distilled = _distill_with_llm(sample, scores)
        if distilled is not None:
            return distilled
    return _template_feedback(sample.essay, scores, sample.commentary_by_row, seed=seed)


def _template_feedback(
    essay: str,
    scores: RubricScores,
    commentary_by_row: dict[str, str],
    *,
    seed: int,
) -> RubricFeedback:
    """Rule-based feedback grounded in essay text, preserving CB scores."""
    anchor = _safe_anchor(essay)
    if len(anchor) < 8:
        anchor = "the student response"
    _ = random.Random(seed)

    def row_text(criterion: str, earned: int, max_score: int, positive: str, negative: str) -> str:
        if earned >= max_score:
            return positive.format(anchor=anchor)
        if earned == 0:
            return negative.format(anchor=anchor)
        return (
            f"The essay references '{anchor}' but only partially meets the {criterion} criterion."
        )

    thesis = row_text(
        "thesis",
        scores.thesis,
        1,
        "A defensible thesis appears when the essay argues '{anchor}'.",
        "The response restates the topic ('{anchor}') without a defensible claim.",
    )
    contextualization = row_text(
        "contextualization",
        scores.contextualization,
        1,
        "Broader context involving '{anchor}' frames the argument before evidence.",
        "Before discussing '{anchor}', the essay does not establish broader context.",
    )
    evidence = row_text(
        "evidence",
        scores.evidence,
        2,
        "Specific examples such as '{anchor}' support the prompt topic.",
        "The essay names '{anchor}' but does not develop sufficient specific evidence.",
    )
    if scores.evidence == 1:
        evidence = (
            f"The essay identifies '{anchor}' as evidence but needs a second developed example."
        )

    analysis = row_text(
        "analysis_reasoning",
        scores.analysis_reasoning,
        2,
        "The essay links '{anchor}' to argument and notes continuity and change.",
        "References like '{anchor}' are listed without using evidence to support an argument.",
    )
    if scores.analysis_reasoning == 1:
        analysis = (
            f"References like '{anchor}' support a claim, but analysis stays general."
        )

    return RubricFeedback(
        thesis=thesis,
        contextualization=contextualization,
        evidence=evidence,
        analysis_reasoning=analysis,
    )


def _safe_anchor(essay: str) -> str:
    anchor = _essay_anchor(essay)
    if "'" in anchor or '"' in anchor:
        for chunk in re.split(r"[.!?]", essay):
            cleaned = chunk.strip()
            if len(cleaned) >= 15 and "'" not in cleaned and '"' not in cleaned:
                return cleaned[:70]
    if anchor.lower() in essay.lower():
        return anchor
    return _first_substantial_phrase(essay) or essay[:70].strip()


def _distill_with_llm(sample: RawAPCSample, scores: RubricScores) -> RubricFeedback | None:
    """Optional LLM rewrite of CB commentary into essay-anchored feedback."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    import os

    if not os.environ.get("OPENAI_API_KEY"):
        return None

    client = OpenAI()
    rows = {}
    for criterion in ("thesis", "contextualization", "evidence", "analysis_reasoning"):
        cb_text = sample.commentary_by_row.get(criterion, "")
        earned = getattr(scores, criterion)
        prompt = (
            "Rewrite this AP reader commentary into 1-2 sentences of feedback for the student. "
            "Quote or paraphrase the student's essay. Do not change the score. "
            f"Score for {criterion}: {earned}.\n\n"
            f"Essay:\n{sample.essay[:3000]}\n\n"
            f"Reader commentary:\n{cb_text[:1500]}"
        )
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
            )
            rows[criterion] = response.output_text.strip()
        except Exception:
            return None

    feedback = RubricFeedback.model_validate(rows)
    case = FRQCase(
        id="distill-check",
        split="eval",
        prompt=sample.prompt,
        student_response=sample.essay,
        reference_scores=scores,
        reference_feedback=feedback,
        failure_type=FailureType.STRONG,
        difficulty="strong",
        assistant_response=_format_grade_json(scores, feedback),
    )
    ok, _ = passes_quality_gate(case)
    if not ok:
        return None
    return feedback


def distill_payload(sample: RawAPCSample, feedback: RubricFeedback, scores: RubricScores) -> dict[str, Any]:
    return {
        "scores": scores.model_dump(),
        "total": compute_total(scores),
        "feedback": feedback.model_dump(),
    }


def _difficulty(total: int) -> str:
    if total <= 2:
        return "weak"
    if total <= 4:
        return "borderline"
    return "strong"
