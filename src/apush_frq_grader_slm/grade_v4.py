"""Deterministic essay-grounded feedback for v4 target score profiles."""

from __future__ import annotations

import re
from typing import Mapping

from apush_frq_grader_slm.filters import (
    contains_hallucination_pattern,
    feedback_references_essay,
)
from apush_frq_grader_slm.schemas import RubricFeedback, RubricScores

_STOP = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "were",
    "was",
    "are",
    "for",
    "into",
    "their",
    "they",
    "have",
    "had",
    "been",
    "which",
    "while",
    "about",
    "after",
    "before",
    "because",
    "through",
    "during",
    "also",
    "than",
    "then",
    "when",
    "where",
    "what",
    "would",
    "could",
    "should",
    "united",
    "states",
    "american",
    "america",
    "people",
    "history",
    "period",
}


def _sentences(essay: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", essay.strip())
    return [part.strip() for part in parts if len(part.strip().split()) >= 4]


def _anchors(essay: str, n: int = 4) -> list[str]:
    """Pick short distinctive spans that appear verbatim in the essay."""
    words = essay.split()
    spans: list[str] = []
    step = max(1, len(words) // max(n, 1))
    for start in range(0, max(1, len(words) - 7), step):
        chunk = " ".join(words[start : start + 8])
        if len(chunk) > 12 and chunk not in spans and chunk in essay:
            spans.append(chunk)
        if len(spans) >= n:
            break
    if len(spans) < n:
        for sentence in _sentences(essay):
            chunk = " ".join(sentence.split()[:8])
            if len(chunk) > 12 and chunk not in spans and chunk in essay:
                spans.append(chunk)
            if len(spans) >= n:
                break
    if not spans:
        spans = [" ".join(words[:8])] if words else ["this essay"]
    while len(spans) < n:
        spans.append(spans[-1])
    return spans[:n]


def _safe_quote(span: str) -> str:
    """Prefer double quotes; fall back to no quotes if the span has awkward marks."""
    if '"' in span or "\u201c" in span or "\u201d" in span:
        return span
    return f'"{span}"'


def _draft_feedback(essay: str, scores: Mapping[str, int]) -> RubricFeedback:
    a0, a1, a2, a3 = _anchors(essay, 4)
    q0, q1, q2, q3 = (_safe_quote(a) for a in (a0, a1, a2, a3))
    thesis = int(scores["thesis"])
    contextualization = int(scores["contextualization"])
    evidence = int(scores["evidence"])
    analysis = int(scores["analysis_reasoning"])

    if thesis:
        thesis_fb = (
            f"A historically defensible claim appears when the essay argues {q0}, "
            "establishing a clear line of reasoning rather than restating the prompt."
        )
    else:
        thesis_fb = (
            f"The opening material around {q0} restates the topic or stays too vague "
            "to count as a historically defensible thesis with a line of reasoning."
        )

    if contextualization:
        ctx_fb = (
            f"Broader context is present where the essay situates {q1} against earlier "
            "or surrounding developments before developing the main argument."
        )
    else:
        ctx_fb = (
            f"Although {q1} appears, the essay does not set the stage with broader "
            "historical context before or beyond the prompt's immediate timeframe."
        )

    if evidence >= 2:
        ev_fb = (
            f"Specific evidence such as {q2} is used to support the argument rather than "
            "merely listed, helping prove the thesis in response to the prompt."
        )
    elif evidence == 1:
        ev_fb = (
            f"The essay names relevant examples including {q2}, earning evidence for "
            "specificity, but does not consistently use those examples to prove a thesis."
        )
    else:
        ev_fb = (
            f"References like {q2} remain too general or off-topic to provide two "
            "specific pieces of historical evidence relevant to the prompt."
        )

    if analysis >= 2:
        ar_fb = (
            f"The essay structures historical reasoning around {q3} and adds complexity "
            "through nuance, qualification, or multiple perspectives."
        )
    elif analysis == 1:
        ar_fb = (
            f"Historical reasoning is visible in how {q3} is organized through causation, "
            "comparison, or continuity/change, but complexity remains limited."
        )
    else:
        ar_fb = (
            f"Discussion of {q3} stays descriptive and does not structure an argument "
            "with causation, comparison, or continuity and change."
        )

    return RubricFeedback(
        thesis=thesis_fb,
        contextualization=ctx_fb,
        evidence=ev_fb,
        analysis_reasoning=ar_fb,
    )


def _fallback_feedback(essay: str, scores: Mapping[str, int]) -> RubricFeedback:
    """Quote-free fallback that overlaps multiple distinctive essay tokens."""
    tokens = [word for word in re.findall(r"[A-Za-z]{5,}", essay)]
    seen: set[str] = set()
    uniq: list[str] = []
    for token in tokens:
        key = token.lower()
        if key not in seen and key not in _STOP:
            seen.add(key)
            uniq.append(token)
    while len(uniq) < 8:
        uniq.append(uniq[-1] if uniq else "essay")
    thesis = int(scores["thesis"])
    contextualization = int(scores["contextualization"])
    evidence = int(scores["evidence"])
    analysis = int(scores["analysis_reasoning"])
    return RubricFeedback(
        thesis=(
            f"The essay builds a defensible claim using {uniq[0]} and {uniq[1]} with a line of reasoning."
            if thesis
            else f"The essay mentions {uniq[0]} and {uniq[1]} but never states a defensible thesis."
        ),
        contextualization=(
            f"Before the main argument, {uniq[2]} and {uniq[3]} help set broader historical context."
            if contextualization
            else f"The essay jumps to {uniq[2]} and {uniq[3]} without broader contextualization."
        ),
        evidence=(
            f"Specific examples involving {uniq[4]} and {uniq[5]} are used to support the argument."
            if evidence >= 2
            else (
                f"The essay names {uniq[4]} and {uniq[5]} but mostly lists them without proving a thesis."
                if evidence == 1
                else f"Mentions of {uniq[4]} and {uniq[5]} stay too vague for two specific evidence points."
            )
        ),
        analysis_reasoning=(
            f"The handling of {uniq[6]} and {uniq[7]} shows historical reasoning plus complexity."
            if analysis >= 2
            else (
                f"Organization around {uniq[6]} and {uniq[7]} shows historical reasoning without complexity."
                if analysis == 1
                else (
                    f"Discussion of {uniq[6]} and {uniq[7]} stays descriptive without historical "
                    "reasoning structure."
                )
            )
        ),
    )


def grounded_feedback_for_scores(essay: str, scores: Mapping[str, int]) -> RubricFeedback:
    """Build feedback that passes grounding and hallucination checks."""
    candidates = [_draft_feedback(essay, scores), _fallback_feedback(essay, scores)]
    for feedback in candidates:
        ok = True
        for text in feedback.model_dump().values():
            if not feedback_references_essay(text, essay):
                ok = False
                break
            if contains_hallucination_pattern(text, essay):
                ok = False
                break
        if ok:
            return feedback
    return candidates[-1]


def grade_payload_for_target(essay: str, scores: Mapping[str, int]) -> dict:
    rubric_scores = RubricScores.model_validate(scores)
    feedback = grounded_feedback_for_scores(essay, scores)
    return {
        "scores": rubric_scores.model_dump(),
        "total": rubric_scores.total,
        "feedback": feedback.model_dump(),
        "labeling_method": "target_profile_grounded",
        "grader_ids": ["v4_grounded_feedback"],
    }
