"""Deterministic essay-grounded feedback for v4 target score profiles."""

from __future__ import annotations

import re
from typing import Mapping

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
    """Pick short distinctive spans from the essay for grounded feedback."""
    spans: list[str] = []
    for sentence in _sentences(essay):
        words = sentence.split()
        if len(words) >= 8:
            chunk = " ".join(words[:12])
        else:
            chunk = sentence
        chunk = chunk.strip(" \"'")
        if len(chunk) > 12 and chunk.lower() not in {s.lower() for s in spans}:
            spans.append(chunk)
        if len(spans) >= n:
            break
    if not spans:
        words = essay.split()
        spans = [" ".join(words[:8])] if words else ["the response"]
    return spans


def grounded_feedback_for_scores(essay: str, scores: Mapping[str, int]) -> RubricFeedback:
    """Build one-sentence-per-criterion feedback that quotes/paraphrases the essay."""
    a0, a1, a2, a3 = (_anchors(essay, 4) + ["the response"] * 4)[:4]
    thesis = int(scores["thesis"])
    contextualization = int(scores["contextualization"])
    evidence = int(scores["evidence"])
    analysis = int(scores["analysis_reasoning"])

    if thesis:
        thesis_fb = (
            f"A historically defensible claim appears when the essay argues '{a0}', "
            "establishing a clear line of reasoning rather than restating the prompt."
        )
    else:
        thesis_fb = (
            f"The opening material around '{a0}' restates the topic or stays too vague "
            "to count as a historically defensible thesis with a line of reasoning."
        )

    if contextualization:
        ctx_fb = (
            f"Broader context is present where the essay situates '{a1}' against earlier "
            "or surrounding developments before developing the main argument."
        )
    else:
        ctx_fb = (
            f"Although '{a1}' appears, the essay does not set the stage with broader "
            "historical context before or beyond the prompt's immediate timeframe."
        )

    if evidence >= 2:
        ev_fb = (
            f"Specific evidence such as '{a2}' is used to support the argument rather than "
            "merely listed, helping prove the thesis in response to the prompt."
        )
    elif evidence == 1:
        ev_fb = (
            f"The essay names relevant examples including '{a2}', earning evidence for "
            "specificity, but does not consistently use those examples to prove a thesis."
        )
    else:
        ev_fb = (
            f"References like '{a2}' remain too general or off-topic to provide two "
            "specific pieces of historical evidence relevant to the prompt."
        )

    if analysis >= 2:
        ar_fb = (
            f"The essay structures historical reasoning around '{a3}' and adds complexity "
            "through nuance, qualification, or multiple perspectives."
        )
    elif analysis == 1:
        ar_fb = (
            f"Historical reasoning is visible in how '{a3}' is organized through causation, "
            "comparison, or continuity/change, but complexity remains limited."
        )
    else:
        ar_fb = (
            f"Discussion of '{a3}' stays descriptive and does not structure an argument "
            "with causation, comparison, or continuity and change."
        )

    return RubricFeedback(
        thesis=thesis_fb,
        contextualization=ctx_fb,
        evidence=ev_fb,
        analysis_reasoning=ar_fb,
    )


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
