"""College Board APUSH LEQ rubric definitions and score validation."""

from __future__ import annotations

from typing import Any

CRITERIA = ("thesis", "contextualization", "evidence", "analysis_reasoning")

SCORE_RANGES: dict[str, tuple[int, int]] = {
    "thesis": (0, 1),
    "contextualization": (0, 1),
    "evidence": (0, 2),
    "analysis_reasoning": (0, 2),
}

LEQ_RUBRIC = {
    "thesis": {
        "max": 1,
        "description": "Makes a historically defensible claim that establishes a line of reasoning.",
        "0": "No defensible thesis or merely restates the prompt.",
        "1": "Clear, defensible thesis with a line of reasoning.",
    },
    "contextualization": {
        "max": 1,
        "description": "Describes broader historical context relevant to the prompt.",
        "0": "No meaningful broader context.",
        "1": "Accurate broader context that frames the argument.",
    },
    "evidence": {
        "max": 2,
        "description": "Provides specific examples that address the topic of the prompt.",
        "0": "No relevant evidence.",
        "1": "One piece of relevant evidence.",
        "2": "Two or more specific, relevant pieces of evidence.",
    },
    "analysis_reasoning": {
        "max": 2,
        "description": "Uses evidence to support an argument and demonstrates complex understanding.",
        "0": "Lists facts without argument or analysis.",
        "1": "Uses evidence to support an argument.",
        "2": "Demonstrates complex understanding (causation, continuity/change, comparison).",
    },
}


def compute_total(scores: Any) -> int:
    if hasattr(scores, "thesis"):
        return (
            int(scores.thesis)
            + int(scores.contextualization)
            + int(scores.evidence)
            + int(scores.analysis_reasoning)
        )
    return sum(int(scores[criterion]) for criterion in CRITERIA)


def validate_scores(scores: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for criterion, (low, high) in SCORE_RANGES.items():
        if criterion not in scores:
            reasons.append(f"missing_{criterion}")
            continue
        value = scores[criterion]
        if not isinstance(value, int) or value < low or value > high:
            reasons.append(f"out_of_range_{criterion}")
    return not reasons, reasons


def validate_grade_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "scores" not in payload or not isinstance(payload["scores"], dict):
        reasons.append("missing_scores")
        return False, reasons
    if "feedback" not in payload or not isinstance(payload["feedback"], dict):
        reasons.append("missing_feedback")
        return False, reasons

    ok, score_reasons = validate_scores(payload["scores"])
    reasons.extend(score_reasons)

    for criterion in CRITERIA:
        if criterion not in payload["feedback"]:
            reasons.append(f"missing_feedback_{criterion}")
        elif not isinstance(payload["feedback"][criterion], str) or not payload["feedback"][criterion].strip():
            reasons.append(f"empty_feedback_{criterion}")

    expected_total = compute_total(payload["scores"])
    if "total" not in payload:
        reasons.append("missing_total")
    elif payload["total"] != expected_total:
        reasons.append("total_mismatch")

    return not reasons, reasons
