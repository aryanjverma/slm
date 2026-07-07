"""Quality gates for LEQ grading examples and model outputs."""

from __future__ import annotations

import json
import re

from apush_frq_grader_slm.rubric import validate_grade_payload
from apush_frq_grader_slm.schemas import FRQCase

REWRITE_PATTERNS = [
    r"\bhere is a revised\b",
    r"\bimproved (thesis|essay|version)\b",
    r"\byou should rewrite\b",
    r"\btry this thesis instead\b",
]

HALLUCINATION_PATTERNS = [
    r"\baccording to document\b",
    r"\bprimary source quote\b",
]


def parse_grade_json(text: str) -> tuple[dict | None, list[str]]:
    text = text.strip()
    reasons: list[str] = []
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return None, ["not_json_object"]
        return payload, reasons
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                reasons.append("json_extracted_from_prose")
                return payload, reasons
        except json.JSONDecodeError:
            pass
    return None, ["invalid_json"]


def feedback_references_essay(feedback: str, essay: str) -> bool:
    feedback_words = _content_words(feedback)
    essay_words = _content_words(essay)
    if not feedback_words or not essay_words:
        return False
    overlap = feedback_words & essay_words
    if len(overlap) >= 2:
        return True
    for chunk in _essay_chunks(essay):
        if chunk.lower() in feedback.lower():
            return True
    return any(word in essay.lower() for word in feedback_words if len(word) > 5)


def contains_rewrite_pattern(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in REWRITE_PATTERNS)


def contains_hallucination_pattern(text: str, essay: str) -> bool:
    lowered = text.lower()
    if re.search(r"\baccording to document\b", lowered):
        return True
    if re.search(r"\bprimary source quote\b", lowered):
        return True
    for quoted in re.findall(r"'([^']{12,})'", text):
        if quoted.lower() not in essay.lower():
            return True
    for quoted in re.findall(r'"([^"]{12,})"', text):
        if quoted.lower() not in essay.lower():
            return True
    return False


def passes_quality_gate(case: FRQCase) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    payload, parse_reasons = parse_grade_json(case.assistant_response)
    reasons.extend(parse_reasons)
    if payload is None:
        return False, reasons

    ok, validation_reasons = validate_grade_payload(payload)
    reasons.extend(validation_reasons)
    if not ok:
        return False, reasons

    feedback = payload["feedback"]
    for criterion, text in feedback.items():
        if not feedback_references_essay(text, case.student_response):
            reasons.append(f"feedback_not_grounded_{criterion}")

    if contains_rewrite_pattern(case.assistant_response):
        reasons.append("rewrites_essay")

    feedback_text = " ".join(str(value) for value in feedback.values())
    if contains_hallucination_pattern(feedback_text, case.student_response):
        reasons.append("hallucinated_quote")

    return not reasons, reasons


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']{4,}", text.lower())
    stop = {"this", "that", "with", "from", "have", "were", "their", "about", "which", "essay"}
    cleaned = {word.strip("'\"") for word in words}
    return {word for word in cleaned if word and word not in stop}


def _essay_chunks(essay: str) -> list[str]:
    sentences = re.split(r"[.!?]", essay)
    return [sentence.strip() for sentence in sentences if 12 <= len(sentence.strip()) <= 120]
