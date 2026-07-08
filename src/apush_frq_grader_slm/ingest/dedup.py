"""Deduplicate ingested essays against existing eval cases."""

from __future__ import annotations

import re

from apush_frq_grader_slm.schemas import FRQCase


def normalize_essay(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def essay_fingerprint(text: str) -> set[str]:
    words = normalize_essay(text).split()
    return {word for word in words if len(word) >= 4}


def is_duplicate_essay(
    essay: str,
    existing: list[FRQCase],
    *,
    jaccard_threshold: float = 0.82,
    prompt: str | None = None,
) -> bool:
    """Return True if essay substantially overlaps an existing case."""
    if len(normalize_essay(essay)) < 80:
        return True
    essay_fp = essay_fingerprint(essay)
    if not essay_fp:
        return True
    norm_prompt = normalize_essay(prompt or "")
    for case in existing:
        if norm_prompt and normalize_essay(case.prompt) == norm_prompt:
            overlap = essay_fp & essay_fingerprint(case.student_response)
            union = essay_fp | essay_fingerprint(case.student_response)
            if union and len(overlap) / len(union) >= jaccard_threshold:
                return True
        else:
            overlap = essay_fp & essay_fingerprint(case.student_response)
            if len(overlap) >= min(40, len(essay_fp) * 0.75):
                return True
    return False
