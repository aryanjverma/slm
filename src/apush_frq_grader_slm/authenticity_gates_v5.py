"""Deterministic authenticity hard gates for the v5 regeneration campaign.

Cloud authenticity readers may approve an essay, but these gates still reject
meta/process language, instruction leakage, style-reference over-copying, and
invalid length. Accepted regeneration candidates must pass every check.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from apush_frq_grader_slm.filters import contains_generation_leakage
from apush_frq_grader_slm.ingest.dedup import normalize_essay

# Categories observed in the first eight contaminated-corpus rejections / spotchecks.
PROHIBITED_ARTIFACT_CATEGORIES: tuple[str, ...] = (
    "memory_or_notes",
    "planning_or_draft_process",
    "physical_test_conditions",
    "knowledge_admission",
    "generator_mattering_stub",
    "prompt_or_instruction_leakage",
    "timing_theater_filler",
    "stock_classroom_filler",
)

ESSAY_ONLY_CONTRACT = {
    "return_only": "student_essay_text",
    "forbid_mention": sorted(
        {
            "memory",
            "notes",
            "planning",
            "drafts",
            "outlines",
            "checklists",
            "margins",
            "rewriting",
            "pens",
            "pencils",
            "erasers",
            "seating",
            "clocks",
            "bluebooks",
            "test conditions",
            "missing knowledge",
            "how the essay is being written",
        }
    ),
    "weak_knowledge_via": [
        "omission",
        "vagueness",
        "plausible_factual_mistakes",
        "underdeveloped_arguments",
    ],
    "forbid_knowledge_admissions": [
        "I cannot recall",
        "I can't remember",
        "I forget",
        "I do not know",
    ],
    "style_copy_limits": {
        "max_contiguous_normalized_words": 20,
        "max_sentences_with_eight_word_overlap": 1,
    },
}

_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "memory_or_notes": (
        re.compile(r"\b(?:class\s+)?notes?\b", re.I),
        re.compile(r"\bfrom memory\b", re.I),
        re.compile(r"\bmy memory\b", re.I),
        re.compile(r"\bin my notes\b", re.I),
        re.compile(r"\bcoming up in notes\b", re.I),
    ),
    "planning_or_draft_process": (
        re.compile(r"\b(?:outline|outlining|checklist|draft|drafting|rewrite|rewriting)\b", re.I),
        re.compile(r"\bwriting process\b", re.I),
        re.compile(r"\bas I (?:write|plan|draft)\b", re.I),
        re.compile(r"\bI am (?:writing|planning|drafting)\b", re.I),
        re.compile(r"\bmargins?\b", re.I),
    ),
    "physical_test_conditions": (
        re.compile(r"\b(?:bluebooks?|pencils?|erasers?|seating|clocks?)\b", re.I),
        re.compile(r"\b(?:test conditions?|timed test|exam room)\b", re.I),
        re.compile(r"(?<![A-Za-z])pens?(?![A-Za-z])", re.I),
    ),
    "knowledge_admission": (
        re.compile(r"\bI (?:cannot|can't|can not) recall\b", re.I),
        re.compile(r"\bI (?:don't|do not|cant|can't) remember\b", re.I),
        re.compile(r"\bI forget\b", re.I),
        re.compile(r"\bI (?:don't|do not) know (?:the|enough|much)\b", re.I),
        re.compile(r"\bmissing knowledge\b", re.I),
        re.compile(r"\bfuzzy in my (?:notes|memory)\b", re.I),
    ),
    "generator_mattering_stub": (
        re.compile(r"\bmattering\b", re.I),
    ),
    "prompt_or_instruction_leakage": (
        re.compile(r"\bthese generation instructions\b", re.I),
        re.compile(r"\bI was asked to write\b", re.I),
        re.compile(r"\btarget (?:total|score|rubric)\b", re.I),
        re.compile(r"\bstudent persona\b", re.I),
        re.compile(r"\bstyle reference\b", re.I),
        re.compile(r"\bfact cards?\b", re.I),
    ),
    "timing_theater_filler": (
        re.compile(r"Hard to finish in time", re.I),
        re.compile(r"lost a little time", re.I),
        re.compile(r"running out of time", re.I),
        re.compile(r"not enough time", re.I),
    ),
    "stock_classroom_filler": (
        re.compile(r"Teachers say", re.I),
        re.compile(r"classmates argued", re.I),
        re.compile(r"My outline was short", re.I),
        re.compile(r"worksheet", re.I),
    ),
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def detect_artifact_categories(text: str) -> list[str]:
    """Return prohibited artifact category ids present in ``text``."""
    found: list[str] = []
    for category in PROHIBITED_ARTIFACT_CATEGORIES:
        patterns = _CATEGORY_PATTERNS[category]
        if any(pattern.search(text) for pattern in patterns):
            found.append(category)
    if contains_generation_leakage(text) and "prompt_or_instruction_leakage" not in found:
        found.append("prompt_or_instruction_leakage")
    return found


def meta_process_gate_reasons(essay: str) -> list[str]:
    """Hard-reject essays that narrate the writing process or exam conditions."""
    categories = detect_artifact_categories(essay)
    return [f"meta_process:{category}" for category in categories]


def _normalized_words(text: str) -> list[str]:
    return normalize_essay(text).split()


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def max_contiguous_overlap_words(candidate: str, source: str) -> int:
    """Longest contiguous normalized-word overlap between candidate and source."""
    cand = _normalized_words(candidate)
    src = _normalized_words(source)
    if not cand or not src:
        return 0
    max_n = min(len(cand), len(src))
    for n in range(max_n, 0, -1):
        source_spans = {tuple(src[i : i + n]) for i in range(len(src) - n + 1)}
        for i in range(len(cand) - n + 1):
            if tuple(cand[i : i + n]) in source_spans:
                return n
    return 0


def _has_eight_gram(sentence: str, source_grams: set[tuple[str, ...]]) -> bool:
    words = _normalized_words(sentence)
    if len(words) < 8:
        return False
    for i in range(len(words) - 7):
        if tuple(words[i : i + 8]) in source_grams:
            return True
    return False


def style_copy_gate_reasons(
    essay: str,
    style_reference_essay: str,
    *,
    max_contiguous: int = 20,
    max_eight_gram_sentences: int = 1,
) -> list[str]:
    """Allow one short borrowed span from the matched golden essay; reject more."""
    if not style_reference_essay.strip():
        return []
    reasons: list[str] = []
    overlap = max_contiguous_overlap_words(essay, style_reference_essay)
    if overlap > max_contiguous:
        reasons.append("style_copy_contiguous_overlap_exceeded")
    source_words = _normalized_words(style_reference_essay)
    source_grams = {
        tuple(source_words[i : i + 8]) for i in range(max(0, len(source_words) - 7))
    }
    hit_sentences = sum(
        1 for sentence in _sentences(essay) if _has_eight_gram(sentence, source_grams)
    )
    if hit_sentences > max_eight_gram_sentences:
        reasons.append("style_copy_eight_gram_sentence_quota_exceeded")
    return reasons


def length_band_for_reference(reference_word_count: int) -> tuple[int, int]:
    """Per-row length band vs matched golden essay (~±20%, wider at extremes)."""
    ref = max(1, int(reference_word_count))
    if ref < 120:
        lo = max(70, int(round(ref * 0.70)))
        # Floor at 70 can exceed 1.4x for very short goldens; keep the band usable.
        hi = max(lo, int(round(ref * 1.40)))
        if hi - lo < 25:
            hi = lo + 25
    elif ref > 420:
        lo = int(round(ref * 0.75))
        hi = int(round(ref * 1.25))
    else:
        lo = int(round(ref * 0.80))
        hi = int(round(ref * 1.20))
    return lo, hi


def length_match_gate_reasons(
    essay: str,
    reference_word_count: int | None,
) -> list[str]:
    """Reject essays whose length diverges too far from the matched golden essay."""
    if reference_word_count is None or reference_word_count <= 0:
        return []
    words = len(essay.split())
    lo, hi = length_band_for_reference(int(reference_word_count))
    if words < lo or words > hi:
        return ["length_outside_matched_golden_band"]
    if words < 40:
        return ["essay_too_short"]
    return []


def hard_gate_reasons(
    essay: str,
    *,
    style_reference_essay: str = "",
    reference_word_count: int | None = None,
) -> list[str]:
    """Union of deterministic hard gates that override cloud-reader approval."""
    reasons: list[str] = []
    reasons.extend(meta_process_gate_reasons(essay))
    reasons.extend(
        style_copy_gate_reasons(essay, style_reference_essay)
    )
    reasons.extend(length_match_gate_reasons(essay, reference_word_count))
    if not essay.strip():
        reasons.append("empty_essay")
    return sorted(set(reasons))


def writer_instructions(
    *,
    has_boundary_behavior: bool,
) -> str:
    """Score-blind essay-only instructions shown to cloud writers."""
    boundary = (
        " Follow the content-level boundary behavior described under student_capability "
        "without naming rubric points or scores."
        if has_boundary_behavior
        else ""
    )
    return (
        "Write one authentic timed APUSH student LEQ response to the prompt. "
        "Use the full style_reference_essay only as a tone/length/mechanics reference; "
        "do not copy more than one short span (at most 20 contiguous words; at most one "
        "sentence may share an eight-word sequence). Paraphrase semantic_fact_cards; "
        "never quote them. Show weak knowledge through omission, vagueness, plausible "
        "mistakes, or thin arguments—never by saying you cannot recall or describing "
        "notes, outlines, drafts, pens, clocks, bluebooks, or the writing process. "
        "Return only the essay text with no preamble or commentary."
        + boundary
    )


def aggregate_artifact_audit(
    essays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate contamination rates without retaining private essay text."""
    total = 0
    contaminated = 0
    category_counts: dict[str, int] = {key: 0 for key in PROHIBITED_ARTIFACT_CATEGORIES}
    for row in essays:
        essay = str(row.get("student_response") or row.get("essay") or "")
        if not essay.strip():
            continue
        total += 1
        cats = detect_artifact_categories(essay)
        if cats:
            contaminated += 1
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
    rate = (contaminated / total) if total else 0.0
    return {
        "campaign": "v5_r1_authenticity_failure",
        "essays_scanned": total,
        "contaminated_essays": contaminated,
        "contamination_rate": round(rate, 4),
        "artifact_category_counts": category_counts,
        "artifact_category_rates": {
            key: round((category_counts[key] / total) if total else 0.0, 4)
            for key in PROHIBITED_ARTIFACT_CATEGORIES
        },
        "why_previous_authenticity_judge_failed": (
            "Feature-based authenticity readers rewarded informal/timed surface cues and "
            "missed systematic generator meta/process language (notes, outlines, memory "
            "admissions, timing theater, mattering stubs). Deterministic hard gates for "
            "those categories were absent, so contaminated essays passed into selection."
        ),
        "private_essay_text_retained": False,
        "redistribution_authorized": False,
    }


def quartile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def aggregate_length_realism_audit(
    candidate_word_counts: Sequence[int],
    golden_word_counts: Sequence[int],
) -> dict[str, Any]:
    """Mean within 10%; median and quartiles within 15% of golden length stats."""
    if not candidate_word_counts or not golden_word_counts:
        return {"passed": False, "reason": "empty_inputs"}
    c = [float(x) for x in candidate_word_counts]
    g = [float(x) for x in golden_word_counts]
    c_mean = sum(c) / len(c)
    g_mean = sum(g) / len(g)
    metrics: dict[str, Any] = {}
    passed = True
    mean_delta = abs(c_mean - g_mean)
    mean_allowed = abs(g_mean) * 0.10
    mean_ok = mean_delta <= max(mean_allowed, 1.0)
    passed = passed and mean_ok
    metrics["mean"] = {
        "candidate": round(c_mean, 4),
        "golden": round(g_mean, 4),
        "allowed_delta": round(max(mean_allowed, 1.0), 4),
        "passed": mean_ok,
    }
    for name, q in (("median", 0.5), ("q1", 0.25), ("q3", 0.75)):
        cv = quartile(c, q)
        gv = quartile(g, q)
        allowed = max(abs(gv) * 0.15, 1.0)
        ok = abs(cv - gv) <= allowed
        passed = passed and ok
        metrics[name] = {
            "candidate": round(cv, 4),
            "golden": round(gv, 4),
            "allowed_delta": round(allowed, 4),
            "passed": ok,
        }
    return {"passed": passed, "metrics": metrics, "private_essay_text_retained": False}
