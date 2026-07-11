"""College Board golden-set seed profiles for v4 synthetic generation.

Real CB essays never enter training text. This module emits structural /
score / style metadata plus short cleaned style excerpts (writer-only
references). Full cleaned essays are not written to any training artifact.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.filters import SOURCE_CONTAMINATION_PATTERNS
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.rubric import compute_total
from apush_frq_grader_slm.schemas import FailureType, RubricScores

# Known parse failures in eval_cb_cases.jsonl → exact CB prompt text.
KNOWN_PROMPT_REPAIRS: dict[str, str] = {
    "explain how changes in debates over the role of the • Explaining the nuance "
    "of an issue by analyzing multiple variables.": (
        "Evaluate the extent to which changes in debates over the role of the "
        "federal government contributed to the growth of political parties from "
        "1800 to 1854."
    ),
}

# Extra markers beyond filters.SOURCE_CONTAMINATION_PATTERNS for essay cleaning.
_CLEANING_EXTRA_PATTERNS: tuple[str, ...] = (
    r"(?:history\s+)?\d{4}\s+scoring\s+commentary",
    r"\blong\s+essay\s+question\s+\d+(?:\s*\(continued\))?",
    r"\bap(?:®|\s+)?\s*united\s+states\s+history\s+20\d{2}\b",
    r"\bsample:\s*\d+[a-c]\b",
    r"\bthe\s+response\s+(?:earned|did\s+not\s+earn)\b",
    r"\bstudent\s+samples\s+are\s+quoted\s+verbatim\b",
    r"visit\s+college\s+board\s+on\s+the\s+web",
    r"\bapcentral\.collegeboard\.org\b",
    r"\bcollegeboard\.org\b",
    r"(?:^|\n)\s*[a-d]\.\s*(?:thesis(?:/claim)?|contextualization|evidence|"
    r"analysis\s+and\s+reasoning)\b",
    r"\bearned\s+(?:0|1|2|one|two)\s+points?\b",
    r"\bof\s+significance:\s*",
)

_STYLE_MIN_WORDS = 80
_STYLE_EXCERPT_CHARS = 400
_LENGTH_BAND_PCT = 0.15

# APUSH period boundaries (inclusive start, exclusive end except period 9).
_PERIOD_BOUNDS: tuple[tuple[int, int, int], ...] = (
    (1, 1491, 1607),
    (2, 1607, 1754),
    (3, 1754, 1800),
    (4, 1800, 1848),
    (5, 1844, 1877),
    (6, 1865, 1898),
    (7, 1890, 1945),
    (8, 1945, 1980),
    (9, 1980, 2100),
)

_YEAR_RANGE_RE = re.compile(
    r"\b((?:1[4-9]|20)\d{2})\s*(?:to|through|–|-|—)\s*((?:1[4-9]|20)\d{2})\b",
    re.IGNORECASE,
)
_SINGLE_YEAR_RE = re.compile(r"\b((?:1[4-9]|20)\d{2})\b")

_THESIS_CUES = re.compile(
    r"\b(?:although|while|whereas|despite|to\s+(?:a\s+)?(?:large|great|some|limited)"
    r"\s+extent|the\s+(?:most\s+)?important|argues?\s+that|claim(?:s|ed)?\s+that)\b",
    re.IGNORECASE,
)
_CONTEXT_CUES = re.compile(
    r"\b(?:before|prior\s+to|leading\s+up|earlier|previously|in\s+the\s+years\s+before|"
    r"context|background|preceded)\b",
    re.IGNORECASE,
)
_COMPLEXITY_CUES = re.compile(
    r"\b(?:although|however|nevertheless|on\s+the\s+other\s+hand|while|whereas|"
    r"despite|conversely|nuance|qualify|qualification)\b",
    re.IGNORECASE,
)
_EVIDENCE_NAME_RE = re.compile(
    r"\b(?:Act|Treaty|Doctrine|War|Compromise|Amendment|Proclamation|Movement|"
    r"Revolution|Bank|Tariff|Purchase|Colony|Party|Deal|Plan|Code|Laws?)\b"
)

_STOPWORDS = frozenset(
    "the a an of to in for and or on from with that which how evaluate extent "
    "relative importance causes effects growth changes change united states "
    "america american north between during period".split()
)


def _contamination_cut_regex() -> re.Pattern[str]:
    parts = list(SOURCE_CONTAMINATION_PATTERNS) + list(_CLEANING_EXTRA_PATTERNS)
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


_CONTAMINATION_CUT_RE = _contamination_cut_regex()


def clean_student_response(text: str) -> str:
    """Strip scoring commentary, page headers, copyright, and score lines.

    Keeps only the student prose before the first contamination marker.
    """
    if not text or not text.strip():
        return ""
    match = _CONTAMINATION_CUT_RE.search(text)
    cleaned = text[: match.start()] if match else text
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Trailing orphan row labels left by PDF joins (" ... England. C.")
    cleaned = re.sub(r"(?:\s+[A-D]\.)+\s*$", "", cleaned).strip()
    # Drop leading sample letter crumbs ("B. the arrival...")
    cleaned = re.sub(r"^[A-D]\.\s+", "", cleaned).strip()
    return cleaned


def repair_prompt(prompt: str) -> str:
    """Return exact CB prompt, repairing known ingestion corruptions."""
    stripped = prompt.strip()
    return KNOWN_PROMPT_REPAIRS.get(stripped, stripped)


def make_seed_id(case_id: str, tags: Sequence[str]) -> str:
    """Build seed_id like cb-2023-leq2-set1-sampleA from case id/tags."""
    year = _tag_year(tags)
    leq = _tag_leq(tags)
    set_num = _parse_set_number(case_id, tags)
    sample = _parse_sample_letter(case_id)
    parts = ["cb"]
    if year is not None:
        parts.append(str(year))
    if leq is not None:
        parts.append(f"leq{leq}")
    if set_num is not None:
        parts.append(f"set{set_num}")
    if sample:
        parts.append(f"sample{sample}")
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", case_id.lower()).strip("-")
        parts.append(slug)
    return "-".join(parts)


def prompt_family_id(prompt: str) -> str:
    """Stable slug from prompt wording + date range."""
    years = _YEAR_RANGE_RE.search(prompt)
    year_suffix = f"_{years.group(1)}_{years.group(2)}" if years else ""
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9]+", prompt.lower())
        if tok not in _STOPWORDS and not tok.isdigit() and len(tok) > 2
    ]
    # Prefer distinctive content words; keep order, dedupe.
    seen: set[str] = set()
    key_tokens: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        key_tokens.append(tok)
        if len(key_tokens) >= 6:
            break
    if not key_tokens:
        key_tokens = ["prompt"]
    return "cb_" + "_".join(key_tokens) + year_suffix


def infer_period(prompt: str) -> int | None:
    """Infer APUSH period 1–9 from the prompt date range midpoint/overlap."""
    match = _YEAR_RANGE_RE.search(prompt)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            start, end = end, start
        return _period_for_range(start, end)
    years = [int(y) for y in _SINGLE_YEAR_RE.findall(prompt)]
    if years:
        mid = sum(years) // len(years)
        return _period_for_year(mid)
    return None


def infer_reasoning_skill(prompt: str) -> str:
    """Infer causation | comparison | ccot | relative_importance | extent."""
    lowered = prompt.lower()
    if "relative importance" in lowered:
        return "relative_importance"
    if re.search(r"\bcompar(?:e|ing|ison)\b|\bsimilarit|\bdifference", lowered):
        return "comparison"
    if re.search(
        r"continuity\s+and\s+change|\bccot\b|\bcontinuities\b|"
        r"\badapt(?:ed|ation|ing)\b",
        lowered,
    ):
        return "ccot"
    if "extent to which" in lowered or re.search(r"\bevaluate the extent\b", lowered):
        return "extent"
    if re.search(
        r"\bcauses?\b|\bcaused\b|\bcontributed\b|\bresponded\b|\bshaped\b|\binfluenced\b",
        lowered,
    ):
        return "causation"
    return "causation"


def infer_failure_type(scores: Mapping[str, int] | RubricScores) -> str:
    """Map rubric score pattern to a FailureType-aligned slice label."""
    if isinstance(scores, RubricScores):
        thesis = scores.thesis
        contextualization = scores.contextualization
        evidence = scores.evidence
        analysis = scores.analysis_reasoning
        total = scores.total
    else:
        thesis = int(scores["thesis"])
        contextualization = int(scores["contextualization"])
        evidence = int(scores["evidence"])
        analysis = int(scores["analysis_reasoning"])
        total = int(
            scores.get(
                "total",
                compute_total(
                    RubricScores(
                        thesis=thesis,
                        contextualization=contextualization,
                        evidence=evidence,
                        analysis_reasoning=analysis,
                    )
                ),
            )
        )

    if thesis == 0:
        return FailureType.WEAK_THESIS.value
    if contextualization == 0:
        return FailureType.MISSING_CONTEXT.value
    if evidence <= 1:
        return FailureType.EVIDENCE_LIST.value
    if total >= 5:
        return FailureType.STRONG.value
    if analysis <= 1:
        return FailureType.BORDERLINE_COMPLEXITY.value
    return FailureType.BORDERLINE_COMPLEXITY.value


def structural_notes(cleaned_essay: str, scores: Mapping[str, int]) -> dict[str, Any]:
    """Brief structural heuristics for writer conditioning (not full essay)."""
    paragraphs = _paragraphs(cleaned_essay)
    word_count = len(cleaned_essay.split()) if cleaned_essay else 0
    intro = paragraphs[0] if paragraphs else cleaned_essay
    evidence_hits = len(_EVIDENCE_NAME_RE.findall(cleaned_essay))
    # Rough distinct evidence mentions: named entities + capitalized multiword cues.
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", cleaned_essay)
    evidence_count_estimate = max(evidence_hits, min(len(set(proper)), 6))
    if word_count < 40:
        evidence_count_estimate = min(evidence_count_estimate, 1)
    return {
        "has_intro_thesis": bool(_THESIS_CUES.search(intro)) or int(scores.get("thesis", 0)) == 1,
        "has_context_block": bool(_CONTEXT_CUES.search(cleaned_essay[:500])),
        "evidence_count_estimate": int(evidence_count_estimate),
        "has_complexity_attempt": bool(_COMPLEXITY_CUES.search(cleaned_essay))
        or int(scores.get("analysis_reasoning", 0)) >= 1,
        "paragraph_count": max(1, len(paragraphs)) if cleaned_essay else 0,
    }


def length_band(word_count: int, pct: float = _LENGTH_BAND_PCT) -> tuple[int, int]:
    """Inclusive (lo, hi) band around word_count ± pct."""
    if word_count <= 0:
        return (0, 0)
    lo = max(1, int(math.floor(word_count * (1.0 - pct))))
    hi = int(math.ceil(word_count * (1.0 + pct)))
    return (lo, hi)


def style_excerpt(cleaned_essay: str) -> tuple[str, bool]:
    """First ~400 chars of cleaned essay; empty if under min word threshold."""
    words = cleaned_essay.split()
    if len(words) < _STYLE_MIN_WORDS:
        return "", False
    excerpt = cleaned_essay[:_STYLE_EXCERPT_CHARS].rstrip()
    if len(cleaned_essay) > _STYLE_EXCERPT_CHARS:
        # Prefer breaking on a word boundary.
        if " " in excerpt:
            excerpt = excerpt.rsplit(" ", 1)[0]
        excerpt = excerpt.rstrip(" ,;:") + "…"
    return excerpt, True


def build_seed_profile(case: Mapping[str, Any]) -> dict[str, Any]:
    """Emit one CB seed profile dict from an eval_cb_cases row."""
    case_id = str(case["id"])
    tags = list(case.get("tags") or [])
    prompt = repair_prompt(str(case["prompt"]))
    scores_raw = case["reference_scores"]
    scores = {
        "thesis": int(scores_raw["thesis"]),
        "contextualization": int(scores_raw["contextualization"]),
        "evidence": int(scores_raw["evidence"]),
        "analysis_reasoning": int(scores_raw["analysis_reasoning"]),
    }
    total = compute_total(RubricScores(**scores))

    cleaned = clean_student_response(str(case.get("student_response") or ""))
    word_count = len(cleaned.split()) if cleaned else 0
    excerpt, usable = style_excerpt(cleaned)

    return {
        "seed_id": make_seed_id(case_id, tags),
        "source_case_id": case_id,
        "prompt": prompt,
        "prompt_family_id": prompt_family_id(prompt),
        "year": _tag_year(tags),
        "leq_number": _tag_leq(tags),
        "set_number": _parse_set_number(case_id, tags),
        "sample_letter": _parse_sample_letter(case_id),
        "reference_scores": scores,
        "total": total,
        "word_count": word_count,
        "length_band": list(length_band(word_count)),
        "structural_notes": structural_notes(cleaned, scores),
        "style_excerpt": excerpt,
        "style_usable": usable,
        "period": infer_period(prompt),
        "reasoning_skill": infer_reasoning_skill(prompt),
        "failure_type": infer_failure_type(scores),
        "tags": tags,
    }


def build_seed_profiles(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_seed_profile(case) for case in cases]


def build_prompt_families(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Unique prompts with period, reasoning_skill, and member seed_ids."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for profile in profiles:
        family_id = str(profile["prompt_family_id"])
        if family_id not in grouped:
            order.append(family_id)
            grouped[family_id] = {
                "prompt_family_id": family_id,
                "prompt": profile["prompt"],
                "period": profile.get("period"),
                "reasoning_skill": profile.get("reasoning_skill"),
                "seed_ids": [],
            }
        grouped[family_id]["seed_ids"].append(profile["seed_id"])
    return [grouped[key] for key in order]


def summarize_seed_profiles(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter(int(p["total"]) for p in profiles)
    periods = Counter(
        str(p["period"]) if p.get("period") is not None else "unknown" for p in profiles
    )
    skills = Counter(str(p.get("reasoning_skill") or "unknown") for p in profiles)
    style = Counter(bool(p.get("style_usable")) for p in profiles)
    failures = Counter(str(p.get("failure_type") or "unknown") for p in profiles)
    return {
        "n_profiles": len(profiles),
        "n_prompt_families": len({p["prompt_family_id"] for p in profiles}),
        "by_total": {str(k): totals[k] for k in sorted(totals)},
        "by_period": dict(sorted(periods.items(), key=lambda kv: kv[0])),
        "by_reasoning_skill": dict(sorted(skills.items())),
        "by_failure_type": dict(sorted(failures.items())),
        "style_usable": {"true": style.get(True, 0), "false": style.get(False, 0)},
        "seed_ids": [p["seed_id"] for p in profiles],
    }


def load_cb_cases(path: Path | str) -> list[dict[str, Any]]:
    return read_jsonl(Path(path))


def write_v4_seed_artifacts(
    *,
    cases_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Load CB cases, write seed profiles, summary, and prompt families."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cases = load_cb_cases(cases_path)
    profiles = build_seed_profiles(cases)
    families = build_prompt_families(profiles)
    summary = summarize_seed_profiles(profiles)

    profiles_path = output / "cb_seed_profiles.jsonl"
    summary_path = output / "cb_seed_profiles_summary.json"
    families_path = output / "prompt_families_v4.jsonl"

    write_jsonl(profiles_path, profiles)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(families_path, families)

    return {
        "n_profiles": len(profiles),
        "n_prompt_families": len(families),
        "profiles_path": str(profiles_path),
        "summary_path": str(summary_path),
        "families_path": str(families_path),
        "summary": summary,
    }


def _tag_year(tags: Sequence[str]) -> int | None:
    for tag in tags:
        if re.fullmatch(r"20\d{2}", tag):
            return int(tag)
    return None


def _tag_leq(tags: Sequence[str]) -> int | None:
    for tag in tags:
        match = re.fullmatch(r"leq(\d+)", tag, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _parse_set_number(case_id: str, tags: Sequence[str]) -> int | None:
    for tag in tags:
        match = re.fullmatch(r"set(\d+)", tag, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    match = re.search(r"set[_-]?(\d+)", case_id, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_sample_letter(case_id: str) -> str:
    match = re.search(r"(\d)([A-Ca-c])$", case_id)
    if match:
        return match.group(2).upper()
    match = re.search(r"sample[_-]?([A-Ca-c])", case_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def _period_for_year(year: int) -> int:
    for period, start, end in _PERIOD_BOUNDS:
        if start <= year < end or (period == 9 and year >= start):
            return period
    if year < 1491:
        return 1
    return 9


def _period_for_range(start: int, end: int) -> int:
    """Choose the period with maximum overlap; ties → midpoint period."""
    best_period = _period_for_year((start + end) // 2)
    best_overlap = -1
    for period, p_start, p_end in _PERIOD_BOUNDS:
        overlap_start = max(start, p_start)
        overlap_end = min(end, p_end if period != 9 else end + 1)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_period = period
    return best_period


def _paragraphs(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) > 1:
        return parts
    # Many APC extractions are single-line; approximate paragraphs by sentence groups.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 3:
        return [" ".join(sentences)] if sentences else [text.strip()]
    groups: list[str] = []
    for i in range(0, len(sentences), 3):
        groups.append(" ".join(sentences[i : i + 3]))
    return groups
