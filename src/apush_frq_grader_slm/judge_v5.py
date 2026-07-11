"""Feature-based blind judging for v5 external candidate records.

Readers see only the prompt, essay, and corrected v5 rubric rules encoded here.
They never consume personas, seed ids, style excerpts, coverage class, or
boundary metadata — the validator restores planner fields later.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from random import Random
from typing import Any

from apush_frq_grader_slm.filters import contains_generation_leakage
from apush_frq_grader_slm.grade_v4 import grounded_feedback_for_scores
from apush_frq_grader_slm.prompts_v5 import V5_RUBRIC_TEXT
from apush_frq_grader_slm.rubric import CRITERIA

# Rubric text is the sole scoring contract for blind readers (imported for callers/tests).
_ = V5_RUBRIC_TEXT

AUTH_REVIEWER_IDS = ("auth-a", "auth-b", "auth-c")
RUBRIC_READER_IDS = ("reader-a", "reader-b", "reader-c")
FACT_CHECKER_ID = "facts-a"
CONFIDENCE_ADJUDICATION_FLOOR = 0.85

# Timed APUSH LEQ student length band (words).
_STUDENT_LENGTH_LO = 90
_STUDENT_LENGTH_HI = 520

_THESIS_CUES = re.compile(
    r"\b(?:because|although|despite|whereas|while|however|therefore|thus|"
    r"to (?:a |some )?(?:great|large|limited|significant)?\s*extent|"
    r"primarily|mainly|led to|caused|resulted in|due to|as a result|"
    r"argue[sd]?|claim[sd]?|contends?|this (?:shows|demonstrates|proves))\b",
    re.I,
)
_CONTEXT_CUES = re.compile(
    r"\b(?:prior to|before|earlier|previously|by the time|in the (?:years|decades) "
    r"(?:before|leading)|following the|in the (?:wake|aftermath) of|origins?|"
    r"roots?|background|setting the stage|leading up to|coming out of|"
    r"building on|inherited|long[- ]standing)\b",
    re.I,
)
_CAUSATION = re.compile(
    r"\b(?:caused|led to|resulting in|resulted in|because of this|as a result|"
    r"consequently|thereby|sparked|triggered|fueled)\b",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(?:unlike|whereas|similarly|in contrast|compared (?:to|with)|"
    r"on the other hand|both .+ and|differ(?:ed|ence|ent))\b",
    re.I,
)
_CCOT = re.compile(
    r"\b(?:continued|continuity|changed|change over time|remained|transformation|"
    r"shifted|evolved|persisted|broke from|departure from)\b",
    re.I,
)
_COMPLEXITY = re.compile(
    r"\b(?:however|although|while also|nuance|qualified|qualification|"
    r"multiple perspectives|on the other hand|at the same time|not only|"
    r"even as|despite this|complex)\b",
    re.I,
)
_INFORMAL = re.compile(
    r"\b(?:wasnt|didnt|couldnt|wouldnt|dont|cant|wont|alot|goverment|"
    r"kinda|gonna|wanna|idk|tho|bc)\b|(?<![A-Za-z])i(?![A-Za-z])",
    re.I,
)
_YEAR = re.compile(r"\b((?:1[6-9]|20)\d{2})\b")
_PROPER_SPAN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_DATE_RANGE = re.compile(
    r"(?:from|between|during|in|over)?\s*((?:1[6-9]|20)\d{2})\s*"
    r"(?:to|[-–—]|and|through)\s*((?:1[6-9]|20)\d{2})",
    re.I,
)
_NONSENSE = re.compile(r"[aeiou]", re.I)
# Generator stubs / pad clusters from compose_v5 (not natural timed student prose).
_MATTERING_STUB = re.compile(r"\bmattering\b", re.I)
_FILLER_HARD_TO_FINISH = re.compile(r"Hard to finish in time", re.I)
_FILLER_TEACHERS_SAY = re.compile(r"Teachers say", re.I)
_FILLER_CLASSMATES_ARGUED = re.compile(r"classmates argued", re.I)
_GENERATOR_FILLER_MARKERS = (
    _FILLER_HARD_TO_FINISH,
    _FILLER_TEACHERS_SAY,
    _FILLER_CLASSMATES_ARGUED,
    re.compile(r"My outline was short", re.I),
    re.compile(r"lost a little time", re.I),
)
_REPEATED_SPAN_MIN_WORDS = 10
_REPEATED_SPAN_MIN_COUNT = 3
_DENSE_FILLER_WORD_CAP = 360


@dataclass(frozen=True)
class EssayFeatures:
    word_count: int
    paragraph_count: int
    sentence_count: int
    thesis_signal: float
    context_signal: float
    evidence_count: int
    evidence_used: bool
    analysis_signal: float
    complexity_signal: float
    informal_count: int
    sentence_length_std: float
    has_instruction_leakage: bool
    unique_ratio: float
    years: tuple[int, ...]
    has_mattering_stub: bool
    has_repeated_long_span: bool
    has_dense_generator_filler: bool


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _has_repeated_long_span(
    text: str,
    *,
    min_words: int = _REPEATED_SPAN_MIN_WORDS,
    min_count: int = _REPEATED_SPAN_MIN_COUNT,
) -> bool:
    """True when the same 10+ word span appears 3+ times (prompt-clause paste)."""
    words = [w.lower() for w in text.split()]
    if len(words) < min_words * min_count:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - min_words + 1):
        gram = tuple(words[i : i + min_words])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= min_count:
            return True
    return False


def _has_dense_generator_filler(text: str, word_count: int) -> bool:
    """Fail short essays that pack multiple known compose_v5 pad phrases."""
    hard = bool(_FILLER_HARD_TO_FINISH.search(text))
    teachers = bool(_FILLER_TEACHERS_SAY.search(text))
    classmates = bool(_FILLER_CLASSMATES_ARGUED.search(text))
    # Classic generator cluster in one short essay.
    if hard and teachers and classmates and word_count <= _DENSE_FILLER_WORD_CAP:
        return True
    hits = sum(1 for pat in _GENERATOR_FILLER_MARKERS if pat.search(text))
    return hits >= 3 and word_count <= _DENSE_FILLER_WORD_CAP


def _prompt_date_range(prompt: str) -> tuple[int, int] | None:
    match = _DATE_RANGE.search(prompt)
    if not match:
        years = [int(y) for y in _YEAR.findall(prompt)]
        if len(years) >= 2:
            return min(years), max(years)
        if len(years) == 1:
            return years[0], years[0]
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return (start, end) if start <= end else (end, start)


def extract_essay_features(prompt: str, essay: str) -> EssayFeatures:
    """Detect rubric-relevant cues from essay text only."""
    text = essay.strip()
    words = text.split()
    sentences = _sentences(text)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()] or ([text] if text else [])
    sentence_lengths = [len(s.split()) for s in sentences] or [0]
    length_std = float(statistics.pstdev(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0

    bookends = " ".join((sentences[:2] + sentences[-2:]) if sentences else [])
    thesis_hits = len(_THESIS_CUES.findall(bookends))
    prompt_tokens = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", prompt)}
    bookend_tokens = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", bookends)}
    overlap = len(prompt_tokens & bookend_tokens) / max(len(prompt_tokens), 1)
    # Restatement-only openings look like high prompt overlap with weak claim cues.
    thesis_signal = min(1.0, 0.15 * thesis_hits + (0.55 if thesis_hits else 0.0))
    if overlap > 0.55 and thesis_hits == 0:
        thesis_signal = min(thesis_signal, 0.2)
    elif thesis_hits and len(bookends.split()) >= 12:
        thesis_signal = min(1.0, thesis_signal + 0.25)

    context_hits = len(_CONTEXT_CUES.findall(text))
    early = " ".join(sentences[:3]) if sentences else ""
    early_context = len(_CONTEXT_CUES.findall(early))
    context_signal = min(1.0, 0.35 * context_hits + 0.4 * early_context)

    years = tuple(int(y) for y in _YEAR.findall(text))
    proper = [
        span
        for span in _PROPER_SPAN.findall(text)
        if span.lower() not in {"the", "this", "that", "united states", "american", "america"}
    ]
    # Specific evidence: distinct years + named examples.
    evidence_items = set(years) | {p.lower() for p in proper if len(p.split()) >= 1 and len(p) > 3}
    evidence_count = min(6, len(evidence_items))
    support_cues = len(
        re.findall(
            r"\b(?:this (?:shows|proves|supports|demonstrates)|for example|such as|"
            r"which (?:helped|allowed|forced|showed))\b",
            text,
            flags=re.I,
        )
    )
    evidence_used = evidence_count >= 2 and support_cues >= 1

    causation = len(_CAUSATION.findall(text))
    comparison = len(_COMPARISON.findall(text))
    ccot = len(_CCOT.findall(text))
    complexity = len(_COMPLEXITY.findall(text))
    structure_hits = sum(1 for count in (causation, comparison, ccot) if count > 0)
    analysis_signal = min(2.0, 0.7 * structure_hits + 0.25 * (causation + comparison + ccot))
    complexity_signal = min(1.0, 0.35 * complexity + (0.4 if evidence_count >= 4 else 0.0))

    unique = {w.lower() for w in words}
    unique_ratio = len(unique) / max(len(words), 1)
    word_count = len(words)
    return EssayFeatures(
        word_count=word_count,
        paragraph_count=len(paragraphs),
        sentence_count=len(sentences),
        thesis_signal=thesis_signal,
        context_signal=context_signal,
        evidence_count=evidence_count,
        evidence_used=evidence_used,
        analysis_signal=analysis_signal,
        complexity_signal=complexity_signal,
        informal_count=len(_INFORMAL.findall(text)),
        sentence_length_std=length_std,
        has_instruction_leakage=contains_generation_leakage(text),
        unique_ratio=unique_ratio,
        years=years,
        has_mattering_stub=bool(_MATTERING_STUB.search(text)),
        has_repeated_long_span=_has_repeated_long_span(text),
        has_dense_generator_filler=_has_dense_generator_filler(text, word_count),
    )


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def _reader_scores(features: EssayFeatures, reader_id: str, rng: Random) -> dict[str, int]:
    """Map features to criterion scores with reader-specific thresholds + light noise."""
    # Distinct heuristics so readers are not identical clones.
    profiles = {
        "reader-a": {"thesis": 0.45, "context": 0.40, "evidence": 2.2, "analysis": 0.85, "complex": 0.45},
        "reader-b": {"thesis": 0.55, "context": 0.55, "evidence": 1.8, "analysis": 0.70, "complex": 0.35},
        "reader-c": {"thesis": 0.40, "context": 0.35, "evidence": 2.0, "analysis": 1.00, "complex": 0.55},
    }
    profile = profiles[reader_id]
    noise = lambda scale=0.08: rng.uniform(-scale, scale)  # noqa: E731

    thesis_raw = features.thesis_signal + noise()
    if features.word_count < 25:
        thesis_raw = 0.0
    thesis = 1 if thesis_raw >= profile["thesis"] else 0

    context_raw = features.context_signal + noise()
    if features.word_count < 40:
        context_raw *= 0.5
    contextualization = 1 if context_raw >= profile["context"] else 0

    evidence_raw = features.evidence_count + noise(0.35)
    if features.word_count < 30:
        evidence_raw = 0.0
    if evidence_raw < profile["evidence"] * 0.45:
        evidence = 0
    elif features.evidence_used and evidence_raw >= profile["evidence"]:
        evidence = 2
    elif evidence_raw >= 1.5:
        evidence = 1
    else:
        evidence = 0

    analysis_raw = features.analysis_signal + noise(0.2)
    complexity_raw = features.complexity_signal + noise(0.15)
    if analysis_raw < profile["analysis"] * 0.55 or features.word_count < 40:
        analysis = 0
    elif complexity_raw >= profile["complex"] and analysis_raw >= profile["analysis"]:
        analysis = 2
    else:
        analysis = 1 if analysis_raw >= profile["analysis"] * 0.75 else 0

    return {
        "thesis": thesis,
        "contextualization": contextualization,
        "evidence": evidence,
        "analysis_reasoning": analysis,
    }


def _reader_confidence(features: EssayFeatures, scores: dict[str, int], rng: Random) -> float:
    """Lower confidence near decision boundaries; higher for clear signals."""
    margins: list[float] = []
    margins.append(abs(features.thesis_signal - 0.5))
    margins.append(abs(features.context_signal - 0.45))
    if scores["evidence"] == 0:
        margins.append(min(1.0, abs(features.evidence_count - 1.0) / 2))
    elif scores["evidence"] == 1:
        margins.append(0.25 if features.evidence_used else 0.45)
    else:
        margins.append(0.55 if features.evidence_used else 0.2)
    margins.append(abs(features.analysis_signal - 0.9) / 2)
    if features.word_count < 40:
        base = 0.55
    else:
        base = 0.72 + 0.35 * (sum(margins) / len(margins))
    base += rng.uniform(-0.04, 0.04)
    return round(max(0.35, min(0.98, base)), 3)


def _authenticity_review(features: EssayFeatures, reviewer_id: str, rng: Random) -> dict[str, Any]:
    """Judge student-like + timed-AP consistency from surface writing cues."""
    # Slight per-reviewer threshold shifts.
    shifts = {"auth-a": -0.05, "auth-b": 0.05, "auth-c": 0.0}
    shift = shifts[reviewer_id] + rng.uniform(-0.03, 0.03)

    polished = (
        features.informal_count == 0
        and features.sentence_length_std < 4.0
        and features.paragraph_count >= 3
        and features.word_count >= 220
        and features.unique_ratio > 0.62
    )
    student_like = (not features.has_instruction_leakage) and (not polished or features.informal_count > 0)
    # auth-c is slightly more willing to call polished-but-uneven essays student-like.
    if reviewer_id == "auth-c" and not features.has_instruction_leakage and features.sentence_length_std >= 3.0:
        student_like = True
    if features.word_count < 15:
        student_like = False

    length_ok = _STUDENT_LENGTH_LO <= features.word_count <= _STUDENT_LENGTH_HI
    uneven = features.paragraph_count <= 2 or features.sentence_length_std >= (5.0 + shift * 10)
    informal = features.informal_count >= max(0, 1 + int(shift > 0.04))
    timed = bool(length_ok and (informal or uneven or features.word_count <= 360))
    if features.word_count < 20:
        timed = False
    if features.has_instruction_leakage:
        timed = False

    # Generator artifacts override reviewer leniency (still allow natural informal timed prose).
    generator_artifact = (
        features.has_mattering_stub
        or features.has_repeated_long_span
        or features.has_dense_generator_filler
    )
    if generator_artifact:
        student_like = False

    return {
        "reviewer_id": reviewer_id,
        "student_like": bool(student_like),
        "timed_ap_consistent": bool(timed),
    }


def _auth_pass(review: dict[str, Any]) -> bool:
    return bool(review.get("student_like")) and bool(review.get("timed_ap_consistent"))


def _resolve_scores(reviews: list[dict[str, Any]]) -> tuple[dict[str, int], bool]:
    score_rows = [dict(r["scores"]) for r in reviews]
    signatures = {tuple(row[c] for c in CRITERIA) for row in score_rows}
    low_confidence = any(float(r["confidence"]) < CONFIDENCE_ADJUDICATION_FLOOR for r in reviews)
    disagreement = len(signatures) > 1
    adjudicated = disagreement or low_confidence

    resolved: dict[str, int] = {}
    for criterion in CRITERIA:
        values = [int(row[criterion]) for row in score_rows]
        # Majority; ties break toward median.
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        best = max(counts.values())
        winners = sorted(v for v, n in counts.items() if n == best)
        if len(winners) == 1:
            resolved[criterion] = winners[0]
        else:
            resolved[criterion] = int(statistics.median(values))
    return resolved, adjudicated


def fact_check_essay(prompt: str, essay: str) -> dict[str, Any]:
    """Local fact gate: empty/nonsense essays or obvious anachronistic years fail."""
    text = essay.strip()
    words = text.split()
    if len(words) < 8:
        return {"checker_id": FACT_CHECKER_ID, "passed": False, "reason": "empty_or_too_short"}

    letters = re.findall(r"[A-Za-z]", text)
    if len(letters) < 20 or not _NONSENSE.search(text):
        return {"checker_id": FACT_CHECKER_ID, "passed": False, "reason": "nonsense_text"}
    unique_ratio = len({w.lower() for w in words}) / len(words)
    if unique_ratio < 0.18 and len(words) > 20:
        return {"checker_id": FACT_CHECKER_ID, "passed": False, "reason": "nonsense_repetition"}

    prompt_range = _prompt_date_range(prompt)
    if prompt_range:
        start, end = prompt_range
        essay_years = [int(y) for y in _YEAR.findall(text)]
        # Years long after the prompt window with no surrounding "later/legacy" cue.
        late = [y for y in essay_years if y > end + 40]
        if late:
            late_ok = 0
            for year in late:
                for match in re.finditer(rf"\b{year}\b", text):
                    window = text[max(0, match.start() - 40) : match.end() + 40].lower()
                    if re.search(r"\b(?:later|legacy|eventually|would later|afterward|today)\b", window):
                        late_ok += 1
                        break
            if late_ok < len(late):
                return {
                    "checker_id": FACT_CHECKER_ID,
                    "passed": False,
                    "reason": "anachronistic_year",
                    "years": late,
                }
        absurd = [y for y in essay_years if y < start - 350 or y > end + 120]
        if absurd and len(absurd) >= 2:
            return {
                "checker_id": FACT_CHECKER_ID,
                "passed": False,
                "reason": "anachronistic_year",
                "years": absurd,
            }

    return {"checker_id": FACT_CHECKER_ID, "passed": True}


def judge_essay(prompt: str, essay: str, *, task_id: str) -> dict[str, Any]:
    """Produce one external-candidate judgment record (no planner metadata)."""
    prompt = str(prompt or "").strip()
    essay = str(essay or "").strip()
    features = extract_essay_features(prompt, essay)

    auth_reviews: list[dict[str, Any]] = []
    for reviewer_id in AUTH_REVIEWER_IDS[:2]:
        rng = Random(f"{task_id}:{reviewer_id}:{features.word_count}")
        auth_reviews.append(_authenticity_review(features, reviewer_id, rng))
    if _auth_pass(auth_reviews[0]) != _auth_pass(auth_reviews[1]):
        rng = Random(f"{task_id}:auth-c:{features.word_count}")
        auth_reviews.append(_authenticity_review(features, "auth-c", rng))

    rubric_reviews: list[dict[str, Any]] = []
    for reader_id in RUBRIC_READER_IDS:
        rng = Random(f"{task_id}:{reader_id}:rubric")
        scores = _reader_scores(features, reader_id, rng)
        confidence = _reader_confidence(features, scores, rng)
        rubric_reviews.append(
            {"reader_id": reader_id, "scores": scores, "confidence": confidence}
        )

    resolved_scores, adjudicated = _resolve_scores(rubric_reviews)
    feedback = grounded_feedback_for_scores(essay or "this essay", resolved_scores)
    fact = fact_check_essay(prompt, essay)

    return {
        "task_id": task_id,
        "student_response": essay,
        "authenticity_reviews": auth_reviews,
        "rubric_reviews": rubric_reviews,
        "resolved_grade": {
            "scores": resolved_scores,
            "feedback": feedback.model_dump(),
            "adjudicated": adjudicated,
        },
        "fact_check": {
            "checker_id": fact["checker_id"],
            "passed": bool(fact["passed"]),
        },
    }
