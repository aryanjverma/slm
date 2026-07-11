"""Semantic AMSCO fact cards and overlap-exemption helpers for v5 data production.

Cards are memory-style paraphrases of AMSCO chapter material. They must not copy
source sentences verbatim; generators receive concepts, not textbook wording.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from apush_frq_grader_slm.ingest.dedup import normalize_essay
from apush_frq_grader_slm.knowledge.amsco import (
    facts_for_period,
    facts_for_prompt,
    load_kb,
)

CONCEPT_MAX_CHARS = 200
SOURCE_KIND = "semantic_rewrite"
MAX_SEED_CHAPTERS = 3

_YEAR_RE = re.compile(
    r"\b((?:1[4-9]\d{2}|20[0-2]\d)(?:\s*[-–]\s*(?:1[4-9]\d{2}|20[0-2]\d))?)\b"
)
_LEAD_IN_RE = re.compile(
    r"^(?:In|By|During|After|Before|Among|Throughout|Over|From)\s+[^,]{0,40},\s*",
    re.I,
)
_PROPER_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)?)\b"
)

# Structural swaps deliberately break AMSCO phrasing while keeping the claim.
_STRUCTURAL_SWAPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmoved the pope'?s line a few degrees to the west and signed an agreement called\b", re.I),
     "renegotiated the papal demarcation and formalized it as"),
    (re.compile(r"\bsigned an agreement called\b", re.I), "codified a settlement known as"),
    (re.compile(r"\bimplemented a mercantilist policy with a series of\b", re.I),
     "enforced mercantile controls through"),
    (re.compile(r"\bprovided the colony with\b", re.I), "issued the settlement"),
    (re.compile(r"\bguaranteed a representative assembly\b", re.I), "secured an elected assembly"),
    (re.compile(r"\bpersuaded the king to institute\b", re.I), "pressed the crown to enact"),
    (re.compile(r"\bthe population increase was even more dramatic\b", re.I),
     "demographic growth proved still sharper"),
    (re.compile(r"\bonly five newspapers existed in the colonies, but\b", re.I),
     "colonial newspapers were scarce at first, yet"),
    (re.compile(r"\bthe number had grown to more than\b", re.I), "the count rose above"),
    (re.compile(r"\bfrom about\b", re.I), "rising from roughly"),
    (re.compile(r"\bEngland'?s government\b", re.I), "the English state"),
    (re.compile(r"\bwhich guaranteed\b", re.I), "securing"),
    (re.compile(r"\band signed\b", re.I), "and ratified"),
    (re.compile(r"\bmoved\b", re.I), "shifted"),
    (re.compile(r"\bdeveloped a powerful empire\b", re.I), "built an expansive imperial order"),
    (re.compile(r"\boriginal discovery, exploration, and settlement\b", re.I),
     "earlier peopling and settlement"),
    (re.compile(r"\boccurred at least\b", re.I), "began no later than roughly"),
    (re.compile(r"\bbefore Christopher Columbus was born\b", re.I), "prior to Columbus"),
    (re.compile(r"\bSeveral centuries after the decline of\b", re.I), "Long after the fall of"),
)

_COMMON_HISTORICAL_TERMS: tuple[str, ...] = (
    "United States",
    "Native American",
    "Native Americans",
    "African American",
    "African Americans",
    "British North America",
    "New England",
    "Great Britain",
    "Federal government",
    "Supreme Court",
    "Civil War",
    "World War",
    "Cold War",
    "New Deal",
    "Great Depression",
    "Manifest Destiny",
    "Columbian Exchange",
    "Trail of Tears",
    "Bill of Rights",
    "Constitution",
    "Congress",
    "President",
    "Republican",
    "Democrat",
    "Federalist",
)


def _cap_concept(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,;:")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    if len(cleaned) > CONCEPT_MAX_CHARS:
        cleaned = cleaned[: CONCEPT_MAX_CHARS - 1].rstrip(" .,;:") + "."
    return cleaned[0].upper() + cleaned[1:]


def _years_in(text: str) -> list[str]:
    return [m.replace("–", "-") for m in _YEAR_RE.findall(text)]


def _eight_grams(text: str) -> set[tuple[str, ...]]:
    words = normalize_essay(text).split()
    return {tuple(words[i : i + 8]) for i in range(max(0, len(words) - 7))}


def has_eight_gram_overlap(left: str, right: str) -> bool:
    """True when two texts share a normalized eight-word span."""
    return bool(_eight_grams(left) & _eight_grams(right))


def _entity_memory_card(text: str, *, period: int | None = None) -> str:
    years = _years_in(text)
    names = _PROPER_RE.findall(text)
    # Prefer multi-word names; fall back to first capitalized token sequence.
    name = names[0] if names else "This development"
    year_bit = f" ({years[0]})" if years else ""
    period_bit = f" in period {period}" if period is not None else ""
    return _cap_concept(
        f"{name} is a remembered reference point{period_bit}{year_bit}"
    )


def rewrite_source_sentence(text: str, *, period: int | None = None) -> str:
    """Rewrite an AMSCO sentence as a short memory-style concept (not a copy)."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    years = _years_in(raw)
    body = _LEAD_IN_RE.sub("", raw)
    body = _YEAR_RE.sub(" ", body)
    body = re.sub(r"\s+", " ", body).strip(" .,;")
    for pattern, replacement in _STRUCTURAL_SWAPS:
        body = pattern.sub(replacement, body)
    # Drop leftover textbook connective fluff.
    body = re.sub(r"\b(?:called|known as)\b", "labeled", body, flags=re.I)
    body = re.sub(r"\s+", " ", body).strip(" .,;")
    year_tag = f" ({years[0]})" if years else ""
    candidate = _cap_concept(f"{body}{year_tag}")
    if not candidate or has_eight_gram_overlap(candidate, raw):
        candidate = _entity_memory_card(raw, period=period)
    if has_eight_gram_overlap(candidate, raw):
        # Last resort: highly compressed entity+year card.
        years = _years_in(raw)
        names = _PROPER_RE.findall(raw) or ["Key development"]
        candidate = _cap_concept(
            f"Memory cue{f' for period {period}' if period is not None else ''}: "
            f"{names[0]}{f', {years[0]}' if years else ''}"
        )
    return candidate


def rewrite_evidence_term(term: str, *, period: int | None = None) -> str:
    """Turn a short evidence-bank label into a paraphrased concept card."""
    raw = str(term or "").strip()
    if not raw:
        return ""
    years = _years_in(raw)
    name = _YEAR_RE.sub(" ", raw)
    name = re.sub(r"[()]", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .,;")
    if not name:
        return ""
    if years:
        return _cap_concept(
            f"{name} marks a concrete period-{period if period is not None else '?'} "
            f"development dated {years[0]}"
        )
    return _cap_concept(
        f"{name} is a concrete period-{period if period is not None else '?'} "
        "reference students should recall in their own words"
    )


def _near_duplicate(a: str, b: str, *, threshold: float = 0.78) -> bool:
    wa = set(normalize_essay(a).split())
    wb = set(normalize_essay(b).split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def chapter_to_semantic_cards(chapter: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rewrite one AMSCO chapter into deduplicated semantic concept cards."""
    chapter_id = str(chapter.get("id") or chapter.get("chapter") or "").strip()
    if not chapter_id:
        raise ValueError("chapter missing id")
    period_raw = chapter.get("period")
    period = int(period_raw) if period_raw is not None else None
    concepts: list[str] = []

    for field in ("key_facts", "context_hooks"):
        for item in chapter.get(field) or ():
            concept = rewrite_source_sentence(str(item), period=period)
            if concept:
                concepts.append(concept)
    for item in chapter.get("evidence_bank") or ():
        concept = rewrite_evidence_term(str(item), period=period)
        if concept:
            concepts.append(concept)

    deduped: list[str] = []
    for concept in concepts:
        if any(_near_duplicate(concept, prior) for prior in deduped):
            continue
        # Reject residual copies of any source sentence.
        sources = [
            str(s)
            for field in ("key_facts", "context_hooks")
            for s in (chapter.get(field) or ())
        ]
        if any(has_eight_gram_overlap(concept, source) for source in sources if source):
            continue
        deduped.append(concept)

    cards: list[dict[str, Any]] = []
    for concept in deduped:
        card: dict[str, Any] = {
            "chapter_id": chapter_id,
            "concept": concept,
            "source_kind": SOURCE_KIND,
        }
        if period is not None:
            card["period"] = period
        cards.append(card)
    return cards


def kb_to_semantic_cards(kb: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert a full AMSCO KB into semantic fact cards."""
    cards: list[dict[str, Any]] = []
    for chapter in kb:
        cards.extend(chapter_to_semantic_cards(chapter))
    return cards


def build_semantic_fact_cards(
    kb_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load AMSCO KB from disk and emit semantic concept cards."""
    return kb_to_semantic_cards(load_kb(kb_path))


def _chapter_id(chapter: Mapping[str, Any]) -> str:
    return str(chapter.get("id") or chapter.get("chapter") or "").strip()


def amsco_chapter_ids_for_prompt(
    prompt: str,
    *,
    period: int | None = None,
    kb: Sequence[Mapping[str, Any]] | None = None,
    kb_path: str | Path | None = None,
    max_chapters: int = MAX_SEED_CHAPTERS,
) -> list[str]:
    """Map an LEQ prompt (and optional APUSH period) to 1–3 AMSCO chapter IDs.

    Ranking prefers ``facts_for_prompt`` keyword/date overlap, then fills gaps
    with same-period chapters when a period is provided.
    """
    records: list[Mapping[str, Any]]
    if kb is not None:
        records = list(kb)
    else:
        records = load_kb(kb_path)
    if not records:
        return []
    n = max(1, min(int(max_chapters), MAX_SEED_CHAPTERS))
    # facts_for_prompt may truncate chapter_ids once key_facts hit max_facts;
    # re-rank with a high fact ceiling so we still see the top overlapping chapters.
    bundle = facts_for_prompt(list(records), str(prompt or ""), max_facts=10_000)
    ranked_ids = [str(item).strip() for item in (bundle.get("chapter_ids") or ()) if str(item).strip()]
    selected: list[str] = []
    seen: set[str] = set()

    def _take(candidate: str) -> None:
        if not candidate or candidate in seen or len(selected) >= n:
            return
        seen.add(candidate)
        selected.append(candidate)

    if period is not None:
        period_ids = {_chapter_id(ch) for ch in facts_for_period(list(records), int(period))}
        for chapter_id in ranked_ids:
            if chapter_id in period_ids:
                _take(chapter_id)
        for chapter_id in ranked_ids:
            _take(chapter_id)
        for chapter in facts_for_period(list(records), int(period)):
            _take(_chapter_id(chapter))
    else:
        for chapter_id in ranked_ids:
            _take(chapter_id)

    if not selected:
        # Absolute fallback: first KB chapters so packets never attach zero cards.
        for chapter in records[:n]:
            _take(_chapter_id(chapter))
    return selected[:n]


def attach_amsco_chapter_ids_to_seeds(
    seeds: Sequence[Mapping[str, Any]],
    *,
    kb: Sequence[Mapping[str, Any]] | None = None,
    kb_path: str | Path | None = None,
    max_chapters: int = MAX_SEED_CHAPTERS,
) -> list[dict[str, Any]]:
    """Ensure each seed profile carries 1–3 ``amsco_chapter_ids`` for packet export."""
    records = list(kb) if kb is not None else load_kb(kb_path)
    updated: list[dict[str, Any]] = []
    for seed in seeds:
        row = dict(seed)
        existing = [
            str(item).strip()
            for item in (row.get("amsco_chapter_ids") or row.get("chapter_ids") or ())
            if str(item).strip()
        ]
        if existing:
            row["amsco_chapter_ids"] = existing[: max(1, min(int(max_chapters), MAX_SEED_CHAPTERS))]
        else:
            period_raw = row.get("period")
            period = int(period_raw) if period_raw is not None else None
            prompt = str(row.get("prompt") or row.get("prompt_text") or "")
            row["amsco_chapter_ids"] = amsco_chapter_ids_for_prompt(
                prompt,
                period=period,
                kb=records,
                max_chapters=max_chapters,
            )
        updated.append(row)
    return updated


def evidence_terms_from_kb(kb: Sequence[Mapping[str, Any]]) -> list[str]:
    """Short evidence-bank labels and years suitable as overlap exemptions."""
    terms: list[str] = []
    seen: set[str] = set()
    for chapter in kb:
        for item in chapter.get("evidence_bank") or ():
            text = str(item).strip()
            if not text:
                continue
            key = normalize_essay(text)
            if key in seen:
                continue
            seen.add(key)
            terms.append(text)
            for year in _years_in(text):
                if year not in seen:
                    seen.add(year)
                    terms.append(year)
        for field in ("key_facts", "context_hooks", "topic_keywords"):
            for item in chapter.get(field) or ():
                for year in _years_in(str(item)):
                    if year not in seen:
                        seen.add(year)
                        terms.append(year)
    return terms


def evidence_terms_from_fact_cards(cards: Sequence[Mapping[str, Any]]) -> list[str]:
    """Extract exemption phrases from already-built semantic cards."""
    terms: list[str] = []
    seen: set[str] = set()
    for card in cards:
        concept = str(card.get("concept") or "")
        # Parenthetical cues often hold the canonical short name/year.
        for match in re.findall(r"\(([^)]{3,60})\)", concept):
            text = match.strip()
            key = normalize_essay(text)
            if key and key not in seen:
                seen.add(key)
                terms.append(text)
        for year in _years_in(concept):
            if year not in seen:
                seen.add(year)
                terms.append(year)
        # Leading proper-noun chunk before "is/marks/stands".
        lead = re.match(
            r"^([A-Z][^.]{2,50}?)(?:\s+(?:is|marks|stands|serves)\b)",
            concept,
        )
        if lead:
            text = lead.group(1).strip(" ,")
            key = normalize_essay(text)
            if key and key not in seen and len(key.split()) <= 8:
                seen.add(key)
                terms.append(text)
    return terms


def default_allowed_overlap_phrases(
    *,
    kb: Sequence[Mapping[str, Any]] | None = None,
    fact_cards: Sequence[Mapping[str, Any]] | None = None,
    kb_path: str | Path | None = None,
) -> list[str]:
    """Names, dates, and short historical terms that should not fail overlap gates.

    Pulls AMSCO ``evidence_bank`` labels and years when a KB is available, plus a
    small set of unavoidable textbook terms. Fact-card evidence cues are merged
    when provided.
    """
    phrases: list[str] = list(_COMMON_HISTORICAL_TERMS)
    seen = {normalize_essay(p) for p in phrases}
    if kb is not None:
        records: list[Mapping[str, Any]] = list(kb)
    elif kb_path is not None:
        records = load_kb(kb_path)
    else:
        records = []
    for term in evidence_terms_from_kb(records):
        key = normalize_essay(term)
        if key and key not in seen:
            seen.add(key)
            phrases.append(term)
    if fact_cards:
        for term in evidence_terms_from_fact_cards(fact_cards):
            key = normalize_essay(term)
            if key and key not in seen:
                seen.add(key)
                phrases.append(term)
    return phrases


def load_allowed_phrases_file(path: Path) -> list[str]:
    """Load allowed overlap phrases from a JSONL (``phrase``/``text``) or text file."""
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if lines[0].startswith("{"):
        phrases: list[str] = []
        for line in lines:
            row = json.loads(line)
            value = row.get("phrase") or row.get("text") or row.get("term") or ""
            if str(value).strip():
                phrases.append(str(value).strip())
        return phrases
    return lines


def merge_allowed_phrases(*groups: Iterable[str]) -> list[str]:
    """Stable de-duplication of allowed-phrase lists."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item).strip()
            key = normalize_essay(text)
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(text)
    return merged
