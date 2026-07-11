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
MIN_CONCEPT_CHARS = 40
MAX_CARDS_PER_CHAPTER = 40
# Soft floor: keep filling from ranked sources until this many when possible.
TARGET_CARDS_PER_CHAPTER = 25

_YEAR_RE = re.compile(
    r"\b((?:1[4-9]\d{2}|20[0-2]\d)(?:\s*[-–]\s*(?:1[4-9]\d{2}|20[0-2]\d))?)\b"
)
_LEAD_IN_RE = re.compile(
    r"^(?:In|By|During|After|Before|Among|Throughout|Over|From)\s+[^,]{0,40},\s*",
    re.I,
)
# Multi-word proper names, or Title + of + Title (Treaty of Paris, Act of Toleration).
_PROPER_RE = re.compile(
    r"\b("
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}"
    r"|[A-Z][a-z]+\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
    r")\b"
)
# Historical labeled entities. Title-case in prose; lowercase bank labels are
# accepted separately via evidence-term rewriting.
_HIST_LABEL_RE = re.compile(
    r"\b(?:"
    r"(?:Treaty|Act|War|Battle|Compromise|Doctrine|Purchase|Revolution|"
    r"Constitution|Amendment|Proclamation|Ordinance|Crisis|System|"
    r"Debate|Exchange|Confederation|Compact|Bank|Party|Movement|"
    r"Election|Tariff|Rebellion|Massacre|Convention|Congress|"
    r"Court|Plan|Code|Laws?)"
    r"(?:\s+(?:of|v\.?|versus)\s+[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,4})?"
    r"(?:\s+[A-Z][A-Za-z'’\-]+){0,4}"
    r"|"
    # "Indian Removal Act", "New Laws", "Stamp Act"
    r"(?:[A-Z][A-Za-z'’\-]+\s+){1,4}(?:Act|Laws?|Treaty|War|Crisis|System|Debate|"
    r"Compromise|Doctrine|Proclamation|Ordinance|Rebellion|Bank|Plan)"
    r")"
)
_PERSON_RE = re.compile(
    r"\b(?:[A-Z][a-z]+\s+){0,2}"
    r"(?:de\s+|van\s+|von\s+)?"
    r"[A-Z][a-z]+(?:'[A-Z]?[a-z]+)?\b"
)
_WEAK_LEAD_WORDS = frozenset(
    {
        "few",
        "most",
        "this",
        "the",
        "he",
        "she",
        "they",
        "it",
        "by",
        "in",
        "among",
        "these",
        "those",
        "some",
        "many",
        "one",
        "another",
        "his",
        "her",
        "their",
        "its",
        "a",
        "an",
        "and",
        "or",
        "but",
        "with",
        "from",
        "for",
        "that",
        "which",
        "when",
        "while",
        "after",
        "before",
        "during",
        "over",
        "under",
        "into",
        "onto",
        "also",
        "then",
        "thus",
        "such",
        "other",
        "new",
        "old",
        "first",
        "second",
        "third",
        "later",
        "earlier",
        "southern",
        "northern",
        "western",
        "eastern",
        "conservative",
        "liberal",
        "english",
        "french",
        "spanish",
        "dutch",
        "portuguese",
        "american",
        "british",
        "european",
        "african",
        "native",
        "colonial",
        "federal",
        "national",
        "local",
        "popular",
        "political",
        "economic",
        "social",
        "religious",
        "military",
        "foreign",
        "domestic",
        "memory",
        "cue",
        "key",
        "development",
        "prelude",
        "september",
        "october",
        "november",
        "december",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
    }
)
_REMEMBERED_REF_RE = re.compile(r"\bis a remembered reference point\b", re.I)
_MEMORY_CUE_RE = re.compile(r"^memory cue\b", re.I)
_JAMMED_LIST_RE = re.compile(r"(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){4,}")

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

_EVIDENCE_TEMPLATES: tuple[str, ...] = (
    "{name}{year_bit} stands as a concrete historical marker students should paraphrase.",
    "{name}{year_bit} shaped outcomes in the era and belongs in evidence, not as a vague label.",
    "{name}{year_bit} is a dated development writers can cite with a clear causal claim.",
    "Students should recall {name}{year_bit} as a named episode with consequences in the period.",
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


def _token_is_weak(token: str) -> bool:
    return token.lower().strip(".,;:'\"") in _WEAK_LEAD_WORDS


def _looks_like_proper_entity(name: str) -> bool:
    """True for multi-word names, labeled acts/treaties, or person-like spans."""
    cleaned = re.sub(r"\s+", " ", (name or "").strip(" .,;:"))
    if not cleaned or len(cleaned) < 3:
        return False
    words = cleaned.split()
    if words and _token_is_weak(words[0]) and len(words) < 2:
        return False
    if _HIST_LABEL_RE.search(cleaned):
        return True
    # Court citations.
    if re.search(r"\bv\.?\s+[A-Z]", cleaned):
        return True
    # Possessive titled phrases: Sherman's March, Freedmen's Bureau
    if re.search(r"[A-Z][A-Za-z'’\-]*'s\s+[A-Z]", cleaned):
        return True
    meaningful = [w for w in words if w.lower() not in {"of", "the", "and", "v", "v.", "de", "van", "von", "to", "for", "in"}]
    if len(meaningful) >= 2 and all(
        (w[:1].isupper() or w[:1].islower() and "-" in w) for w in meaningful
    ):
        # Allow Scotch-Irish migration style when at least one token is capitalized.
        if any(w[:1].isupper() for w in meaningful):
            return True
    if len(words) >= 2 and all(
        w[:1].isupper() for w in words if w.lower() not in {"of", "the", "and", "v", "v.", "de", "van", "von", "to"}
    ):
        return True
    # Single capitalized token only if long enough and not a weak/generic word.
    if len(meaningful) == 1:
        w = meaningful[0]
        core = w.replace("'", "").replace("’", "")
        return len(core) >= 5 and not _token_is_weak(w) and w[0].isupper()
    # Lowercase institutional labels: encomienda system, spoils system, mercantilism
    if len(words) <= 4 and words[0][0].islower() and not _token_is_weak(words[0]):
        if any(tok in cleaned.lower() for tok in ("system", "trade", "suffrage", "farming", "migration", "nativism")):
            return True
        if len(words) == 1 and len(words[0]) >= 8:
            return True
    return False


def extract_concrete_entities(text: str) -> list[str]:
    """Pull concrete historical entities (acts, treaties, people, places) from text."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        name = re.sub(r"\s+", " ", raw).strip(" .,;:")
        if not _looks_like_proper_entity(name):
            return
        key = normalize_essay(name)
        if not key or key in seen:
            return
        parts = name.split()
        while parts and _token_is_weak(parts[0]):
            parts = parts[1:]
        if not parts or not _looks_like_proper_entity(" ".join(parts)):
            return
        name = " ".join(parts)
        seen.add(key)
        found.append(name)

    for match in _HIST_LABEL_RE.finditer(text):
        _add(match.group(0))
    for match in _PROPER_RE.finditer(text):
        _add(match.group(0))
    # Court-style citations: Worcester v. Georgia
    for match in re.finditer(
        r"\b([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)*\s+v\.?\s+[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)*)\b",
        text,
    ):
        _add(match.group(1))
    # Significant single CapWords / hyphenated names not caught above.
    for match in re.finditer(r"\b([A-Z][A-Za-z'’\-]{4,})\b", text):
        token = match.group(1)
        if _token_is_weak(token):
            continue
        # Skip if already inside a longer accepted entity.
        if any(token in ent for ent in found):
            continue
        _add(token)
    return found


def source_has_concrete_entity(text: str) -> bool:
    """Whether a source sentence/term contains a usable historical entity or year+claim."""
    raw = str(text or "").strip()
    if not raw:
        return False
    if extract_concrete_entities(raw):
        return True
    # Year alone is not enough for a short vague sentence.
    if _years_in(raw) and len(raw) >= MIN_CONCEPT_CHARS and _HIST_LABEL_RE.search(raw):
        return True
    return bool(_years_in(raw) and extract_concrete_entities(raw))


def _weak_lead_without_entity(concept: str) -> bool:
    words = concept.strip().split()
    if not words:
        return True
    lead = words[0].strip(".,;:\"'")
    if lead.lower() not in _WEAK_LEAD_WORDS:
        return False
    # Allow "The Treaty of …" / "The Indian Removal Act …" when an entity follows.
    remainder = concept[len(words[0]) :].lstrip()
    if extract_concrete_entities(remainder) or extract_concrete_entities(concept):
        # Still reject if the *only* "entity" is the weak lead itself somehow.
        entities = extract_concrete_entities(concept)
        if not entities:
            return True
        # "The Spanish established…" with no named act/person/place beyond adjective+noun.
        if lead.lower() == "the" and len(entities) == 1:
            ent_words = entities[0].split()
            if len(ent_words) == 1 and _token_is_weak(ent_words[0]):
                return True
        return False
    return True


def is_acceptable_card(concept: str) -> bool:
    """Reject useless / underspecified concept cards."""
    text = re.sub(r"\s+", " ", str(concept or "")).strip()
    if not text:
        return False
    if _REMEMBERED_REF_RE.search(text) or _MEMORY_CUE_RE.search(text):
        return False
    if _weak_lead_without_entity(text):
        return False
    entities = extract_concrete_entities(text)
    years = _years_in(text)
    if len(text) < MIN_CONCEPT_CHARS and not entities and not years:
        return False
    if len(text) < MIN_CONCEPT_CHARS and not entities:
        # Year alone on a tiny card is still useless for generators.
        return False
    if not entities and not years:
        return False
    if not entities and years and len(text) < MIN_CONCEPT_CHARS + 20:
        return False
    # Reject jammed multi-name dumps that never form a sentence.
    if text.count(" ") >= 12 and not re.search(r"\b(?:is|was|were|shaped|marks|stands|forced|split|created|established|led|caused)\b", text, re.I):
        if len(extract_concrete_entities(text)) >= 4:
            return False
    return True


def _entity_concept_sentence(
    entity: str,
    *,
    years: Sequence[str] | None = None,
    period: int | None = None,
    claim: str | None = None,
) -> str:
    """Build a clear concept sentence that names the entity."""
    name = re.sub(r"\s+", " ", entity).strip(" .,;:")
    if not name:
        return ""
    year_bit = f" ({years[0]})" if years else ""
    # Avoid double-year if name already includes one.
    if years and _years_in(name):
        year_bit = ""
    if claim:
        claim_clean = re.sub(r"\s+", " ", claim).strip(" .,;")
        # Avoid leading weak pronoun claims.
        claim_clean = re.sub(r"^(?:he|she|they|it|this|that)\s+", "", claim_clean, flags=re.I)
        if claim_clean and name.lower() not in claim_clean.lower():
            return _cap_concept(f"{name}{year_bit} {claim_clean}")
        if claim_clean:
            return _cap_concept(f"{name}{year_bit}: {claim_clean}")
    period_bit = f" in period {period}" if period is not None else " in this era"
    return _cap_concept(
        f"{name}{year_bit} marks a concrete development{period_bit} that writers should name and explain"
    )


def rewrite_source_sentence(text: str, *, period: int | None = None) -> str:
    """Rewrite an AMSCO sentence as a short memory-style concept (not a copy)."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if not source_has_concrete_entity(raw):
        return ""

    years = _years_in(raw)
    entities = extract_concrete_entities(raw)
    if not entities:
        return ""

    body = _LEAD_IN_RE.sub("", raw)
    # Keep years in body for sense, but also tag primary year at entity.
    body = re.sub(r"\s+", " ", body).strip(" .,;")
    for pattern, replacement in _STRUCTURAL_SWAPS:
        body = pattern.sub(replacement, body)
    body = re.sub(r"\b(?:called|known as)\b", "labeled", body, flags=re.I)
    body = re.sub(r"\s+", " ", body).strip(" .,;")

    # Prefer rewriting around the strongest entity rather than keeping weak leads.
    primary = entities[0]
    # Strip a leading year clause leftover ("1494, Spain...") already handled by lead-in.
    body_no_year = _YEAR_RE.sub(" ", body)
    body_no_year = re.sub(r"\s+", " ", body_no_year).strip(" .,;")

    # If body still starts with a weak pronoun, rebuild from entity + residual claim.
    first = body_no_year.split()[0] if body_no_year else ""
    if first and _token_is_weak(first):
        # Try to salvage a verb phrase after the weak subject.
        salvaged = re.sub(
            r"^(?:he|she|they|it|this|that|the|few|most|these|those|some|many)\s+",
            "",
            body_no_year,
            flags=re.I,
        )
        salvaged = re.sub(r"\s+", " ", salvaged).strip(" .,;")
        candidate = _entity_concept_sentence(
            primary, years=years, period=period, claim=salvaged or None
        )
    else:
        year_tag = f" ({years[0]})" if years and not _years_in(body) else ""
        candidate = _cap_concept(f"{body}{year_tag}")
        if not is_acceptable_card(candidate):
            candidate = _entity_concept_sentence(
                primary, years=years, period=period, claim=body_no_year
            )

    if not candidate or has_eight_gram_overlap(candidate, raw):
        # Compress to entity-centered paraphrase without copying source n-grams.
        claim = None
        lowered = body_no_year.lower()
        if "treaty" in primary.lower() or "tordesillas" in lowered:
            claim = "split Iberian claims in the Americas along a negotiated line"
        elif "act" in primary.lower() or "law" in primary.lower():
            claim = "created a binding policy change with lasting political effects"
        elif "war" in primary.lower() or "battle" in primary.lower():
            claim = "reordered power and forced new diplomatic settlements"
        elif "crisis" in primary.lower():
            claim = "exposed sectional or institutional conflict under pressure"
        else:
            claim = "shaped political and social outcomes writers can explain"
        candidate = _entity_concept_sentence(
            primary, years=years, period=period, claim=claim
        )

    if has_eight_gram_overlap(candidate, raw):
        candidate = _entity_concept_sentence(primary, years=years, period=period)

    if not is_acceptable_card(candidate):
        return ""
    return candidate


def _split_jammed_evidence_terms(term: str) -> list[str]:
    """Split accidental concatenated evidence-bank dumps into separate labels."""
    raw = str(term or "").strip()
    if not raw:
        return []
    # Slash alternatives: keep as one concept if short; else split.
    if " / " in raw and len(raw) > 48:
        parts = [p.strip() for p in raw.split("/") if p.strip()]
        if 2 <= len(parts) <= 4 and all(len(p.split()) <= 8 for p in parts):
            return parts
    # Space-separated CapWord dumps without punctuation.
    if _JAMMED_LIST_RE.fullmatch(raw) or (
        len(re.findall(r"\b[A-Z][a-z]+\b", raw)) >= 5 and "," not in raw and "(" not in raw
    ):
        # Prefer known multiword chunks via entity extraction on lightly punctuated text.
        # Insert separators before CapWords after the first.
        spaced = re.sub(r"(?<=[a-z])\s+(?=[A-Z])", " | ", raw)
        chunks = [c.strip() for c in spaced.split("|") if c.strip()]
        # Re-join bigrams that look like First Last names / Title Noun pairs.
        merged: list[str] = []
        i = 0
        while i < len(chunks):
            if i + 1 < len(chunks) and len(chunks[i].split()) == 1 and len(chunks[i + 1].split()) == 1:
                pair = f"{chunks[i]} {chunks[i + 1]}"
                if _looks_like_proper_entity(pair):
                    merged.append(pair)
                    i += 2
                    continue
            if _looks_like_proper_entity(chunks[i]) or _years_in(chunks[i]):
                merged.append(chunks[i])
            i += 1
        if merged:
            return merged
    return [raw]


def rewrite_evidence_term(term: str, *, period: int | None = None) -> str:
    """Turn a short evidence-bank label into a paraphrased concept card."""
    raw = str(term or "").strip()
    if not raw:
        return ""
    # Reject jammed dumps at the term level; caller may split first.
    if len(re.findall(r"\b[A-Z][a-z]+\b", raw)) >= 5 and "/" not in raw and "(" not in raw:
        if not _HIST_LABEL_RE.search(raw) and len(raw.split()) >= 5:
            return ""

    years = _years_in(raw)
    # Keep inline years (New Laws of 1542); strip only trailing parenthetical years.
    name = re.sub(
        r"\s*\((?:1[4-9]\d{2}|20[0-2]\d)(?:\s*[-–]\s*(?:1[4-9]\d{2}|20[0-2]\d))?\)\s*$",
        "",
        raw,
    )
    name = re.sub(r"\s+", " ", name).strip(" .,;")
    name = re.sub(r"\s*/\s*", " / ", name)
    if not name:
        return ""
    if not (_looks_like_proper_entity(name) or _HIST_LABEL_RE.search(name) or years):
        if len(name) >= 5 and not _token_is_weak(name.split()[0]):
            pass
        else:
            return ""

    year_bit = f" ({years[0]})" if years and years[0] not in name else ""
    lowered = name.lower()
    if (
        _HIST_LABEL_RE.search(name)
        or _looks_like_proper_entity(name)
        or re.search(r"\bv\.?\s+", name)
        or years
    ):
        if "treaty" in lowered:
            claim = "split or settled competing territorial claims along negotiated terms"
        elif re.search(r"\bact\b|\blaws?\b", name, re.I):
            claim = "imposed a policy change with enforceable political consequences"
        elif any(tok in lowered for tok in ("war", "battle", "rebellion", "march")):
            claim = "reordered power through conflict and aftermath settlements"
        elif "crisis" in lowered or "panic" in lowered:
            claim = "exposed institutional strain and contested authority"
        elif any(tok in lowered for tok in ("suffrage", "spoils", "nativism", "migration", "immigration")):
            claim = "reshaped who held political voice and how parties mobilized supporters"
        elif "system" in lowered:
            claim = "organized labor, trade, or governance into a lasting structure"
        elif "exchange" in lowered:
            claim = "moved people, goods, microbes, and ideas across the Atlantic world"
        elif "debate" in lowered or "trial" in lowered:
            claim = "forced public argument over rights, power, and legitimacy"
        elif re.search(r"\bv\.?\s+", name):
            claim = "set a contested legal precedent over sovereignty and rights"
        else:
            claim = "shaped political and social outcomes across the period"
        candidate = _cap_concept(f"{name}{year_bit} {claim}")
    else:
        period_label = str(period) if period is not None else "this"
        candidate = _cap_concept(
            f"{name}{year_bit} is a concrete period-{period_label} development "
            "students should name and explain in their own words"
        )

    if not is_acceptable_card(candidate):
        return ""
    return candidate


def _near_duplicate(a: str, b: str, *, threshold: float = 0.72) -> bool:
    wa = set(normalize_essay(a).split())
    wb = set(normalize_essay(b).split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _entity_keys(concept: str) -> set[str]:
    return {normalize_essay(e) for e in extract_concrete_entities(concept)}


def _dedupe_concepts(concepts: Sequence[str], sources: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen_entities: set[str] = set()
    for concept in concepts:
        if not is_acceptable_card(concept):
            continue
        if any(_near_duplicate(concept, prior) for prior in deduped):
            continue
        if any(has_eight_gram_overlap(concept, source) for source in sources if source):
            continue
        # Aggressive entity-level dedupe: one card per primary entity when possible.
        keys = _entity_keys(concept)
        if keys and keys <= seen_entities:
            continue
        deduped.append(concept)
        seen_entities |= keys
    return deduped


def chapter_to_semantic_cards(chapter: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rewrite one AMSCO chapter into deduplicated semantic concept cards.

    Prefers ``evidence_bank`` labels, then well-formed ``key_facts`` /
    ``context_hooks`` that name concrete historical entities. Caps at
    ``MAX_CARDS_PER_CHAPTER`` after aggressive dedupe.
    """
    chapter_id = str(chapter.get("id") or chapter.get("chapter") or "").strip()
    if not chapter_id:
        raise ValueError("chapter missing id")
    period_raw = chapter.get("period")
    period = int(period_raw) if period_raw is not None else None

    sources = [
        str(s)
        for field in ("key_facts", "context_hooks")
        for s in (chapter.get(field) or ())
    ]

    ranked: list[str] = []

    # 1) Evidence bank first — highest-precision concrete entities.
    for item in chapter.get("evidence_bank") or ():
        for piece in _split_jammed_evidence_terms(str(item)):
            concept = rewrite_evidence_term(piece, period=period)
            if concept:
                ranked.append(concept)

    # 2) Key facts with concrete entities.
    for item in chapter.get("key_facts") or ():
        text = str(item)
        if not source_has_concrete_entity(text):
            continue
        # Skip pronoun-led sentences unless they still contain a strong named entity.
        lead = text.split()[0].strip(".,;:\"'") if text.split() else ""
        if lead.lower() in {"he", "she", "they", "this", "few", "most"} and not extract_concrete_entities(text):
            continue
        concept = rewrite_source_sentence(text, period=period)
        if concept:
            ranked.append(concept)

    # 3) Context hooks last.
    for item in chapter.get("context_hooks") or ():
        text = str(item)
        if not source_has_concrete_entity(text):
            continue
        concept = rewrite_source_sentence(text, period=period)
        if concept:
            ranked.append(concept)

    deduped = _dedupe_concepts(ranked, sources)
    if len(deduped) > MAX_CARDS_PER_CHAPTER:
        # Keep evidence-forward cards first (they were appended first).
        deduped = deduped[:MAX_CARDS_PER_CHAPTER]

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
            for piece in _split_jammed_evidence_terms(str(item)):
                text = piece.strip()
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
        for entity in extract_concrete_entities(concept):
            key = normalize_essay(entity)
            if key and key not in seen and len(key.split()) <= 8:
                seen.add(key)
                terms.append(entity)
        # Leading proper-noun chunk before common predicates.
        lead = re.match(
            r"^([A-Z][^.]{2,50}?)(?:\s+(?:is|marks|stands|serves|shaped|split|imposed|reordered|exposed|organized|moved|forced|created)\b)",
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
