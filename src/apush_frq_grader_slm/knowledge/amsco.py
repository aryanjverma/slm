"""APUSH knowledge-base helpers (AMSCO-derived)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_KB_PATH = Path("artifacts/knowledge/amsco_2016_kb.jsonl")

_YEAR_RE = re.compile(r"\b((?:1[4-9]\d{2}|20[0-2]\d)(?:\s*[-–]\s*(?:1[4-9]\d{2}|20[0-2]\d))?)\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]{2,}")

_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "were",
        "was",
        "are",
        "have",
        "had",
        "has",
        "their",
        "they",
        "them",
        "which",
        "when",
        "what",
        "into",
        "also",
        "been",
        "more",
        "than",
        "only",
        "over",
        "after",
        "before",
        "between",
        "under",
        "would",
        "could",
        "about",
        "other",
        "these",
        "those",
        "such",
        "many",
        "most",
        "some",
        "while",
        "where",
        "during",
        "united",
        "states",
        "american",
        "america",
        "people",
        "evaluate",
        "extent",
        "analyze",
        "compare",
        "contrast",
        "explain",
        "how",
        "why",
        "did",
        "does",
        "essay",
        "prompt",
        "response",
        "period",
    }
)


def load_kb(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load AMSCO knowledge-base JSONL into a list of chapter dicts."""
    kb_path = Path(path) if path is not None else DEFAULT_KB_PATH
    rows: list[dict[str, Any]] = []
    with kb_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("chapter", 0)))
    return rows


def facts_for_period(kb: list[dict[str, Any]], period: int) -> list[dict[str, Any]]:
    """Return chapter records whose APUSH period matches ``period`` (1–9)."""
    return [ch for ch in kb if int(ch.get("period", -1)) == int(period)]


def _prompt_years(prompt: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _YEAR_RE.finditer(prompt):
        raw = match.group(1).replace("–", "-")
        if "-" in raw:
            a, b = raw.split("-", 1)
            try:
                spans.append((int(a.strip()), int(b.strip())))
            except ValueError:
                continue
        else:
            try:
                y = int(raw.strip())
            except ValueError:
                continue
            spans.append((y, y))
    return spans


def _parse_date_range(date_range: str) -> tuple[int, int] | None:
    text = date_range.lower().replace("present", "2020")
    nums = [int(x) for x in re.findall(r"\b(?:1[4-9]\d{2}|20[0-2]\d)\b", text)]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums), max(nums)


def _prompt_tokens(prompt: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(prompt) if w.lower() not in _STOP}


def _chapter_overlap_score(chapter: dict[str, Any], tokens: set[str], year_spans: list[tuple[int, int]]) -> float:
    score = 0.0
    title = str(chapter.get("title", "")).lower()
    keywords = [str(k).lower() for k in chapter.get("topic_keywords", [])]
    evidence = [str(e).lower() for e in chapter.get("evidence_bank", [])[:15]]
    haystack = " ".join([title, " ".join(keywords), " ".join(evidence)])

    for tok in tokens:
        if len(tok) < 4:
            continue
        if tok in title:
            score += 4.0
        elif any(tok in kw or kw in tok for kw in keywords):
            score += 2.5
        elif any(tok in ev for ev in evidence):
            score += 2.0
        elif tok in haystack:
            score += 1.0

    ch_span = _parse_date_range(str(chapter.get("date_range", "")))
    if ch_span and year_spans:
        c0, c1 = ch_span
        for y0, y1 in year_spans:
            latest_start = max(c0, y0)
            earliest_end = min(c1, y1)
            if latest_start <= earliest_end:
                overlap = earliest_end - latest_start + 1
                # Strong preference for chapters whose range covers the prompt years
                score += 8.0 + min(overlap, 50) / 5.0
            elif y1 < c0 and c0 - y1 <= 15:
                score += 1.5
            elif y0 > c1 and y0 - c1 <= 15:
                score += 1.5
    elif year_spans and not ch_span:
        score -= 1.0
    return score


def facts_for_prompt(
    kb: list[dict[str, Any]],
    prompt: str,
    max_facts: int = 30,
) -> dict[str, Any]:
    """Return relevant chapter facts/evidence/misconceptions/hooks for an LEQ prompt.

    Ranking uses keyword overlap with title/topic_keywords/evidence_bank and
    date-range overlap with years mentioned in the prompt.
    """
    tokens = _prompt_tokens(prompt)
    year_spans = _prompt_years(prompt)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for ch in kb:
        score = _chapter_overlap_score(ch, tokens, year_spans)
        if score > 0:
            ranked.append((score, ch))
    ranked.sort(key=lambda x: (-x[0], int(x[1].get("chapter", 0))))

    if not ranked:
        # Fallback: period hint in prompt text ("Period 4", etc.)
        period_m = re.search(r"\bperiod\s*([1-9])\b", prompt, re.I)
        if period_m:
            ranked = [(1.0, ch) for ch in facts_for_period(kb, int(period_m.group(1)))]
        else:
            ranked = [(0.0, ch) for ch in kb[:3]]

    selected = [ch for _, ch in ranked[:4]]
    key_facts: list[str] = []
    evidence_bank: list[str] = []
    misconceptions: list[str] = []
    context_hooks: list[str] = []
    chapter_ids: list[str] = []

    for ch in selected:
        chapter_ids.append(str(ch.get("id", ch.get("chapter"))))
        for fact in ch.get("key_facts", []):
            if fact not in key_facts:
                key_facts.append(fact)
            if len(key_facts) >= max_facts:
                break
        for item in ch.get("evidence_bank", []):
            if item not in evidence_bank:
                evidence_bank.append(item)
        for item in ch.get("misconceptions", []):
            if item not in misconceptions:
                misconceptions.append(item)
        for item in ch.get("context_hooks", []):
            if item not in context_hooks:
                context_hooks.append(item)
        if len(key_facts) >= max_facts:
            break

    return {
        "chapter_ids": chapter_ids,
        "key_facts": key_facts[:max_facts],
        "evidence_bank": evidence_bank[:20],
        "misconceptions": misconceptions[:12],
        "context_hooks": context_hooks[:12],
    }
