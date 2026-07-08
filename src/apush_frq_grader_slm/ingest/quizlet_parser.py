"""Parse Quizlet APUSH LEQ study sets into RawAPCSample records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.ingest.apc_parser import RawAPCSample
from apush_frq_grader_slm.ingest.scoring import total_to_row_scores

PROMPT_PATTERN = re.compile(
    r"((?:Evaluate|Explain|Compare|Analyze)\s+(?:the\s+)?"
    r"(?:extent to which|relative importance|impact|causes|how)[^.\n?]{10,280}[.?])",
    re.IGNORECASE,
)

LEQ_PROMPT_LINE = re.compile(r"^\(?(LEQ)\)?[:\s]", re.IGNORECASE)
STRUCTURED_LINE = re.compile(r"^(CONTEXT|CLAIM|EVIDENCE|THESIS|CC)\s*:", re.IGNORECASE)
BODY_PART = re.compile(r"^(Introduction|Thesis|Body\s*\d+|Conclusion)", re.IGNORECASE)
THESIS_CC = re.compile(
    r"thesis:\s*(.+?)(?:\s+cc:\s*(.+))?$",
    re.IGNORECASE | re.DOTALL,
)


def parse_quizlet_set(data: dict[str, Any]) -> list[RawAPCSample]:
    """Parse a Quizlet set JSON export into essay samples."""
    set_id = str(data.get("set_id", data.get("id", "unknown")))
    title = str(data.get("title", ""))
    terms = data.get("terms") or data.get("cards") or []
    cards = [_normalize_card(row) for row in terms if _normalize_card(row)]
    if not cards:
        return []

    meta = {
        "set_id": set_id,
        "title": title,
        "provider": "quizlet",
        "url": data.get("url") or f"https://quizlet.com/{set_id}",
    }
    if data.get("prompt"):
        meta["prompt"] = data["prompt"]

    grouped = _group_cards(cards)
    samples: list[RawAPCSample] = []
    for idx, group in enumerate(grouped):
        prompt = group.get("prompt") or meta.get("prompt") or _prompt_from_title(title)
        essay = group.get("essay", "").strip()
        if not prompt or len(essay) < 120:
            continue
        total = int(group.get("total", 4))
        scores = group.get("scores") or total_to_row_scores(total)
        sample_id = group.get("sample_id") or f"Q{idx + 1}"
        samples.append(
            RawAPCSample(
                sample_id=sample_id,
                prompt=prompt,
                essay=essay,
                scores=scores,
                total_score=sum(scores.values()) if "scores" in group else total,
                commentary_by_row={},
                metadata={
                    **meta,
                    "sample_id": sample_id,
                    "source": f"quizlet_{set_id}_{sample_id}",
                    "essay_source": "quizlet_cards",
                    "provider": "quizlet",
                },
            )
        )
    return samples


def load_quizlet_json(path: Path) -> list[RawAPCSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_quizlet_set(data)


def _normalize_card(row: dict[str, Any]) -> dict[str, str] | None:
    term = str(row.get("term") or row.get("word") or row.get("front") or "").strip()
    definition = str(
        row.get("definition") or row.get("back") or row.get("def") or ""
    ).strip()
    if not term and not definition:
        return None
    return {"term": term, "definition": definition}


def _group_cards(cards: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current and (
            current.get("essay") or current.get("parts") or current.get("structured")
        ):
            groups.append(current)
        current = None

    for card in cards:
        term = card["term"]
        definition = card["definition"]
        combined = f"{term}\n{definition}".strip()

        if LEQ_PROMPT_LINE.match(term) or (term.upper().startswith("(LEQ)") and definition):
            flush()
            prompt_text = definition if LEQ_PROMPT_LINE.match(term) else definition
            current = {
                "prompt": _clean_prompt(prompt_text),
                "parts": [],
                "sample_id": f"L{len(groups) + 1}",
            }
            continue

        thesis_match = THESIS_CC.search(definition) if definition else None
        if thesis_match:
            flush()
            essay = thesis_match.group(1).strip()
            if thesis_match.group(2):
                essay = f"{essay} {thesis_match.group(2).strip()}"
            prompt_guess = term if PROMPT_PATTERN.search(term) else ""
            current = {
                "prompt": _clean_prompt(prompt_guess) if prompt_guess else "",
                "essay": essay,
                "total": 4,
                "sample_id": f"T{len(groups) + 1}",
            }
            continue

        prompt_match = PROMPT_PATTERN.search(combined)
        if prompt_match and len(definition) > 80 and not definition.lower().startswith("thesis:"):
            flush()
            current = {
                "prompt": re.sub(r"\s+", " ", prompt_match.group(1)).strip(),
                "essay": definition,
                "sample_id": f"P{len(groups) + 1}",
            }
            continue

        if STRUCTURED_LINE.match(definition) or STRUCTURED_LINE.match(term):
            text = definition if STRUCTURED_LINE.match(definition) else term
            if current is None:
                current = {"parts": [], "sample_id": f"S{len(groups) + 1}"}
            current.setdefault("structured", []).append(text)
            continue

        if BODY_PART.match(term):
            if current is None:
                current = {"parts": [], "sample_id": f"B{len(groups) + 1}"}
            current.setdefault("parts", []).append(definition)
            continue

        if term.lower() in {"introduction", "thesis", "conclusion"} or term.lower().startswith("body"):
            if current is None:
                current = {"parts": [], "sample_id": f"M{len(groups) + 1}"}
            current.setdefault("parts", []).append(definition or term)
            continue

        if PROMPT_PATTERN.search(term) and len(definition) >= 120:
            flush()
            current = {
                "prompt": _clean_prompt(term),
                "essay": definition,
                "sample_id": f"D{len(groups) + 1}",
            }
            continue

    flush()

    finalized: list[dict[str, Any]] = []
    for group in groups:
        if group.get("essay"):
            finalized.append(group)
            continue
        if group.get("structured"):
            structured = group["structured"]
            prompt = group.get("prompt") or _prompt_from_structured(structured)
            essay = " ".join(
                line.split(":", 1)[1].strip() for line in structured if ":" in line
            )
            group["prompt"] = prompt
            group["essay"] = essay
            group["total"] = 4
            finalized.append(group)
            continue
        if group.get("parts"):
            essay = " ".join(part.strip() for part in group["parts"] if part.strip())
            group["essay"] = essay
            group["total"] = 4
            finalized.append(group)
    return finalized


def _prompt_from_structured(lines: list[str]) -> str:
    for line in lines:
        if PROMPT_PATTERN.search(line):
            return _clean_prompt(PROMPT_PATTERN.search(line).group(1))  # type: ignore[union-attr]
    return "Evaluate the extent to which the topic described in the prompt changed United States history."


def _prompt_from_title(title: str) -> str:
    if PROMPT_PATTERN.search(title):
        return _clean_prompt(PROMPT_PATTERN.search(title).group(1))  # type: ignore[union-attr]
    return f"APUSH LEQ practice set: {title}"


def _clean_prompt(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^\(LEQ\)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned
