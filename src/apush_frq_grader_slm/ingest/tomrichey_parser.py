"""Parse Tom Richey labeled APUSH LEQ sample PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.ingest.apc_parser import RawAPCSample
from apush_frq_grader_slm.ingest.scoring import total_to_row_scores

ESSAY_BLOCK_PATTERN = re.compile(
    r"(?:SAMPLE RESPONSE|"
    r"EXEMPLAR(?:\s*ESSAY)?|"
    r"ABOVE[- ]AVERAGE(?:\s*ESSAY)?|"
    r"BELOW[- ]AVERAGE(?:\s*ESSAY)?|"
    r"FULL CREDIT)"
    r"\s*(?:\((\d+)\s*/\s*6\))?"
    r"(?:\s*(\d+)\s*Words?)?",
    re.IGNORECASE,
)

PROMPT_PATTERN = re.compile(
    r"(Evaluate the extent to which[^.]+\.)",
    re.IGNORECASE,
)


def parse_tomrichey_pdf(path: Path, *, metadata: dict[str, Any] | None = None) -> list[RawAPCSample]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required. Install with: pip install apush-frq-grader-slm[ingest]"
        ) from exc

    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    meta = {**_metadata_from_filename(path), **(metadata or {})}
    return parse_tomrichey_text(text, metadata=meta)


def parse_tomrichey_text(text: str, *, metadata: dict[str, Any] | None = None) -> list[RawAPCSample]:
    meta = dict(metadata or {})
    prompt = meta.get("prompt") or _extract_prompt(text)
    if not prompt:
        raise ValueError("Could not locate LEQ prompt for Tom Richey PDF")

    samples: list[RawAPCSample] = []
    markers = list(ESSAY_BLOCK_PATTERN.finditer(text))
    for idx, marker in enumerate(markers):
        total = int(marker.group(1)) if marker.group(1) else _label_total(marker.group(0))
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        essay = _clean_essay(text[start:end])
        if len(essay) < 100:
            continue
        sample_id = _sample_id(marker.group(0), idx)
        scores = total_to_row_scores(total)
        source = _source_tag(meta, sample_id)
        samples.append(
            RawAPCSample(
                sample_id=sample_id,
                prompt=prompt,
                essay=essay,
                scores=scores,
                total_score=total,
                commentary_by_row={},
                metadata={
                    **meta,
                    "sample_id": sample_id,
                    "source": source,
                    "essay_source": "tom_richey_pdf",
                    "provider": "tom_richey",
                },
            )
        )
    return samples


def _metadata_from_filename(path: Path) -> dict[str, Any]:
    name = path.stem.lower()
    meta: dict[str, Any] = {"filename": path.name, "provider": "tom_richey"}
    year_match = re.search(r"(20\d{2})", name)
    if year_match:
        meta["year"] = int(year_match.group(1))
    leq_match = re.search(r"leq[_\s-]*(\d)", name)
    if leq_match:
        meta["leq_num"] = int(leq_match.group(1))
    set_match = re.search(r"set[_\s-]*(\d)", name)
    if set_match:
        meta["set"] = int(set_match.group(1))
    return meta


def _source_tag(meta: dict[str, Any], sample_id: str) -> str:
    year = meta.get("year", "unknown")
    leq_num = meta.get("leq_num", "x")
    set_num = meta.get("set", "x")
    return f"tom_richey_{year}_leq{leq_num}_set{set_num}_{sample_id}"


def _extract_prompt(text: str) -> str:
    match = PROMPT_PATTERN.search(text)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    culture_match = re.search(
        r"development of a [“\"]?national culture[”\"]?",
        text,
        re.IGNORECASE,
    )
    if culture_match:
        return (
            "Evaluate the extent to which developments in the period contributed to "
            "the growth of a distinct national culture in the United States."
        )
    return ""


def _label_total(label: str) -> int:
    lowered = label.lower()
    if "full credit" in lowered or "exemplar" in lowered:
        return 6
    if "above" in lowered:
        return 5
    if "below" in lowered:
        return 2
    match = re.search(r"\((\d+)\s*/\s*6\)", label)
    if match:
        return int(match.group(1))
    return 3


def _sample_id(label: str, index: int) -> str:
    match = re.search(r"SAMPLE RESPONSE\s+([A-E])", label, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    lowered = label.lower()
    if "exemplar" in lowered or "full credit" in lowered:
        return "A"
    if "above" in lowered:
        return "B"
    if "below" in lowered:
        return "C"
    return chr(ord("A") + index)


def _clean_essay(text: str) -> str:
    cleaned = re.sub(r"Advanced Placement.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"Visit tomrichey\.net.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"202\d APUSH Sample Responses.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
