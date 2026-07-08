"""Parse College Board AP Central LEQ sample PDFs into structured records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROMPT_PATTERN = re.compile(
    r"((?:Evaluate|Explain|Compare|Describe|Analyze)\s+(?:the\s+)?"
    r"(?:extent to which|relative importance|impact|causes and effects|how)[^.\n]{10,260}\.)",
    re.IGNORECASE,
)

PAGE_MARKER_PATTERN = re.compile(
    r"Page\s+\d+\s+of\s+\d+\s+(\d+)\s*\n?\s*([A-C])\b",
    re.IGNORECASE,
)

SAMPLE_ESSAY_PATTERN = re.compile(
    r"Sample\s+(\d+[A-C])\s+(\d+)\s+of\s+(\d+)\s*\n",
    re.IGNORECASE,
)

SAMPLE_HEADER_CLASSIC = re.compile(
    r"Sample:\s*(\d+[A-C])\s*\n"
    r"Thesis/Claim:\s*(\d+)\s*\n"
    r"Contextualization:\s*(\d+)\s*\n"
    r"Evidence:\s*(\d+)\s*\n"
    r"Analysis and Reasoning:\s*(\d+)\s*\n"
    r"Total Score:\s*(\d+)",
    re.IGNORECASE,
)

SAMPLE_HEADER_SHORT = re.compile(
    r"Sample:\s*(\d+[A-C])\s*\n"
    r"Thesis:\s*(\d+)\s*\n"
    r"Contextualization:\s*(\d+)\s*\n"
    r"Evidence:\s*(\d+)\s*\n"
    r"Analysis and Reasoning:\s*(\d+)\s*\n"
    r"Total Score:\s*(\d+)",
    re.IGNORECASE,
)

SAMPLE_HEADER_2025 = re.compile(
    r"Sample:\s*(\d+[A-C])\s*\n"
    r"Thesis Score:\s*(\d+)\s*\n"
    r"Contextualization Score:\s*(\d+)\s*\n"
    r"Evidence Score:\s*(\d+)\s*\n"
    r"Analysis and Reasoning Score:\s*(\d+)\s*\n"
    r"Total Score:\s*(\d+)",
    re.IGNORECASE,
)

SAMPLE_HEADER_PATTERNS = (
    SAMPLE_HEADER_CLASSIC,
    SAMPLE_HEADER_2025,
    SAMPLE_HEADER_SHORT,
)

ROW_COMMENTARY_PATTERNS = {
    "thesis": re.compile(
        r"A\.\s*Thesis/Claim[^:]*:\s*\d+\s*\n(.*?)(?=B\.\s*Contextualization|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    "contextualization": re.compile(
        r"B\.\s*Contextualization[^:]*:\s*\d+\s*\n(.*?)(?=C\.\s*Evidence|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    "evidence": re.compile(
        r"C\.\s*Evidence[^:]*:\s*\d+\s*\n(.*?)(?=D\.\s*Analysis|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    "analysis_reasoning": re.compile(
        r"D\.\s*Analysis and Reasoning[^:]*:\s*\d+\s*\n(.*?)(?=Sample:\s*\d+[A-C]|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
}


@dataclass
class RawAPCSample:
    sample_id: str
    prompt: str
    essay: str
    scores: dict[str, int]
    total_score: int
    commentary_by_row: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SampleHeaderMatch:
    sample_id: str
    scores: dict[str, int]
    total_score: int
    start: int
    end: int


def parse_apc_pdf(path: Path) -> list[RawAPCSample]:
    """Extract structured samples from an AP Central LEQ PDF."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF parsing. Install with: pip install apush-frq-grader-slm[ingest]"
        ) from exc

    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    metadata = _metadata_from_filename(path)
    return parse_apc_text(text, metadata=metadata)


def parse_apc_text(text: str, *, metadata: dict[str, Any] | None = None) -> list[RawAPCSample]:
    """Parse plain text extracted from an AP Central LEQ PDF."""
    meta = dict(metadata or {})
    prompt = _extract_prompt(text)
    commentary_text = _extract_commentary_section(text)
    essay_map = _extract_essays(text, meta.get("leq_num"))
    headers = _find_sample_headers(commentary_text)
    samples: list[RawAPCSample] = []

    for idx, header in enumerate(headers):
        block_end = headers[idx + 1].start if idx + 1 < len(headers) else len(commentary_text)
        block = commentary_text[header.start : block_end]
        commentary_by_row = _extract_row_commentary(block)
        essay = essay_map.get(header.sample_id, "")
        essay_source = "pdf_text"
        if not essay or len(essay) < 80:
            essay = _reconstruct_essay_from_commentary(block)
            essay_source = "commentary_quotes"
        sample_meta = {
            **meta,
            "sample_id": header.sample_id,
            "source": _source_tag(meta, header.sample_id),
            "essay_source": essay_source,
        }
        samples.append(
            RawAPCSample(
                sample_id=header.sample_id,
                prompt=prompt,
                essay=_clean_essay(essay),
                scores=header.scores,
                total_score=header.total_score,
                commentary_by_row=commentary_by_row,
                metadata=sample_meta,
            )
        )
    return samples


def _metadata_from_filename(path: Path) -> dict[str, Any]:
    match = re.search(
        r"ap(\d{2})-apc-us-history-leq(\d)-set-(\d)",
        path.name,
        re.IGNORECASE,
    )
    if not match:
        return {"filename": path.name}
    year_suffix, leq_num, set_num = match.groups()
    year = 2000 + int(year_suffix)
    return {
        "year": year,
        "leq_num": int(leq_num),
        "set": int(set_num),
        "filename": path.name,
        "url": f"https://apcentral.collegeboard.org/media/pdf/{path.name}",
    }


def _source_tag(meta: dict[str, Any], sample_id: str) -> str:
    year = meta.get("year", "unknown")
    leq_num = meta.get("leq_num", "x")
    set_num = meta.get("set", "x")
    return f"ap_central_{year}_leq{leq_num}_set{set_num}_{sample_id}"


def _extract_prompt(text: str) -> str:
    matches = PROMPT_PATTERN.findall(text)
    if not matches:
        question_match = re.search(
            r"Question \d+:\s*Long Essay Question,\s*([^\n]+)\n",
            text,
            re.IGNORECASE,
        )
        if question_match:
            topic = question_match.group(1).strip()
            return f"Long Essay Question about {topic}."
        raise ValueError("Could not locate LEQ prompt in PDF text")
    prompt = matches[0].strip()
    return re.sub(r"\s+", " ", prompt)


def _extract_commentary_section(text: str) -> str:
    markers = list(re.finditer(r"Scoring Commentary", text, re.IGNORECASE))
    best_start: int | None = None
    best_score = -1
    for marker in markers:
        window = text[marker.start() : marker.start() + 12000]
        sample_count = len(re.findall(r"Sample:\s*\d+[A-C]", window))
        score = sample_count
        if "Note: Student samples" in window[:800]:
            score += 10
        if score > best_score:
            best_score = score
            best_start = marker.start()

    if best_start is not None and best_score > 0:
        return text[best_start:]

    sample_match = re.search(r"Sample:\s*\d+[A-C]", text)
    if sample_match:
        return text[sample_match.start() :]

    if markers:
        return text[markers[0].start() :]
    raise ValueError("Could not locate scoring commentary section")


def _find_sample_headers(commentary_text: str) -> list[_SampleHeaderMatch]:
    headers: list[_SampleHeaderMatch] = []
    for pattern in SAMPLE_HEADER_PATTERNS:
        for match in pattern.finditer(commentary_text):
            headers.append(
                _SampleHeaderMatch(
                    sample_id=match.group(1).upper(),
                    scores={
                        "thesis": int(match.group(2)),
                        "contextualization": int(match.group(3)),
                        "evidence": int(match.group(4)),
                        "analysis_reasoning": int(match.group(5)),
                    },
                    total_score=int(match.group(6)),
                    start=match.start(),
                    end=match.end(),
                )
            )
        if headers:
            break
    return headers


def _extract_essays(text: str, leq_num: int | None) -> dict[str, str]:
    commentary_start = _commentary_boundary(text)
    essay_section = text[:commentary_start]
    essays: dict[str, str] = {}

    by_sample_header = _extract_essays_by_sample_header(essay_section)
    essays.update(by_sample_header)

    if not essays:
        by_page_marker = _extract_essays_by_page_marker(essay_section)
        essays.update(by_page_marker)

    return {sample_id: body for sample_id, body in essays.items() if body.strip()}


def _commentary_boundary(text: str) -> int:
    markers = list(re.finditer(r"Scoring Commentary", text, re.IGNORECASE))
    best_start: int | None = None
    best_score = -1
    for marker in markers:
        window = text[marker.start() : marker.start() + 12000]
        sample_count = len(re.findall(r"Sample:\s*\d+[A-C]", window))
        score = sample_count
        if "Note: Student samples" in window[:800]:
            score += 10
        if score > best_score:
            best_score = score
            best_start = marker.start()
    if best_start is not None and best_score > 0:
        return best_start
    sample_match = re.search(r"Sample:\s*\d+[A-C]", text)
    if sample_match:
        return sample_match.start()
    return len(text)


def _extract_essays_by_sample_header(essay_section: str) -> dict[str, str]:
    markers = list(SAMPLE_ESSAY_PATTERN.finditer(essay_section))
    if not markers:
        return {}

    grouped: dict[str, list[tuple[int, int]]] = {}
    for idx, marker in enumerate(markers):
        sample_id = marker.group(1).upper()
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(essay_section)
        grouped.setdefault(sample_id, []).append((start, end))

    essays: dict[str, str] = {}
    for sample_id, spans in grouped.items():
        parts = [essay_section[start:end].strip() for start, end in spans]
        essays[sample_id] = "\n\n".join(part for part in parts if part)
    return essays


def _extract_essays_by_page_marker(essay_section: str) -> dict[str, str]:
    rubric_end = essay_section.rfind("Additional Notes:")
    if rubric_end == -1:
        rubric_end = 0
    body = essay_section[rubric_end:]
    markers = list(PAGE_MARKER_PATTERN.finditer(body))
    if not markers:
        return {}

    first_gap = markers[0].start() - 0
    if first_gap > 200:
        return _extract_essays_marker_prefix(body, markers)
    return _extract_essays_marker_suffix(body, markers)


def _extract_essays_marker_prefix(body: str, markers: list[re.Match[str]]) -> dict[str, str]:
    essays: dict[str, str] = {}
    for idx, marker in enumerate(markers):
        sample_id = f"{marker.group(1)}{marker.group(2).upper()}"
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(body)
        chunk = body[start:end].strip()
        if len(chunk) > len(essays.get(sample_id, "")):
            essays[sample_id] = chunk
    return essays


def _extract_essays_marker_suffix(body: str, markers: list[re.Match[str]]) -> dict[str, str]:
    if not markers:
        return {}
    prefix = body[: markers[0].start()].strip()
    if len(prefix) < 200:
        return {}

    sample_order = []
    seen: set[str] = set()
    for marker in markers:
        sample_id = f"{marker.group(1)}{marker.group(2).upper()}"
        if sample_id not in seen:
            sample_order.append(sample_id)
            seen.add(sample_id)

    if len(sample_order) <= 1:
        return {sample_order[0]: prefix} if sample_order else {}

    chunks = re.split(r"\n{2,}", prefix)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 120]
    essays: dict[str, str] = {}
    if len(chunks) >= len(sample_order):
        for sample_id, chunk in zip(sample_order, chunks[-len(sample_order) :], strict=False):
            essays[sample_id] = chunk
    else:
        size = max(len(prefix) // len(sample_order), 1)
        for idx, sample_id in enumerate(sample_order):
            essays[sample_id] = prefix[idx * size : (idx + 1) * size].strip()
    return essays


def _reconstruct_essay_from_commentary(block: str) -> str:
    quotes = re.findall(r'["\u201c]([^\u201d"]{12,})[\u201d"]', block)
    sentences = re.findall(
        r"(?:states|notes|claims|argues|identifies|describes|explains),?\s+([^\.]{20,200}\.)",
        block,
        flags=re.IGNORECASE,
    )
    parts = quotes + sentences
    if not parts:
        return ""
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = part.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(part.strip())
    return " ".join(unique)


def _extract_row_commentary(block: str) -> dict[str, str]:
    commentary: dict[str, str] = {}
    for row, pattern in ROW_COMMENTARY_PATTERNS.items():
        match = pattern.search(block)
        if match:
            commentary[row] = _clean_commentary(match.group(1))
    return commentary


def _clean_essay(text: str) -> str:
    cleaned = re.sub(
        r"Sample\s+\d+[A-C]\s+\d+\s+of\s+\d+\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Page\s+\d+\s+of\s+\d+\s+\d+\s*\n?\s*[A-C]\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _clean_commentary(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"© \d{4} College Board.*", "", cleaned).strip()
    return cleaned


def _normalize_sample_id(sample_id: str) -> str:
    return sample_id.upper().replace(" ", "")
