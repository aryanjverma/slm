"""Parse College Board AP Central LEQ sample PDFs into structured records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROMPT_PATTERN = re.compile(
    r"((?:Evaluate|Explain|Compare|Describe|Analyze)\s+(?:the\s+)?"
    r"(?:extent to which|relative importance|impact|causes and effects|how)[^.\n]{10,260}\.)",
    re.IGNORECASE,
)

PAGE_MARKER_PATTERN = re.compile(
    r"^[ \t]*Page[ \t]+(?P<page>\d+)[ \t]+of[ \t]+(?P<page_count>\d+)"
    r"[ \t]+(?P<question>\d+)[ \t]*(?:\r?\n[ \t]*)?(?P<label>[A-C])[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

SAMPLE_ESSAY_PATTERN = re.compile(
    r"^[ \t]*Sample[ \t]+(?P<sample_id>\d+[ \t]*[A-C])[ \t]+"
    r"(?P<page>\d+)[ \t]+of[ \t]+(?P<page_count>\d+)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

ESSAY_CONTAMINATION_PATTERNS = (
    ("scoring_commentary", re.compile(r"\bScoring\s+Commentary\b", re.IGNORECASE)),
    (
        "commentary_page_header",
        re.compile(
            r"(?:^|\n)[ \t]*Long\s+Essay\s+Question\s+\d+"
            r"(?:\s*\(continued\))?[ \t]*(?:\n|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "ap_page_header",
        re.compile(
            r"\bAP(?:®|\s+)?\s*United\s+States\s+History\s+20\d{2}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sample_score_header",
        re.compile(
            r"(?:^|\n)[ \t]*Sample:\s*\d+[A-C]\b|"
            r"(?:^|\n)[ \t]*(?:Thesis(?:/Claim)?|Contextualization|Evidence|"
            r"Analysis\s+and\s+Reasoning|Total)(?:\s+Score)?:\s*\d+\b",
            re.IGNORECASE,
        ),
    ),
    (
        "commentary_row_label",
        re.compile(
            r"(?:^|\n)[ \t]*[A-D]\.[ \t]+(?:Thesis/Claim|Contextualization|Evidence|"
            r"Analysis\s+and\s+Reasoning)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "document_page_marker",
        re.compile(
            r"(?:^|\n)[ \t]*(?:Page[ \t]+\d+[ \t]+of[ \t]+\d+[ \t]+\d+"
            r"[ \t]*(?:\r?\n[ \t]*)?[A-C]|Sample[ \t]+\d+[ \t]*[A-C]"
            r"[ \t]+\d+[ \t]+of[ \t]+\d+)[ \t]*(?:\n|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "copyright_footer",
        re.compile(
            r"©\s*20\d{2}\s+College\s+Board|"
            r"Visit\s+College\s+Board\s+on\s+the\s+web|"
            r"\bapcentral\.collegeboard\.org\b|\bcollegeboard\.org\b",
            re.IGNORECASE,
        ),
    ),
    (
        "commentary_boilerplate",
        re.compile(
            r"Student\s+samples\s+are\s+quoted\s+verbatim|"
            r"\bThe\s+response\s+(?:earned|did\s+not\s+earn)\b|"
            r"\bearned\s+(?:0|1|2|one|two)\s+points?\b",
            re.IGNORECASE,
        ),
    ),
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
    essay_source: str = ""
    parser_confidence: float = 0.0

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        essay_source = self.essay_source or str(metadata.get("essay_source", "unknown"))
        confidence_value = metadata.get("parser_confidence", self.parser_confidence)
        parser_confidence = float(confidence_value)
        if not 0.0 <= parser_confidence <= 1.0:
            raise ValueError("parser_confidence must be between 0.0 and 1.0")
        metadata["essay_source"] = essay_source
        metadata["parser_confidence"] = parser_confidence
        self.metadata = metadata
        self.essay_source = essay_source
        self.parser_confidence = parser_confidence


class EssayExtractionError(ValueError):
    """Raised when a complete student essay cannot be extracted safely."""


class EssayContaminationError(ValueError):
    """Raised when non-student document text appears in an extracted essay."""

    def __init__(self, sample_id: str, markers: tuple[str, ...]) -> None:
        marker_text = ", ".join(markers)
        super().__init__(f"Rejected contaminated essay {sample_id}: {marker_text}")
        self.sample_id = sample_id
        self.markers = markers


@dataclass
class _SampleHeaderMatch:
    sample_id: str
    scores: dict[str, int]
    total_score: int
    start: int
    end: int


@dataclass(frozen=True)
class _EssayExtraction:
    text: str
    layout: str
    confidence: float


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
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return parse_apc_text(text, metadata=metadata)


def parse_apc_text(text: str, *, metadata: dict[str, Any] | None = None) -> list[RawAPCSample]:
    """Parse plain text extracted from an AP Central LEQ PDF."""
    meta = dict(metadata or {})
    prompt = _extract_prompt(text)
    commentary_text = _extract_commentary_section(text)
    essay_map = _extract_essays(text, meta.get("leq_num"))
    headers = _find_sample_headers(commentary_text)
    if not headers:
        raise EssayExtractionError("Could not locate sample score headers in scoring commentary")
    samples: list[RawAPCSample] = []

    for idx, header in enumerate(headers):
        block_end = headers[idx + 1].start if idx + 1 < len(headers) else len(commentary_text)
        block = commentary_text[header.start : block_end]
        commentary_by_row = _extract_row_commentary(block)
        extraction = essay_map.get(header.sample_id)
        if extraction is None or not extraction.text.strip():
            raise EssayExtractionError(
                f"Could not extract complete PDF essay for sample {header.sample_id}; "
                "commentary reconstruction is prohibited"
            )
        reject_contaminated_essay(extraction.text, sample_id=header.sample_id)
        essay = _clean_essay(extraction.text)
        reject_contaminated_essay(essay, sample_id=header.sample_id)
        sample_meta = {
            **meta,
            "sample_id": header.sample_id,
            "source": _source_tag(meta, header.sample_id),
            "essay_source": "pdf_text",
            "parser_confidence": extraction.confidence,
            "extraction_layout": extraction.layout,
        }
        samples.append(
            RawAPCSample(
                sample_id=header.sample_id,
                prompt=prompt,
                essay=essay,
                scores=header.scores,
                total_score=header.total_score,
                commentary_by_row=commentary_by_row,
                metadata=sample_meta,
                essay_source="pdf_text",
                parser_confidence=extraction.confidence,
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
        if score >= best_score:
            best_score = score
            best_start = marker.start()

    if best_start is not None and best_score > 0:
        line_start = text.rfind("\n", 0, best_start) + 1
        return text[line_start:]

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


def detect_essay_contamination(text: str) -> tuple[str, ...]:
    """Return stable marker names for document or commentary text in an essay."""
    return tuple(name for name, pattern in ESSAY_CONTAMINATION_PATTERNS if pattern.search(text))


def reject_contaminated_essay(text: str, *, sample_id: str = "unknown") -> None:
    """Reject essay text containing any known non-student contamination marker."""
    markers = detect_essay_contamination(text)
    if markers:
        raise EssayContaminationError(sample_id, markers)


def _extract_essays(text: str, leq_num: int | None) -> dict[str, _EssayExtraction]:
    commentary_start = _commentary_boundary(text)
    essay_section = text[:commentary_start]
    candidates = (
        _extract_essays_by_sample_header(essay_section, leq_num),
        _extract_essays_by_page_marker(essay_section, leq_num),
    )
    return max(
        candidates,
        key=lambda essays: (len(essays), sum(len(item.text) for item in essays.values())),
    )


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
        if score >= best_score:
            best_score = score
            best_start = marker.start()
    if best_start is not None and best_score > 0:
        return text.rfind("\n", 0, best_start) + 1
    sample_match = re.search(r"Sample:\s*\d+[A-C]", text)
    if sample_match:
        return sample_match.start()
    return len(text)


def _extract_essays_by_sample_header(
    essay_section: str,
    leq_num: int | None,
) -> dict[str, _EssayExtraction]:
    markers = list(SAMPLE_ESSAY_PATTERN.finditer(essay_section))
    return _collect_marked_essays(
        essay_section,
        markers,
        sample_id=lambda marker: _normalize_sample_id(marker.group("sample_id")),
        leq_num=leq_num,
        layout="sample_page_header",
        base_confidence=0.99,
    )


def _extract_essays_by_page_marker(
    essay_section: str,
    leq_num: int | None,
) -> dict[str, _EssayExtraction]:
    markers = list(PAGE_MARKER_PATTERN.finditer(essay_section))
    return _collect_marked_essays(
        essay_section,
        markers,
        sample_id=lambda marker: (
            f"{marker.group('question')}{marker.group('label').upper()}"
        ),
        leq_num=leq_num,
        layout="page_marker",
        base_confidence=0.98,
    )


def _collect_marked_essays(
    essay_section: str,
    markers: list[re.Match[str]],
    *,
    sample_id: Callable[[re.Match[str]], str],
    leq_num: int | None,
    layout: str,
    base_confidence: float,
) -> dict[str, _EssayExtraction]:
    grouped: dict[str, list[tuple[int, int, str]]] = {}
    for idx, marker in enumerate(markers):
        current_id = sample_id(marker)
        if leq_num is not None and not current_id.startswith(str(leq_num)):
            continue
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(essay_section)
        chunk = essay_section[marker.end() : end].strip()
        if chunk:
            grouped.setdefault(current_id, []).append(
                (int(marker.group("page")), int(marker.group("page_count")), chunk)
            )

    essays: dict[str, _EssayExtraction] = {}
    for current_id, pages in grouped.items():
        expected_count = max(item[1] for item in pages)
        page_text: dict[int, str] = {}
        for page_number, _, chunk in pages:
            if len(chunk) > len(page_text.get(page_number, "")):
                page_text[page_number] = chunk
        if set(page_text) != set(range(1, expected_count + 1)):
            continue
        ordered_text = (page_text[page_number] for page_number in range(1, expected_count + 1))
        essays[current_id] = _EssayExtraction(
            text="\n\n".join(ordered_text),
            layout=layout,
            confidence=base_confidence,
        )
    return essays


def _extract_row_commentary(block: str) -> dict[str, str]:
    commentary: dict[str, str] = {}
    for row, pattern in ROW_COMMENTARY_PATTERNS.items():
        match = pattern.search(block)
        if match:
            commentary[row] = _clean_commentary(match.group(1))
    return commentary


def _clean_essay(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_commentary(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"© \d{4} College Board.*", "", cleaned).strip()
    return cleaned


def _normalize_sample_id(sample_id: str) -> str:
    return sample_id.upper().replace(" ", "")
