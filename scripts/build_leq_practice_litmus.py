"""Extract the nine human-scored LEQs from LEQ_Grading_Practice.docx."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ESSAY_RE = re.compile(r"^Essay (?P<id>[1-3]\.[1-3])$")
PROMPT_RE = re.compile(r"^Prompt (?P<number>[1-3])$")
ROWS = {
    "Thesis / Claim (0-1)": "thesis",
    "Contextualization (0-1)": "contextualization",
    "Evidence (0-2)": "evidence",
    "Analysis & Reasoning (0-2)": "analysis_reasoning",
}


def extract_cases(source: Path) -> list[dict]:
    paragraphs = _paragraphs(source)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    prompts: dict[str, str] = {}
    cases: list[dict] = []

    for index, paragraph in enumerate(paragraphs):
        prompt_match = PROMPT_RE.fullmatch(paragraph)
        if prompt_match:
            prompts[prompt_match.group("number")] = _next_text(paragraphs, index + 1)[1]
            continue
        essay_match = ESSAY_RE.fullmatch(paragraph)
        if not essay_match:
            continue

        essay_id = essay_match.group("id")
        prompt_number = essay_id.split(".", 1)[0]
        score_heading = paragraphs.index("Score using the rubric below:", index + 1)
        essay = "\n\n".join(text for text in paragraphs[index + 1 : score_heading] if text)
        scores: dict[str, int] = {}
        feedback: dict[str, str] = {}
        for row_index in range(score_heading + 1, len(paragraphs)):
            label = paragraphs[row_index]
            if label.startswith("TOTAL (0-6)"):
                total = int(_next_text(paragraphs, row_index + 1)[1])
                break
            criterion = ROWS.get(label)
            if criterion:
                score_index, score = _next_text(paragraphs, row_index + 1)
                _, note = _next_text(paragraphs, score_index + 1)
                scores[criterion] = int(score)
                feedback[criterion] = note
        else:  # pragma: no cover - malformed source guard
            raise ValueError(f"Missing total for Essay {essay_id}")

        if set(scores) != set(ROWS.values()) or sum(scores.values()) != total:
            raise ValueError(f"Inconsistent rubric rows for Essay {essay_id}")
        difficulty = "strong" if total >= 5 else "weak" if total <= 2 else "borderline"
        if difficulty == "strong":
            failure_type = "strong"
        elif scores["contextualization"] == 0:
            failure_type = "missing_context"
        else:
            failure_type = "evidence_list"
        expected = {"scores": scores, "total": total, "feedback": feedback}
        cases.append(
            {
                "id": f"leq_practice_{essay_id.replace('.', '_')}",
                "split": "eval",
                "prompt": prompts[prompt_number],
                "student_response": essay,
                "reference_scores": scores,
                "reference_feedback": feedback,
                "failure_type": failure_type,
                "difficulty": difficulty,
                "assistant_response": json.dumps(expected, ensure_ascii=True),
                "tags": ["external", "human_scored", "leq_grading_practice"],
                "provenance": {
                    "source_type": "external",
                    "source_id": f"LEQ_Grading_Practice.docx#Essay-{essay_id}",
                    "file_sha256": source_sha256,
                    "rubric_version": "2024_2026_leq",
                    "extraction_method": "docx_xml",
                    "extraction_confidence": 1.0,
                    "prompt_family_id": f"leq_grading_practice_prompt_{prompt_number}",
                    "review_status": "human_verified",
                },
                "labeling": {
                    "method": "source_scores",
                    "grader_ids": ["document_reference_grader"],
                    "confidence": 1.0,
                    "adjudicated": False,
                    "human_reviewed": True,
                    "protocol_version": "leq_grading_practice_v1",
                    "resolution": "Rubric scores and justifications transcribed from source document.",
                },
            }
        )

    if len(cases) != 9 or set(prompts) != {"1", "2", "3"}:
        raise ValueError(f"Expected three prompts and nine essays; got {len(prompts)} and {len(cases)}")
    return cases


def _paragraphs(source: Path) -> list[str]:
    with ZipFile(source) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return [
        "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)).strip()
        for paragraph in root.findall(".//w:p", WORD_NS)
    ]


def _next_text(paragraphs: list[str], start: int) -> tuple[int, str]:
    for index in range(start, len(paragraphs)):
        if paragraphs[index]:
            return index, paragraphs[index]
    raise ValueError("Unexpected end of DOCX")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("LEQ_Grading_Practice.docx"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/leq_grading_practice_v1.jsonl")
    )
    args = parser.parse_args()
    cases = extract_cases(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
