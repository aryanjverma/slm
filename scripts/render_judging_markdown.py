"""Render judging JSONL records with their source essays as readable Markdown."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ID_KEYS = ("task_id", "case_id", "id")
ESSAY_KEYS = ("student_response", "essay", "text")
PROMPT_KEYS = ("prompt", "prompt_text")
CRITERIA = ("thesis", "contextualization", "evidence", "analysis_reasoning")
ROLE_TOKENS = {
    "adjudication",
    "adjudicator",
    "assembly",
    "cases",
    "grade",
    "grades",
    "grading",
    "packet",
    "packets",
    "raw",
    "reader",
    "rejects",
    "resolved",
    "validated",
    "validation",
}


@dataclass(frozen=True)
class EssayRecord:
    item_id: str
    essay: str
    prompt: str
    source: Path


@dataclass(frozen=True)
class JudgmentRecord:
    item_id: str
    row: dict[str, Any]
    source: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def row_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def discover_jsonl_files(
    input_path: Path,
    *,
    sources: list[Path] | None = None,
    search_roots: list[Path] | None = None,
) -> list[Path]:
    files: list[Path] = [input_path]
    files.extend(sources or [])
    roots = list(search_roots or [])
    roots.append(input_path.parent)
    for root in roots:
        if root.is_file() and root.suffix == ".jsonl":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.jsonl")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _name_tokens(path: Path) -> list[str]:
    return re.findall(r"[a-z]+\d*|\d+", path.stem.lower())


def _run_prefix(path: Path) -> tuple[str, ...]:
    prefix: list[str] = []
    for token in _name_tokens(path):
        if token in ROLE_TOKENS:
            break
        prefix.append(token)
    return tuple(prefix)


def _source_priority(preferred_path: Path | None, source_path: Path) -> tuple[int, int]:
    if preferred_path is None:
        return (0, 0)
    preferred_prefix = _run_prefix(preferred_path)
    source_prefix = _run_prefix(source_path)
    prefix_match = int(preferred_prefix == source_prefix)
    shared_tokens = len(set(_name_tokens(preferred_path)) & set(_name_tokens(source_path)))
    return (prefix_match, shared_tokens)


def _grade_richness(row: dict[str, Any]) -> int:
    if isinstance(row.get("accepted_grade"), dict):
        return 5
    if isinstance(row.get("reader_grades"), list):
        return 4
    if isinstance(row.get("reader_a"), dict) or isinstance(row.get("reader_b"), dict):
        return 3
    if isinstance(row.get("reference_scores"), dict):
        return 2
    if isinstance(row.get("scores"), dict):
        return 1
    return 0


def build_essay_index(
    paths: Iterable[Path],
    *,
    preferred_path: Path | None = None,
    required_ids: set[str] | None = None,
) -> dict[str, EssayRecord]:
    index: dict[str, EssayRecord] = {}
    priorities: dict[str, tuple[int, int]] = {}
    for path in paths:
        for row in read_jsonl(path):
            item_id = row_id(row)
            essay = row_text(row, ESSAY_KEYS)
            if not item_id or not essay:
                continue
            if required_ids is not None and item_id not in required_ids:
                continue
            prompt = row_text(row, PROMPT_KEYS)
            existing = index.get(item_id)
            if existing is not None and existing.essay != essay:
                existing_priority = priorities[item_id]
                new_priority = _source_priority(preferred_path, path)
                if new_priority < existing_priority:
                    continue
                if new_priority == existing_priority:
                    raise ValueError(
                        f"Conflicting essays for {item_id}: {existing.source} and {path}. "
                        "Pass the intended file with --source to disambiguate."
                    )
            if existing is None or (not existing.prompt and prompt):
                index[item_id] = EssayRecord(item_id, essay, prompt, path)
                priorities[item_id] = _source_priority(preferred_path, path)
            elif existing.essay != essay:
                index[item_id] = EssayRecord(item_id, essay, prompt, path)
                priorities[item_id] = _source_priority(preferred_path, path)
    return index


def build_judgment_index(
    paths: Iterable[Path],
    *,
    preferred_path: Path | None = None,
    required_ids: set[str] | None = None,
) -> dict[str, JudgmentRecord]:
    index: dict[str, JudgmentRecord] = {}
    priorities: dict[str, tuple[int, int, int]] = {}
    for path in paths:
        for row in read_jsonl(path):
            item_id = row_id(row)
            richness = _grade_richness(row)
            if not item_id or not richness:
                continue
            if required_ids is not None and item_id not in required_ids:
                continue
            affinity = _source_priority(preferred_path, path)
            priority = (richness, affinity[0], affinity[1])
            if item_id not in index or priority > priorities[item_id]:
                index[item_id] = JudgmentRecord(item_id, row, path)
                priorities[item_id] = priority
    return index


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _blockquote(text: str) -> str:
    lines: list[str] = []
    for paragraph in text.replace("\r\n", "\n").split("\n"):
        lines.append(f"> {paragraph}" if paragraph else ">")
    return "\n".join(lines)


def _grade_sections(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    sections: list[tuple[str, dict[str, Any]]] = []
    accepted = row.get("accepted_grade")
    if isinstance(accepted, dict):
        sections.append(("Consensus Grade", accepted))
    elif isinstance(row.get("reference_scores"), dict):
        scores = row["reference_scores"]
        labeling = row.get("labeling") if isinstance(row.get("labeling"), dict) else {}
        sections.append(
            (
                "Consensus Grade",
                {
                    "scores": scores,
                    "total": sum(int(scores.get(key, 0)) for key in CRITERIA),
                    "feedback": row.get("reference_feedback", {}),
                    "evidence_spans": labeling.get("feedback_spans", {}),
                    "confidence": labeling.get("confidence"),
                    "grader_id": ", ".join(labeling.get("grader_ids", [])),
                },
            )
        )
    elif isinstance(row.get("scores"), dict):
        sections.append(("Grade", row))
    reader_grades = row.get("reader_grades")
    if isinstance(reader_grades, list):
        for index, grade in enumerate(reader_grades, start=1):
            if isinstance(grade, dict):
                grader_id = str(grade.get("grader_id") or f"Reader {index}")
                sections.append((grader_id.replace("_", " ").title(), grade))
    for key, title in (
        ("reader_a", "Reader A"),
        ("reader_b", "Reader B"),
        ("adjudicator_grade", "Adjudicator"),
    ):
        grade = row.get(key)
        if isinstance(grade, dict):
            sections.append((title, grade))
    return sections


def _render_grade(title: str, grade: dict[str, Any]) -> list[str]:
    scores = grade.get("scores") if isinstance(grade.get("scores"), dict) else {}
    feedback = grade.get("feedback") if isinstance(grade.get("feedback"), dict) else {}
    spans = grade.get("evidence_spans") if isinstance(grade.get("evidence_spans"), dict) else {}
    total = grade.get("total")
    lines = [f"### {title}", ""]
    metadata = []
    if total is not None:
        metadata.append(f"**Total:** {total}/6")
    if grade.get("confidence") is not None:
        metadata.append(f"**Confidence:** {float(grade['confidence']):.2f}")
    if grade.get("grader_id"):
        metadata.append(f"**Grader:** `{grade['grader_id']}`")
    if metadata:
        lines.extend((" · ".join(metadata), ""))
    if scores:
        lines.extend(("| Criterion | Score | Feedback |", "|---|---:|---|"))
        for criterion in CRITERIA:
            if criterion not in scores:
                continue
            lines.append(
                f"| {_escape_table(criterion.replace('_', ' ').title())} "
                f"| {_escape_table(scores[criterion])} "
                f"| {_escape_table(feedback.get(criterion, ''))} |"
            )
        lines.append("")
    rendered_spans = []
    for criterion in CRITERIA:
        values = spans.get(criterion)
        if isinstance(values, list) and values:
            rendered_spans.append(
                f"- **{criterion.replace('_', ' ').title()}:** "
                + "; ".join(f"“{str(value).strip()}”" for value in values)
            )
    if rendered_spans:
        lines.extend(("**Evidence spans**", "", *rendered_spans, ""))
    return lines


def render_markdown(
    judging_rows: list[dict[str, Any]],
    essay_index: dict[str, EssayRecord],
    *,
    title: str,
    allow_missing: bool = False,
    judgment_index: dict[str, JudgmentRecord] | None = None,
) -> str:
    lines = [f"# {title}", "", f"Judging records: **{len(judging_rows)}**", ""]
    for position, row in enumerate(judging_rows, start=1):
        item_id = row_id(row)
        if not item_id:
            raise ValueError(f"Judging row {position} has no task_id, case_id, or id")
        direct_essay = row_text(row, ESSAY_KEYS)
        direct_prompt = row_text(row, PROMPT_KEYS)
        located = essay_index.get(item_id)
        if direct_essay:
            essay = direct_essay
            prompt = direct_prompt or (located.prompt if located else "")
            source = Path("<judging row>")
        elif located:
            essay, prompt, source = located.essay, direct_prompt or located.prompt, located.source
        elif allow_missing:
            essay, prompt, source = "_Essay not found._", direct_prompt, Path("<missing>")
        else:
            raise KeyError(f"Could not locate an essay for {item_id}")

        lines.extend((f"## {position}. `{item_id}`", ""))
        if row.get("status"):
            lines.extend((f"**Status:** {_escape_table(row['status'])}", ""))
        if row.get("reasons"):
            reasons = row["reasons"] if isinstance(row["reasons"], list) else [row["reasons"]]
            lines.extend(("**Reasons:** " + ", ".join(map(str, reasons)), ""))
        lines.extend((f"**Essay source:** `{source.as_posix()}`", ""))
        if prompt:
            lines.extend(("### Prompt", "", prompt, ""))
        lines.extend(("### Essay", "", _blockquote(essay), ""))
        grade_row = row
        judgment = (judgment_index or {}).get(item_id)
        if not _grade_sections(grade_row) and judgment is not None:
            grade_row = judgment.row
            lines.extend((f"**Judgment source:** `{judgment.source.as_posix()}`", ""))
        for section_title, grade in _grade_sections(grade_row):
            lines.extend(_render_grade(section_title, grade))
        lines.extend(("---", ""))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".md")
    judging_rows = read_jsonl(input_path)
    required_ids = {item_id for row in judging_rows if (item_id := row_id(row))}
    search_roots = list(args.search_root)
    if any(item_id.startswith("v4-") for item_id in required_ids):
        v4_root = Path("artifacts/data/v4")
        if v4_root.exists():
            search_roots.append(v4_root)
    source_files = discover_jsonl_files(
        input_path,
        sources=args.source,
        search_roots=search_roots,
    )
    essay_index = build_essay_index(
        source_files,
        preferred_path=input_path,
        required_ids=required_ids,
    )
    judgment_index = build_judgment_index(
        source_files,
        preferred_path=input_path,
        required_ids=required_ids,
    )
    markdown = render_markdown(
        judging_rows,
        essay_index,
        title=args.title or f"Judging Review: {input_path.name}",
        allow_missing=args.allow_missing,
        judgment_index=judgment_index,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(
        f"Wrote {len(judging_rows)} records to {output_path} "
        f"using {len(source_files)} searched JSONL files"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Judging JSONL to render")
    parser.add_argument("--output", type=Path, help="Markdown output path")
    parser.add_argument("--title", help="Markdown document title")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="Explicit essay-source JSONL; may be repeated",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="Directory to search recursively for essay JSONL files; may be repeated",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Render a placeholder instead of failing when an essay cannot be found",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
