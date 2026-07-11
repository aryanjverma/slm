import json
from pathlib import Path

import pytest

from scripts.render_judging_markdown import (
    build_essay_index,
    build_judgment_index,
    discover_jsonl_files,
    render_markdown,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_render_finds_essay_and_formats_grade(tmp_path: Path) -> None:
    judging = tmp_path / "judging" / "reader.jsonl"
    source = tmp_path / "raw_essays_v4.jsonl"
    _write_jsonl(
        judging,
        [{
            "task_id": "case-1",
            "scores": {"thesis": 1, "contextualization": 0, "evidence": 1, "analysis_reasoning": 0},
            "total": 2,
            "feedback": {"thesis": "Defensible claim.", "contextualization": "No context.", "evidence": "One example.", "analysis_reasoning": "No reasoning."},
            "evidence_spans": {"thesis": ["A clear claim"]},
            "confidence": 0.8,
        }],
    )
    _write_jsonl(
        source,
        [{"task_id": "case-1", "prompt": "Evaluate change.", "student_response": "A clear claim.\n\nOne example."}],
    )
    paths = discover_jsonl_files(judging, search_roots=[tmp_path])
    markdown = render_markdown(
        [json.loads(judging.read_text().strip())],
        build_essay_index(paths),
        title="Review",
    )
    assert "# Review" in markdown
    assert "Evaluate change." in markdown
    assert "> A clear claim." in markdown
    assert "| Thesis | 1 | Defensible claim. |" in markdown
    assert "**Total:** 2/6" in markdown


def test_render_uses_consensus_grade(tmp_path: Path) -> None:
    source = tmp_path / "packets.jsonl"
    _write_jsonl(source, [{"task_id": "case-2", "student_response": "Essay text."}])
    row = {
        "task_id": "case-2",
        "status": "accepted",
        "accepted_grade": {
            "scores": {"thesis": 1, "contextualization": 1, "evidence": 2, "analysis_reasoning": 1},
            "total": 5,
            "feedback": {},
        },
    }
    markdown = render_markdown([row], build_essay_index([source]), title="Consensus")
    assert "### Consensus Grade" in markdown
    assert "**Status:** accepted" in markdown
    assert "**Total:** 5/6" in markdown


def test_conflicting_essay_sources_fail(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"task_id": "same", "student_response": "First essay."}])
    _write_jsonl(second, [{"task_id": "same", "student_response": "Second essay."}])
    with pytest.raises(ValueError, match="Conflicting essays"):
        build_essay_index([first, second])


def test_retry_round_prefers_matching_source_prefix(tmp_path: Path) -> None:
    judging = tmp_path / "retry2_reader_a_part_1.jsonl"
    original = tmp_path / "grading_packet_part_1.jsonl"
    retry = tmp_path / "retry2_adjudication_packet_part_1.jsonl"
    _write_jsonl(judging, [{"task_id": "same", "scores": {}, "total": 0}])
    _write_jsonl(original, [{"task_id": "same", "student_response": "Original essay."}])
    _write_jsonl(retry, [{"task_id": "same", "student_response": "Retry essay."}])

    index = build_essay_index(
        [original, retry],
        preferred_path=judging,
    )

    assert index["same"].essay == "Retry essay."


def test_unrequested_conflicts_are_ignored(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"task_id": "irrelevant", "student_response": "First."}])
    _write_jsonl(second, [{"task_id": "irrelevant", "student_response": "Second."}])

    assert build_essay_index([first, second], required_ids={"wanted"}) == {}


def test_missing_essay_fails_by_default() -> None:
    with pytest.raises(KeyError, match="Could not locate"):
        render_markdown([{"task_id": "missing"}], {}, title="Missing")


def test_review_row_joins_consensus_and_reader_feedback(tmp_path: Path) -> None:
    essay_source = tmp_path / "grading_packets.jsonl"
    judgment_source = tmp_path / "resolved_grades.jsonl"
    _write_jsonl(
        essay_source,
        [{"task_id": "reviewed", "prompt": "Evaluate change.", "student_response": "Essay."}],
    )
    reader_grade = {
        "grader_id": "reader_a:offline",
        "scores": {"thesis": 1, "contextualization": 0, "evidence": 1, "analysis_reasoning": 0},
        "total": 2,
        "feedback": {"thesis": "Reader feedback."},
        "evidence_spans": {"thesis": ["Essay"]},
        "confidence": 0.8,
    }
    _write_jsonl(
        judgment_source,
        [{
            "task_id": "reviewed",
            "accepted_grade": {**reader_grade, "grader_id": "consensus"},
            "reader_grades": [reader_grade],
        }],
    )
    paths = [essay_source, judgment_source]
    markdown = render_markdown(
        [{"case_id": "reviewed", "reviewer": ""}],
        build_essay_index(paths),
        judgment_index=build_judgment_index(paths),
        title="Human Review",
    )
    assert "### Consensus Grade" in markdown
    assert "### Reader A:Offline" in markdown
    assert "Reader feedback." in markdown
    assert "**Evidence spans**" in markdown
