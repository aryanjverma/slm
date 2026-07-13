from __future__ import annotations

import json
from pathlib import Path

from apush_frq_grader_slm.eval import score_response
from apush_frq_grader_slm.schemas import FRQCase
from scripts.build_leq_practice_litmus import extract_cases
from scripts.compare_leq_practice import compare, render_markdown


ROOT = Path(__file__).resolve().parents[1]


def test_docx_extracts_expected_nine_reference_totals() -> None:
    rows = extract_cases(ROOT / "LEQ_Grading_Practice.docx")

    assert [sum(row["reference_scores"].values()) for row in rows] == [4, 6, 2, 5, 3, 1, 6, 4, 3]
    assert len({row["id"] for row in rows}) == 9


def test_comparison_reports_tuned_mae_reduction() -> None:
    cases = [
        FRQCase.model_validate(json.loads(line))
        for line in (ROOT / "artifacts/data/leq_grading_practice_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    base = [score_response(case, _response(0, 0, 0, 0), "base") for case in cases]
    tuned = [
        score_response(
            case,
            _response(
                case.reference_scores.thesis,
                case.reference_scores.contextualization,
                case.reference_scores.evidence,
                case.reference_scores.analysis_reasoning,
            ),
            "tuned",
        )
        for case in cases
    ]

    report = compare(cases, base, tuned)

    assert report["tuned"]["total_mae"] == 0
    assert report["improvement"]["total_mae_reduction"] > 0
    assert "Directional external-validity check only" in render_markdown(report)


def _response(thesis: int, context: int, evidence: int, analysis: int) -> str:
    scores = {
        "thesis": thesis,
        "contextualization": context,
        "evidence": evidence,
        "analysis_reasoning": analysis,
    }
    return json.dumps(
        {
            "scores": scores,
            "total": sum(scores.values()),
            "feedback": {criterion: "Essay-grounded feedback." for criterion in scores},
        }
    )
