from __future__ import annotations

from apush_frq_grader_slm.rubric import (
    RubricVersion,
    get_leq_rubric,
    rubric_version_for_year,
)
from apush_frq_grader_slm.schemas import CaseProvenance, FRQCase


def test_rubric_versions_follow_year_boundary() -> None:
    assert rubric_version_for_year(2023) == RubricVersion.LEQ_2023
    assert rubric_version_for_year(2024) == RubricVersion.LEQ_2024_2026
    assert "four" in get_leq_rubric(RubricVersion.LEQ_2024_2026)["analysis_reasoning"]["2"]


def test_existing_case_without_provenance_remains_compatible() -> None:
    case = FRQCase.model_validate(
        {
            "id": "legacy",
            "split": "train",
            "prompt": "Evaluate change.",
            "student_response": "A short response about change in the period.",
            "reference_scores": {
                "thesis": 0,
                "contextualization": 0,
                "evidence": 0,
                "analysis_reasoning": 0,
            },
            "reference_feedback": {
                "thesis": "No thesis about change appears.",
                "contextualization": "No broader period context appears.",
                "evidence": "No specific evidence appears.",
                "analysis_reasoning": "No historical reasoning appears.",
            },
            "failure_type": "weak_thesis",
            "difficulty": "weak",
            "assistant_response": "{}",
        }
    )
    assert case.provenance == CaseProvenance()
