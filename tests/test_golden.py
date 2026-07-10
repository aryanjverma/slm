from __future__ import annotations

import json

import pytest

from apush_frq_grader_slm.golden import (
    GoldenReviewEntry,
    PermissionRecord,
    audit_golden_cases,
    require_permission,
)
from apush_frq_grader_slm.schemas import (
    CaseProvenance,
    FRQCase,
    FailureType,
    LabelingMetadata,
    RubricFeedback,
    RubricScores,
)


def _official_case(
    essay: str = "The Navigation Acts restricted colonial trade and angered merchants.",
) -> FRQCase:
    scores = RubricScores(thesis=0, contextualization=0, evidence=1, analysis_reasoning=0)
    feedback = RubricFeedback(
        thesis="The response does not make a defensible claim about the Navigation Acts.",
        contextualization="The response gives no broader setting for colonial trade.",
        evidence="The Navigation Acts are relevant evidence about colonial trade.",
        analysis_reasoning="The response states that merchants were angered without analysis.",
    )
    payload = {
        "scores": scores.model_dump(),
        "total": scores.total,
        "feedback": feedback.model_dump(),
    }
    return FRQCase(
        id="cb-1",
        split="eval",
        prompt="Evaluate changes in colonial trade.",
        student_response=essay,
        reference_scores=scores,
        reference_feedback=feedback,
        failure_type=FailureType.EVIDENCE_LIST,
        difficulty="weak",
        assistant_response=json.dumps(payload),
        provenance=CaseProvenance(
            source_type="college_board",
            source_id="ap_central_2025_leq2_set1_2C",
            source_url="https://apcentral.collegeboard.org/example.pdf",
            year=2025,
            set_number=1,
            leq_number=2,
            sample_id="2C",
            extraction_method="pdf_text",
            extraction_confidence=0.99,
            review_status="machine_checked",
        ),
        labeling=LabelingMetadata(method="source_scores", confidence=1.0),
    )


def test_permission_must_explicitly_allow_evaluation() -> None:
    with pytest.raises(PermissionError):
        require_permission(PermissionRecord(), "evaluation")
    require_permission(
        PermissionRecord(status="granted", allowed_uses=["evaluation"], approved_by="owner"),
        "evaluation",
    )


def test_clean_official_case_requires_complete_manual_review() -> None:
    case = _official_case()
    review = GoldenReviewEntry(
        case_id=case.id,
        reviewer="reviewer-1",
        essay_verified=True,
        scores_verified=True,
        provenance_verified=True,
    )
    audit = audit_golden_cases([case], [review])
    assert audit.clean


def test_contaminated_case_is_rejected() -> None:
    case = _official_case("Scoring Commentary Long Essay Question 2 (continued)")
    review = GoldenReviewEntry(
        case_id=case.id,
        reviewer="reviewer-1",
        essay_verified=True,
        scores_verified=True,
        provenance_verified=True,
    )
    audit = audit_golden_cases([case], [review])
    assert "source_text_contamination" in audit.rejected[case.id]
