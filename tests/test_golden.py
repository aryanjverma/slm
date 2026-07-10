from __future__ import annotations

import hashlib
import json

import pytest

from apush_frq_grader_slm.golden import (
    GoldenReviewEntry,
    PermissionRecord,
    audit_golden_cases,
    require_permission,
    split_official_cases,
    write_official_artifacts,
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
            file_sha256="a" * 64,
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


def _official_split_cases() -> list[FRQCase]:
    cases: list[FRQCase] = []
    for set_number, group_sizes in ((1, [3] * 9), (2, [3] * 8 + [2])):
        for group_index, group_size in enumerate(group_sizes):
            for sample_index in range(group_size):
                case = _official_case(
                    f"Unique essay for set {set_number}, group {group_index}, sample {sample_index}."
                ).model_copy(deep=True)
                case.id = f"set{set_number}-group{group_index}-sample{sample_index}"
                case.prompt = f"Official prompt set {set_number}, group {group_index}."
                case.provenance.source_id = case.id
                case.provenance.year = 2023 + (group_index % 2)
                case.provenance.set_number = set_number
                case.provenance.leq_number = 2 + (group_index % 3)
                case.provenance.sample_id = f"{case.provenance.leq_number}{'ABC'[sample_index]}"
                case.provenance.review_status = "human_verified"
                case.labeling.human_reviewed = True
                cases.append(case)
    return cases


def test_official_split_is_27_dev_26_final_with_no_prompt_overlap() -> None:
    dev, final = split_official_cases(_official_split_cases())
    assert len(dev) == 27
    assert len(final) == 26
    assert {case.provenance.set_number for case in dev} == {1}
    assert {case.provenance.set_number for case in final} == {2}
    assert {case.prompt for case in dev}.isdisjoint({case.prompt for case in final})


def test_official_artifact_manifest_has_portable_hashes_and_counts(tmp_path) -> None:
    manifest = write_official_artifacts(tmp_path, _official_split_cases())
    assert {name: row["count"] for name, row in manifest["artifacts"].items()} == {
        "combined": 53,
        "dev": 27,
        "final": 26,
    }
    for row in manifest["artifacts"].values():
        assert "\\" not in row["path"]
        path = tmp_path / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_official_artifact_writer_rejects_unverified_rows(tmp_path) -> None:
    cases = _official_split_cases()
    cases[0].labeling.human_reviewed = False
    with pytest.raises(ValueError, match="unverified rows"):
        write_official_artifacts(tmp_path, cases)


def test_official_split_rejects_prompt_overlap() -> None:
    cases = _official_split_cases()
    cases[-1].prompt = cases[0].prompt
    with pytest.raises(ValueError, match="prompt overlap"):
        split_official_cases(cases)
