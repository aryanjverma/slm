from __future__ import annotations

import json

import pytest

from apush_frq_grader_slm.dataset_v2 import (
    SyntheticReviewEntry,
    audit_case_collection,
    apply_human_reviews,
    assemble_training_mix,
    audit_training_cases,
    select_human_review_sample,
    write_training_artifacts,
)
from apush_frq_grader_slm.schemas import (
    CaseProvenance,
    FRQCase,
    FailureType,
    LabelingMetadata,
    RubricFeedback,
    RubricScores,
)


def _case(case_id: str, essay: str, failure_type: FailureType = FailureType.STRONG) -> FRQCase:
    scores = RubricScores(thesis=1, contextualization=1, evidence=2, analysis_reasoning=1)
    feedback = RubricFeedback(
        thesis=f"The claim about {essay.split()[0]} establishes a defensible argument.",
        contextualization=f"The discussion of {essay.split()[1]} supplies broader context.",
        evidence=f"Specific evidence about {essay.split()[2]} supports the argument.",
        analysis_reasoning=f"The response connects {essay.split()[3]} to historical change.",
    )
    spans = {
        "thesis": [essay.split()[0]],
        "contextualization": [essay.split()[1]],
        "evidence": [essay.split()[2]],
        "analysis_reasoning": [essay.split()[3]],
    }
    return FRQCase(
        id=case_id,
        split="train",
        prompt="Evaluate historical change in this period.",
        student_response=essay,
        reference_scores=scores,
        reference_feedback=feedback,
        failure_type=failure_type,
        difficulty="strong",
        assistant_response=json.dumps(
            {
                "scores": scores.model_dump(),
                "total": scores.total,
                "feedback": feedback.model_dump(),
            }
        ),
        provenance=CaseProvenance(
            source_type="synthetic", prompt_family_id=f"family-{case_id}", generator_name="test"
        ),
        labeling=LabelingMetadata(
            method="independent_consensus",
            grader_ids=["a", "b"],
            agreement=1.0,
            confidence=1.0,
            human_reviewed=True,
            feedback_spans=spans,
        ),
    )


def test_audit_rejects_exact_duplicates() -> None:
    essay = (
        "Markets canals factories labor changed regional society through sustained "
        "economic growth."
    )
    audit = audit_training_cases([_case("a", essay), _case("b", essay)])
    assert not audit.clean
    assert audit.exact_duplicate_count == 1


def test_mix_is_deterministic_and_respects_target() -> None:
    realistic = [
        _case(
            f"r-{index}",
            f"Markets{index} canals{index} factories{index} labor{index} changed region.",
        )
        for index in range(8)
    ]
    adversarial = [
        _case(
            f"a-{index}",
            f"Taxes{index} protests{index} merchants{index} parliament{index} changed colony.",
            FailureType.PROMPT_INJECTION,
        )
        for index in range(4)
    ]
    first = assemble_training_mix(realistic, adversarial, target_count=10, seed=7)
    second = assemble_training_mix(realistic, adversarial, target_count=10, seed=7)
    assert [case.id for case in first] == [case.id for case in second]
    assert len(first) == 10


def test_mix_skips_repeated_feedback_candidates() -> None:
    realistic = [
        _case(
            f"r-{index}",
            f"Markets{index} canals{index} factories{index} labor{index} changed region.",
        )
        for index in range(2)
    ]
    adversarial = [
        _case(
            "a-duplicate-1",
            "Taxes protests merchants parliament changed colony alpha.",
            FailureType.PROMPT_INJECTION,
        ),
        _case(
            "a-duplicate-2",
            "Taxes protests merchants parliament changed colony beta.",
            FailureType.PROMPT_INJECTION,
        ),
        _case(
            "a-unique",
            "Tariffs petitions artisans congress reshaped republic gamma.",
            FailureType.PROMPT_INJECTION,
        ),
    ]

    mix = assemble_training_mix(
        realistic,
        adversarial,
        target_count=4,
        adversarial_ratio=0.5,
        seed=7,
    )

    assert len(mix) == 4
    assert {case.id for case in mix}.isdisjoint({"a-duplicate-2"})
    assert audit_training_cases(mix).repeated_feedback_count == 0


def test_manifest_refuses_overwrite(tmp_path) -> None:
    case = _case("unique", "Markets canals factories labor changed regional society substantially.")
    audit = audit_training_cases([case])
    assert audit.clean
    manifest = write_training_artifacts(tmp_path, [case], audit, seed=13, settings={})
    assert len(manifest.artifacts) == 3
    assert all("\\" not in artifact.path for artifact in manifest.artifacts)
    assert all(
        b"\r\n" not in (tmp_path / artifact.path).read_bytes()
        for artifact in manifest.artifacts
    )
    with pytest.raises(FileExistsError):
        write_training_artifacts(tmp_path, [case], audit, seed=13, settings={})


def test_review_sample_and_application() -> None:
    cases = [
        _case(f"r-{index}", f"Markets canals factories labor changed region number {index}.")
        for index in range(20)
    ]
    selected = select_human_review_sample(cases, rate=0.10, seed=7)
    assert len(selected) == 2
    review = SyntheticReviewEntry(
        case_id=selected[0].id,
        reviewer="human-1",
        scores_verified=True,
        feedback_verified=True,
        historical_accuracy_verified=True,
    )
    apply_human_reviews(cases, [review])
    assert selected[0].labeling.human_reviewed


def test_eval_collection_rejects_train_split() -> None:
    case = _case("train-only", "Markets canals factories labor changed one region substantially.")
    audit = audit_case_collection([case], allowed_splits={"eval"})
    assert not audit.clean
    assert "unexpected_split" in audit.rejected[case.id]
