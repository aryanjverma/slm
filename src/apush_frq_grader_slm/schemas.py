"""Shared data schemas for training and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from apush_frq_grader_slm.rubric import compute_total
from apush_frq_grader_slm.rubric import DEFAULT_RUBRIC_VERSION, RubricVersion


class FailureType(StrEnum):
    WEAK_THESIS = "weak_thesis"
    MISSING_CONTEXT = "missing_context"
    EVIDENCE_LIST = "evidence_list"
    WRONG_PERIOD = "wrong_period"
    BORDERLINE_COMPLEXITY = "borderline_complexity"
    GRADE_INFLATION_REQUEST = "grade_inflation_request"
    PROMPT_INJECTION = "prompt_injection"
    STRONG = "strong"


class RubricScores(BaseModel):
    thesis: int = Field(ge=0, le=1)
    contextualization: int = Field(ge=0, le=1)
    evidence: int = Field(ge=0, le=2)
    analysis_reasoning: int = Field(ge=0, le=2)

    @property
    def total(self) -> int:
        return compute_total(self)


class RubricFeedback(BaseModel):
    thesis: str
    contextualization: str
    evidence: str
    analysis_reasoning: str


class CaseProvenance(BaseModel):
    source_type: Literal["synthetic", "college_board", "external", "unknown"] = "unknown"
    source_id: str = ""
    source_url: str = ""
    file_sha256: str = ""
    year: int | None = None
    set_number: int | None = None
    leq_number: int | None = None
    sample_id: str = ""
    rubric_version: RubricVersion = DEFAULT_RUBRIC_VERSION
    extraction_method: str = ""
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    prompt_family_id: str = ""
    generator_name: str = ""
    generator_config: dict[str, Any] = Field(default_factory=dict)
    review_status: Literal["unreviewed", "machine_checked", "human_verified", "rejected"] = (
        "unreviewed"
    )


class LabelingMetadata(BaseModel):
    method: Literal[
        "rule_based", "source_scores", "independent_consensus", "adjudicated", "unknown"
    ] = "unknown"
    grader_ids: list[str] = Field(default_factory=list)
    agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    adjudicated: bool = False
    human_reviewed: bool = False
    feedback_spans: dict[str, list[str]] = Field(default_factory=dict)
    protocol_version: str = ""
    resolution: str = ""
    criterion_agreement: dict[str, bool] = Field(default_factory=dict)
    generation_target_total: int | None = Field(default=None, ge=0, le=6)
    target_distance: int | None = Field(default=None, ge=0, le=6)


class FRQCase(BaseModel):
    id: str
    split: Literal["train", "eval", "adversarial"]
    prompt: str
    student_response: str
    reference_scores: RubricScores
    reference_feedback: RubricFeedback
    failure_type: FailureType
    difficulty: Literal["weak", "borderline", "strong"]
    assistant_response: str
    tags: list[str] = Field(default_factory=list)
    provenance: CaseProvenance = Field(default_factory=CaseProvenance)
    labeling: LabelingMetadata = Field(default_factory=LabelingMetadata)

    @field_validator("reference_scores", mode="before")
    @classmethod
    def _coerce_scores(cls, value: RubricScores | dict) -> RubricScores:
        if isinstance(value, RubricScores):
            return value
        return RubricScores.model_validate(value)

    @field_validator("reference_feedback", mode="before")
    @classmethod
    def _coerce_feedback(cls, value: RubricFeedback | dict) -> RubricFeedback:
        if isinstance(value, RubricFeedback):
            return value
        return RubricFeedback.model_validate(value)


class EvalResult(BaseModel):
    case_id: str
    model_name: str
    response: str
    structured_output_valid: bool
    rubric_accuracy: float
    evidence_grounding: bool
    no_hallucination: bool
    robustness: int
    total_score: float
    notes: str


class EvalSummary(BaseModel):
    model_name: str
    count: int
    structured_output_valid_rate: float
    rubric_accuracy_mean: float
    evidence_grounding_rate: float
    no_hallucination_rate: float
    robustness_mean: float
    total_score_mean: float
