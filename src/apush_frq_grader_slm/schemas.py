"""Shared data schemas for training and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from apush_frq_grader_slm.rubric import compute_total


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
