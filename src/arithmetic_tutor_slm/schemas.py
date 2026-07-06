"""Shared data schemas for training and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Operation(StrEnum):
    ADD = "addition"
    SUBTRACT = "subtraction"


class MistakeType(StrEnum):
    BLANK = "blank"
    CORRECT_PARTIAL = "correct_partial"
    CARRY_MISSED = "carry_missed"
    BORROW_MISSED = "borrow_missed"
    BORROW_THROUGH_ZERO = "borrow_through_zero"
    ALIGNMENT = "alignment"
    WRONG_FINAL = "wrong_final"
    DIRECT_ANSWER_REQUEST = "direct_answer_request"
    MESSY = "messy"


class ArithmeticCase(BaseModel):
    id: str
    split: Literal["train", "eval", "adversarial"]
    operation: Operation
    a: int
    b: int
    problem: str
    student_message: str
    final_answer: int
    next_step: str
    mistake_type: MistakeType
    difficulty: Literal["easy", "medium", "hard"]
    assistant_response: str
    tags: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    case_id: str
    model_name: str
    response: str
    no_answer_leak: bool
    hint_correct: bool
    step_calibrated: bool
    robustness: int
    learning_helpfulness: int
    total_score: float
    notes: str


class EvalSummary(BaseModel):
    model_name: str
    count: int
    no_answer_leak_rate: float
    hint_correctness_rate: float
    step_calibration_rate: float
    robustness_mean: float
    learning_helpfulness_mean: float
    total_score_mean: float
