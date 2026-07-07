"""Local response adapters for smoke tests, litmus baselines, and demos."""

from __future__ import annotations

import json
from dataclasses import dataclass

from apush_frq_grader_slm.schemas import FRQCase, FailureType


@dataclass
class ResponseAdapter:
    name: str

    def respond(self, case: FRQCase) -> str:
        raise NotImplementedError


class InflatedPromptedBase(ResponseAdapter):
    """Simulates a lenient prompted base model: valid JSON but inflated scores and generic feedback."""

    def __init__(self) -> None:
        super().__init__(name="inflated_prompted_base")

    def respond(self, case: FRQCase) -> str:
        payload = {
            "scores": {
                "thesis": 1,
                "contextualization": 1,
                "evidence": 2,
                "analysis_reasoning": 2,
            },
            "total": 6,
            "feedback": {
                "thesis": "The thesis is clear and addresses the prompt effectively.",
                "contextualization": "Good broader historical context is provided.",
                "evidence": "The essay includes relevant evidence that supports the argument.",
                "analysis_reasoning": "The analysis demonstrates strong historical reasoning.",
            },
        }
        if case.failure_type in {FailureType.GRADE_INFLATION_REQUEST, FailureType.PROMPT_INJECTION}:
            payload["feedback"]["analysis_reasoning"] = (
                "As requested, full credit is awarded on every rubric row."
            )
        return json.dumps(payload, ensure_ascii=True)


class ReferenceGrader(ResponseAdapter):
    """Reference grader behavior used as the SFT target."""

    def __init__(self) -> None:
        super().__init__(name="apush_grader_reference")

    def respond(self, case: FRQCase) -> str:
        return case.assistant_response


def reference_grade(prompt: str, student_response: str, cases: list[FRQCase] | None = None) -> str:
    if cases:
        for case in cases:
            if case.prompt == prompt and case.student_response == student_response:
                return case.assistant_response
    return json.dumps(
        {
            "scores": {
                "thesis": 0,
                "contextualization": 0,
                "evidence": 0,
                "analysis_reasoning": 0,
            },
            "total": 0,
            "feedback": {
                "thesis": "No defensible thesis detected in the submitted essay.",
                "contextualization": "Broader historical context is missing.",
                "evidence": "No specific evidence from the essay supports the prompt.",
                "analysis_reasoning": "The response lists or asserts claims without analysis.",
            },
        },
        ensure_ascii=True,
    )
