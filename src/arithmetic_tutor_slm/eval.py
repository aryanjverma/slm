"""Deterministic evaluation harness for tutor behavior."""

from __future__ import annotations

from collections.abc import Iterable

from arithmetic_tutor_slm.baselines import ResponseAdapter
from arithmetic_tutor_slm.filters import is_calibrated_hint, leaks_final_answer
from arithmetic_tutor_slm.schemas import ArithmeticCase, EvalResult, EvalSummary, MistakeType


NEXT_STEP_KEYWORDS = {
    "ones": ["ones", "rightmost"],
    "carry": ["carry", "carried"],
    "borrow": ["borrow", "lend"],
    "line": ["line", "place", "column", "vertical"],
    "zero": ["zero", "moving left"],
}


def score_response(case: ArithmeticCase, response: str, model_name: str) -> EvalResult:
    lowered = response.lower()
    no_leak = not leaks_final_answer(response, case.final_answer)
    calibrated = is_calibrated_hint(response)
    hint_correct = _hint_matches_next_step_for_case(case, lowered)
    robustness = 2 if no_leak and calibrated else 1 if no_leak else 0
    helpfulness = 2 if hint_correct and no_leak else 1 if no_leak else 0
    total = (
        float(no_leak)
        + float(hint_correct)
        + float(calibrated)
        + robustness / 2
        + helpfulness / 2
    ) / 5
    notes = "ok"
    if not no_leak:
        notes = "leaked final answer"
    elif not hint_correct:
        notes = "hint did not match expected next step"
    elif not calibrated:
        notes = "hint not calibrated"
    return EvalResult(
        case_id=case.id,
        model_name=model_name,
        response=response,
        no_answer_leak=no_leak,
        hint_correct=hint_correct,
        step_calibrated=calibrated,
        robustness=robustness,
        learning_helpfulness=helpfulness,
        total_score=round(total, 4),
        notes=notes,
    )


def evaluate_adapter(cases: Iterable[ArithmeticCase], adapter: ResponseAdapter) -> list[EvalResult]:
    return [score_response(case, adapter.respond(case), adapter.name) for case in cases]


def summarize(results: list[EvalResult], model_name: str) -> EvalSummary:
    count = len(results)
    if count == 0:
        return EvalSummary(
            model_name=model_name,
            count=0,
            no_answer_leak_rate=0,
            hint_correctness_rate=0,
            step_calibration_rate=0,
            robustness_mean=0,
            learning_helpfulness_mean=0,
            total_score_mean=0,
        )
    return EvalSummary(
        model_name=model_name,
        count=count,
        no_answer_leak_rate=_mean(r.no_answer_leak for r in results),
        hint_correctness_rate=_mean(r.hint_correct for r in results),
        step_calibration_rate=_mean(r.step_calibrated for r in results),
        robustness_mean=_mean(r.robustness for r in results),
        learning_helpfulness_mean=_mean(r.learning_helpfulness for r in results),
        total_score_mean=_mean(r.total_score for r in results),
    )


def _hint_matches_next_step_for_case(case: ArithmeticCase, response: str) -> bool:
    if case.mistake_type == MistakeType.CARRY_MISSED:
        return "carry" in response or "carried" in response
    if case.mistake_type in {MistakeType.BORROW_MISSED, MistakeType.BORROW_THROUGH_ZERO}:
        return "borrow" in response or "lend" in response
    if case.mistake_type == MistakeType.ALIGNMENT:
        return any(word in response for word in ["line", "place", "column", "vertical"])
    if case.mistake_type == MistakeType.DIRECT_ANSWER_REQUEST:
        return "won't" in response or "ones" in response or "column" in response
    return _hint_matches_next_step(case.next_step, response)


def _hint_matches_next_step(next_step: str, response: str) -> bool:
    expected = next_step.lower()
    for family in NEXT_STEP_KEYWORDS.values():
        if any(word in expected for word in family):
            return any(word in response for word in family)
    return "column" in response or "start" in response


def _mean(values: Iterable[bool | int | float]) -> float:
    numbers = [float(value) for value in values]
    return round(sum(numbers) / len(numbers), 4)
