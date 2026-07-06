"""Quality gates for tutor examples and model outputs."""

from __future__ import annotations

import re

from arithmetic_tutor_slm.schemas import ArithmeticCase


def leaks_final_answer(text: str, final_answer: int) -> bool:
    answer = str(final_answer)
    return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", text) is not None


def too_many_result_digits(text: str, final_answer: int) -> bool:
    digits = set(str(abs(final_answer)))
    if not digits:
        return False
    text_digits = [char for char in text if char in digits]
    return len(text_digits) >= max(3, len(digits) + 1)


def is_calibrated_hint(text: str) -> bool:
    text = text.strip()
    sentence_count = len(re.findall(r"[.!?]", text))
    return 20 <= len(text) <= 260 and sentence_count <= 3


def passes_quality_gate(case: ArithmeticCase) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if leaks_final_answer(case.assistant_response, case.final_answer):
        reasons.append("leaks_final_answer")
    if too_many_result_digits(case.assistant_response, case.final_answer):
        reasons.append("too_many_result_digits")
    if not is_calibrated_hint(case.assistant_response):
        reasons.append("not_calibrated")
    if "?" not in case.assistant_response and "try" not in case.assistant_response.lower():
        reasons.append("not_socratic")
    return not reasons, reasons
