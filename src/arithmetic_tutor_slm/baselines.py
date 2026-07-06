"""Local response adapters for smoke tests and demos."""

from __future__ import annotations

import re
from dataclasses import dataclass

from arithmetic_tutor_slm.arithmetic import first_step, format_problem, solve
from arithmetic_tutor_slm.schemas import ArithmeticCase, Operation


@dataclass
class ResponseAdapter:
    name: str

    def respond(self, case: ArithmeticCase) -> str:
        raise NotImplementedError


class LeakyPromptedBase(ResponseAdapter):
    """A deliberately plausible base-model failure mode: helpful, but answer-leaky."""

    def __init__(self) -> None:
        super().__init__(name="prompted_base_leaky")

    def respond(self, case: ArithmeticCase) -> str:
        return (
            f"The answer is {case.final_answer}. To get there, work from the ones column "
            "and handle any carry or borrow."
        )


class SocraticTutorReference(ResponseAdapter):
    """Reference tutor behavior used for smoke tests before a real fine-tuned model exists."""

    def __init__(self) -> None:
        super().__init__(name="socratic_tutor_reference")

    def respond(self, case: ArithmeticCase) -> str:
        return case.assistant_response


def parse_problem(text: str) -> tuple[int, int, Operation] | None:
    match = re.search(r"(\d+)\s*([+-])\s*(\d+)", text)
    if not match:
        return None
    a = int(match.group(1))
    b = int(match.group(3))
    operation = Operation.ADD if match.group(2) == "+" else Operation.SUBTRACT
    if operation == Operation.SUBTRACT and b > a:
        return None
    return a, b, operation


def reference_reply(user_message: str) -> str:
    parsed = parse_problem(user_message)
    if parsed is None:
        return "Please send one addition or subtraction problem, and tell me where you are stuck."
    a, b, operation = parsed
    answer = solve(a, b, operation)
    if re.search(rf"(?<!\d){answer}(?!\d)", user_message):
        return "Yes, that final answer is correct. Can you explain how you handled the ones column?"
    if "answer" in user_message.lower() or "just tell" in user_message.lower():
        return "I won't give the final answer yet. What do you get when you start with the ones column?"
    step = first_step(a, b, operation)
    return step.next_step


def format_reference_problem(a: int, b: int, operation: Operation) -> str:
    return format_problem(a, b, operation)
