"""Synthetic data generation for Socratic addition/subtraction tutoring."""

from __future__ import annotations

import random
from collections.abc import Iterable

from arithmetic_tutor_slm.arithmetic import (
    contains_borrow_through_zero,
    first_step,
    format_problem,
    solve,
    vertical_problem,
)
from arithmetic_tutor_slm.schemas import ArithmeticCase, MistakeType, Operation


def build_case(
    *,
    case_id: str,
    split: str,
    a: int,
    b: int,
    operation: Operation,
    mistake_type: MistakeType,
) -> ArithmeticCase:
    info = first_step(a, b, operation)
    problem = format_problem(a, b, operation)
    final_answer = solve(a, b, operation)
    difficulty = _difficulty(a, b, operation, mistake_type)
    student_message = _student_message(a, b, operation, mistake_type, final_answer)
    assistant_response = _assistant_response(a, b, operation, mistake_type, info.next_step)
    tags = [operation.value, mistake_type.value, difficulty]
    if operation == Operation.SUBTRACT and contains_borrow_through_zero(a, b):
        tags.append("borrow_through_zero")
    if info.requires_carry_or_borrow:
        tags.append("regrouping")

    return ArithmeticCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        operation=operation,
        a=a,
        b=b,
        problem=problem,
        student_message=student_message,
        final_answer=final_answer,
        next_step=info.next_step,
        mistake_type=mistake_type,
        difficulty=difficulty,
        assistant_response=assistant_response,
        tags=tags,
    )


def generate_cases(
    *,
    count: int,
    split: str,
    seed: int = 13,
    adversarial_ratio: float = 0.1,
) -> list[ArithmeticCase]:
    rng = random.Random(seed)
    cases: list[ArithmeticCase] = []
    for idx in range(count):
        operation = rng.choice([Operation.ADD, Operation.SUBTRACT])
        a, b = _numbers(rng, operation)
        mistake_type = _mistake_type(rng, operation, adversarial_ratio)
        case_id = f"{split}-{idx:05d}"
        cases.append(
            build_case(
                case_id=case_id,
                split=split,
                a=a,
                b=b,
                operation=operation,
                mistake_type=mistake_type,
            )
        )
    return cases


def to_chat_rows(cases: Iterable[ArithmeticCase]) -> list[dict]:
    rows = []
    for case in cases:
        rows.append(
            {
                "id": case.id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a Socratic tutor for addition and subtraction. Never give "
                            "the final answer unless the student has already produced it. Give "
                            "one short next-step hint or question."
                        ),
                    },
                    {"role": "user", "content": case.student_message},
                    {"role": "assistant", "content": case.assistant_response},
                ],
                "metadata": {
                    "problem": case.problem,
                    "final_answer": case.final_answer,
                    "next_step": case.next_step,
                    "tags": case.tags,
                },
            }
        )
    return rows


def _numbers(rng: random.Random, operation: Operation) -> tuple[int, int]:
    digits = rng.choices([2, 3, 4], weights=[0.35, 0.45, 0.2])[0]
    low = 10 ** (digits - 1)
    high = (10**digits) - 1
    a = rng.randint(low, high)
    b = rng.randint(low, high)
    if operation == Operation.SUBTRACT and b > a:
        a, b = b, a
    return a, b


def _mistake_type(rng: random.Random, operation: Operation, adversarial_ratio: float) -> MistakeType:
    if rng.random() < adversarial_ratio:
        return rng.choice([MistakeType.DIRECT_ANSWER_REQUEST, MistakeType.MESSY])
    if operation == Operation.ADD:
        return rng.choices(
            [
                MistakeType.BLANK,
                MistakeType.CORRECT_PARTIAL,
                MistakeType.CARRY_MISSED,
                MistakeType.ALIGNMENT,
                MistakeType.WRONG_FINAL,
            ],
            weights=[0.25, 0.25, 0.25, 0.1, 0.15],
        )[0]
    return rng.choices(
        [
            MistakeType.BLANK,
            MistakeType.CORRECT_PARTIAL,
            MistakeType.BORROW_MISSED,
            MistakeType.BORROW_THROUGH_ZERO,
            MistakeType.ALIGNMENT,
            MistakeType.WRONG_FINAL,
        ],
        weights=[0.2, 0.2, 0.25, 0.15, 0.1, 0.1],
    )[0]


def _difficulty(a: int, b: int, operation: Operation, mistake_type: MistakeType) -> str:
    if mistake_type in {MistakeType.BORROW_THROUGH_ZERO, MistakeType.ALIGNMENT}:
        return "hard"
    if operation == Operation.SUBTRACT and contains_borrow_through_zero(a, b):
        return "hard"
    if max(a, b) >= 100 or mistake_type in {MistakeType.CARRY_MISSED, MistakeType.BORROW_MISSED}:
        return "medium"
    return "easy"


def _student_message(
    a: int, b: int, operation: Operation, mistake_type: MistakeType, final_answer: int
) -> str:
    problem = format_problem(a, b, operation)
    if mistake_type == MistakeType.BLANK:
        return f"I need help with {problem}. I don't know where to start."
    if mistake_type == MistakeType.CORRECT_PARTIAL:
        return f"For {problem}, I started with the ones column. Am I on the right track?"
    if mistake_type == MistakeType.CARRY_MISSED:
        return f"I'm doing {problem}. I added the ones column but forgot what to do with the extra ten."
    if mistake_type == MistakeType.BORROW_MISSED:
        return f"For {problem}, the top ones digit is smaller, but I tried subtracting anyway."
    if mistake_type == MistakeType.BORROW_THROUGH_ZERO:
        return f"I'm stuck on this subtraction with a zero in the middle:\n{vertical_problem(a, b, operation)}"
    if mistake_type == MistakeType.ALIGNMENT:
        return f"I wrote {problem} sideways and I might be lining up the digits wrong. What should I check?"
    if mistake_type == MistakeType.WRONG_FINAL:
        offsets = [-11, -9, -1, 1, 9, 11]
        wrong = final_answer + offsets[abs(final_answer) % len(offsets)]
        return f"I think {problem} equals {wrong}. Can you check it?"
    if mistake_type == MistakeType.DIRECT_ANSWER_REQUEST:
        return f"Just tell me the answer to {problem}. I don't want hints."
    return f"pls help?? {problem}\nmy columns look weird and I am confused"


def _assistant_response(
    a: int,
    b: int,
    operation: Operation,
    mistake_type: MistakeType,
    next_step: str,
) -> str:
    if mistake_type == MistakeType.DIRECT_ANSWER_REQUEST:
        return "I won't give the answer yet. Let's do just the ones column first: what do you get there?"
    if mistake_type == MistakeType.ALIGNMENT:
        return "First check place value: are the ones digits lined up under each other?"
    if mistake_type == MistakeType.WRONG_FINAL:
        return "Let's verify one column at a time. What did you get in the ones column?"
    if mistake_type == MistakeType.CARRY_MISSED:
        return "You found a two-digit sum in one column. Which digit stays, and which digit gets carried?"
    if mistake_type == MistakeType.BORROW_MISSED:
        return "Before subtracting, compare the ones digits. Do you need to borrow from the next column?"
    if mistake_type == MistakeType.BORROW_THROUGH_ZERO:
        return "When the next column is zero, keep moving left until you find a place to borrow from. Which column can lend?"
    if mistake_type == MistakeType.MESSY:
        return "Rewrite it vertically first. Which digits belong in the ones column?"
    return next_step
