"""Small arithmetic engine used to generate ground-truth tutor states."""

from __future__ import annotations

from dataclasses import dataclass

from arithmetic_tutor_slm.schemas import Operation


@dataclass(frozen=True)
class StepInfo:
    final_answer: int
    next_step: str
    focus_digit: int
    requires_carry_or_borrow: bool


def format_problem(a: int, b: int, operation: Operation) -> str:
    symbol = "+" if operation == Operation.ADD else "-"
    return f"{a} {symbol} {b}"


def solve(a: int, b: int, operation: Operation) -> int:
    return a + b if operation == Operation.ADD else a - b


def first_step(a: int, b: int, operation: Operation) -> StepInfo:
    """Return a human-readable first column step without exposing the final answer."""
    ones_a = a % 10
    ones_b = b % 10
    if operation == Operation.ADD:
        total = ones_a + ones_b
        carry = total >= 10
        if carry:
            step = (
                f"Start in the ones column: {ones_a} + {ones_b} makes a two-digit number. "
                "What digit stays in the ones place, and what gets carried?"
            )
        else:
            step = (
                f"Start in the ones column: what is {ones_a} + {ones_b}, and where should "
                "that digit go?"
            )
        return StepInfo(solve(a, b, operation), step, ones_a, carry)

    borrow = ones_a < ones_b
    if borrow:
        step = (
            f"Look at the ones column: {ones_a} is smaller than {ones_b}. "
            "Where can you borrow from before subtracting?"
        )
    else:
        step = (
            f"Start in the ones column: what is {ones_a} - {ones_b}, and where should "
            "that digit go?"
        )
    return StepInfo(solve(a, b, operation), step, ones_a, borrow)


def contains_borrow_through_zero(a: int, b: int) -> bool:
    """Detect subtraction cases such as 407 - 168 where borrowing crosses zero."""
    if a < b:
        return False
    a_digits = [int(d) for d in str(a)][::-1]
    b_digits = [int(d) for d in str(b)][::-1]
    for idx, b_digit in enumerate(b_digits):
        a_digit = a_digits[idx] if idx < len(a_digits) else 0
        if a_digit < b_digit and idx + 1 < len(a_digits) and a_digits[idx + 1] == 0:
            return True
    return False


def vertical_problem(a: int, b: int, operation: Operation) -> str:
    symbol = "+" if operation == Operation.ADD else "-"
    width = max(len(str(a)), len(str(b))) + 2
    return f"{a:>{width}}\n{symbol}{b:>{width - 1}}\n{'-' * width}"
