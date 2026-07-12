"""Interactively review the 60 private v5 cases and write a human approval receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from apush_frq_grader_slm.manual_review_v5 import (
    build_human_approval,
    create_review_backup,
    load_review_packet,
    review_status,
    set_review_decision,
    write_json_atomic,
    write_packet_atomic,
)
from apush_frq_grader_slm.rubric import CRITERIA, SCORE_RANGES

ACTION_OPTIONS = (
    ("a", "Accept", "Essay, scores, and feedback are all correct"),
    ("c", "Correct", "Change scores and/or grounded feedback"),
    ("r", "Reject", "Essay is unsuitable and must be replaced"),
    ("s", "Skip", "Leave this row unchanged for now"),
    ("b", "Back", "Return to the preceding displayed row"),
    ("h", "Help", "Show the detailed review instructions"),
    ("q", "Quit", "Save completed actions and exit"),
)


def main() -> None:
    args = parse_args()
    console = Console(no_color=args.plain, width=args.width)
    rows = load_review_packet(args.packet)
    reviewer = args.reviewer or Prompt.ask("Your name or reviewer ID", console=console).strip()
    if not reviewer:
        raise SystemExit("A reviewer name is required.")
    indices = select_indices(rows, reviewer, args)
    if not indices:
        console.print("No rows match the requested filter.")
        finish_review(console, rows, args, reviewer)
        return

    backup: Path | None = None
    position = 0
    while 0 <= position < len(indices):
        index = indices[position]
        row = rows[index]
        render_case(console, row, index=index, total=len(rows), reviewer=reviewer, plain=args.plain)
        render_action_menu(console, args.plain)
        command = Prompt.ask(
            "Enter action",
            choices=[key for key, _, _ in ACTION_OPTIONS],
            default="a",
            console=console,
            show_choices=False,
        )
        if command == "q":
            break
        if command == "h":
            render_help(console, args.plain)
            continue
        if command == "b":
            position = max(0, position - 1)
            continue
        if command == "s":
            position += 1
            continue

        if backup is None:
            backup = create_review_backup(args.packet, args.approval)
            console.print(f"Private backup created: {backup}")
        if command == "a":
            notes = optional_input(console, "Optional note")
            rows[index] = set_review_decision(
                row, decision="accept", reviewer=reviewer, notes=notes
            )
        elif command == "r":
            notes = required_input(
                console,
                "Why must this essay be replaced? Do not use reject for a score-only correction",
            )
            rows[index] = set_review_decision(
                row, decision="reject", reviewer=reviewer, notes=notes
            )
        else:
            corrections = prompt_corrections(console, row)
            if not corrections:
                console.print("No values changed; the row was not updated.")
                continue
            notes = optional_input(console, "Correction note")
            rows[index] = set_review_decision(
                row,
                decision="corrected",
                reviewer=reviewer,
                notes=notes,
                corrections=corrections,
            )
        write_packet_atomic(args.packet, rows)
        console.print(f"Saved {row['task_id']} immediately.")
        position += 1

    finish_review(console, rows, args, reviewer)


def select_indices(rows: list[dict[str, Any]], reviewer: str, args: argparse.Namespace) -> list[int]:
    if args.task_id:
        matches = [index for index, row in enumerate(rows) if row["task_id"] == args.task_id]
        if not matches:
            raise SystemExit(f"Task ID not found: {args.task_id}")
        return matches
    indices = list(range(max(0, args.start - 1), len(rows)))
    if args.only_unverified:
        indices = [
            index
            for index in indices
            if (rows[index].get("manual_review") or {}).get("reviewed_by") != reviewer
            or (rows[index].get("manual_review") or {}).get("decision")
            not in {"accept", "corrected"}
        ]
    return indices


def effective_grade(row: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    resolved = row.get("resolved_grade") or {}
    review = row.get("manual_review") or {}
    corrections = review.get("corrections") or {}
    scores = dict(corrections.get("scores") or resolved.get("scores") or {})
    feedback = dict(corrections.get("feedback") or resolved.get("feedback") or {})
    return scores, feedback


def render_case(
    console: Console,
    row: Mapping[str, Any],
    *,
    index: int,
    total: int,
    reviewer: str,
    plain: bool,
) -> None:
    scores, feedback = effective_grade(row)
    review = row.get("manual_review") or {}
    heading = (
        f"CASE {index + 1}/{total} | {row['task_id']} | "
        f"decision={review.get('decision', 'pending')} | "
        f"reviewed_by={review.get('reviewed_by', 'not human-verified')}"
    )
    console.rule(heading)
    boundary = ""
    if row.get("selection_class") == "boundary":
        boundary = (
            f"Boundary: {row.get('boundary_type')} | pair={row.get('contrast_pair_id')} "
            f"| side={row.get('contrast_side')}\n"
        )
    show_text(console, "PROMPT", boundary + str(row.get("prompt") or ""), plain)
    show_text(console, "STUDENT ESSAY", str(row.get("student_response") or ""), plain)
    render_scores(console, scores, row.get("rubric_reviews") or [], plain)
    feedback_text = "\n\n".join(
        f"{criterion}: {feedback.get(criterion, '')}" for criterion in CRITERIA
    )
    show_text(console, "RESOLVED FEEDBACK", feedback_text, plain)
    authenticity = row.get("authenticity_reviews") or []
    fact = row.get("fact_check") or {}
    audit_text = (
        f"Authenticity readers: {len(authenticity)} | "
        f"student-like passes: {sum(bool(item.get('student_like')) for item in authenticity)} | "
        f"timed-AP passes: {sum(bool(item.get('timed_ap_consistent')) for item in authenticity)}\n"
        f"Historical fact check: {'PASS' if fact.get('passed') else 'FAIL'} "
        f"({fact.get('checker_id', 'unknown checker')})\n"
        f"Current reviewer: {reviewer}"
    )
    show_text(console, "AUDIT SUMMARY", audit_text, plain)


def show_text(console: Console, title: str, body: str, plain: bool) -> None:
    if plain:
        console.print(f"\n{title}\n{'-' * len(title)}")
        console.print(body, markup=False)
    else:
        console.print(Panel(Text(body), title=title, expand=False))


def render_scores(
    console: Console,
    scores: Mapping[str, int],
    reader_reviews: list[Mapping[str, Any]],
    plain: bool,
) -> None:
    if plain:
        console.print("\nSCORES\n------")
        for criterion in CRITERIA:
            readers = ", ".join(
                f"{review.get('reader_id')}={str(review.get('scores', {}).get(criterion, '?'))}"
                for review in reader_reviews
            )
            console.print(f"{criterion}: resolved={scores.get(criterion, '?')} | {readers}")
        console.print(f"total: {sum(int(scores.get(key, 0)) for key in CRITERIA)}")
        return
    table = Table(title="SCORES", show_lines=True)
    table.add_column("Criterion")
    table.add_column("Resolved", justify="center")
    for review in reader_reviews:
        table.add_column(str(review.get("reader_id") or "reader"), justify="center")
    for criterion in CRITERIA:
        table.add_row(
            criterion,
            str(scores.get(criterion, "?")),
            *(str(review.get("scores", {}).get(criterion, "?")) for review in reader_reviews),
        )
    table.add_row(
        "total",
        str(sum(int(scores.get(key, 0)) for key in CRITERIA)),
        *([""] * len(reader_reviews)),
    )
    console.print(table)


def prompt_corrections(console: Console, row: Mapping[str, Any]) -> dict[str, Any]:
    current_scores, current_feedback = effective_grade(row)
    new_scores: dict[str, int] = {}
    for criterion in CRITERIA:
        low, high = SCORE_RANGES[criterion]
        while True:
            value = IntPrompt.ask(
                f"{criterion} score ({low}-{high})",
                default=int(current_scores[criterion]),
                console=console,
            )
            if low <= value <= high:
                new_scores[criterion] = value
                break
            console.print(f"Enter an integer from {low} through {high}.")
    corrections: dict[str, Any] = {}
    if new_scores != current_scores:
        corrections["scores"] = new_scores
    if Confirm.ask("Edit grounded feedback sentences?", default=bool(corrections), console=console):
        new_feedback: dict[str, str] = {}
        for criterion in CRITERIA:
            console.print(f"Current {criterion}: {current_feedback[criterion]}", markup=False)
            replacement = optional_input(console, "Replacement (Enter keeps current)")
            new_feedback[criterion] = replacement or current_feedback[criterion]
        if new_feedback != current_feedback:
            corrections["feedback"] = new_feedback
    return corrections


def optional_input(console: Console, label: str) -> str:
    return console.input(f"{label}: ").strip()


def required_input(console: Console, label: str) -> str:
    while True:
        value = optional_input(console, label)
        if value:
            return value
        console.print("A note is required for rejected rows.")


def render_help(console: Console, plain: bool) -> None:
    show_text(
        console,
        "REVIEW HELP",
        "a = accept the essay, scores, and feedback\n"
        "c = correct scores and/or feedback\n"
        "r = reject a fundamentally unsuitable essay; approval will remain blocked\n"
        "s = leave unchanged and move forward\n"
        "b = return to the preceding displayed row\n"
        "q = save completed actions and quit\n\n"
        "Every action is written atomically. A private backup is created before the first edit.",
        plain,
    )


def render_action_menu(console: Console, plain: bool) -> None:
    """Display every valid input before the terminal asks for one."""

    if plain:
        console.print("\nINPUT OPTIONS\n-------------")
        for key, label, description in ACTION_OPTIONS:
            console.print(f"{key} - {label}: {description}")
        return
    table = Table(title="INPUT OPTIONS", show_header=True, header_style="bold")
    table.add_column("Key", justify="center", width=5)
    table.add_column("Action", width=10)
    table.add_column("What it does")
    for key, label, description in ACTION_OPTIONS:
        table.add_row(key, label, description)
    console.print(table)


def finish_review(
    console: Console,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    reviewer: str,
) -> None:
    status = review_status(rows, reviewer)
    console.print("Review status: " + json.dumps(status, sort_keys=True))
    complete = status["human_verified"] == status["total"] and status["reject"] == 0
    if not complete:
        console.print(
            "Approval not written. Every row must be personally accepted or corrected, and "
            "rejected essays must be replaced first."
        )
        return
    if args.approve or Confirm.ask(
        "All rows are human-verified. Write the hash-bound approval receipt?",
        default=True,
        console=console,
    ):
        approval = build_human_approval(rows, packet_path=args.packet, reviewer=reviewer)
        write_json_atomic(args.approval, approval)
        console.print(f"Human approval written: {args.approval}")
        console.print("Rerun the v5 assembler finalize stage before training.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    private = Path("artifacts/data/v5/private")
    parser.add_argument(
        "--packet", type=Path, default=private / "manual_review_packet_v5.jsonl"
    )
    parser.add_argument(
        "--approval", type=Path, default=private / "manual_review_approval_v5.json"
    )
    parser.add_argument("--reviewer", help="Your name or stable reviewer ID")
    parser.add_argument("--start", type=int, default=1, help="One-based packet row")
    parser.add_argument("--task-id", help="Display one exact task ID")
    parser.add_argument("--only-unverified", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Screen-reader-friendly plain text")
    parser.add_argument("--width", type=int, default=110)
    parser.add_argument(
        "--approve", action="store_true", help="Write approval without the final confirmation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
