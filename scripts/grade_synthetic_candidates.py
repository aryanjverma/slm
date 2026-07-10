"""Run two anonymous graders and adjudicate disagreements for synthetic essays."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.independent_grading import (
    GradeDecision,
    parse_grader_output,
    render_adjudication_prompt,
    render_grader_prompt,
    resolve_independent_grades,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.synth_realistic import GenTask, parse_candidate_row


def _request(client: Any, model: str, prompt: str) -> str:
    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip()


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the 'judge' extra to run independent grading.") from exc

    client = OpenAI()
    tasks = {row["task_id"]: GenTask.from_row(row) for row in read_jsonl(args.tasks)}
    decisions: list[dict] = []
    rejects: list[dict] = []
    raw_rows = read_jsonl(args.raw)
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]

    for row in raw_rows:
        task_id = row.get("task_id")
        task = tasks.get(task_id)
        if task is None:
            rejects.append({"task_id": task_id, "reasons": ["unknown_task_id"]})
            continue
        try:
            candidate = parse_candidate_row(row, task)
            prompt = render_grader_prompt(candidate)
            with ThreadPoolExecutor(max_workers=2) as executor:
                response_a = executor.submit(_request, client, args.grader_a_model, prompt)
                response_b = executor.submit(_request, client, args.grader_b_model, prompt)
                reader_a = parse_grader_output(
                    response_a.result(), candidate, f"reader_a:{args.grader_a_model}"
                )
                reader_b = parse_grader_output(
                    response_b.result(), candidate, f"reader_b:{args.grader_b_model}"
                )
            decision = resolve_independent_grades(
                task_id,
                reader_a,
                reader_b,
                minimum_confidence=args.minimum_confidence,
            )
            needs_adjudication = (
                decision.status == "rejected"
                and "grader_disagreement_requires_adjudication" in decision.reasons
            )
            if needs_adjudication:
                adjudicator_response = _request(
                    client,
                    args.adjudicator_model,
                    render_adjudication_prompt(candidate, reader_a, reader_b),
                )
                adjudicator = parse_grader_output(
                    adjudicator_response,
                    candidate,
                    f"adjudicator:{args.adjudicator_model}",
                )
                decision = resolve_independent_grades(
                    task_id,
                    reader_a,
                    reader_b,
                    adjudicator,
                    minimum_confidence=args.minimum_confidence,
                )
        except Exception as exc:
            decision = GradeDecision(
                str(task_id),
                "rejected",
                None,
                None,
                None,
                {"resolution": "grading_error"},
                (f"grading_error:{exc}",),
            )
        decisions.append(decision.to_row())
        if decision.status == "rejected":
            rejects.append({"task_id": task_id, "reasons": list(decision.reasons)})

    write_jsonl(args.output, decisions)
    write_jsonl(args.rejects, rejects)
    accepted = sum(row["status"] == "accepted" for row in decisions)
    adjudicated = sum(
        row["consensus_metadata"].get("resolution") == "adjudicated" for row in decisions
    )
    print(
        f"Wrote {len(decisions)} decisions to {args.output}: "
        f"accepted={accepted}, rejected={len(decisions) - accepted}, adjudicated={adjudicated}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently grade synthetic LEQ candidates.")
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/synth_tasks_train_v2.jsonl")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("artifacts/data/synth_realistic_validated_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/synth_realistic_grades_v2.jsonl")
    )
    parser.add_argument(
        "--rejects", type=Path, default=Path("artifacts/data/synth_grading_rejects_v2.jsonl")
    )
    parser.add_argument("--grader-a-model", default="gpt-4.1-mini")
    parser.add_argument("--grader-b-model", default="gpt-4.1-mini")
    parser.add_argument("--adjudicator-model", default="gpt-4.1")
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
