"""Rank v5 scorer/feedback checkpoint eval summaries and write the selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.checkpoint_rank_v5 import (
    build_feedback_ranking,
    build_scorer_ranking,
    build_v5_checkpoint_ranking,
    normalize_ranking_candidate,
)


def load_candidates(paths: list[Path]) -> list[dict]:
    candidates = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates.append(normalize_ranking_candidate(payload, source=str(path)))
    return candidates


def main() -> None:
    args = parse_args()
    if args.task:
        candidates = load_candidates(args.summaries)
        if args.task == "scorer":
            output = build_scorer_ranking(candidates)
            selected = output["selected_adapter"]
        else:
            output = build_feedback_ranking(candidates)
            selected = output["selected_adapter"]
    else:
        output = build_v5_checkpoint_ranking(
            scorer_candidates=load_candidates(args.scorer_summaries),
            feedback_candidates=load_candidates(args.feedback_summaries),
        )
        selected = {
            "scorer": output["selected_scorer"],
            "feedback": output["selected_feedback"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "output": str(args.output)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("scorer", "feedback"),
        help="Rank one task's summaries (positional files). Omit to rank both via flags.",
    )
    parser.add_argument(
        "summaries",
        nargs="*",
        type=Path,
        help="Summary JSON files when --task is set (raw or {adapter,summary}).",
    )
    parser.add_argument("--scorer-summaries", nargs="+", type=Path, default=[])
    parser.add_argument("--feedback-summaries", nargs="+", type=Path, default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/eval/v5/checkpoint_ranking.json"),
    )
    args = parser.parse_args()
    if args.task:
        if not args.summaries:
            parser.error("provide one or more summary files when --task is set")
    elif not args.scorer_summaries and not args.feedback_summaries:
        parser.error("provide --task summaries or --scorer-summaries/--feedback-summaries")
    return args


if __name__ == "__main__":
    main()
