"""Rewrite AMSCO KB chapters into semantic concept cards for v5 generation packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.fact_cards_v5 import build_semantic_fact_cards
from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.knowledge.amsco import DEFAULT_KB_PATH


def main() -> None:
    args = parse_args()
    cards = build_semantic_fact_cards(args.amsco_kb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, cards)
    summary = {
        "n_cards": len(cards),
        "n_chapters": len({card["chapter_id"] for card in cards}),
        "source_kind": "semantic_rewrite",
        "amsco_kb": str(args.amsco_kb),
        "output": str(args.output),
        "contains_essays": False,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amsco-kb",
        type=Path,
        default=DEFAULT_KB_PATH,
        help="AMSCO knowledge-base JSONL (default: artifacts/knowledge/amsco_2016_kb.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/v5/planning/semantic_fact_cards_v5.jsonl"),
        help="Planning-level semantic fact cards JSONL (no essays)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
