"""Export blinded v5 writer packets; labels and score targets are never included."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from apush_frq_grader_slm.dataset_v5 import V5GenerationTask, generator_packet
from apush_frq_grader_slm.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"))
    parser.add_argument("--fact-cards", type=Path, required=True,
                        help="Private JSONL semantic cards with chapter_id and concept/fact.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/v5/packets"))
    args = parser.parse_args()
    cards: dict[str, list[dict]] = defaultdict(list)
    for card in read_jsonl(args.fact_cards):
        cards[str(card.get("chapter_id") or "")].append(card)
    packets: dict[str, list[dict]] = defaultdict(list)
    forbidden = {"target_scores", "target_total", "rubric_text", "score"}
    for row in read_jsonl(args.tasks):
        task = V5GenerationTask(
            task_id=row["task_id"], shard_id=row["shard_id"], prompt=row["prompt"],
            prompt_family_id=row["prompt_family_id"], style_seed_id=row["style_seed_id"],
            style_excerpt=row.get("style_excerpt", ""), period=row.get("period"),
            reasoning_skill=row.get("reasoning_skill", ""),
            capability_profile=row["capability_profile"], composition_profile=row["composition_profile"],
            amsco_chapter_ids=tuple(row.get("amsco_chapter_ids") or ()),
            coverage_class=row.get("coverage_class", "golden_matched"),
            boundary_type=row.get("boundary_type", ""),
            contrast_pair_id=row.get("contrast_pair_id", ""),
            contrast_side=row.get("contrast_side", ""),
        )
        task_cards = [card for chapter in task.amsco_chapter_ids for card in cards.get(chapter, [])]
        packet = generator_packet(task, task_cards)
        if forbidden & set(packet):
            raise AssertionError("writer packet leaked scoring information")
        packets[task.shard_id].append(packet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard, rows in sorted(packets.items()):
        if len(rows) != 50:
            raise ValueError(f"{shard} contains {len(rows)} packets, expected 50")
        write_jsonl(args.output_dir / f"{shard}.jsonl", rows)
    (args.output_dir / "README_PRIVATE.json").write_text(json.dumps({
        "private_use": True, "redistribute": False, "contains_style_excerpts": True,
        "writer_sees_score_targets": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {sum(map(len, packets.values()))} blinded packets to {args.output_dir}")


if __name__ == "__main__":
    main()
