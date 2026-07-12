"""Export private v5 writer packets with full matched golden essays.

The deterministic composer is not used. Packets are score-blind and include the
complete style-reference essay, semantic fact cards, capability/composition
cues, and an essay-only contract.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from apush_frq_grader_slm.dataset_v5 import (
    V5GenerationTask,
    WRITER_FORBIDDEN_PACKET_KEYS,
    attach_style_reference,
    generator_packet,
    load_style_reference_essays,
    select_v5_pilot_tasks,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl


def task_from_row(row: dict) -> V5GenerationTask:
    return V5GenerationTask(
        task_id=row["task_id"],
        shard_id=row["shard_id"],
        prompt=row["prompt"],
        prompt_family_id=row["prompt_family_id"],
        style_seed_id=row["style_seed_id"],
        style_excerpt=row.get("style_excerpt", ""),
        period=row.get("period"),
        reasoning_skill=row.get("reasoning_skill", ""),
        capability_profile=dict(row["capability_profile"]),
        composition_profile=dict(row["composition_profile"]),
        amsco_chapter_ids=tuple(row.get("amsco_chapter_ids") or ()),
        coverage_class=row.get("coverage_class", "golden_matched"),
        boundary_type=row.get("boundary_type", ""),
        contrast_pair_id=row.get("contrast_pair_id", ""),
        contrast_side=row.get("contrast_side", ""),
        reference_word_count=row.get("reference_word_count"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"),
    )
    parser.add_argument(
        "--fact-cards",
        type=Path,
        required=True,
        help="Private JSONL semantic cards with chapter_id and concept/fact.",
    )
    parser.add_argument(
        "--seed-profiles",
        type=Path,
        default=Path("artifacts/data/v5/planning/cb_seed_profiles_v5.jsonl"),
    )
    parser.add_argument(
        "--golden-cases",
        type=Path,
        default=Path("artifacts/data/eval_cb_cases.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v5/packets"),
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Export only the 30-essay pilot packets (required before full production).",
    )
    parser.add_argument(
        "--pilot-approval",
        type=Path,
        default=Path("artifacts/data/v5/private/pilot_approval_v5.json"),
        help="Hash-bound pilot approval required before exporting non-pilot packets.",
    )
    parser.add_argument(
        "--pilot-essays",
        type=Path,
        default=Path("artifacts/data/v5/private/pilot_essays_v5.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    if not args.pilot_only:
        from apush_frq_grader_slm.dataset_v5 import assert_pilot_approval

        assert_pilot_approval(args.pilot_essays, args.pilot_approval)

    references = load_style_reference_essays(args.seed_profiles, args.golden_cases)
    cards: dict[str, list[dict]] = defaultdict(list)
    for card in read_jsonl(args.fact_cards):
        cards[str(card.get("chapter_id") or "")].append(card)

    planned = [task_from_row(row) for row in read_jsonl(args.tasks)]
    if args.pilot_only:
        planned = select_v5_pilot_tasks(planned, seed=args.seed)
        args.output_dir = (
            args.output_dir
            if args.output_dir.name == "pilot_packets"
            else args.output_dir.parent / "pilot_packets"
            if args.output_dir.name == "packets"
            else args.output_dir
        )

    packets: dict[str, list[dict]] = defaultdict(list)
    for task in planned:
        enriched = attach_style_reference(task, references)
        if not enriched.style_reference_essay:
            raise ValueError(
                f"{task.task_id} has no matched golden essay for seed {task.style_seed_id}"
            )
        task_cards = [
            card for chapter in enriched.amsco_chapter_ids for card in cards.get(chapter, [])
        ]
        packet = generator_packet(enriched, task_cards)
        if WRITER_FORBIDDEN_PACKET_KEYS & set(packet):
            raise AssertionError("writer packet leaked scoring or identity information")
        if "style_reference_essay" not in packet:
            raise AssertionError("writer packet missing full style_reference_essay")
        # Truncated excerpts must not replace the full essay.
        if len(packet["style_reference_essay"].split()) < 40:
            raise AssertionError(f"{task.task_id} style reference essay is too short")
        packets[enriched.shard_id].append(packet)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard, rows in sorted(packets.items()):
        write_jsonl(args.output_dir / f"{shard}.jsonl", rows)

    manifest = {
        "private_use": True,
        "redistribute": False,
        "contains_full_style_reference_essays": True,
        "writer_sees_score_targets": False,
        "writer_sees_source_ids": False,
        "deterministic_composer_used": False,
        "pilot_only": bool(args.pilot_only),
        "packet_count": sum(map(len, packets.values())),
        "shard_count": len(packets),
        "essay_only_contract": True,
        "fresh_cloud_agent_context_required_per_essay": True,
    }
    (args.output_dir / "README_PRIVATE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.pilot_only:
        index_path = args.output_dir / "pilot_task_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "task_ids": [task.task_id for task in planned],
                    "count": len(planned),
                    "seed": args.seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        f"Exported {sum(map(len, packets.values()))} private writer packets "
        f"(pilot_only={args.pilot_only}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
