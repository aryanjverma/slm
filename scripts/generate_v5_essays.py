"""Deprecated local composer entrypoint.

V5 regeneration requires independent cloud writer agents with a fresh context
per essay. The deterministic composer must not produce replacement essays.
"""

from __future__ import annotations

import argparse
import sys


_BLOCK_MESSAGE = """\
generate_v5_essays.py is retired from the production workflow.

The v5 authentic regeneration plan requires:
  - independent cloud writer agents
  - a fresh agent context for every essay
  - private packets from export_v5_generation_packets.py (full golden style essays)

Use:
  python scripts/export_v5_generation_packets.py --fact-cards ... --pilot-only
then return external JSONL rows with only task_id + student_response for hard-gate
validation. Do not use compose_v5 for replacement candidates.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-legacy-composer",
        action="store_true",
        help="Escape hatch for offline experiments only; never for production regeneration.",
    )
    args, unknown = parser.parse_known_args()
    if not args.allow_legacy_composer:
        print(_BLOCK_MESSAGE, file=sys.stderr)
        raise SystemExit(2)
    # Legacy path kept only for explicit offline experiments.
    from apush_frq_grader_slm.compose_v4 import rng_for_task
    from apush_frq_grader_slm.compose_v5 import (
        GENERATOR_NAME,
        compose_essay,
        resolve_observable_behavior,
    )
    from apush_frq_grader_slm.io import read_jsonl, write_jsonl
    from pathlib import Path

    # Minimal re-parse for the legacy experimental path.
    legacy = argparse.ArgumentParser()
    legacy.add_argument("--packets-dir", type=Path, default=Path("artifacts/data/v5/packets"))
    legacy.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"),
    )
    legacy.add_argument("--shard", type=str, default=None)
    legacy.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v5/private/raw_essays_legacy_composer"),
    )
    legacy_args = legacy.parse_args(unknown)
    tasks = {
        str(row["task_id"]): row
        for row in read_jsonl(legacy_args.tasks)
        if row.get("task_id")
    }
    paths = sorted(legacy_args.packets_dir.glob("v5-shard-*.jsonl"))
    if legacy_args.shard and legacy_args.shard != "all":
        name = (
            legacy_args.shard
            if legacy_args.shard.endswith(".jsonl")
            else f"{legacy_args.shard}.jsonl"
        )
        paths = [legacy_args.packets_dir / name]
    legacy_args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in paths:
        packets = read_jsonl(path)
        rows = []
        for packet in packets:
            task_id = str(packet["task_id"])
            behavior = resolve_observable_behavior(packet, tasks.get(task_id))
            # New packets use style_reference_essay; composer still accepts style_reference.
            if "style_reference" not in packet and packet.get("style_reference_essay"):
                packet = dict(packet)
                packet["style_reference"] = ""
            essay = compose_essay(
                packet,
                observable_writing_behavior=behavior,
                rng=rng_for_task(task_id),
            )
            rows.append(
                {
                    "task_id": task_id,
                    "student_response": essay,
                    "generator_name": GENERATOR_NAME,
                    "shard_id": path.stem,
                    "legacy_composer": True,
                }
            )
        out = legacy_args.output_dir / f"{path.stem}.jsonl"
        write_jsonl(out, rows)
        total += len(rows)
        print(f"LEGACY wrote {len(rows)} essays -> {out}")
    print(f"LEGACY done. {total} essays. generator={GENERATOR_NAME}")


if __name__ == "__main__":
    main()
