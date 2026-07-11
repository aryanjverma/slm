"""Generate score-blind v5 essays from writer packets (optional task metadata restore)."""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.compose_v4 import rng_for_task
from apush_frq_grader_slm.compose_v5 import (
    GENERATOR_NAME,
    compose_essay,
    resolve_observable_behavior,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl

_SCORE_LEAK_KEYS = frozenset(
    {
        "target_scores",
        "target_total",
        "scores",
        "score",
        "rubric_text",
        "resolved_grade",
        "reference_scores",
    }
)


def _load_tasks_by_id(tasks_path: Path | None) -> dict[str, dict]:
    if tasks_path is None or not tasks_path.is_file():
        return {}
    return {str(row["task_id"]): row for row in read_jsonl(tasks_path) if row.get("task_id")}


def _packet_paths(packets_dir: Path, shard: str | None) -> list[Path]:
    if shard and shard != "all":
        name = shard if shard.endswith(".jsonl") else f"{shard}.jsonl"
        path = packets_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"packet shard not found: {path}")
        return [path]
    paths = sorted(packets_dir.glob("v5-shard-*.jsonl"))
    if not paths:
        paths = sorted(p for p in packets_dir.glob("*.jsonl") if p.name != "README_PRIVATE.json")
    if not paths:
        raise FileNotFoundError(f"no packet shards under {packets_dir}")
    return paths


def _shard_id_from_path(path: Path, packet: dict, tasks: dict[str, dict]) -> str:
    task_id = str(packet.get("task_id") or "")
    if task_id and task_id in tasks and tasks[task_id].get("shard_id"):
        return str(tasks[task_id]["shard_id"])
    stem = path.stem
    if stem.startswith("v5-shard-"):
        return stem
    return str(packet.get("shard_id") or stem)


def generate_rows(
    packets: list[dict],
    *,
    shard_id: str,
    tasks_by_id: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for packet in packets:
        task_id = str(packet["task_id"])
        task = tasks_by_id.get(task_id)
        behavior = resolve_observable_behavior(packet, task)
        essay = compose_essay(
            packet,
            observable_writing_behavior=behavior,
            rng=rng_for_task(task_id),
        )
        row = {
            "task_id": task_id,
            "student_response": essay,
            "generator_name": GENERATOR_NAME,
            "shard_id": shard_id,
        }
        leaked = _SCORE_LEAK_KEYS & set(row)
        if leaked:
            raise AssertionError(f"essay output leaked scoring keys: {sorted(leaked)}")
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    tasks_by_id = _load_tasks_by_id(args.tasks)
    paths = _packet_paths(args.packets_dir, args.shard)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in paths:
        packets = read_jsonl(path)
        if not packets:
            continue
        shard_id = _shard_id_from_path(path, packets[0], tasks_by_id)
        rows = generate_rows(packets, shard_id=shard_id, tasks_by_id=tasks_by_id)
        out_path = args.output_dir / f"{shard_id}.jsonl"
        write_jsonl(out_path, rows)
        total += len(rows)
        print(f"Wrote {len(rows)} essays -> {out_path}")
    print(f"Done. {total} essays across {len(paths)} shard file(s). generator={GENERATOR_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets-dir",
        type=Path,
        default=Path("artifacts/data/v5/packets"),
        help="Directory of blinded writer packet shards (v5-shard-XX.jsonl).",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/data/v5/planning/generation_tasks_v5.jsonl"),
        help="Private task plan used to restore boundary observable_writing_behavior.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help='Optional shard id such as "v5-shard-00", or "all" for every shard.',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/v5/private/raw_essays"),
        help="Destination for private essay JSONL shards (do not commit).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
