"""Materialize and optionally execute 200/500/1,200-row v2 train/eval checkpoints."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from apush_frq_grader_slm.checkpoints import (
    build_checkpoint_plan,
    materialize_checkpoint_data,
    write_checkpoint_plan,
)


def main() -> None:
    args = parse_args()
    rows = sum(1 for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip())
    eval_paths = [("litmus", args.litmus, False)]
    if args.golden.exists():
        eval_paths.append(("golden", args.golden, True))
    if args.external.exists():
        eval_paths.append(("external", args.external, True))
    runs = build_checkpoint_plan(
        counts=args.counts,
        available_rows=rows,
        checkpoint_root=args.output_root,
        model=args.model,
        eval_paths=eval_paths,
    )
    manifest = materialize_checkpoint_data(args.data, runs)
    plan_path = args.output_root / "checkpoint_plan.json"
    write_checkpoint_plan(plan_path, runs, manifest)
    print(f"Prepared {len(runs)} checkpoints from {rows} rows; plan={plan_path}")
    if not args.execute:
        return
    for run in runs:
        subprocess.run(run.train_command, check=True)
        for command in run.eval_commands:
            subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("artifacts/data/v2/train_chat_v2.jsonl")
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[200, 500, 1200])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--litmus", type=Path, default=Path("artifacts/data/eval_cases.jsonl")
    )
    parser.add_argument(
        "--golden", type=Path, default=Path("artifacts/data/v2/eval_cb_golden_v2.jsonl")
    )
    parser.add_argument(
        "--external", type=Path, default=Path("artifacts/data/v2/eval_external_v2.jsonl")
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/checkpoints/v2"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
