"""Day 2 smoke pipeline: generate -> train -> eval (baselines + tuned model).

Runs the full end-to-end loop on 50 synthetic cases (30 train / 20 eval).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA = ROOT / "artifacts" / "smoke"
SMOKE_EVAL = ROOT / "artifacts" / "smoke_eval"
SMOKE_MODEL = ROOT / "artifacts" / "models" / "apush-frq-grader-v1-smoke"


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_generate:
        _run(
            [
                python,
                "-m",
                "apush_frq_grader_slm.cli.generate_dataset",
                "--train-count",
                str(args.train_count),
                "--eval-count",
                str(args.eval_count),
                "--output-dir",
                str(SMOKE_DATA),
            ]
        )

    if not args.skip_train:
        _run(
            [
                python,
                str(ROOT / "scripts" / "train_smoke.py"),
                "--data",
                str(SMOKE_DATA / "train_chat.jsonl"),
                "--output",
                str(SMOKE_MODEL),
                "--max-steps",
                str(args.max_steps),
            ]
        )

    _run(
        [
            python,
            "-m",
            "apush_frq_grader_slm.cli.run_eval",
            "--eval-path",
            str(SMOKE_DATA / "eval_cases.jsonl"),
            "--output-dir",
            str(SMOKE_EVAL),
        ]
    )

    _run(
        [
            python,
            str(ROOT / "scripts" / "eval_hf_model.py"),
            "--model",
            str(SMOKE_MODEL),
            "--model-name",
            "apush_frq_grader_smoke",
            "--eval-path",
            str(SMOKE_DATA / "eval_cases.jsonl"),
            "--output-dir",
            str(SMOKE_EVAL),
        ]
    )

    _merge_summaries(SMOKE_EVAL)
    print(f"Smoke pipeline complete. Results in {SMOKE_EVAL}")


def _merge_summaries(output_dir: Path) -> None:
    from apush_frq_grader_slm.io import read_jsonl, write_jsonl

    by_name: dict[str, dict] = {}
    summary_path = output_dir / "summary.jsonl"
    if summary_path.exists():
        for row in read_jsonl(summary_path):
            by_name[row["model_name"]] = row
    for path in sorted(output_dir.glob("*_summary.jsonl")):
        if path.name == "summary.jsonl" or "slice" in path.name:
            continue
        for row in read_jsonl(path):
            by_name[row["model_name"]] = row

    rows = list(by_name.values())
    write_jsonl(summary_path, rows)
    for row in rows:
        print(
            f"{row['model_name']}: json={row['structured_output_valid_rate']:.2f}, "
            f"rubric={row['rubric_accuracy_mean']:.2f}, "
            f"grounding={row['evidence_grounding_rate']:.2f}, "
            f"robustness={row['robustness_mean']:.2f}, "
            f"total={row['total_score_mean']:.2f}"
        )


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
