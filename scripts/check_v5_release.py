"""Apply the locked v5 release gates to an aggregate evaluation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.release_v5 import evaluate_v5_release


def main() -> None:
    args = parse_args()
    summary = read_summary(args.summary)
    decision = evaluate_v5_release(summary)
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not decision["release_ready"]:
        raise SystemExit(1)


def read_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty evaluation summary: {path}")
    first = text.splitlines()[0]
    value = json.loads(first if path.suffix == ".jsonl" else text)
    if not isinstance(value, dict):
        raise ValueError("Evaluation summary must be a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
