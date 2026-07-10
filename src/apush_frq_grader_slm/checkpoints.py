"""Reproducible planning for v2 data-scaling train/eval checkpoints."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel


class CheckpointRun(BaseModel):
    count: int
    data_path: Path
    model_output: Path
    train_command: list[str]
    eval_commands: list[list[str]]


def build_checkpoint_plan(
    *,
    counts: list[int],
    available_rows: int,
    checkpoint_root: Path,
    model: str,
    eval_paths: list[tuple[str, Path, bool]],
    eval_output_root: Path | None = None,
) -> list[CheckpointRun]:
    runs: list[CheckpointRun] = []
    for count in counts:
        if count > available_rows:
            continue
        data_path = checkpoint_root / "data" / f"train_chat_v2_{count}.jsonl"
        model_output = checkpoint_root / "models" / f"apush-frq-grader-v2-{count}"
        model_name = f"apush_frq_grader_v2_{count}"
        train_command = [
            sys.executable,
            "scripts/train_qlora.py",
            "--model",
            model,
            "--data",
            str(data_path),
            "--output",
            str(model_output),
        ]
        eval_commands = []
        for track_name, eval_path, real_eval in eval_paths:
            command = [
                sys.executable,
                "scripts/eval_hf_model.py",
                "--model",
                str(model_output),
                "--model-name",
                f"{model_name}_{track_name}",
                "--eval-path",
                str(eval_path),
                "--output-dir",
                str((eval_output_root or checkpoint_root / "eval") / str(count)),
            ]
            if real_eval:
                command.append("--real-eval")
            eval_commands.append(command)
        runs.append(
            CheckpointRun(
                count=count,
                data_path=data_path,
                model_output=model_output,
                train_command=train_command,
                eval_commands=eval_commands,
            )
        )
    return runs


def materialize_checkpoint_data(source: Path, runs: list[CheckpointRun]) -> dict[str, dict]:
    rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest: dict[str, dict] = {}
    for run in runs:
        selected = rows[: run.count]
        run.data_path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("\n".join(selected) + "\n").encode("utf-8")
        run.data_path.write_bytes(payload)
        manifest[str(run.count)] = {
            "path": str(run.data_path),
            "rows": len(selected),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return manifest


def write_checkpoint_plan(path: Path, runs: list[CheckpointRun], manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "runs": [run.model_dump(mode="json") for run in runs],
                "data_manifest": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
