"""Generate train/eval JSONL files for the arithmetic tutor."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from arithmetic_tutor_slm.data import generate_cases, to_chat_rows
from arithmetic_tutor_slm.filters import passes_quality_gate
from arithmetic_tutor_slm.io import write_jsonl

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    output_dir: Path = typer.Option(Path("artifacts/data"), help="Directory for generated JSONL."),
    train_count: int = typer.Option(1000, min=1, help="Training example count."),
    eval_count: int = typer.Option(200, min=1, help="Held-out eval example count."),
    seed: int = typer.Option(13, help="Random seed."),
) -> None:
    train_cases = _filtered(generate_cases(count=train_count, split="train", seed=seed))
    eval_cases = _filtered(
        generate_cases(count=eval_count, split="eval", seed=seed + 1, adversarial_ratio=0.25)
    )

    write_jsonl(output_dir / "train_cases.jsonl", train_cases)
    write_jsonl(output_dir / "eval_cases.jsonl", eval_cases)
    write_jsonl(output_dir / "train_chat.jsonl", to_chat_rows(train_cases))

    console.print(
        f"Wrote {len(train_cases)} train cases, {len(eval_cases)} eval cases, "
        f"and chat-format SFT data to {output_dir}."
    )


def _filtered(cases):
    accepted = []
    for case in cases:
        ok, _ = passes_quality_gate(case)
        if ok:
            accepted.append(case)
    return accepted


if __name__ == "__main__":
    app()
