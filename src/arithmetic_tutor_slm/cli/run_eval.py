"""Run base-vs-tutor behavioral evaluation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from arithmetic_tutor_slm.baselines import LeakyPromptedBase, SocraticTutorReference
from arithmetic_tutor_slm.eval import evaluate_adapter, summarize
from arithmetic_tutor_slm.io import read_jsonl, write_jsonl
from arithmetic_tutor_slm.schemas import ArithmeticCase

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    eval_path: Path = typer.Option(Path("artifacts/data/eval_cases.jsonl"), help="Eval JSONL."),
    output_dir: Path = typer.Option(Path("artifacts/eval"), help="Directory for eval results."),
) -> None:
    cases = [ArithmeticCase.model_validate(row) for row in read_jsonl(eval_path)]
    adapters = [LeakyPromptedBase(), SocraticTutorReference()]
    summaries = []
    for adapter in adapters:
        results = evaluate_adapter(cases, adapter)
        summaries.append(summarize(results, adapter.name))
        write_jsonl(output_dir / f"{adapter.name}_results.jsonl", results)

    write_jsonl(output_dir / "summary.jsonl", summaries)
    for summary in summaries:
        console.print(
            f"{summary.model_name}: no-leak={summary.no_answer_leak_rate:.2f}, "
            f"hint={summary.hint_correctness_rate:.2f}, "
            f"calibration={summary.step_calibration_rate:.2f}, "
            f"total={summary.total_score_mean:.2f}"
        )


if __name__ == "__main__":
    app()
