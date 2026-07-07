"""Run reference-vs-inflated-base behavioral evaluation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from apush_frq_grader_slm.baselines import InflatedPromptedBase, ReferenceGrader
from apush_frq_grader_slm.eval import evaluate_adapter, summarize, summarize_by_slice
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    eval_path: Path = typer.Option(Path("artifacts/data/eval_cases.jsonl"), help="Eval JSONL."),
    output_dir: Path = typer.Option(Path("artifacts/eval"), help="Directory for eval results."),
) -> None:
    cases = [FRQCase.model_validate(row) for row in read_jsonl(eval_path)]
    adapters = [InflatedPromptedBase(), ReferenceGrader()]
    summaries = []
    for adapter in adapters:
        results = evaluate_adapter(cases, adapter)
        summaries.append(summarize(results, adapter.name))
        write_jsonl(output_dir / f"{adapter.name}_results.jsonl", results)
        slice_summary = summarize_by_slice(results, cases)
        write_jsonl(
            output_dir / f"{adapter.name}_slice_summary.jsonl",
            [{"failure_type": key, **value} for key, value in slice_summary.items()],
        )

    write_jsonl(output_dir / "summary.jsonl", summaries)
    for summary in summaries:
        console.print(
            f"{summary.model_name}: json={summary.structured_output_valid_rate:.2f}, "
            f"rubric={summary.rubric_accuracy_mean:.2f}, "
            f"grounding={summary.evidence_grounding_rate:.2f}, "
            f"robustness={summary.robustness_mean:.2f}, "
            f"total={summary.total_score_mean:.2f}"
        )


if __name__ == "__main__":
    app()
