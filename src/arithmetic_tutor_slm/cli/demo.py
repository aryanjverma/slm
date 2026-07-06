"""Tiny local demo for the reference tutor behavior."""

from __future__ import annotations

import typer
from rich.console import Console

from arithmetic_tutor_slm.baselines import reference_reply

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main() -> None:
    console.print("Arithmetic Tutor Demo. Enter addition/subtraction questions, or 'quit'.")
    while True:
        message = console.input("[bold]Student[/bold]: ")
        if message.strip().lower() in {"q", "quit", "exit"}:
            break
        console.print(f"[bold]Tutor[/bold]: {reference_reply(message)}")


if __name__ == "__main__":
    app()
