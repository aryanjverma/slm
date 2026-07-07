"""Interactive demo: paste an LEQ prompt and essay, receive a JSON grade."""

from __future__ import annotations

import typer
from rich.console import Console

from apush_frq_grader_slm.data import format_user_message, grade_essay

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main() -> None:
    console.print(
        "APUSH LEQ Grader Demo. Paste a prompt and essay, or type 'quit' to exit."
    )
    while True:
        prompt = console.input("[bold]LEQ Prompt[/bold] (or 'quit'): ")
        if prompt.strip().lower() in {"q", "quit", "exit"}:
            break
        essay = console.input("[bold]Student Essay[/bold]: ")
        if essay.strip().lower() in {"q", "quit", "exit"}:
            break
        response = grade_essay(prompt, essay)
        console.print("[bold]Grade JSON[/bold]:")
        console.print_json(response)
        console.print(f"[dim]User message format:[/dim]\n{format_user_message(prompt, essay)}")


if __name__ == "__main__":
    app()
