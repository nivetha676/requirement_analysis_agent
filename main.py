#!/usr/bin/env python3
# main.py
# ─────────────────────────────────────────────────────────────
# CLI entry point.
# Run:  python main.py --help
# ─────────────────────────────────────────────────────────────

import json
import typer
from pathlib import Path

from rich.console import Console

import knowledge.vector_store as vs
from analyser  import analyse_all
from reporter  import print_report
from config    import DEFAULT_MODEL, DEFAULT_TOP_K

app     = typer.Typer(add_completion=False)
console = Console()

SAMPLE_REQUIREMENTS = """\
1. The device must support Bluetooth connectivity
2. The app should connect to nearby Bluetooth devices quickly
3. Bluetooth pairing must be secure
4. The system must handle multiple Bluetooth connections
5. Users should receive a notification when a Bluetooth device is out of range"""


# ── CLI command ───────────────────────────────────────────────

@app.command()
def main(
    file: Path = typer.Option(
        None, "--file", "-f",
        help="Requirements file (.txt) — one requirement per line or numbered.",
    ),
    knowledge: list[Path] = typer.Option(
        [], "--knowledge", "-k",
        help="Knowledge-base file or directory. Repeatable: -k bt.pdf -k specs/",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m",
        help="Ollama model name. Try: mistral, llama3.1:8b, gemma3:12b",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save JSON report to this path.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Paste requirements interactively in the terminal.",
    ),
    top_k: int = typer.Option(
        DEFAULT_TOP_K, "--top-k",
        help="Number of knowledge-base chunks retrieved per requirement.",
    ),
) -> None:
    """
    Analyse software requirements for ambiguity, missing information,
    vague terms, missing acceptance criteria, and edge cases.

    Pass domain documents with -k to unlock domain-aware analysis.
    """

    console.rule("[bold]Requirements Ambiguity Analysis Agent[/bold]")

    # ── Build knowledge base ──────────────────────────────────
    if knowledge:
        ok = vs.build(list(knowledge))
        if not ok:
            console.print("[yellow]Continuing without knowledge base.[/yellow]\n")
    else:
        console.print(
            "[dim]No knowledge base provided — running without domain context.[/dim]\n"
            "[dim]Tip: use -k path/to/spec.pdf for domain-aware analysis.[/dim]\n"
        )

    # ── Resolve requirements input ────────────────────────────
    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        text = file.read_text(encoding="utf-8").strip()
        console.print(f"[dim]Requirements from:[/dim] {file}\n")

    elif interactive:
        console.print("[bold]Paste requirements below. Press Enter twice when done.\n[/bold]")
        lines, blanks = [], 0
        try:
            while blanks < 2:
                line = input()
                if not line.strip():
                    blanks += 1
                else:
                    blanks = 0
                    lines.append(line)
        except EOFError:
            pass
        text = "\n".join(lines).strip()
        if not text:
            console.print("[red]No requirements entered.[/red]")
            raise typer.Exit(1)

    else:
        console.print("[dim]No input provided — running with Bluetooth sample requirements.[/dim]\n")
        text = SAMPLE_REQUIREMENTS

    # ── Run analysis ──────────────────────────────────────────
    try:
        results = analyse_all(text, model, top_k=top_k)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("[yellow]Is Ollama running?[/yellow]  Start with:  ollama serve")
        raise typer.Exit(1)

    if not results:
        console.print("[red]No requirements were successfully analysed.[/red]")
        raise typer.Exit(1)

    # ── Print report ──────────────────────────────────────────
    print_report(results)

    # ── Save JSON ─────────────────────────────────────────────
    if output:
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        output.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]✓ Report saved to:[/green] {output}\n")


if __name__ == "__main__":
    app()
