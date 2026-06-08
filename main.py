#!/usr/bin/env python3
# main.py
# ─────────────────────────────────────────────────────────────
# CLI entry point for the requirements analysis agent.
#
# Build a knowledge base first:
#   python -m knowledge.builder --name bluetooth --docs ./docs/
#
# Then run the agent:
#   python main.py -f reqs.txt --kb bluetooth
#   python main.py -f reqs.txt --kb bluetooth --pdf report.pdf
#   → saves as:  report_20260608_143022.pdf
# ─────────────────────────────────────────────────────────────

import json
import typer
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box

import knowledge.retriever as retriever
from analyser      import analyse_all
from reporter      import print_report
from pdf_reporter  import save_pdf
from config        import DEFAULT_MODEL, DEFAULT_TOP_K

app     = typer.Typer(add_completion=False)
console = Console()

SAMPLE_REQUIREMENTS = """\
1. The device must support Bluetooth connectivity
2. The app should connect to nearby Bluetooth devices quickly
3. Bluetooth pairing must be secure
4. The system must handle multiple Bluetooth connections
5. Users should receive a notification when a Bluetooth device is out of range"""


# ── Helpers ───────────────────────────────────────────────────

def _stamp() -> str:
    """Return current datetime as a filename-safe string: 20260608_143022"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _with_timestamp(path: Path) -> Path:
    """
    Insert a timestamp before the file extension.
    report.pdf      → report_20260608_143022.pdf
    report.json     → report_20260608_143022.json
    report          → report_20260608_143022
    """
    ts   = _stamp()
    stem = path.stem
    ext  = path.suffix   # includes the dot, e.g. ".pdf"
    return path.with_name(f"{stem}_{ts}{ext}")


# ── CLI ───────────────────────────────────────────────────────

@app.command()
def main(
    file: Path = typer.Option(
        None, "--file", "-f",
        help="Requirements file (.txt) — one per line or numbered.",
    ),
    kb: str = typer.Option(
        None, "--kb",
        help="Name of the knowledge base to use (built with knowledge.builder).",
    ),
    list_kbs: bool = typer.Option(
        False, "--list-kb",
        help="List all available knowledge bases and exit.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m",
        help="Ollama model name. Try: mistral, llama3.1:8b, gemma3:12b",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save JSON report to this path. Timestamp is appended automatically.",
    ),
    pdf: Path = typer.Option(
        None, "--pdf", "-p",
        help="Save PDF report to this path. Timestamp is appended automatically.",
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

    Build a knowledge base first:\n
        python -m knowledge.builder --name bluetooth --docs ./bt_docs/\n

    Then run the agent:\n
        python main.py -f reqs.txt --kb bluetooth\n
        python main.py -f reqs.txt --kb bluetooth --pdf report.pdf\n
        → saves as report_20260608_143022.pdf
    """

    console.rule("[bold]Requirements Ambiguity Analysis Agent[/bold]")

    # ── List available KBs and exit ───────────────────────────
    if list_kbs:
        metas = retriever.list_available()
        if not metas:
            console.print(
                "[dim]No knowledge bases found.[/dim]\n"
                "[dim]Build one with:[/dim]  python -m knowledge.builder --name <name> --docs <path>"
            )
        else:
            table = Table(box=box.ROUNDED, title="[bold]Available Knowledge Bases[/bold]")
            table.add_column("Name",    style="bold cyan")
            table.add_column("Chunks",  justify="right")
            table.add_column("Sources", justify="right")
            table.add_column("Created", style="dim")
            for m in metas:
                table.add_row(
                    m["name"],
                    str(m["total_chunks"]),
                    str(len(m["sources"])),
                    m["created_at"][:19].replace("T", "  "),
                )
            console.print(table)
        return

    # ── Load knowledge base if specified ──────────────────────
    if kb:
        ok = retriever.load(kb)
        if not ok:
            raise typer.Exit(1)
    else:
        console.print(
            "[dim]No knowledge base selected — running without domain context.[/dim]\n"
            "[dim]Tip: build one with  python -m knowledge.builder --name <name> --docs <path>[/dim]\n"
            "[dim]     then use it with  --kb <name>[/dim]\n"
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
        console.print("[dim]No input — running with Bluetooth sample requirements.[/dim]\n")
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

    # ── Print terminal report ─────────────────────────────────
    print_report(results)

    # ── Save JSON ─────────────────────────────────────────────
    if output:
        # Ensure .json extension, then stamp
        json_path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
        json_path = _with_timestamp(json_path)
        clean     = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        json_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]✓ JSON report saved to:[/green] {json_path}\n")

    # ── Save PDF ──────────────────────────────────────────────
    if pdf:
        # Ensure .pdf extension, then stamp
        pdf_path = pdf if pdf.suffix.lower() == ".pdf" else pdf.with_suffix(".pdf")
        pdf_path = _with_timestamp(pdf_path)
        with console.status("[bold green]Generating PDF report...[/bold green]", spinner="dots"):
            save_pdf(results, output_path=pdf_path, kb_name=kb)
        console.print(f"[green]✓ PDF report saved to:[/green] {pdf_path}\n")


if __name__ == "__main__":
    app()