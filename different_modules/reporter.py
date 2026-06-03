# reporter.py
# ─────────────────────────────────────────────────────────────
# Renders analysis results to the terminal using Rich.
# ─────────────────────────────────────────────────────────────

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()

LEVEL_STYLES: dict[str, tuple[str, str]] = {
    "High":   ("red",    "🔴"),
    "Medium": ("yellow", "🟡"),
    "Low":    ("green",  "🟢"),
    "Clear":  ("cyan",   "✅"),
}


# ── Summary table ─────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    counts = {"High": 0, "Medium": 0, "Low": 0, "Clear": 0}
    for r in results:
        lvl = r.get("ambiguityLevel", "Clear")
        counts[lvl] = counts.get(lvl, 0) + 1

    table = Table(box=box.ROUNDED, title="[bold]Analysis Summary[/bold]", show_header=False)
    table.add_column("Metric", style="dim", min_width=36)
    table.add_column("Count",  justify="right", min_width=6)

    table.add_row("Total requirements analysed",                str(len(results)))
    table.add_row("", "")
    table.add_row(
        "[red]Need more info  (High + Medium)[/red]",
        str(counts["High"] + counts["Medium"]),
    )
    table.add_row("[red]  High ambiguity[/red]",                str(counts["High"]))
    table.add_row("[yellow]  Medium ambiguity[/yellow]",        str(counts["Medium"]))
    table.add_row("[green]  Low ambiguity[/green]",             str(counts["Low"]))
    table.add_row("[cyan]  Clear & testable[/cyan]",            str(counts["Clear"]))

    console.print(table)
    console.print()


# ── Per-requirement panel ─────────────────────────────────────

def _requirement_panel(r: dict) -> Panel:
    lvl           = r.get("ambiguityLevel", "Clear")
    color, icon   = LEVEL_STYLES.get(lvl, ("white", "❓"))

    title = Text()
    title.append(f"{icon}  {r.get('id', 'REQ')}  ", style="bold")
    title.append(f"[{lvl}]", style=f"bold {color}")

    body = Text()
    body.append(r.get("requirementText", ""), style="bold white")
    body.append("\n")

    # Knowledge-base sources used
    sources = r.get("_sources_used", [])
    if sources:
        body.append("\n📚  Knowledge base: ", style="bold dim")
        body.append(", ".join(sources) + "\n", style="dim")

    # Missing information
    if r.get("missingInformation"):
        body.append(f"\n🕳  What is missing:\n", style=f"bold {color}")
        for item in r["missingInformation"]:
            body.append(f"  • {item}\n", style=color)

    # Ambiguous terms
    if r.get("ambiguousTerms"):
        body.append("\n🌫  Vague terms:\n", style="bold magenta")
        for t in r["ambiguousTerms"]:
            body.append(
                f"  • \"{t.get('term', '')}\" — {t.get('problem', '')}\n",
                style="magenta",
            )
            if t.get("example"):
                body.append(f"    ↳ {t['example']}\n", style="dim")

    # Missing acceptance criteria
    if r.get("missingAcceptanceCriteria"):
        body.append("\n📏  Missing acceptance criteria:\n", style="bold yellow")
        for ac in r["missingAcceptanceCriteria"]:
            body.append(f"  • {ac}\n", style="yellow")

    # Edge cases
    if r.get("edgeCasesNotCovered"):
        body.append("\n⚠️  Edge cases not covered:\n", style="bold red")
        for ec in r["edgeCasesNotCovered"]:
            body.append(f"  • {ec}\n", style="red")

    # Clarifying questions
    if r.get("clarifyingQuestions"):
        body.append("\n❓  Ask the product owner:\n", style="bold yellow")
        for q in r["clarifyingQuestions"]:
            body.append(f"  {q}\n", style="yellow")

    # Rewrite suggestion
    if r.get("rewriteSuggestion"):
        body.append("\n✏️  Suggested rewrite:\n", style="bold blue")
        body.append(f"  {r['rewriteSuggestion']}\n", style="blue")

    return Panel(body, title=title, border_style=color, expand=False)


# ── Full report ───────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    print_summary(results)
    for r in results:
        console.print(_requirement_panel(r))
        console.print()
