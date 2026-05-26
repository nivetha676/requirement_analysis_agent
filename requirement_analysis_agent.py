#!/usr/bin/env python3
"""
Requirements Ambiguity Analysis Agent
--------------------------------------
Analyses software requirements for ambiguity, missing information,
vague terms, missing acceptance criteria, and edge cases.

Usage:
    python requirements_agent.py                          # demo mode
    python requirements_agent.py -f requirements.txt     # from file
    python requirements_agent.py -i                      # interactive paste
    python requirements_agent.py -f reqs.txt -o out.json # save JSON report
    python requirements_agent.py -f reqs.txt -m mistral  # different model
"""

import json
import sys
import typer
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)
console = Console()

# Ollama runs locally at 11434. For LM Studio change port to 1234.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior QA analyst specialising in requirements engineering.
Your job is to deeply inspect each requirement and identify EXACTLY what information is missing
before a tester can write test cases.

Return a JSON array only — no markdown fences, no prose before or after.

Each object must have these fields:

- id: "REQ-01", "REQ-02", etc.
- requirementText: the requirement exactly as given
- ambiguityLevel: one of "High" | "Medium" | "Low" | "Clear"

- missingInformation: list of strings — concrete facts that are ABSENT from the requirement.
  Each entry must name the specific missing piece, e.g.:
  "No maximum password length specified"
  "No timeout value defined for session expiry"
  "No list of supported languages provided"
  "No error message text defined for failed login"
  NOT vague statements like "more detail needed" or "unclear scope".

- ambiguousTerms: list of objects, each with:
    - term: the exact word or phrase that is vague
    - problem: why it is ambiguous for testing purposes
    - example: what two different testers might assume it means
  Examples of vague terms: "fast", "secure", "relevant", "appropriate", "large number",
  "standard format", "regularly", "easy to use", "should", "etc."

- missingAcceptanceCriteria: list of strings — specific measurable conditions that are absent.
  These are the pass/fail rules a tester needs. e.g.:
  "No response time threshold (e.g. must load within 2 seconds)"
  "No definition of what constitutes a successful payment"
  "No specification of which user roles can access the admin panel"
  "No maximum retry attempts defined for failed logins"

- edgeCasesNotCovered: list of strings — boundary conditions and error scenarios not mentioned.
  e.g.:
  "What happens when a user tries to register with an already-existing email?"
  "What happens when the file being exported is empty?"
  "Is there a maximum number of items the search can return?"
  "What happens if the user session expires mid-transaction?"

- clarifyingQuestions: list of strings — exact questions to ask the product owner or BA.
  Each must be specific and answerable, not open-ended.
  Good: "What is the maximum allowed file size for uploads in MB?"
  Bad:  "Can you clarify the upload requirements?"

- rewriteSuggestion: string — a rewritten version of the requirement that fills the gaps
  using measurable and testable language. Empty string if already Clear.

Ambiguity levels:
- High:   3+ missing pieces, core behaviour undefined — test cases cannot be written at all
- Medium: 1-2 missing pieces — some test cases possible but important paths untestable
- Low:    minor gaps only, main behaviour clear — test cases writable with small assumptions
- Clear:  fully specified, measurable, no vague terms — test cases can be written immediately"""


def _build_user_prompt(requirements_text: str) -> str:
    return (
        "Analyse each of the following requirements in detail.\n"
        "For every vague term, missing value, undefined behaviour, and untestable condition "
        "— call it out explicitly.\n\n"
        f"Requirements:\n{requirements_text}"
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(requirements_text: str, model: str) -> list[dict]:
    console.print(f"\n[dim]Model:[/dim] [bold]{model}[/bold]  "
                  f"[dim]Server:[/dim] [bold]http://localhost:11434[/bold]\n")

    with console.status("[bold green]Analysing requirements...[/bold green]", spinner="dots"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(requirements_text)},
            ],
            temperature=0.2,
        )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences some models add
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

LEVEL_STYLES = {
    "High":   ("red",     "🔴"),
    "Medium": ("yellow",  "🟡"),
    "Low":    ("green",   "🟢"),
    "Clear":  ("cyan",    "✅"),
}


def print_report(results: list[dict]) -> None:
    counts = {"High": 0, "Medium": 0, "Low": 0, "Clear": 0}
    for r in results:
        lvl = r.get("ambiguityLevel", "Clear")
        counts[lvl] = counts.get(lvl, 0) + 1

    # Summary table
    summary = Table(box=box.ROUNDED, title="[bold]Analysis Summary[/bold]", show_header=False)
    summary.add_column("Metric", style="dim", min_width=32)
    summary.add_column("Count", justify="right", min_width=6)
    summary.add_row("Total requirements analysed",            str(len(results)))
    summary.add_row("",                                        "")
    summary.add_row("[red]Need more info  (High + Medium)[/red]",
                    str(counts["High"] + counts["Medium"]))
    summary.add_row("[red]  High ambiguity[/red]",            str(counts["High"]))
    summary.add_row("[yellow]  Medium ambiguity[/yellow]",    str(counts["Medium"]))
    summary.add_row("[green]  Low ambiguity[/green]",         str(counts["Low"]))
    summary.add_row("[cyan]  Clear & testable[/cyan]",        str(counts["Clear"]))
    console.print(summary)
    console.print()

    # Per-requirement panels
    for r in results:
        lvl   = r.get("ambiguityLevel", "Clear")
        color, icon = LEVEL_STYLES.get(lvl, ("white", "❓"))

        title = Text()
        title.append(f"{icon}  {r.get('id', 'REQ')}  ", style="bold")
        title.append(f"[{lvl}]", style=f"bold {color}")

        body = Text()
        body.append(r.get("requirementText", ""), style="bold white")
        body.append("\n")

        # Missing information
        items = r.get("missingInformation", [])
        if items:
            body.append("\n🕳  What is missing:\n", style=f"bold {color}")
            for item in items:
                body.append(f"  • {item}\n", style=color)

        # Ambiguous terms
        terms = r.get("ambiguousTerms", [])
        if terms:
            body.append("\n🌫  Vague terms:\n", style="bold magenta")
            for t in terms:
                body.append(f"  • \"{t.get('term','')}\" — {t.get('problem','')}\n",
                            style="magenta")
                if t.get("example"):
                    body.append(f"    ↳ {t['example']}\n", style="dim")

        # Missing acceptance criteria
        criteria = r.get("missingAcceptanceCriteria", [])
        if criteria:
            body.append("\n📏  Missing acceptance criteria:\n", style="bold yellow")
            for ac in criteria:
                body.append(f"  • {ac}\n", style="yellow")

        # Edge cases
        edges = r.get("edgeCasesNotCovered", [])
        if edges:
            body.append("\n⚠️  Edge cases not covered:\n", style="bold red")
            for ec in edges:
                body.append(f"  • {ec}\n", style="red")

        # Clarifying questions
        questions = r.get("clarifyingQuestions", [])
        if questions:
            body.append("\n❓  Ask the product owner:\n", style="bold yellow")
            for q in questions:
                body.append(f"  {q}\n", style="yellow")

        # Rewrite suggestion
        rewrite = r.get("rewriteSuggestion", "")
        if rewrite:
            body.append("\n✏️  Suggested rewrite:\n", style="bold blue")
            body.append(f"  {rewrite}\n", style="blue")

        console.print(Panel(body, title=title, border_style=color, expand=False))
        console.print()


# ---------------------------------------------------------------------------
# Sample requirements (demo mode)
# ---------------------------------------------------------------------------

SAMPLE_REQUIREMENTS = """\
1. The system should be fast and responsive at all times
2. Users must be able to register and login using their email address and a password that meets security standards
3. The admin panel should allow management of user accounts
4. The application should support multiple languages
5. Payment processing must be secure
6. Users should receive notifications when something important happens
7. The report generation feature must export data in a standard format
8. The search functionality should return accurate results quickly
9. The system must handle a large number of concurrent users
10. All user data must be stored securely and comply with regulations"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    file: Path = typer.Option(
        None, "--file", "-f",
        help="Path to a .txt file with requirements (one per line or numbered).",
    ),
    model: str = typer.Option(
        "llama3.2", "--model", "-m",
        help="Ollama model name. Recommended: llama3.2 (8 GB), mistral (16 GB), gemma3:12b (32 GB).",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save the full JSON report to this file path.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Paste requirements directly in the terminal.",
    ),
):
    """
    Analyse software requirements for ambiguity, missing information,
    vague terms, missing acceptance criteria, and untested edge cases.
    """

    console.rule("[bold]Requirements Ambiguity Analysis Agent[/bold]")

    # Resolve input source
    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        text = file.read_text(encoding="utf-8").strip()
        console.print(f"[dim]Reading from:[/dim] {file}\n")

    elif interactive:
        console.print("[bold]Paste your requirements below.[/bold]")
        console.print("[dim]Press Enter twice on a blank line when done.[/dim]\n")
        lines, blank_streak = [], 0
        try:
            while blank_streak < 2:
                line = input()
                if line.strip() == "":
                    blank_streak += 1
                else:
                    blank_streak = 0
                    lines.append(line)
        except EOFError:
            pass
        text = "\n".join(lines).strip()
        if not text:
            console.print("[red]No requirements entered. Exiting.[/red]")
            raise typer.Exit(1)

    else:
        console.print("[dim]No input provided — running with built-in sample requirements.[/dim]")
        text = SAMPLE_REQUIREMENTS

    # Run analysis
    try:
        results = analyse(text, model)
    except json.JSONDecodeError as e:
        console.print(f"\n[red]Could not parse model output as JSON.[/red]")
        console.print(f"[dim]Detail: {e}[/dim]")
        console.print(
            "\n[yellow]Tips:[/yellow]\n"
            "  • Try a larger model:  [bold]-m mistral[/bold] or [bold]-m gemma3:12b[/bold]\n"
            "  • Make sure Ollama is running:  [bold]ollama serve[/bold]\n"
            "  • Pull the model first:  [bold]ollama pull llama3.2[/bold]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error contacting Ollama:[/red] {e}")
        console.print(
            "\n[yellow]Is Ollama running?[/yellow]  Start it with:  [bold]ollama serve[/bold]"
        )
        raise typer.Exit(1)

    # Print report
    print_report(results)

    # Optionally save JSON
    if output:
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]✓ JSON report saved to:[/green] {output}\n")


if __name__ == "__main__":
    app()