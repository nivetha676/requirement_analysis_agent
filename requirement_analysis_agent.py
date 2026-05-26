import json
import typer
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

app = typer.Typer()
console = Console()

# Point to your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the SDK but not used by Ollama
)

SYSTEM_PROMPT = """You are a senior QA analyst and requirements engineer.
Analyse software requirements and return a JSON array only — no markdown fences, no explanation.

Each object in the array must have:
- id: string like "REQ-01"
- requirementText: string (the requirement as given)
- ambiguityLevel: exactly one of "High", "Medium", "Low", "Clear"
- issues: list of strings describing specific ambiguity problems (empty list if Clear)
- clarifyingQuestions: list of strings a tester needs answered before writing test cases (empty if Clear)
- testabilitySuggestion: string with one concrete rewrite tip to make it testable (empty string if already testable)

Ambiguity levels:
- High: vague, no acceptance criteria, unmeasurable — test cases cannot be written
- Medium: partially defined, some gaps — test cases possible but incomplete
- Low: mostly clear, minor gaps — test cases writable with assumptions
- Clear: fully specified and measurable — test cases can be written immediately

Return ONLY the raw JSON array. No prose before or after it."""


def analyse(requirements_text: str, model: str) -> list[dict]:
    console.print(f"\n[dim]Analysing with model:[/dim] [bold]{model}[/bold]")

    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyse these requirements:\n\n{requirements_text}"},
            ],
            temperature=0.2,  # low temp = more consistent structured output
        )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences some models add anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def print_report(results: list[dict]):
    level_styles = {
        "High":   ("red",    "🔴"),
        "Medium": ("yellow", "🟡"),
        "Low":    ("green",  "🟢"),
        "Clear":  ("cyan",   "✅"),
    }

    counts = {"High": 0, "Medium": 0, "Low": 0, "Clear": 0}
    for r in results:
        counts[r["ambiguityLevel"]] = counts.get(r["ambiguityLevel"], 0) + 1

    # Summary table
    summary = Table(box=box.ROUNDED, title="Summary", show_header=True)
    summary.add_column("Metric", style="dim")
    summary.add_column("Count", justify="right")

    summary.add_row("Total requirements", str(len(results)))
    summary.add_row("[red]Need more info (High + Medium)[/red]",
                    str(counts["High"] + counts["Medium"]))
    summary.add_row("[red]High ambiguity[/red]", str(counts["High"]))
    summary.add_row("[yellow]Medium ambiguity[/yellow]", str(counts["Medium"]))
    summary.add_row("[green]Low ambiguity[/green]", str(counts["Low"]))
    summary.add_row("[cyan]Clear & testable[/cyan]", str(counts["Clear"]))
    console.print(summary)
    console.print()

    # Per-requirement panels
    for r in results:
        lvl = r.get("ambiguityLevel", "Clear")
        color, icon = level_styles.get(lvl, ("white", "❓"))

        title = Text()
        title.append(f"{icon} {r['id']}  ", style="bold")
        title.append(f"[{lvl}]", style=f"bold {color}")

        body = Text()
        body.append(r["requirementText"] + "\n", style="bold white")

        if r.get("issues"):
            body.append("\nIssues found:\n", style=f"bold {color}")
            for issue in r["issues"]:
                body.append(f"  • {issue}\n", style=color)

        if r.get("clarifyingQuestions"):
            body.append("\nClarifying questions needed:\n", style="bold yellow")
            for q in r["clarifyingQuestions"]:
                body.append(f"  ❓ {q}\n", style="yellow")

        if r.get("testabilitySuggestion"):
            body.append(f"\n💡 Suggestion: ", style="bold blue")
            body.append(r["testabilitySuggestion"] + "\n", style="blue")

        console.print(Panel(body, title=title, border_style=color, expand=False))
        console.print()


@app.command()
def main(
    file: Path = typer.Option(None, "--file", "-f", help="Path to a .txt file with requirements"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model name to use"),
    output: Path = typer.Option(None, "--output", "-o", help="Save JSON report to this path"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Type/paste requirements interactively"),
):
    """Analyse project requirements for ambiguity and testability gaps."""

    if file:
        text = file.read_text()
    elif interactive:
        console.print("[bold]Paste your requirements below. Enter a blank line twice to finish.[/bold]")
        lines, blanks = [], 0
        while blanks < 2:
            line = input()
            if line == "":
                blanks += 1
            else:
                blanks = 0
                lines.append(line)
        text = "\n".join(lines)
    else:
        # Demo mode with sample requirements
        text = """1. The system should be fast and responsive
2. Users must register with email and a password meeting security standards
3. Admins can manage user accounts
4. The app must support multiple languages
5. Payment processing must be secure and PCI-DSS compliant
6. Users get notified when something important happens
7. Reports export data in CSV and PDF format with all visible columns
8. Search returns results within 2 seconds for datasets under 1 million records"""
        console.print("[dim]No input provided. Running with sample requirements.[/dim]\n")

    try:
        results = analyse(text, model)
        print_report(results)

        if output:
            output.write_text(json.dumps(results, indent=2))
            console.print(f"[green]Report saved to {output}[/green]")

    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse model output as JSON: {e}[/red]")
        console.print("[dim]Try a larger model with --model mistral or --model gemma3:12b[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()