#!/usr/bin/env python3
"""
Requirements Ambiguity Analysis Agent — with RAG Knowledge Base
----------------------------------------------------------------
Analyses software requirements using domain knowledge files for
better accuracy on specialised topics (Bluetooth, payments, etc.)

Usage:
    python requirements_agent.py -f reqs.txt
    python requirements_agent.py -f reqs.txt -k docs/bluetooth.pdf -k docs/spec.md
    python requirements_agent.py -f reqs.txt -k ./knowledge/ -m mistral
    python requirements_agent.py -f reqs.txt -k docs/ -o report.json
"""

import json
import os
import re
import typer
from pathlib import Path
from typing import Optional
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

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, small, good quality — runs fully offline
CHUNK_SIZE    = 400                    # characters per chunk
CHUNK_OVERLAP = 80                     # overlap between chunks to preserve context
TOP_K         = 5                      # how many chunks to retrieve per requirement

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_markdown(path: Path) -> str:
    import markdown, re
    html = markdown.markdown(load_text_file(path))
    return re.sub(r"<[^>]+>", " ", html)


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".md", ".markdown"):
        return load_markdown(path)
    elif ext in (".txt", ".rst", ".text"):
        return load_text_file(path)
    else:
        # Try plain text for anything else
        try:
            return load_text_file(path)
        except Exception:
            console.print(f"  [yellow]Skipping unsupported file: {path.name}[/yellow]")
            return ""


def collect_files(paths: list[Path]) -> list[Path]:
    """Expand directories and collect all supported files."""
    supported = {".pdf", ".txt", ".md", ".markdown", ".rst", ".text"}
    files = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in supported and f.is_file():
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            console.print(f"  [yellow]Path not found: {p}[/yellow]")
    return files

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with source metadata."""
    chunks = []
    text   = re.sub(r"\s+", " ", text).strip()
    start  = 0
    idx    = 0

    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:   # skip tiny fragments
            chunks.append({
                "id":     f"{source}::chunk_{idx}",
                "text":   chunk,
                "source": source,
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# ---------------------------------------------------------------------------
# Vector store (ChromaDB + local embeddings)
# ---------------------------------------------------------------------------

_collection = None   # module-level singleton


def build_knowledge_base(kb_paths: list[Path]) -> bool:
    """Load documents, chunk, embed, and store in a local ChromaDB collection.
    Returns True if any documents were loaded."""
    global _collection

    import chromadb
    from sentence_transformers import SentenceTransformer

    files = collect_files(kb_paths)
    if not files:
        console.print("  [yellow]No knowledge base files found.[/yellow]")
        return False

    console.print(f"\n[bold]Building knowledge base[/bold] from {len(files)} file(s)...\n")

    # Load embedding model (downloads once, cached locally at ~/.cache/huggingface)
    console.print(f"  Loading embedding model [bold]{EMBED_MODEL}[/bold]...")
    embedder = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    for f in files:
        console.print(f"  Reading [dim]{f.name}[/dim]")
        text = load_document(f)
        if text.strip():
            chunks = chunk_text(text, f.name)
            all_chunks.extend(chunks)
            console.print(f"    → {len(chunks)} chunks")

    if not all_chunks:
        console.print("  [yellow]No content extracted from knowledge base files.[/yellow]")
        return False

    # Embed all chunks
    console.print(f"\n  Embedding [bold]{len(all_chunks)}[/bold] chunks...")
    texts      = [c["text"]   for c in all_chunks]
    ids        = [c["id"]     for c in all_chunks]
    metadatas  = [{"source": c["source"]} for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # Build in-memory ChromaDB collection (no disk, no server)
    chroma     = chromadb.Client()
    _collection = chroma.create_collection("knowledge_base")
    _collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    console.print(f"\n  [green]✓ Knowledge base ready[/green] — {len(all_chunks)} chunks indexed\n")
    return True


def retrieve_context(requirement: str, top_k: int = TOP_K) -> str:
    """Return the top-k most relevant chunks for a requirement as a formatted string."""
    if _collection is None:
        return ""

    from sentence_transformers import SentenceTransformer
    embedder   = SentenceTransformer(EMBED_MODEL)
    query_vec  = embedder.encode([requirement]).tolist()
    results    = _collection.query(query_embeddings=query_vec, n_results=top_k)

    docs     = results.get("documents", [[]])[0]
    metas    = results.get("metadatas",  [[]])[0]
    context  = []

    for doc, meta in zip(docs, metas):
        context.append(f"[Source: {meta['source']}]\n{doc}")

    return "\n\n---\n\n".join(context)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior QA analyst and requirements engineer with deep domain expertise.
Your job is to deeply inspect each requirement and identify EXACTLY what information is missing
before a tester can write test cases.

When domain knowledge is provided, use it to:
- Identify domain-specific values, thresholds, and standards that are missing
- Spot violations of known protocols or specifications
- Reference specific standards (e.g. Bluetooth Core Spec section numbers, PCI-DSS clauses)
- Detect assumptions that contradict the domain specification

Return a JSON array only — no markdown fences, no prose before or after.

Each object must have these fields:

- id: "REQ-01", "REQ-02", etc.
- requirementText: the requirement exactly as given
- ambiguityLevel: one of "High" | "Medium" | "Low" | "Clear"

- missingInformation: list of strings — concrete facts ABSENT from the requirement.
  Be specific. Reference domain knowledge where relevant. e.g.:
  "No maximum password length specified (NIST SP 800-63B recommends at least 64 chars)"
  "No Bluetooth version specified — behaviour differs between BT 4.2, 5.0, and 5.3"
  "No timeout value defined — Bluetooth spec section 7.8.5 defines supervision timeout range"
  NOT vague statements like "more detail needed".

- ambiguousTerms: list of objects, each with:
    - term: the exact word or phrase that is vague
    - problem: why it is ambiguous for testing
    - example: two conflicting interpretations a tester might make
  e.g. "fast", "secure", "nearby", "compatible", "standard", "regularly", "large"

- missingAcceptanceCriteria: list of strings — measurable pass/fail rules that are absent.
  Reference domain standards where known. e.g.:
  "No RSSI threshold defined for 'nearby' — typical BLE proximity is -70 dBm to -90 dBm"
  "No pairing timeout specified — Bluetooth Core Spec allows 30 seconds default"
  "No definition of which Bluetooth profiles must be supported (A2DP, HFP, GATT, etc.)"

- edgeCasesNotCovered: list of strings — boundary conditions and error scenarios absent. e.g.:
  "Behaviour when Bluetooth is disabled on the device"
  "What happens when two devices attempt to pair simultaneously"
  "Behaviour at maximum connection range boundary"
  "What happens when bonding information is lost on one side only"

- clarifyingQuestions: list of strings — specific, answerable questions for the product owner.
  Good: "Which Bluetooth version (4.2 / 5.0 / 5.3) must be supported?"
  Bad:  "Can you clarify the Bluetooth requirements?"

- rewriteSuggestion: string — a rewritten requirement that is fully testable and measurable,
  incorporating domain knowledge where relevant. Empty string if already Clear.

Ambiguity levels:
- High:   3+ missing pieces, core behaviour undefined — test cases cannot be written
- Medium: 1-2 missing pieces — partial test coverage possible
- Low:    minor gaps — test cases writable with small stated assumptions
- Clear:  fully specified and measurable — test cases can be written immediately"""


def _build_user_prompt(requirement: str, context: str) -> str:
    if context:
        return (
            "Use the following domain knowledge to inform your analysis. "
            "Reference specific details from it where relevant.\n\n"
            f"=== DOMAIN KNOWLEDGE ===\n{context}\n=== END DOMAIN KNOWLEDGE ===\n\n"
            "Now analyse this requirement:\n\n"
            f"{requirement}\n\n"
            "Think step by step:\n"
            "1. What exact values, thresholds, or states are missing?\n"
            "2. Which words are vague — what would two testers assume differently?\n"
            "3. Which measurable pass/fail conditions are absent?\n"
            "4. What error paths and edge cases are unspecified?\n"
            "5. What domain standards apply that the requirement ignores?\n\n"
            "Then output a single JSON object (not array) for this requirement."
        )
    else:
        return (
            f"Analyse this requirement:\n\n{requirement}\n\n"
            "Think step by step:\n"
            "1. What exact values, thresholds, or states are missing?\n"
            "2. Which words are vague — what would two testers assume differently?\n"
            "3. Which measurable pass/fail conditions are absent?\n"
            "4. What error paths and edge cases are unspecified?\n\n"
            "Then output a single JSON object (not array) for this requirement."
        )

# ---------------------------------------------------------------------------
# Analysis — one requirement at a time for best quality
# ---------------------------------------------------------------------------

def analyse_single(req_text: str, req_id: str, model: str) -> dict:
    context = retrieve_context(req_text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(req_text, context)},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Extract the JSON object even if the model adds prose around it
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model output:\n{raw}")

    obj = json.loads(raw[start:end])
    obj["id"] = req_id

    # Show which sources were used
    if context and _collection is not None:
        sources = list({m["source"] for m in
                        _collection.query(
                            query_embeddings=SentenceTransformer(EMBED_MODEL)
                                            .encode([req_text]).tolist(),
                            n_results=TOP_K
                        ).get("metadatas", [[]])[0]})
        obj["_sources_used"] = sources

    return obj


def parse_requirements(text: str) -> list[tuple[str, str]]:
    """Return list of (req_id, req_text) pairs from free-form input."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result = []
    for i, line in enumerate(lines, 1):
        # Strip leading numbering like "1." "1)" "REQ-01:" etc.
        clean = re.sub(r"^(REQ-?\d+[:\.\)]\s*|\d+[:\.\)]\s*)", "", line, flags=re.I).strip()
        if clean:
            result.append((f"REQ-{i:02d}", clean))
    return result


def analyse(requirements_text: str, model: str) -> list[dict]:
    reqs    = parse_requirements(requirements_text)
    results = []

    console.print(f"[bold]Analysing {len(reqs)} requirement(s)[/bold] one at a time...\n")

    for req_id, req_text in reqs:
        console.print(f"  → [bold]{req_id}[/bold]: {req_text[:70]}{'...' if len(req_text) > 70 else ''}")
        try:
            result = analyse_single(req_text, req_id, model)
            results.append(result)
        except json.JSONDecodeError as e:
            console.print(f"    [red]JSON parse error on {req_id}: {e}[/red]")
        except Exception as e:
            console.print(f"    [red]Error on {req_id}: {e}[/red]")

    console.print()
    return results

# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

LEVEL_STYLES = {
    "High":   ("red",    "🔴"),
    "Medium": ("yellow", "🟡"),
    "Low":    ("green",  "🟢"),
    "Clear":  ("cyan",   "✅"),
}


def print_report(results: list[dict]) -> None:
    counts = {"High": 0, "Medium": 0, "Low": 0, "Clear": 0}
    for r in results:
        lvl = r.get("ambiguityLevel", "Clear")
        counts[lvl] = counts.get(lvl, 0) + 1

    summary = Table(box=box.ROUNDED, title="[bold]Analysis Summary[/bold]", show_header=False)
    summary.add_column("Metric", style="dim", min_width=36)
    summary.add_column("Count", justify="right", min_width=6)
    summary.add_row("Total requirements analysed",                   str(len(results)))
    summary.add_row("", "")
    summary.add_row("[red]Need more info  (High + Medium)[/red]",    str(counts["High"] + counts["Medium"]))
    summary.add_row("[red]  High ambiguity[/red]",                   str(counts["High"]))
    summary.add_row("[yellow]  Medium ambiguity[/yellow]",           str(counts["Medium"]))
    summary.add_row("[green]  Low ambiguity[/green]",                str(counts["Low"]))
    summary.add_row("[cyan]  Clear & testable[/cyan]",               str(counts["Clear"]))
    console.print(summary)
    console.print()

    for r in results:
        lvl   = r.get("ambiguityLevel", "Clear")
        color, icon = LEVEL_STYLES.get(lvl, ("white", "❓"))

        title = Text()
        title.append(f"{icon}  {r.get('id', 'REQ')}  ", style="bold")
        title.append(f"[{lvl}]", style=f"bold {color}")

        body = Text()
        body.append(r.get("requirementText", ""), style="bold white")
        body.append("\n")

        # Sources used from knowledge base
        sources = r.get("_sources_used", [])
        if sources:
            body.append(f"\n📚  Knowledge base used: ", style="bold dim")
            body.append(", ".join(sources) + "\n", style="dim")

        if r.get("missingInformation"):
            body.append(f"\n🕳  What is missing:\n", style=f"bold {color}")
            for item in r["missingInformation"]:
                body.append(f"  • {item}\n", style=color)

        if r.get("ambiguousTerms"):
            body.append("\n🌫  Vague terms:\n", style="bold magenta")
            for t in r["ambiguousTerms"]:
                body.append(f"  • \"{t.get('term','')}\" — {t.get('problem','')}\n", style="magenta")
                if t.get("example"):
                    body.append(f"    ↳ {t['example']}\n", style="dim")

        if r.get("missingAcceptanceCriteria"):
            body.append("\n📏  Missing acceptance criteria:\n", style="bold yellow")
            for ac in r["missingAcceptanceCriteria"]:
                body.append(f"  • {ac}\n", style="yellow")

        if r.get("edgeCasesNotCovered"):
            body.append("\n⚠️  Edge cases not covered:\n", style="bold red")
            for ec in r["edgeCasesNotCovered"]:
                body.append(f"  • {ec}\n", style="red")

        if r.get("clarifyingQuestions"):
            body.append("\n❓  Ask the product owner:\n", style="bold yellow")
            for q in r["clarifyingQuestions"]:
                body.append(f"  {q}\n", style="yellow")

        if r.get("rewriteSuggestion"):
            body.append("\n✏️  Suggested rewrite:\n", style="bold blue")
            body.append(f"  {r['rewriteSuggestion']}\n", style="blue")

        console.print(Panel(body, title=title, border_style=color, expand=False))
        console.print()

# ---------------------------------------------------------------------------
# Sample requirements
# ---------------------------------------------------------------------------

SAMPLE_REQUIREMENTS = """\
1. The device must support Bluetooth connectivity
2. The app should connect to nearby Bluetooth devices quickly
3. Bluetooth pairing must be secure
4. The system must handle multiple Bluetooth connections
5. Users should receive a notification when a Bluetooth device is out of range"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    file: Path = typer.Option(
        None, "--file", "-f",
        help="Path to requirements file (.txt).",
    ),
    knowledge: list[Path] = typer.Option(
        [], "--knowledge", "-k",
        help="Knowledge base file or directory. Can be repeated: -k bt.pdf -k specs/",
    ),
    model: str = typer.Option(
        "llama3.2", "--model", "-m",
        help="Ollama model. Recommended: mistral, llama3.1:8b, gemma3:12b.",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save JSON report to this path.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Paste requirements interactively.",
    ),
    top_k: int = typer.Option(
        TOP_K, "--top-k",
        help="Number of knowledge base chunks to retrieve per requirement.",
    ),
):
    """
    Analyse software requirements using local LLM + optional RAG knowledge base.
    Pass domain documents (-k) to get domain-aware analysis.
    """

    console.rule("[bold]Requirements Ambiguity Analysis Agent[/bold]")

    # Build knowledge base if provided
    if knowledge:
        has_kb = build_knowledge_base(list(knowledge))
        if not has_kb:
            console.print("[yellow]Continuing without knowledge base.[/yellow]\n")
    else:
        console.print("[dim]No knowledge base provided. Running without domain context.[/dim]")
        console.print("[dim]Tip: use -k path/to/spec.pdf to improve domain accuracy.\n[/dim]")

    # Resolve input
    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        text = file.read_text(encoding="utf-8").strip()
        console.print(f"[dim]Requirements from:[/dim] {file}\n")
    elif interactive:
        console.print("[bold]Paste requirements. Press Enter twice when done.\n[/bold]")
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

    # Run analysis
    try:
        results = analyse(text, model)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("[yellow]Is Ollama running?[/yellow]  ollama serve")
        raise typer.Exit(1)

    if not results:
        console.print("[red]No requirements were successfully analysed.[/red]")
        raise typer.Exit(1)

    print_report(results)

    if output:
        # Remove internal keys before saving
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        output.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]✓ Report saved to:[/green] {output}\n")


if __name__ == "__main__":
    app()