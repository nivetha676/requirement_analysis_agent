# knowledge/builder.py
# ─────────────────────────────────────────────────────────────
# Builds a knowledge base from documents and saves it to disk.
# Run this ONCE per knowledge base — not on every agent run.
#
# Usage:
#   python -m knowledge.builder --name bluetooth --docs ./docs/bluetooth/
#   python -m knowledge.builder --name payments  --docs spec.pdf --docs guide.md
#   python -m knowledge.builder --list
#   python -m knowledge.builder --delete bluetooth
# ─────────────────────────────────────────────────────────────

import json
import shutil
import pickle
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from config import (
    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_BATCH_SIZE, KB_STORE_PATH, SUPPORTED_EXTENSIONS,
)
from knowledge.loader import collect_files, load_document, chunk_text

app     = typer.Typer(add_completion=False)
console = Console()

# ── Paths ─────────────────────────────────────────────────────

def _kb_dir(name: str) -> Path:
    return KB_STORE_PATH / name

def _meta_path(name: str) -> Path:
    return _kb_dir(name) / "meta.json"

def _chroma_path(name: str) -> Path:
    return _kb_dir(name) / "chroma"

def _embedder_path(name: str) -> Path:
    return _kb_dir(name) / "embedder.pkl"


# ── Core build logic ──────────────────────────────────────────

def build_knowledge_base(name: str, doc_paths: list[Path], force: bool = False) -> bool:
    """
    Build a named knowledge base from *doc_paths* and persist it to disk.

    Directory layout:
        kb_store/
        └── <name>/
            ├── meta.json       ← name, date, sources, chunk count
            ├── chroma/         ← ChromaDB persistent storage
            └── embedder.pkl    ← serialised embedder model path (for reload)

    Returns True on success.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    kb_dir = _kb_dir(name)

    # Guard: refuse to overwrite unless --force
    if kb_dir.exists() and not force:
        console.print(
            f"[yellow]Knowledge base '[bold]{name}[/bold]' already exists.[/yellow]\n"
            f"Use [bold]--force[/bold] to overwrite it."
        )
        return False

    if kb_dir.exists():
        shutil.rmtree(kb_dir)

    kb_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect files ─────────────────────────────────────────
    files = collect_files(doc_paths)
    if not files:
        console.print("[red]No supported files found in the provided paths.[/red]")
        return False

    console.print(f"\n[bold]Building knowledge base:[/bold] [cyan]{name}[/cyan]")
    console.print(f"  Files found : {len(files)}")
    console.print(f"  Chunk size  : {CHUNK_SIZE} chars  (overlap {CHUNK_OVERLAP})\n")

    # ── Read + chunk ──────────────────────────────────────────
    all_chunks: list[dict] = []
    sources: list[str]     = []

    for f in files:
        console.print(f"  Reading  [dim]{f.name}[/dim]")
        text = load_document(f)
        if text.strip():
            chunks = chunk_text(text, f.name)
            all_chunks.extend(chunks)
            sources.append(f.name)
            console.print(f"           → {len(chunks)} chunks")

    if not all_chunks:
        console.print("[red]No content could be extracted from the files.[/red]")
        return False

    total = len(all_chunks)
    console.print(f"\n  Total chunks: [bold]{total}[/bold]\n")

    # ── Embed ─────────────────────────────────────────────────
    console.print(f"  Loading embedding model [bold]{EMBED_MODEL}[/bold] ...")
    embedder   = SentenceTransformer(EMBED_MODEL)
    texts      = [c["text"]   for c in all_chunks]
    ids        = [c["id"]     for c in all_chunks]
    metadatas  = [{"source": c["source"]} for c in all_chunks]

    console.print(f"  Embedding {total} chunks ...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # ── Store in persistent ChromaDB ──────────────────────────
    chroma_dir = _chroma_path(name)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma     = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.create_collection("knowledge_base")

    batches = (total + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE
    console.print(f"\n  Storing in ChromaDB ({batches} batch(es)) ...")

    for i in range(batches):
        s = i * CHROMA_BATCH_SIZE
        e = min(s + CHROMA_BATCH_SIZE, total)
        collection.add(
            ids        = ids[s:e],
            documents  = texts[s:e],
            embeddings = embeddings[s:e],
            metadatas  = metadatas[s:e],
        )
        console.print(f"    Batch {i+1}/{batches} — chunks {s+1}–{e} stored")

    # ── Save embedder model name for reload ───────────────────
    with open(_embedder_path(name), "wb") as f:
        pickle.dump(EMBED_MODEL, f)

    # ── Write metadata ────────────────────────────────────────
    meta = {
        "name":        name,
        "created_at":  datetime.now().isoformat(),
        "embed_model": EMBED_MODEL,
        "chunk_size":  CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_chunks": total,
        "sources":     sources,
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2))

    console.print(f"\n  [green]✓ Knowledge base '[bold]{name}[/bold]' saved[/green] → {kb_dir}\n")
    return True


# ── List / delete helpers ─────────────────────────────────────

def list_knowledge_bases() -> list[dict]:
    """Return metadata for all saved knowledge bases."""
    if not KB_STORE_PATH.exists():
        return []
    metas = []
    for meta_file in sorted(KB_STORE_PATH.glob("*/meta.json")):
        try:
            metas.append(json.loads(meta_file.read_text()))
        except Exception:
            pass
    return metas


def delete_knowledge_base(name: str) -> bool:
    kb_dir = _kb_dir(name)
    if not kb_dir.exists():
        console.print(f"[red]Knowledge base '{name}' not found.[/red]")
        return False
    shutil.rmtree(kb_dir)
    console.print(f"[green]✓ Deleted knowledge base '[bold]{name}[/bold]'[/green]")
    return True


def knowledge_base_exists(name: str) -> bool:
    return _kb_dir(name).exists() and _meta_path(name).exists()


# ── CLI ───────────────────────────────────────────────────────

@app.command()
def main(
    name: str = typer.Option(
        None, "--name", "-n",
        help="Name for the knowledge base (e.g. bluetooth, payments).",
    ),
    docs: list[Path] = typer.Option(
        [], "--docs", "-d",
        help="Document file or directory. Repeatable: -d spec.pdf -d ./guides/",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite an existing knowledge base with the same name.",
    ),
    list_kbs: bool = typer.Option(
        False, "--list", "-l",
        help="List all saved knowledge bases.",
    ),
    delete: str = typer.Option(
        None, "--delete",
        help="Delete a knowledge base by name.",
    ),
) -> None:
    """
    Build and manage persistent knowledge bases for the requirements agent.

    Examples:\n
        python -m knowledge.builder --name bluetooth --docs ./bt_docs/\n
        python -m knowledge.builder --name payments  --docs pci.pdf --docs guide.md\n
        python -m knowledge.builder --list\n
        python -m knowledge.builder --delete bluetooth
    """

    console.rule("[bold]Knowledge Base Builder[/bold]")

    # ── List ──────────────────────────────────────────────────
    if list_kbs:
        metas = list_knowledge_bases()
        if not metas:
            console.print("[dim]No knowledge bases found. Build one with --name and --docs.[/dim]")
            return

        table = Table(box=box.ROUNDED, title="[bold]Saved Knowledge Bases[/bold]")
        table.add_column("Name",         style="bold cyan")
        table.add_column("Chunks",       justify="right")
        table.add_column("Sources",      justify="right")
        table.add_column("Embed Model",  style="dim")
        table.add_column("Created",      style="dim")

        for m in metas:
            table.add_row(
                m["name"],
                str(m["total_chunks"]),
                str(len(m["sources"])),
                m["embed_model"],
                m["created_at"][:19].replace("T", "  "),
            )
        console.print(table)
        return

    # ── Delete ────────────────────────────────────────────────
    if delete:
        delete_knowledge_base(delete)
        return

    # ── Build ─────────────────────────────────────────────────
    if not name:
        console.print("[red]Provide --name for the knowledge base.[/red]")
        raise typer.Exit(1)

    if not docs:
        console.print("[red]Provide at least one --docs path.[/red]")
        raise typer.Exit(1)

    build_knowledge_base(name, list(docs), force=force)


if __name__ == "__main__":
    app()
