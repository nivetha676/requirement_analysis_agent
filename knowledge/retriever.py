# knowledge/retriever.py
# ─────────────────────────────────────────────────────────────
# Loads a previously built knowledge base from disk and exposes
# retrieve() for use by the analyser.
#
# This replaces vector_store.py for the agent runtime —
# no building, no embedding, just load and query.
# ─────────────────────────────────────────────────────────────

import json
import pickle
from pathlib import Path

from rich.console import Console

from config import EMBED_MODEL, DEFAULT_TOP_K, KB_STORE_PATH

console = Console()

# Module-level singletons
_collection  = None
_embedder    = None
_active_name = None   # which KB is currently loaded


# ── Internal path helpers (mirror builder.py) ─────────────────

def _kb_dir(name: str)      -> Path: return KB_STORE_PATH / name
def _meta_path(name: str)   -> Path: return _kb_dir(name) / "meta.json"
def _chroma_path(name: str) -> Path: return _kb_dir(name) / "chroma"
def _embedder_path(name: str) -> Path: return _kb_dir(name) / "embedder.pkl"


# ── Public API ────────────────────────────────────────────────

def load(name: str) -> bool:
    """
    Load a saved knowledge base by name into memory.
    Must be called before retrieve().
    Returns True on success.
    """
    global _collection, _embedder, _active_name

    kb_dir = _kb_dir(name)

    if not kb_dir.exists():
        console.print(
            f"[red]Knowledge base '[bold]{name}[/bold]' not found.[/red]\n"
            f"[dim]Available: run  python -m knowledge.builder --list[/dim]\n"
            f"[dim]Build one: run  python -m knowledge.builder --name {name} --docs ./your_docs/[/dim]"
        )
        return False

    # ── Load metadata ─────────────────────────────────────────
    meta = json.loads(_meta_path(name).read_text())
    console.print(f"\n[bold]Loading knowledge base:[/bold] [cyan]{name}[/cyan]")
    console.print(f"  Chunks      : {meta['total_chunks']}")
    console.print(f"  Sources     : {', '.join(meta['sources'])}")
    console.print(f"  Embed model : {meta['embed_model']}")
    console.print(f"  Created     : {meta['created_at'][:19].replace('T', '  ')}\n")

    # ── Load embedding model ──────────────────────────────────
    embed_model = EMBED_MODEL
    if _embedder_path(name).exists():
        with open(_embedder_path(name), "rb") as f:
            embed_model = pickle.load(f)

    console.print(f"  Loading embedding model [bold]{embed_model}[/bold] ...")
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(embed_model)

    # ── Load persistent ChromaDB ──────────────────────────────
    import chromadb
    console.print(f"  Loading ChromaDB from disk ...")
    chroma      = chromadb.PersistentClient(path=str(_chroma_path(name)))
    _collection = chroma.get_collection("knowledge_base")
    _active_name = name

    console.print(f"  [green]✓ Ready[/green] — {_collection.count()} chunks loaded\n")
    return True


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Return the top-k most relevant chunks for *query* as a formatted string.
    Returns empty string if no knowledge base is loaded.
    """
    if _collection is None or _embedder is None:
        return ""

    query_vec = _embedder.encode([query]).tolist()
    results   = _collection.query(query_embeddings=query_vec, n_results=top_k)

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas",  [[]])[0]

    parts = [
        f"[Source: {meta['source']}]\n{doc}"
        for doc, meta in zip(docs, metas)
    ]
    return "\n\n---\n\n".join(parts)


def sources_for(query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
    """Return unique source filenames for the top-k chunks matching *query*."""
    if _collection is None or _embedder is None:
        return []

    query_vec = _embedder.encode([query]).tolist()
    results   = _collection.query(query_embeddings=query_vec, n_results=top_k)
    metas     = results.get("metadatas", [[]])[0]

    seen: list[str] = []
    for m in metas:
        if m["source"] not in seen:
            seen.append(m["source"])
    return seen


def is_ready() -> bool:
    return _collection is not None


def active_name() -> str | None:
    return _active_name


def get_meta(name: str) -> dict | None:
    """Return metadata for a saved knowledge base without loading it."""
    mp = _meta_path(name)
    if not mp.exists():
        return None
    return json.loads(mp.read_text())


def list_available() -> list[dict]:
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
