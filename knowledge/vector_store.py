# knowledge/vector_store.py
# ─────────────────────────────────────────────────────────────
# Builds and queries a local ChromaDB vector store using
# sentence-transformers embeddings — fully offline, no servers.
# ─────────────────────────────────────────────────────────────

from pathlib import Path

from rich.console import Console

from config import EMBED_MODEL, DEFAULT_TOP_K, CHROMA_BATCH_SIZE
from knowledge.loader import collect_files, load_document, chunk_text

console = Console()

# Module-level singletons — initialised by build()
_collection = None
_embedder   = None


# ── Public API ────────────────────────────────────────────────

def build(kb_paths: list[Path]) -> bool:
    """
    Load documents, chunk, embed, and store in an in-memory ChromaDB collection.
    Returns True if at least one chunk was indexed.
    """
    global _collection, _embedder

    import chromadb
    from sentence_transformers import SentenceTransformer

    files = collect_files(kb_paths)
    if not files:
        console.print("  [yellow]No knowledge-base files found.[/yellow]")
        return False

    console.print(f"\n[bold]Building knowledge base[/bold] — {len(files)} file(s) found\n")

    # Load embedding model (cached at ~/.cache/huggingface after first download)
    console.print(f"  Loading embedding model [bold]{EMBED_MODEL}[/bold] ...")
    _embedder = SentenceTransformer(EMBED_MODEL)

    # Read + chunk every file
    all_chunks: list[dict] = []
    for f in files:
        console.print(f"  Reading  [dim]{f.name}[/dim]")
        text = load_document(f)
        if text.strip():
            chunks = chunk_text(text, f.name)
            all_chunks.extend(chunks)
            console.print(f"           → {len(chunks)} chunks")

    if not all_chunks:
        console.print("  [yellow]No content extracted from knowledge-base files.[/yellow]")
        return False

    # Embed
    console.print(f"\n  Embedding [bold]{len(all_chunks)}[/bold] chunks ...")
    texts      = [c["text"]   for c in all_chunks]
    ids        = [c["id"]     for c in all_chunks]
    metadatas  = [{"source": c["source"]} for c in all_chunks]
    embeddings = _embedder.encode(texts, show_progress_bar=True).tolist()

    # Store in-memory (no disk, no external server)
    chroma      = chromadb.Client()
    _collection = chroma.create_collection("knowledge_base")
    # _collection.add(
    #     ids=ids,
    #     documents=texts,
    #     embeddings=embeddings,
    #     metadatas=metadatas,
    # )

    console.print(f"  Storing chunks in ChromaDB ...")
    total   = len(all_chunks)
    batches = (total + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE   # ceiling division

    for i in range(batches):
        start = i * CHROMA_BATCH_SIZE
        end   = min(start + CHROMA_BATCH_SIZE, total)

        _collection.add(
            ids        = ids[start:end],
            documents  = texts[start:end],
            embeddings = embeddings[start:end],
            metadatas  = metadatas[start:end],
        )
        console.print(f"    Batch {i+1}/{batches} — chunks {start+1} to {end} stored")

    console.print(f"\n  [green]✓ Knowledge base ready[/green] — {total} chunks indexed\n")

    # console.print(f"\n  [green]✓ Knowledge base ready[/green] — {len(all_chunks)} chunks indexed\n")
    return True


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Return the top-k most relevant chunks for *query* as a formatted string.
    Returns an empty string when no knowledge base has been built.
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
