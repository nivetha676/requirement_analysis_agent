# knowledge/loader.py
# ─────────────────────────────────────────────────────────────
# Loads and chunks knowledge-base documents (PDF, TXT, MD, RST).
# ─────────────────────────────────────────────────────────────

import re
from pathlib import Path

from rich.console import Console

from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

console = Console()


# ── File readers ──────────────────────────────────────────────

def _read_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_markdown(path: Path) -> str:
    import markdown
    html = markdown.markdown(_read_plain(path))
    return re.sub(r"<[^>]+>", " ", html)


_READERS = {
    ".pdf":      _read_pdf,
    ".md":       _read_markdown,
    ".markdown": _read_markdown,
    ".txt":      _read_plain,
    ".rst":      _read_plain,
    ".text":     _read_plain,
}


def load_document(path: Path) -> str:
    """Load a single file and return its text content."""
    ext    = path.suffix.lower()
    reader = _READERS.get(ext)

    if reader is None:
        try:
            return _read_plain(path)
        except Exception:
            console.print(f"  [yellow]Skipping unsupported file: {path.name}[/yellow]")
            return ""

    try:
        return reader(path)
    except Exception as e:
        console.print(f"  [yellow]Could not read {path.name}: {e}[/yellow]")
        return ""


# ── Directory / path expansion ────────────────────────────────

def collect_files(paths: list[Path]) -> list[Path]:
    """Expand a mix of files and directories into a flat list of supported files."""
    files: list[Path] = []

    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file():
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            console.print(f"  [yellow]Path not found: {p}[/yellow]")

    return files


# ── Chunking ──────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping fixed-size chunks.

    Returns a list of dicts:
        { "id": str, "text": str, "source": str }
    """
    chunks: list[dict] = []
    text  = re.sub(r"\s+", " ", text).strip()
    start = 0
    idx   = 0

    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if len(chunk) > 50:          # skip tiny tail fragments
            chunks.append({
                "id":     f"{source}::chunk_{idx}",
                "text":   chunk,
                "source": source,
            })
            idx += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
