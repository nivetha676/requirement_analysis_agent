# analyser.py
# ─────────────────────────────────────────────────────────────
# Core analysis logic.
# Uses knowledge.retriever (not vector_store) — KB is pre-built,
# just loaded and queried here.
# ─────────────────────────────────────────────────────────────

import json
import re
import time

from openai import OpenAI
from rich.console import Console

import knowledge.retriever as retriever
from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, DEFAULT_TOP_K
from prompts import SYSTEM_PROMPT, build_user_prompt

console = Console()

_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    timeout=120.0,
)


# ── JSON cleaning ─────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model output:\n{raw[:400]}")
    text = raw[start:end]

    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"(?<![\\])'([^']*)'", r'"\1"', text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\.\.\.", "", text)
    text = re.sub(r'(?<!")(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*:', r'"\1":', text)
    text = re.sub(r'""([^"]+)""', r'"\1"', text)
    return text.strip()


def _parse_json_safe(raw: str) -> dict:
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError as e:
        cleaned = _clean_json(raw)
        line_no = e.lineno
        lines   = cleaned.splitlines()
        context = "\n".join(
            f"  {'>>>' if i + 1 == line_no else '   '} {l}"
            for i, l in enumerate(lines[max(0, line_no - 3): line_no + 2],
                                  start=max(0, line_no - 3))
        )
        raise json.JSONDecodeError(
            f"Could not fix JSON.\nError at line {e.lineno} col {e.colno}: {e.msg}\n\n"
            f"Context:\n{context}",
            e.doc, e.pos,
        )


# ── Single-requirement analysis ───────────────────────────────

def analyse_one(req_text: str, req_id: str, model: str, top_k: int = DEFAULT_TOP_K) -> dict:
    context  = retriever.retrieve(req_text, top_k=top_k)
    user_msg = build_user_prompt(req_text, context)

    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
    )

    raw    = response.choices[0].message.content.strip()
    result = _parse_json_safe(raw)
    result["id"] = req_id

    if retriever.is_ready():
        result["_sources_used"] = retriever.sources_for(req_text, top_k=top_k)

    return result


# ── Retry wrapper ─────────────────────────────────────────────

def _analyse_with_retry(
    req_text: str,
    req_id: str,
    model: str,
    top_k: int,
    max_retries: int = 3,
    wait_seconds: float = 5.0,
) -> dict | None:

    for attempt in range(1, max_retries + 1):
        try:
            return analyse_one(req_text, req_id, model, top_k=top_k)

        except json.JSONDecodeError as e:
            console.print(f"    [red]JSON error (attempt {attempt}/{max_retries}):[/red] {e.msg}")
            if attempt == max_retries:
                console.print(
                    f"    [red]Skipping {req_id} after {max_retries} attempts.[/red]\n"
                    f"    [dim]Try a larger model: -m mistral  or  -m llama3.1:8b[/dim]"
                )
                return None

        except Exception as e:
            error_str = str(e).lower()
            is_conn   = any(w in error_str for w in ("connection", "timeout", "refused", "reset"))

            if is_conn:
                console.print(f"    [yellow]Connection error (attempt {attempt}/{max_retries}): {e}[/yellow]")
                if attempt < max_retries:
                    console.print(f"    [dim]Retrying in {wait_seconds}s...[/dim]")
                    time.sleep(wait_seconds)
                else:
                    console.print(f"    [red]Skipping {req_id} — Ollama unreachable.[/red]")
                    return None
            else:
                console.print(f"    [red]Unexpected error on {req_id}: {e}[/red]")
                return None

    return None


# ── Batch analysis ────────────────────────────────────────────

def analyse_all(requirements_text: str, model: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    reqs    = _parse_requirements(requirements_text)
    results = []

    console.print(f"[bold]Analysing {len(reqs)} requirement(s)[/bold] — one at a time\n")

    for req_id, req_text in reqs:
        preview = req_text[:70] + ("..." if len(req_text) > 70 else "")
        console.print(f"  → [bold]{req_id}[/bold]: {preview}")
        result = _analyse_with_retry(req_text, req_id, model, top_k)
        if result:
            results.append(result)

    console.print()
    return results


# ── Helpers ───────────────────────────────────────────────────

def _parse_requirements(text: str) -> list[tuple[str, str]]:
    lines  = [l.strip() for l in text.splitlines() if l.strip()]
    result = []
    for i, line in enumerate(lines, 1):
        clean = re.sub(
            r"^(REQ-?\d+[:\.\)]\s*|\d+[:\.\)]\s*)",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if clean:
            result.append((f"REQ-{i:02d}", clean))
    return result
