# analyser.py
# ─────────────────────────────────────────────────────────────
# Core analysis logic.
# Sends one requirement at a time to the local LLM and returns
# a structured result dict.
# ─────────────────────────────────────────────────────────────

import json
import re

from openai import OpenAI
from rich.console import Console

import knowledge.vector_store as vs
from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, DEFAULT_TOP_K
from prompts import SYSTEM_PROMPT, build_user_prompt

console = Console()

# Shared OpenAI-compatible client (Ollama / LM Studio)
_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


# ── Single-requirement analysis ───────────────────────────────

def analyse_one(req_text: str, req_id: str, model: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Analyse a single requirement and return a result dict.
    Raises json.JSONDecodeError or ValueError on bad model output.
    """
    context  = vs.retrieve(req_text, top_k=top_k)
    user_msg = build_user_prompt(req_text, context)

    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON object even when the model wraps it in prose or fences
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model output:\n{raw[:300]}")

    result       = json.loads(raw[start:end])
    result["id"] = req_id

    # Attach which knowledge-base sources were consulted
    if vs.is_ready():
        result["_sources_used"] = vs.sources_for(req_text, top_k=top_k)

    return result


# ── Batch analysis ────────────────────────────────────────────

def analyse_all(requirements_text: str, model: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Parse *requirements_text* into individual requirements, analyse each one,
    and return a list of result dicts.
    """
    reqs    = _parse_requirements(requirements_text)
    results = []

    console.print(f"[bold]Analysing {len(reqs)} requirement(s)[/bold] — one at a time\n")

    for req_id, req_text in reqs:
        preview = req_text[:70] + ("..." if len(req_text) > 70 else "")
        console.print(f"  → [bold]{req_id}[/bold]: {preview}")
        try:
            result = analyse_one(req_text, req_id, model, top_k=top_k)
            results.append(result)
        except json.JSONDecodeError as e:
            console.print(f"    [red]JSON parse error on {req_id}: {e}[/red]")
        except Exception as e:
            console.print(f"    [red]Error on {req_id}: {e}[/red]")

    console.print()
    return results


# ── Helpers ───────────────────────────────────────────────────

def _parse_requirements(text: str) -> list[tuple[str, str]]:
    """
    Split free-form text into (req_id, req_text) pairs.
    Strips leading numbering like "1." "REQ-01:" "2)" etc.
    """
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
