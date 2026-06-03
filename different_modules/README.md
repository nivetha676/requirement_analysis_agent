# Requirements Ambiguity Analysis Agent

Analyses software requirements for ambiguity, missing information, vague terms,
missing acceptance criteria, and edge cases — using a local LLM via Ollama.
Optionally loads domain documents as a knowledge base (RAG) for domain-aware analysis.

---

## Project structure

```
requirements_agent/
├── main.py                  ← CLI entry point          (run this)
├── config.py                ← All settings in one place
├── prompts.py               ← System prompt + user prompt builder
├── analyser.py              ← Calls Ollama, parses results
├── reporter.py              ← Rich terminal report rendering
├── requirements.txt         ← Python dependencies
└── knowledge/
    ├── __init__.py
    ├── loader.py            ← Reads PDF / TXT / MD / RST files + chunking
    └── vector_store.py      ← ChromaDB + sentence-transformers (fully offline)
```

---

## Setup

```bash
# 1. Install Ollama  →  https://ollama.com
ollama pull llama3.2          # 8 GB RAM
ollama pull mistral           # 16 GB RAM  (better quality)
ollama pull gemma3:12b        # 32 GB RAM  (best local quality)

# 2. Install Python dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Demo mode — built-in Bluetooth sample requirements
python main.py

# From a file
python main.py -f requirements.txt

# Interactive paste
python main.py -i

# With a knowledge base (PDF / TXT / MD)
python main.py -f reqs.txt -k bluetooth_spec.pdf
python main.py -f reqs.txt -k bt_spec.pdf -k company_standards.md
python main.py -f reqs.txt -k ./knowledge_base/          # whole folder

# Choose a better model
python main.py -f reqs.txt -k ./docs/ -m mistral

# Retrieve more context chunks per requirement (default 5)
python main.py -f reqs.txt -k ./docs/ --top-k 8

# Save JSON report
python main.py -f reqs.txt -k ./docs/ -o report.json
```

---

## What each file does

| File | Responsibility |
|---|---|
| `main.py` | CLI flags, input resolution, wires everything together |
| `config.py` | Single place to change model, chunk size, server URL, etc. |
| `prompts.py` | System prompt and user prompt builder — tweak here for better output |
| `analyser.py` | One-at-a-time requirement analysis, JSON extraction, error handling |
| `reporter.py` | Rich terminal rendering — summary table + per-requirement panels |
| `knowledge/loader.py` | PDF/TXT/MD reading, directory expansion, text chunking |
| `knowledge/vector_store.py` | Embedding, ChromaDB indexing, similarity retrieval |

---

## Output fields per requirement

| Field | What it shows |
|---|---|
| `missingInformation` | Specific facts absent from the requirement |
| `ambiguousTerms` | Vague words with two conflicting interpretations |
| `missingAcceptanceCriteria` | Measurable pass/fail rules that are missing |
| `edgeCasesNotCovered` | Boundary conditions and error paths not described |
| `clarifyingQuestions` | Specific questions to bring to the product owner |
| `rewriteSuggestion` | A fully testable rewrite of the requirement |

---

## Changing the LLM server

Edit `config.py`:

```python
# LM Studio
OLLAMA_BASE_URL = "http://localhost:1234/v1"
OLLAMA_API_KEY  = "lm-studio"
```
