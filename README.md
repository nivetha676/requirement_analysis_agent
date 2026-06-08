# Requirements Ambiguity Analysis Agent

Analyses software requirements for ambiguity, missing information, vague terms,
missing acceptance criteria, and edge cases — using a local LLM via Ollama
and an optional persistent RAG knowledge base.

---

## Project structure

```
requirements_agent/
├── main.py                      ← Run the agent
├── config.py                    ← All settings
├── prompts.py                   ← System + user prompt builders
├── analyser.py                  ← Calls Ollama, parses results
├── reporter.py                  ← Rich terminal rendering
├── requirements.txt             ← Python dependencies
└── knowledge/
    ├── __init__.py
    ├── loader.py                ← Reads PDF/TXT/MD + chunking
    ├── builder.py               ← Build & save named knowledge bases  ← run once
    ├── retriever.py             ← Load & query saved knowledge bases  ← used by agent
    └── vector_store.py          ← (legacy, no longer used by agent)
```

---

## Two-step workflow

### Step 1 — Build your knowledge base (once)

```bash
# Single file
python -m knowledge.builder --name bluetooth --docs ./bt_spec.pdf

# Multiple files
python -m knowledge.builder --name bluetooth --docs ./bt_spec.pdf --docs ./bt_guide.md

# Whole folder
python -m knowledge.builder --name bluetooth --docs ./bluetooth_docs/

# Overwrite an existing KB
python -m knowledge.builder --name bluetooth --docs ./docs/ --force
```

The KB is saved to `./kb_store/bluetooth/` and reused on every agent run.
No re-embedding, no re-chunking.

### Step 2 — Run the agent (every time)

```bash
# With a knowledge base
python main.py -f reqs.txt --kb bluetooth

# Without knowledge base
python main.py -f reqs.txt

# Interactive paste
python main.py -i --kb bluetooth

# Different model
python main.py -f reqs.txt --kb bluetooth -m mistral

# Save JSON report
python main.py -f reqs.txt --kb bluetooth -o report.json
```

---

## Managing knowledge bases

```bash
# List all saved knowledge bases
python main.py --list-kb
python -m knowledge.builder --list

# Delete a knowledge base
python -m knowledge.builder --delete bluetooth
```

---

## Setup

```bash
pip install -r requirements.txt

# Install and start Ollama
ollama pull llama3.2      # 8 GB RAM
ollama pull mistral       # 16 GB RAM (better quality)
ollama pull gemma3:12b    # 32 GB RAM (best local quality)
```

---

## What each module does

| Module | Job | When it runs |
|---|---|---|
| `knowledge/builder.py` | Reads docs, chunks, embeds, saves ChromaDB to disk | Once per KB |
| `knowledge/retriever.py` | Loads saved KB, queries for relevant chunks | Every agent run |
| `knowledge/loader.py` | File reading (PDF/TXT/MD) + text chunking | Called by builder |
| `analyser.py` | Calls Ollama per requirement, injects KB context | Every agent run |
| `reporter.py` | Rich terminal output | Every agent run |
| `main.py` | CLI wiring | Every agent run |
| `config.py` | All settings | Imported everywhere |
| `prompts.py` | Prompt strings | Imported by analyser |
