# config.py
# ─────────────────────────────────────────────────────────────
# Central configuration — edit values here, nowhere else.
# ─────────────────────────────────────────────────────────────

# Ollama server  (change port to 1234 for LM Studio)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"           # required by SDK; not validated by Ollama

# Default model  (override with --model / -m at runtime)
DEFAULT_MODEL   = "llama3.2"

# Embedding model — downloaded once, cached at ~/.cache/huggingface
EMBED_MODEL     = "all-MiniLM-L6-v2"

# RAG chunking
CHUNK_SIZE      = 400                # characters per chunk
CHUNK_OVERLAP   = 80                 # overlap between adjacent chunks

# RAG retrieval
DEFAULT_TOP_K   = 5                  # chunks retrieved per requirement

# Supported knowledge-base file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".rst", ".text"}
