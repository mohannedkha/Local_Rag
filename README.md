# Local RAG System

A fully local Retrieval-Augmented Generation (RAG) system powered by [Ollama](https://ollama.com), ChromaDB, and semantic chunking. No cloud API keys required.

---

## Architecture

```
data/raw/          ← drop your PDFs here
    │
    ▼ (pymupdf4llm)
data/processed/    ← Markdown conversions (auto-generated)
    │
    ▼ (semantic chunker)
db/                ← ChromaDB vector store (persistent)
    │
    ▼ (nomic-embed-text via Ollama)
RAG Pipeline
    ├── MMR retrieval      → diversity-aware chunk selection
    ├── Short-term memory  → last N conversation turns (RAM)
    ├── Long-term memory   → ChromaDB collection of past turns
    └── LLM (Ollama)       → answer generation with streamed output
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

Edit `config/config.yaml` to change models, chunk sizes, memory settings, etc.

### 4. Ingest PDFs

Drop PDF files into `data/raw/`, then run:

```bash
python scripts/ingest.py
```

Options:
- `--force` — re-convert already processed files
- `--file report.pdf` — ingest a single file

### 5. Query

```bash
# Interactive REPL (with streaming)
python scripts/query.py

# Single question
python scripts/query.py --question "What does the document say about X?"

# Resume a previous session
python scripts/query.py --session my_session.json
```

---

## Project Structure

```
Local_Rag/
├── config/
│   └── config.yaml          # all tuneable settings
├── data/
│   ├── raw/                 # input PDFs (git-ignored)
│   ├── processed/           # converted Markdown (git-ignored)
│   └── chunks/              # optional debug output
├── db/                      # ChromaDB persistent storage (git-ignored)
├── scripts/
│   ├── ingest.py            # PDF → chunks → vector store
│   └── query.py             # interactive / single-shot querying
├── src/
│   ├── ingestion/
│   │   ├── pdf_converter.py # PDF → Markdown (pymupdf4llm)
│   │   └── chunker.py       # semantic + recursive chunking
│   ├── embeddings/
│   │   └── embedder.py      # Ollama embedding wrapper
│   ├── db/
│   │   └── vector_store.py  # ChromaDB interface + MMR retrieval
│   ├── memory/
│   │   └── memory_manager.py # short-term + long-term memory
│   ├── llm/
│   │   └── ollama_client.py  # Ollama chat + streaming
│   └── rag/
│       ├── pipeline.py       # main RAG orchestration
│       └── factory.py        # build pipeline from config
└── requirements.txt
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| PDF → text | `pymupdf4llm` (Markdown) | Preserves structure (headers, tables) better than plain text |
| Chunking | Semantic (sentence-transformers) + recursive fallback | Keeps related sentences together; avoids mid-sentence breaks |
| Vector DB | ChromaDB (local, persistent) | Zero-infrastructure, fast, supports metadata filtering |
| Embeddings | `nomic-embed-text` via Ollama | State-of-the-art open embedding model, runs fully locally |
| Retrieval | MMR (Maximal Marginal Relevance) | Balances relevance with diversity to avoid redundant context |
| Memory | Short-term (RAM window) + Long-term (ChromaDB) | Handles both in-session and cross-session recall |
| LLM | Any Ollama model | Swappable via config; defaults to `llama3.2` |
