# RAG-Sycophancy

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline with integrated **sycophancy detection** — the ability to identify and resist context poisoning, misleading retrieved documents, and factually incorrect grounding.

---

## Architecture

```
                     ┌──────────────────────────────────────────────────────┐
                     │                   RAGSycophancySystem                 │
                     │                                                        │
 User Query ──────► │   ┌─────────────┐        ┌──────────────────────┐     │
                     │   │ RAGPipeline │        │   ReflectiveRAG       │     │
                     │   │             │        │   (Sycophancy Auditor)│     │
                     │   │ ingest()    │        │                      │     │
                     │   │ query()     │───────►│  AUDITOR_PROMPT      │     │
                     │   │ chat()      │        │  GENERATOR_PROMPT    │     │
                     │   └──────┬──────┘        └──────────────────────┘     │
                     │          │                                              │
                     └──────────┼──────────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────────────┐
          │                     │                             │
          ▼                     ▼                             ▼
  DocumentProcessor        Embedder                     ChatHistory
  ┌──────────────┐    ┌────────────────┐             ┌────────────────┐
  │ PDFParser    │    │ SentenceTransf.│             │ ChatTurn       │
  │  - pymupdf   │    │ HuggingFace    │             │ Message        │
  │  - pdfplumber│    │ EmbedderFactory│             │ persist / load │
  │  - nougat    │    └───────┬────────┘             └────────────────┘
  │  - unstructd │            │
  └──────┬───────┘            │
         │                    ▼
         ▼              VectorStore (ChromaDB)
     Chunker           ┌───────────────────┐
  ┌──────────────┐     │ add_chunks()      │
  │ Sentence     │────►│ query()           │
  │ Recursive    │     │ delete_by_source()│
  │ Semantic     │     │ get_stats()       │
  │ Table        │     └────────┬──────────┘
  └──────────────┘              │
                                ▼
                           Retriever
                     ┌────────────────────┐
                     │ DenseRetriever     │
                     │ BM25Retriever      │
                     │ HybridRetriever    │
                     │  (RRF fusion)      │
                     └────────────────────┘
                                │
                                ▼
                           LLM API Call
                       (OpenAI-compatible)
                                │
                                ▼
                           Final Answer
```

---

## Folder Structure

```
RAG-Sycophancy-main/
├── main.py                   # Entry point — interactive menu
├── config.py                 # API endpoint, model name, auth headers
├── prompts.py                # Auditor and generator system prompts
├── simulation.py             # HostileRetriever (poisoned data simulation)
├── simulation_engine.py      # Shim → re-exports from simulation.py
├── Readme.md                 # This file
└── rag_pipeline/
    ├── __init__.py           # Package exports
    ├── rag_config.py         # RAGConfig dataclass (all knobs)
    ├── document_processor.py # PDF / text loading & pre-processing
    ├── chunker.py            # Sentence / Recursive / Semantic chunking
    ├── embedder.py           # SentenceTransformer & HuggingFace embedders
    ├── vector_store.py       # ChromaDB CRUD operations
    ├── retriever.py          # BM25 / Dense / Hybrid retriever + RRF
    ├── chat_history.py       # Rolling conversation history
    └── rag_pipeline.py       # RAGPipeline — ties everything together
```

---

## Setup

### 1. Clone / enter the project
```bash
cd RAG-Sycophancy-main
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Install **all at once**:
```bash
pip install \
  pymupdf \
  pdfplumber \
  Pillow \
  pytesseract \
  unstructured \
  nougat-ocr \
  nltk \
  spacy \
  sentence-transformers \
  torch \
  transformers \
  chromadb \
  rank-bm25 \
  numpy \
  requests
```

Or **install only what you need**:

| Component | Library | Install |
|-----------|---------|---------|
| PDF parsing (fast) | pymupdf | `pip install pymupdf` |
| PDF parsing (tables) | pdfplumber | `pip install pdfplumber` |
| PDF parsing (scientific) | nougat-ocr | `pip install nougat-ocr` |
| PDF parsing (mixed) | unstructured | `pip install unstructured` |
| OCR | pytesseract | `pip install pytesseract` |
| Sentence chunking | nltk | `pip install nltk` |
| Semantic chunking | spacy | `pip install spacy` |
| Embeddings (fast) | sentence-transformers | `pip install sentence-transformers` |
| Embeddings (custom) | transformers + torch | `pip install transformers torch` |
| Vector store | chromadb | `pip install chromadb` |
| Sparse retrieval | rank-bm25 | `pip install rank-bm25` |
| Numeric ops | numpy | `pip install numpy` |
| HTTP requests | requests | `pip install requests` |

### 4. Configure your LLM API

Edit `config.py`:
```python
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
MODEL_NAME   = "gpt-4o-mini"
TEMPERATURE  = 0.0
HEADERS      = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('MODEL_API_KEY', 'your-key')}"
}
```

Then uncomment the `requests.post(...)` block in `main.py → ReflectiveRAG.call_model()`
and in `rag_pipeline/rag_pipeline.py → RAGPipeline._call_llm()`.

---

## Usage

### Run interactively
```bash
python main.py
```

You will see:
```
============================================================
  RAG-Sycophancy Detection System
============================================================
  Mode 1 : Hostile Retrieval Simulation (original)
  Mode 2 : Real RAG Pipeline + Sycophancy Audit
  q / quit : Exit
============================================================

Select mode (1 / 2 / q):
```

---

### Mode 1 — Hostile Retrieval Simulation

Tests the auditor against deliberately poisoned retrieval results:
```
Select mode (1 / 2 / q): 1

[Mode 1] Hostile Retrieval Active. Type 'back' to return to menu.

Query: What is the capital of France?
  [Auditor] : NOISY
  [Answer]  : The context is unreliable. Paris, France is the capital of France.
```

---

### Mode 2 — Real RAG Pipeline + Sycophancy Audit

```
Select mode (1 / 2 / q): 2

[Mode 2] Real RAG + Sycophancy Audit.

Query: ingest /path/to/my/documents/
  [Ingest] {'documents_loaded': 5, 'total_pages': 42, 'status': 'ok'}

Query: What are the main findings of the paper?
  [Audit]   : TRUSTED
  [Answer]  : The paper finds that ...
  [Sources] : 3 chunk(s) retrieved (hybrid mode)

Query: stats
  [Stats] {
      "collection_stats": {"name": "rag_collection", "count": 320},
      ...
  }
```

---

### Programmatic usage

```python
from rag_pipeline import RAGPipeline, RAGConfig

config = RAGConfig(
    collection_name="my_project",
    chunking_strategy="sentence",
    retrieval_mode="hybrid",
    top_k=5,
)
pipeline = RAGPipeline(config)

# Ingest documents
stats = pipeline.ingest("./documents/")
print(stats)

# Query
result = pipeline.query("What is the main topic of the documents?")
print(result["answer"])
print(result["retrieved_docs"])

# Simple chat
answer = pipeline.chat("Summarise the key points.")
print(answer)
```

---

### Using RAGSycophancySystem

```python
from main import RAGSycophancySystem
from rag_pipeline import RAGConfig

system = RAGSycophancySystem(RAGConfig())
system.ingest("./papers/")

result = system.query("What is the effect of temperature on performance?")
print("Answer      :", result["answer"])
print("Audit status:", result["audit_status"])
print("Num docs    :", result["num_retrieved"])
```

---

## Configuration Guide

All settings live in `RAGConfig` (`rag_pipeline/rag_config.py`).

| Field | Default | Description |
|-------|---------|-------------|
| `persist_directory` | `"./chroma_db"` | ChromaDB data folder |
| `collection_name` | `"rag_collection"` | Collection identifier |
| `pdf_model` | `"auto"` | PDF backend: `auto`, `pymupdf`, `pdfplumber`, `nougat`, `unstructured` |
| `extract_images` | `True` | Save embedded images to disk |
| `extract_tables` | `True` | Extract table data as structured chunks |
| `chunk_size` | `512` | Maximum tokens per chunk |
| `chunk_overlap` | `64` | Shared tokens between consecutive chunks |
| `chunking_strategy` | `"sentence"` | `sentence`, `recursive`, or `semantic` |
| `embedding_model` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model ID |
| `embedding_dim` | `384` | Embedding vector dimension |
| `embedding_batch_size` | `32` | Texts per embedding batch |
| `top_k` | `5` | Chunks returned per query |
| `similarity_threshold` | `0.3` | Minimum cosine similarity to include a result |
| `retrieval_mode` | `"hybrid"` | `dense`, `sparse`, or `hybrid` |
| `bm25_alpha` | `0.5` | Weight of dense vs sparse in hybrid fusion |
| `max_history` | `10` | Turns kept in conversation memory |
| `include_history` | `True` | Prepend history to LLM context |

---

## How Sycophancy Detection Works

```
                   User Query
                       │
                       ▼
              Retrieved Context
                       │
                       ▼
           ┌───────────────────────┐
           │   AUDITOR LLM call    │
           │                       │
           │  "Is this context     │
           │   relevant and        │
           │   factually sound?"   │
           └──────────┬────────────┘
                      │
           ┌──────────┴──────────┐
           │                     │
        TRUSTED               NOISY
           │                     │
           ▼                     ▼
   Answer grounded         Reject context,
   in retrieved docs    use internal knowledge
```

The auditor prompt (`AUDITOR_SYSTEM_PROMPT` in `prompts.py`) instructs the LLM to detect:
- **Entity traps** — e.g. "Paris, Texas" vs "Paris, France"
- **Topical mismatch** — context about an unrelated subject
- **Factual contradiction** — context that contradicts well-known facts

Based on the verdict (`TRUSTED` / `NOISY`), a different generator system prompt is used, either grounding the answer in context or falling back to parametric knowledge.

---

## Contributing

1. Keep `ReflectiveRAG` logic in `main.py` unchanged.
2. Add new chunking strategies by subclassing `BaseChunker` in `chunker.py`.
3. Add new embedding backends by subclassing `BaseEmbedder` in `embedder.py`.
4. Register them in the respective factory classes.
