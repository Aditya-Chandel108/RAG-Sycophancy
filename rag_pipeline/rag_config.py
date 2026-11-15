"""
rag_config.py
=============
Central configuration dataclass for the RAG pipeline.
Inherits LLM-related settings from the top-level config.py and adds
all knobs required by DocumentProcessor, Chunker, Embedder, VectorStore,
Retriever and ChatHistory.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os

# Pull top-level project config (api endpoint, model, headers, temperature)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import API_ENDPOINT, MODEL_NAME, TEMPERATURE, HEADERS as _HEADERS
    _DEFAULT_API_ENDPOINT = API_ENDPOINT
    _DEFAULT_MODEL_NAME = MODEL_NAME
    _DEFAULT_TEMPERATURE = TEMPERATURE
    _DEFAULT_HEADERS = _HEADERS
except ImportError:
    _DEFAULT_API_ENDPOINT = "link of api provider"
    _DEFAULT_MODEL_NAME = "your preferred model"
    _DEFAULT_TEMPERATURE = 0.0
    _DEFAULT_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    }


@dataclass
class RAGConfig:
    """
    Unified configuration for every component in the RAG pipeline.

    Attributes
    ----------
    ChromaDB settings:
        persist_directory   : Folder where ChromaDB persists its data.
        collection_name     : Name of the ChromaDB collection.

    PDF parsing settings:
        pdf_model           : Parser backend to use ('pymupdf', 'pdfplumber',
                              'nougat', 'unstructured', 'auto').
        extract_images      : Whether to extract embedded images from PDFs.
        extract_tables      : Whether to extract tables from PDFs.

    Chunking settings:
        chunk_size          : Maximum token count per chunk.
        chunk_overlap       : Number of tokens shared between consecutive chunks.
        chunking_strategy   : 'sentence', 'recursive', or 'semantic'.

    Embedding settings:
        embedding_model     : HuggingFace model identifier for embeddings.
        embedding_dim       : Dimension of the embedding vector.
        embedding_batch_size: Number of texts to embed in one forward pass.

    Retriever settings:
        top_k               : Number of results to return per query.
        similarity_threshold: Minimum similarity score for a result to be kept.
        retrieval_mode      : 'dense', 'sparse', or 'hybrid'.

    Chat history settings:
        max_history         : Maximum number of turns kept in memory.
        include_history     : Whether to include chat history in LLM context.

    LLM API settings (inherited from config.py):
        api_endpoint        : Full URL of the LLM inference endpoint.
        model_name          : Model identifier sent in the API payload.
        temperature         : Sampling temperature (0.0 = deterministic).
        headers             : HTTP headers, including Authorization.
    """

    # --- ChromaDB ---
    persist_directory: str = "./chroma_db"
    collection_name: str = "rag_collection"

    # --- PDF Parsing ---
    pdf_model: str = "auto"           # 'pymupdf' | 'pdfplumber' | 'nougat' | 'unstructured' | 'auto'
    extract_images: bool = True
    extract_tables: bool = True
    image_output_dir: str = "./extracted_images"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: str = "sentence"   # 'sentence' | 'recursive' | 'semantic'

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # --- Retriever ---
    top_k: int = 5
    similarity_threshold: float = 0.3
    retrieval_mode: str = "hybrid"    # 'dense' | 'sparse' | 'hybrid'
    bm25_alpha: float = 0.5           # weight for dense vs. sparse in hybrid fusion

    # --- Chat History ---
    max_history: int = 10
    include_history: bool = True
    history_persist_path: Optional[str] = None

    # --- LLM API (inherited from top-level config.py) ---
    api_endpoint: str = field(default_factory=lambda: _DEFAULT_API_ENDPOINT)
    model_name: str = field(default_factory=lambda: _DEFAULT_MODEL_NAME)
    temperature: float = field(default_factory=lambda: _DEFAULT_TEMPERATURE)
    headers: dict = field(default_factory=lambda: dict(_DEFAULT_HEADERS))

    def summary(self) -> dict:
        """Return a human-readable summary dict of key settings."""
        return {
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "max_history": self.max_history,
            "model_name": self.model_name,
        }
