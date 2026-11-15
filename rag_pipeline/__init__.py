"""
rag_pipeline/__init__.py
========================
Public API for the rag_pipeline package.

Exports
-------
RAGPipeline      – Main pipeline class (ingest, query, chat).
DocumentProcessor – Document loading and pre-processing.
Embedder         – Alias for the default SentenceTransformerEmbedder.
VectorStore      – ChromaDB-backed vector store.
Retriever        – Unified retriever (dense / sparse / hybrid).
ChatHistory      – Conversation history manager.
RAGConfig        – Central configuration dataclass.
"""

from rag_pipeline.rag_config import RAGConfig
from rag_pipeline.document_processor import DocumentProcessor, PDFParser
from rag_pipeline.embedder import (
    BaseEmbedder,
    EmbedderFactory,
    SentenceTransformerEmbedder as Embedder,
    HuggingFaceEmbedder,
    EmbeddingResult,
)
from rag_pipeline.chunker import (
    Chunk,
    BaseChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    TableChunker,
    ChunkerFactory,
)
from rag_pipeline.vector_store import VectorStore, SearchResult
from rag_pipeline.retriever import Retriever, RetrievalResult, BM25Retriever, DenseRetriever, HybridRetriever
from rag_pipeline.chat_history import ChatHistory, ChatTurn, Message
from rag_pipeline.rag_pipeline import RAGPipeline

__all__ = [
    # Core pipeline
    "RAGPipeline",
    "RAGConfig",
    # Document loading
    "DocumentProcessor",
    "PDFParser",
    # Embedding
    "Embedder",
    "BaseEmbedder",
    "HuggingFaceEmbedder",
    "EmbedderFactory",
    "EmbeddingResult",
    # Chunking
    "Chunk",
    "BaseChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "TableChunker",
    "ChunkerFactory",
    # Vector store
    "VectorStore",
    "SearchResult",
    # Retrieval
    "Retriever",
    "RetrievalResult",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    # Chat history
    "ChatHistory",
    "ChatTurn",
    "Message",
]
