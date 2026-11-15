"""
retriever.py
============
Retrieval strategies for the RAG pipeline.

Strategies
----------
- BM25Retriever   : Sparse lexical retrieval using BM25 (rank-bm25).
- DenseRetriever  : Dense semantic retrieval via VectorStore.
- HybridRetriever : Combines both via Reciprocal Rank Fusion (RRF).
- Retriever       : Unified entry point used by RAGPipeline.

All heavy imports are wrapped in try/except blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[Retriever] rank-bm25 not installed. Run: pip install rank-bm25")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Container for the full output of a retrieval call.

    Attributes
    ----------
    query          : The original query string.
    results        : List of SearchResult objects, ordered by relevance.
    retrieval_mode : 'dense', 'sparse', or 'hybrid'.
    num_retrieved  : Number of results returned.
    metadata       : Timing and debug info.
    """
    query: str
    results: List[Any]           # list[SearchResult] from vector_store.py
    retrieval_mode: str = "hybrid"
    num_retrieved: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.num_retrieved = len(self.results)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Sparse lexical retriever backed by BM25Okapi.

    The index is built from a list of Chunk objects at construction time.
    Call `rebuild(chunks)` to update the index after adding new documents.

    Parameters
    ----------
    chunks : List of Chunk objects to index.
    """

    def __init__(self, chunks: List[Any]):
        self._chunks: List[Any] = []
        self._bm25: Optional[Any] = None
        if chunks:
            self.rebuild(chunks)

    def rebuild(self, chunks: List[Any]) -> None:
        """(Re-)build the BM25 index from the given chunks."""
        self._chunks = list(chunks)
        if not BM25_AVAILABLE:
            print("[BM25Retriever] rank-bm25 unavailable. Sparse retrieval disabled.")
            return
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """
        Return the top-k chunks for *query* ranked by BM25 score.

        Returns a list of SearchResult-compatible objects (duck-typed).
        """
        from rag_pipeline.vector_store import SearchResult  # local import to avoid circular deps

        if self._bm25 is None or not self._chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        if not NUMPY_AVAILABLE:
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in indexed[:top_k]]
        else:
            top_indices = list(np.argsort(scores)[::-1][:top_k])

        results: List[SearchResult] = []
        for idx in top_indices:
            if idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            sc = float(scores[idx])
            if sc <= 0:
                continue
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=sc,
                metadata=chunk.metadata,
                source_doc=chunk.source_doc,
                page_num=chunk.page_num,
            ))

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [tok for tok in text.split() if tok]


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Dense semantic retriever that wraps the VectorStore.

    Parameters
    ----------
    vector_store : VectorStore instance with an indexed collection.
    embedder     : Any embedder exposing `embed_query(str) -> EmbeddingResult`.
    """

    def __init__(self, vector_store: Any, embedder: Any):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Return the top-k chunks via vector similarity search."""
        return self.vector_store.query(query, top_k=top_k)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines dense and sparse retrieval using Reciprocal Rank Fusion (RRF).

    RRF score for a document d:
        RRF(d) = Σ  1 / (k + rank_i(d))
    where the sum is over retrieval systems and rank_i(d) is d's 1-based rank
    in system i.

    Parameters
    ----------
    dense  : DenseRetriever instance.
    sparse : BM25Retriever instance.
    alpha  : Weight for dense scores vs. sparse scores in final ranking.
             alpha=1.0 means purely dense; alpha=0.0 purely sparse.
             (For standard RRF this is ignored; both lists get equal weight.)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        alpha: float = 0.5,
    ):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """
        Retrieve top-k results by fusing dense and sparse rankings.
        """
        fetch_k = top_k * 3  # over-fetch before fusion
        dense_results = self.dense.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        k: int = 60,
    ) -> List[Any]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Parameters
        ----------
        dense_results  : Dense retrieval results (ranked list).
        sparse_results : Sparse retrieval results (ranked list).
        k              : RRF constant (default 60).
        """
        rrf_scores: Dict[str, float] = {}
        id_to_result: Dict[str, Any] = {}

        for rank, result in enumerate(dense_results, start=1):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.alpha / (k + rank)
            id_to_result[cid] = result

        for rank, result in enumerate(sparse_results, start=1):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - self.alpha) / (k + rank)
            id_to_result[cid] = result

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        fused: List[Any] = []
        for cid in sorted_ids:
            result = id_to_result[cid]
            # Attach the fused RRF score for transparency
            result.metadata["rrf_score"] = round(rrf_scores[cid], 6)
            result.score = rrf_scores[cid]
            fused.append(result)

        return fused


# ---------------------------------------------------------------------------
# Retriever (main unified class used by RAGPipeline)
# ---------------------------------------------------------------------------

class Retriever:
    """
    Unified retriever that selects the appropriate strategy based on
    RAGConfig.retrieval_mode ('dense' | 'sparse' | 'hybrid').

    Parameters
    ----------
    config       : RAGConfig.
    vector_store : Initialised VectorStore.
    embedder     : Initialised embedder.
    """

    def __init__(self, config: "RAGConfig", vector_store: Any, embedder: Any):
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self._all_chunks: List[Any] = []

        self._dense = DenseRetriever(vector_store, embedder)
        self._sparse = BM25Retriever([])
        self._hybrid = HybridRetriever(self._dense, self._sparse, alpha=config.bm25_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant chunks for *query* using the configured strategy.

        Returns
        -------
        RetrievalResult with ordered results and metadata.
        """
        mode = self.config.retrieval_mode.lower()
        top_k = self.config.top_k

        if mode == "dense":
            results = self._dense.retrieve(query, top_k=top_k)
        elif mode == "sparse":
            results = self._sparse.retrieve(query, top_k=top_k)
        else:
            # 'hybrid' (default)
            results = self._hybrid.retrieve(query, top_k=top_k)

        return RetrievalResult(
            query=query,
            results=results,
            retrieval_mode=mode,
            metadata={"top_k": top_k},
        )

    def add_documents(self, documents: List[Any]) -> None:
        """
        Chunk, embed, and store a list of ProcessedDocument objects.

        Parameters
        ----------
        documents : List of ProcessedDocument instances from DocumentProcessor.
        """
        from rag_pipeline.chunker import ChunkerFactory, TableChunker

        chunker = ChunkerFactory.create(self.config.chunking_strategy, self.config)
        table_chunker = TableChunker()
        new_chunks: List[Any] = []

        for doc in documents:
            for page in doc.pages:
                meta = {
                    "source_doc": doc.source_path,
                    "page_num": page.page_num,
                    **page.metadata,
                }
                # Text chunks
                text_chunks = chunker.chunk(page.text, metadata=meta)
                new_chunks.extend(text_chunks)

                # Table chunks
                for table in page.tables:
                    t_chunk = table_chunker.chunk_table(table, metadata=meta)
                    new_chunks.append(t_chunk)

        if new_chunks:
            self.vector_store.add_chunks(new_chunks)
            self._all_chunks.extend(new_chunks)
            self._sparse.rebuild(self._all_chunks)
            print(f"[Retriever] Indexed {len(new_chunks)} new chunks "
                  f"(total: {len(self._all_chunks)}).")
        else:
            print("[Retriever] No chunks produced from provided documents.")

    def update_bm25_index(self, chunks: List[Any]) -> None:
        """
        Rebuild the BM25 index from an explicit list of chunks.
        Useful when loading chunks from a persisted session.

        Parameters
        ----------
        chunks : Full list of Chunk objects to index.
        """
        self._all_chunks = list(chunks)
        self._sparse.rebuild(self._all_chunks)
        print(f"[Retriever] BM25 index rebuilt with {len(self._all_chunks)} chunks.")
