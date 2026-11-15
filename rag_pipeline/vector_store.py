"""
vector_store.py
===============
ChromaDB-backed vector store for the RAG pipeline.

Provides
--------
- VectorStore : CRUD operations on a persistent ChromaDB collection.

All heavy imports are wrapped in try/except blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[VectorStore] chromadb not installed. Run: pip install chromadb")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    Represents a single retrieved document chunk.

    Attributes
    ----------
    chunk_id   : UUID of the chunk (matches Chunk.chunk_id).
    text       : Full text of the retrieved chunk.
    score      : Cosine similarity score (higher is more similar).
    metadata   : Arbitrary key/value pairs stored with the chunk.
    source_doc : Originating document path.
    page_num   : Page number within the source document.
    """
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_doc: str = ""
    page_num: int = 0


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Parameters
    ----------
    config   : RAGConfig with persist_directory, collection_name, top_k etc.
    embedder : Any embedder that exposes `embed_query(str) -> EmbeddingResult`
               and `embed(list[str]) -> list[EmbeddingResult]`.
    """

    def __init__(self, config: "RAGConfig", embedder: Any):
        self.config = config
        self.embedder = embedder
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None

        if not CHROMADB_AVAILABLE:
            print("[VectorStore] ChromaDB unavailable — operating in no-op mode.")
            return

        self._init_client()
        self._get_or_create_collection()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Create a persistent ChromaDB client at config.persist_directory."""
        os.makedirs(self.config.persist_directory, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory,
            )
            print(f"[VectorStore] ChromaDB client ready at: {self.config.persist_directory}")
        except Exception as exc:
            print(f"[VectorStore] Failed to initialise ChromaDB client: {exc}")
            self._client = None

    def _get_or_create_collection(self) -> None:
        """Get or create the named collection with cosine distance metric."""
        if self._client is None:
            return
        try:
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[VectorStore] Collection '{self.config.collection_name}' "
                  f"ready ({self._collection.count()} documents).")
        except Exception as exc:
            print(f"[VectorStore] Failed to get/create collection: {exc}")
            self._collection = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List["Chunk"]) -> None:
        """
        Upsert a list of Chunk objects into ChromaDB.
        Each chunk is embedded, then stored with its metadata.

        Parameters
        ----------
        chunks : List of Chunk objects (from chunker.py).
        """
        if not chunks:
            return
        if self._collection is None:
            print("[VectorStore] Collection not available. Skipping add_chunks.")
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source_doc": c.source_doc,
                "page_num": str(c.page_num),
                "chunk_index": str(c.chunk_index),
                "token_count": str(c.token_count),
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        # Embed in one batch call
        embedding_results = self.embedder.embed(texts)
        embeddings = [r.embedding for r in embedding_results]

        # ChromaDB upsert in fixed batches (max 5 000 per call)
        BATCH_SIZE = 500
        total = len(chunks)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            try:
                self._collection.upsert(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )
                print(f"[VectorStore] Upserted chunks {start + 1}–{end}/{total}")
            except Exception as exc:
                print(f"[VectorStore] Upsert error on batch {start}–{end}: {exc}")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Dense retrieval: embed *query_text* and return the *top_k* nearest chunks.

        Parameters
        ----------
        query_text : Natural-language query string.
        top_k      : Number of results (defaults to config.top_k).
        """
        query_embedding = self.embedder.embed_query(query_text)
        return self.query_by_embedding(query_embedding.embedding, top_k=top_k)

    def query_by_embedding(
        self,
        embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Retrieve the *top_k* chunks nearest to a pre-computed embedding vector.

        Parameters
        ----------
        embedding : Dense float vector.
        top_k     : Number of results (defaults to config.top_k).
        """
        if self._collection is None:
            print("[VectorStore] Collection not available.")
            return []

        k = top_k if top_k is not None else self.config.top_k
        try:
            response = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(k, self._collection.count() or 1),
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:
            print(f"[VectorStore] Query error: {exc}")
            return []

        results: List[SearchResult] = []
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        distances = response.get("distances", [[]])[0]
        metas = response.get("metadatas", [[]])[0]

        for cid, doc, dist, meta in zip(ids, docs, distances, metas):
            # ChromaDB cosine distance: similarity = 1 - distance
            score = max(0.0, 1.0 - float(dist))
            if score < self.config.similarity_threshold:
                continue
            results.append(SearchResult(
                chunk_id=cid,
                text=doc,
                score=score,
                metadata=meta or {},
                source_doc=meta.get("source_doc", "") if meta else "",
                page_num=int(meta.get("page_num", 0)) if meta else 0,
            ))

        return results

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------

    def delete_collection(self) -> None:
        """Drop and recreate the collection (destructive!)."""
        if self._client is None:
            return
        try:
            self._client.delete_collection(self.config.collection_name)
            print(f"[VectorStore] Deleted collection '{self.config.collection_name}'.")
            self._get_or_create_collection()
        except Exception as exc:
            print(f"[VectorStore] Failed to delete collection: {exc}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the current collection."""
        if self._collection is None:
            return {"error": "Collection not available"}
        try:
            count = self._collection.count()
            return {
                "name": self.config.collection_name,
                "count": count,
                "persist_directory": self.config.persist_directory,
                "metadata": self._collection.metadata or {},
            }
        except Exception as exc:
            return {"error": str(exc)}

    def list_sources(self) -> List[str]:
        """Return a list of unique source document paths in the collection."""
        if self._collection is None:
            return []
        try:
            # Fetch all with metadata included
            all_data = self._collection.get(include=["metadatas"])
            sources = set()
            for meta in (all_data.get("metadatas") or []):
                if meta and "source_doc" in meta:
                    sources.add(meta["source_doc"])
            return sorted(sources)
        except Exception as exc:
            print(f"[VectorStore] list_sources error: {exc}")
            return []

    def delete_by_source(self, source_path: str) -> None:
        """
        Delete all chunks whose 'source_doc' metadata matches *source_path*.

        Parameters
        ----------
        source_path : Absolute or relative path of the source document to remove.
        """
        if self._collection is None:
            return
        try:
            self._collection.delete(where={"source_doc": source_path})
            print(f"[VectorStore] Deleted all chunks from source: {source_path}")
        except Exception as exc:
            print(f"[VectorStore] delete_by_source error: {exc}")
