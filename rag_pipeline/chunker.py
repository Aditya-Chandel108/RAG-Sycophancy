"""
chunker.py
==========
Text chunking strategies for the RAG pipeline.

Strategies
----------
- sentence   : Groups sentences using NLTK / spaCy with sliding-window overlap.
- recursive  : Splits on hierarchical separators (paragraphs → sentences → words).
- semantic   : Finds semantically meaningful breakpoints via embedding similarity.

All heavy imports are wrapped in try/except blocks.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _nltk_downloaded = False

    def _ensure_nltk_punkt() -> None:
        global _nltk_downloaded
        if not _nltk_downloaded:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            _nltk_downloaded = True

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[Chunker] nltk not installed. Run: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[Chunker] spacy not installed. Run: pip install spacy")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[Chunker] numpy not installed. Run: pip install numpy")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage.

    Attributes
    ----------
    chunk_id    : Globally unique identifier (UUID4 string).
    text        : The raw chunk text.
    source_doc  : Absolute path of the originating document.
    page_num    : Page number within the source document (1-indexed).
    chunk_index : Position of this chunk within the document.
    metadata    : Arbitrary key/value pairs (e.g. section title, table flag).
    token_count : Estimated token count (characters / 4 approximation).
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    source_doc: str = ""
    page_num: int = 0
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        if not self.token_count:
            self.token_count = max(1, len(self.text) // 4)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Abstract interface that all chunkers must implement."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split *text* into a list of Chunk objects."""

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Estimate token count as len(text) // 4 (rough GPT-style approximation)."""
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

class SentenceChunker(BaseChunker):
    """
    Groups sentences into chunks of at most *chunk_size* tokens, with
    *chunk_overlap* tokens of shared content between consecutive chunks.
    Uses NLTK sent_tokenize (falls back to regex splitting if NLTK is absent).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---- public --------------------------------------------------------

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split *text* into sentence-grouped chunks."""
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        grouped = self._group_sentences(sentences)

        source_doc = metadata.get("source_doc", "")
        page_num = metadata.get("page_num", 0)

        chunks: List[Chunk] = []
        for idx, group_text in enumerate(grouped):
            if not group_text.strip():
                continue
            chunks.append(Chunk(
                text=group_text.strip(),
                source_doc=source_doc,
                page_num=page_num,
                chunk_index=idx,
                metadata=dict(metadata),
                token_count=self._count_tokens(group_text),
            ))
        return chunks

    # ---- private -------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Return a list of sentence strings."""
        if NLTK_AVAILABLE:
            _ensure_nltk_punkt()
            return sent_tokenize(text)
        # Regex fallback: split on period/exclamation/question followed by space or end
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Sliding-window grouping: accumulate sentences until the token budget
        (chunk_size) is reached, then start a new chunk — keeping the last
        *overlap_tokens* worth of sentences in the next chunk.
        """
        groups: List[str] = []
        current: List[str] = []
        current_tokens = 0
        overlap_tokens = self.chunk_overlap

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > self.chunk_size and current:
                groups.append(" ".join(current))
                # Build overlap: keep trailing sentences that fit in overlap budget
                overlap_buf: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    t = self._count_tokens(s)
                    if overlap_len + t <= overlap_tokens:
                        overlap_buf.insert(0, s)
                        overlap_len += t
                    else:
                        break
                current = overlap_buf
                current_tokens = overlap_len

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            groups.append(" ".join(current))

        return groups


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class RecursiveChunker(BaseChunker):
    """
    Recursively splits text on a prioritised list of separators until all
    pieces are within the chunk_size budget.  Mimics LangChain's
    RecursiveCharacterTextSplitter without the dependency.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split *text* recursively and return a list of Chunk objects."""
        metadata = metadata or {}
        source_doc = metadata.get("source_doc", "")
        page_num = metadata.get("page_num", 0)

        raw_chunks = self._split(text, self._SEPARATORS)
        # Merge small adjacent pieces and add overlap
        merged = self._merge_with_overlap(raw_chunks)

        chunks: List[Chunk] = []
        for idx, piece in enumerate(merged):
            if not piece.strip():
                continue
            chunks.append(Chunk(
                text=piece.strip(),
                source_doc=source_doc,
                page_num=page_num,
                chunk_index=idx,
                metadata=dict(metadata),
                token_count=self._count_tokens(piece),
            ))
        return chunks

    # ---- private -------------------------------------------------------

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split on the first separator that reduces piece size."""
        if not text.strip():
            return []

        if self._count_tokens(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level fallback
            size = self.chunk_size * 4   # rough char estimate
            return [text[i:i + size] for i in range(0, len(text), size)]

        parts = text.split(sep)
        result: List[str] = []
        for part in parts:
            if self._count_tokens(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split(part, remaining_seps))
        return result

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Merge small adjacent pieces; re-inject overlap between chunks."""
        merged: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for piece in pieces:
            pt = self._count_tokens(piece)
            if current_tokens + pt > self.chunk_size and current:
                merged.append(" ".join(current))
                # Overlap: keep trailing pieces within overlap budget
                overlap_buf: List[str] = []
                overlap_len = 0
                for p in reversed(current):
                    t = self._count_tokens(p)
                    if overlap_len + t <= self.chunk_overlap:
                        overlap_buf.insert(0, p)
                        overlap_len += t
                    else:
                        break
                current = overlap_buf
                current_tokens = overlap_len
            current.append(piece)
            current_tokens += pt

        if current:
            merged.append(" ".join(current))

        return merged


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """
    Splits text at semantic breakpoints detected by a drop in cosine similarity
    between consecutive sentence embeddings.
    Requires an embedder instance (duck-typed: must have an `embed` method
    that accepts a list of strings and returns objects with a `.embedding` attr).
    """

    def __init__(self, embedding_model: Any, breakpoint_threshold: float = 0.3):
        self.embedding_model = embedding_model
        self.threshold = breakpoint_threshold

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split *text* on semantic boundaries."""
        metadata = metadata or {}
        source_doc = metadata.get("source_doc", "")
        page_num = metadata.get("page_num", 0)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text.strip(), source_doc=source_doc,
                          page_num=page_num, metadata=dict(metadata))]

        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities, self.threshold)

        groups: List[str] = []
        prev = 0
        for bp in breakpoints:
            groups.append(" ".join(sentences[prev:bp + 1]))
            prev = bp + 1
        groups.append(" ".join(sentences[prev:]))

        chunks: List[Chunk] = []
        for idx, group in enumerate(groups):
            if not group.strip():
                continue
            chunks.append(Chunk(
                text=group.strip(),
                source_doc=source_doc,
                page_num=page_num,
                chunk_index=idx,
                metadata=dict(metadata),
                token_count=self._count_tokens(group),
            ))
        return chunks

    def _compute_similarities(self, sentences: List[str]) -> List[float]:
        """Return cosine similarity between each consecutive sentence pair."""
        try:
            results = self.embedding_model.embed(sentences)
            embeddings = [r.embedding for r in results]
        except Exception as exc:
            print(f"[SemanticChunker] Embedding error: {exc}")
            return [1.0] * (len(sentences) - 1)

        if not NUMPY_AVAILABLE:
            # Fallback: treat everything as similar (no breakpoints)
            return [1.0] * (len(sentences) - 1)

        similarities: List[float] = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i], dtype=float)
            b = np.array(embeddings[i + 1], dtype=float)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                similarities.append(0.0)
            else:
                sim = float(np.dot(a, b) / (norm_a * norm_b))
                similarities.append(sim)

        return similarities

    def _find_breakpoints(self, similarities: List[float], threshold: float) -> List[int]:
        """Return indices where the similarity drops below *threshold*."""
        return [i for i, sim in enumerate(similarities) if sim < threshold]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if NLTK_AVAILABLE:
            _ensure_nltk_punkt()
            return sent_tokenize(text)
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------------------
# TableChunker
# ---------------------------------------------------------------------------

class TableChunker:
    """
    Converts extracted table dictionaries (from DocumentProcessor) into
    text Chunk objects, preserving column-header structure.
    """

    def chunk_table(
        self,
        table: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Convert a table dict ({headers, rows, raw_text}) into a single Chunk.
        The table is serialised as aligned plain text.
        """
        metadata = metadata or {}
        headers: List[str] = table.get("headers", [])
        rows: List[List[str]] = table.get("rows", [])
        raw_text: str = table.get("raw_text", "")

        if headers:
            header_line = " | ".join(headers)
            separator = "-" * len(header_line)
            row_lines = [" | ".join(str(c) for c in row) for row in rows]
            text = "\n".join([header_line, separator] + row_lines)
        else:
            text = raw_text

        return Chunk(
            text=text.strip(),
            source_doc=metadata.get("source_doc", ""),
            page_num=metadata.get("page_num", 0),
            metadata={**metadata, "is_table": True},
            token_count=BaseChunker._count_tokens(text),
        )


# ---------------------------------------------------------------------------
# ChunkerFactory
# ---------------------------------------------------------------------------

class ChunkerFactory:
    """Factory that constructs the appropriate chunker from a strategy string."""

    @staticmethod
    def create(strategy: str, config: "RAGConfig") -> BaseChunker:
        """
        Parameters
        ----------
        strategy : 'sentence' | 'recursive' | 'semantic'
        config   : RAGConfig instance with chunk_size, chunk_overlap etc.

        Returns
        -------
        A concrete BaseChunker instance.
        """
        s = strategy.lower()
        if s == "sentence":
            return SentenceChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        elif s == "recursive":
            return RecursiveChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        elif s == "semantic":
            # SemanticChunker needs an embedder; defer construction to pipeline
            raise ValueError(
                "SemanticChunker requires an embedder instance. "
                "Construct it directly: SemanticChunker(embedder, threshold=0.3)"
            )
        else:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                "Choose from: sentence, recursive, semantic"
            )
