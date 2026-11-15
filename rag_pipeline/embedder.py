"""
embedder.py
===========
Text embedding module for the RAG pipeline.

Backends
--------
- SentenceTransformerEmbedder : Uses the `sentence-transformers` library.
- HuggingFaceEmbedder         : Uses raw `transformers` with mean pooling.

All heavy imports are wrapped in try/except blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, Exception):
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[Embedder] sentence-transformers could not be loaded. "
          "Run: pip install sentence-transformers  "
          "(also ensure compatible transformers/keras versions)")

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    TORCH_AVAILABLE = False
    print("[Embedder] torch not installed. Run: pip install torch")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except (ImportError, Exception):
    TRANSFORMERS_AVAILABLE = False
    print("[Embedder] transformers not installed or incompatible. Run: pip install transformers")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[Embedder] numpy not installed. Run: pip install numpy")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """
    Container for a single embedding result.

    Attributes
    ----------
    text       : The original input text.
    embedding  : The dense float vector.
    model_name : Identifier of the model that produced the embedding.
    metadata   : Optional arbitrary key/value pairs.
    """
    text: str
    embedding: List[float]
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract interface that all embedders must implement."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a list of texts and return a parallel list of EmbeddingResult."""

    def embed_query(self, query: str) -> EmbeddingResult:
        """Convenience method: embed a single query string."""
        results = self.embed([query])
        return results[0]

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Wraps the `sentence-transformers` library for fast, high-quality embeddings.

    Parameters
    ----------
    model_name : HuggingFace model identifier (e.g. 'all-MiniLM-L6-v2').
    device     : 'cpu' or 'cuda'.
    batch_size : Number of texts encoded per forward pass.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Optional[Any] = None

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"[SentenceTransformerEmbedder] sentence-transformers unavailable. "
                  "Embeddings will be zero-vectors.")
        else:
            self._load_model()

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model."""
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[SentenceTransformerEmbedder] Loaded model: {self.model_name} "
                  f"on device={self.device}")
        except Exception as exc:
            print(f"[SentenceTransformerEmbedder] Failed to load model '{self.model_name}': {exc}")
            self._model = None

    # ---- public --------------------------------------------------------

    def embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Encode *texts* in batches and return EmbeddingResult objects."""
        return self.embed_batch(texts, show_progress=False)

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Encode *texts* in fixed-size batches.

        Parameters
        ----------
        texts         : Input strings to encode.
        show_progress : Print a simple progress line if True.
        """
        if not texts:
            return []

        dim = self.get_embedding_dim()

        if self._model is None:
            # Return zero vectors if model unavailable
            return [
                EmbeddingResult(text=t, embedding=[0.0] * dim, model_name=self.model_name)
                for t in texts
            ]

        results: List[EmbeddingResult] = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            batch = texts[start: start + self.batch_size]
            if show_progress:
                end_idx = min(start + self.batch_size, total)
                print(f"[SentenceTransformerEmbedder] Embedding {start + 1}–{end_idx}/{total}", end="\r")
            try:
                vecs = self._model.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                for text, vec in zip(batch, vecs):
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=vec.tolist(),
                        model_name=self.model_name,
                    ))
            except Exception as exc:
                print(f"\n[SentenceTransformerEmbedder] Encode error: {exc}")
                for text in batch:
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=[0.0] * dim,
                        model_name=self.model_name,
                    ))

        if show_progress:
            print()  # newline after progress

        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single query string."""
        return self.embed([query])[0]

    def get_embedding_dim(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        if self._model is not None:
            try:
                return self._model.get_sentence_embedding_dimension()
            except Exception:
                pass
        return 384  # default for all-MiniLM-L6-v2


# ---------------------------------------------------------------------------
# HuggingFaceEmbedder
# ---------------------------------------------------------------------------

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedding backend using raw `transformers` AutoTokenizer + AutoModel
    with CLS-token / mean pooling and L2 normalisation.

    Parameters
    ----------
    model_name : HuggingFace model identifier (e.g. 'BAAI/bge-small-en-v1.5').
    device     : 'cpu' or 'cuda'.
    batch_size : Texts encoded per forward pass.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

        if not TRANSFORMERS_AVAILABLE:
            print("[HuggingFaceEmbedder] transformers not installed. "
                  "Run: pip install transformers")
            return
        if not TORCH_AVAILABLE:
            print("[HuggingFaceEmbedder] torch not installed. Run: pip install torch")
            return

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model from HuggingFace Hub."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f"[HuggingFaceEmbedder] Loaded model: {self.model_name}")
        except Exception as exc:
            print(f"[HuggingFaceEmbedder] Failed to load '{self.model_name}': {exc}")

    # ---- public --------------------------------------------------------

    def embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Encode *texts* and return EmbeddingResult objects."""
        if not texts:
            return []
        if self._model is None or self._tokenizer is None:
            dim = self.get_embedding_dim()
            return [
                EmbeddingResult(text=t, embedding=[0.0] * dim, model_name=self.model_name)
                for t in texts
            ]

        all_results: List[EmbeddingResult] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            try:
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    model_output = self._model(**encoded)

                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                embeddings = self._normalize(embeddings)

                for text, vec in zip(batch, embeddings):
                    all_results.append(EmbeddingResult(
                        text=text,
                        embedding=vec.cpu().tolist(),
                        model_name=self.model_name,
                    ))
            except Exception as exc:
                print(f"[HuggingFaceEmbedder] Encode error on batch: {exc}")
                dim = self.get_embedding_dim()
                for text in batch:
                    all_results.append(EmbeddingResult(
                        text=text,
                        embedding=[0.0] * dim,
                        model_name=self.model_name,
                    ))

        return all_results

    # ---- private -------------------------------------------------------

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> "torch.Tensor":
        """
        Average token embeddings weighted by the attention mask,
        excluding padding tokens.
        """
        token_embeddings = model_output.last_hidden_state          # (B, T, H)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

    def _normalize(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        """L2-normalise each row of the embedding matrix."""
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def get_embedding_dim(self) -> int:
        """Return hidden dimension of the loaded model, defaulting to 384."""
        if self._model is not None:
            try:
                return self._model.config.hidden_size
            except Exception:
                pass
        return 384


# ---------------------------------------------------------------------------
# EmbedderFactory
# ---------------------------------------------------------------------------

class EmbedderFactory:
    """Factory that constructs the appropriate embedder from RAGConfig."""

    @staticmethod
    def create(config: "RAGConfig") -> BaseEmbedder:
        """
        Select backend based on library availability and config settings.

        Priority: sentence-transformers > transformers (HuggingFace) > zero-vector stub.
        """
        model = config.embedding_model
        device = config.embedding_device
        batch = config.embedding_batch_size

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            return SentenceTransformerEmbedder(
                model_name=model,
                device=device,
                batch_size=batch,
            )
        elif TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            print("[EmbedderFactory] sentence-transformers unavailable; "
                  "falling back to HuggingFaceEmbedder.")
            return HuggingFaceEmbedder(
                model_name=model,
                device=device,
                batch_size=batch,
            )
        else:
            print("[EmbedderFactory] No embedding library found. "
                  "Returning a no-op stub embedder.")
            return _StubEmbedder(model_name=model, dim=config.embedding_dim)


class _StubEmbedder(BaseEmbedder):
    """Zero-vector fallback when no embedding library is installed."""

    def __init__(self, model_name: str = "stub", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def embed(self, texts: List[str]) -> List[EmbeddingResult]:
        return [
            EmbeddingResult(text=t, embedding=[0.0] * self.dim, model_name=self.model_name)
            for t in texts
        ]

    def get_embedding_dim(self) -> int:
        return self.dim
