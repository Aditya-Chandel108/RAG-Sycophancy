"""
rag_pipeline.py
===============
Main RAGPipeline class that ties together all components:
  DocumentProcessor → Chunker → Embedder → VectorStore → Retriever → LLM

Usage
-----
    from rag_pipeline import RAGPipeline, RAGConfig

    config = RAGConfig(collection_name="my_docs")
    pipeline = RAGPipeline(config)

    pipeline.ingest("path/to/documents/")
    result = pipeline.query("What is retrieval-augmented generation?")
    print(result["answer"])
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[RAGPipeline] requests not installed. Run: pip install requests")

from rag_pipeline.rag_config import RAGConfig
from rag_pipeline.document_processor import DocumentProcessor
from rag_pipeline.embedder import EmbedderFactory
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.retriever import Retriever, RetrievalResult
from rag_pipeline.chat_history import ChatHistory
from rag_pipeline.chunker import ChunkerFactory


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Lifecycle
    ---------
    1. __init__         : Parse config, build all sub-components.
    2. ingest(source)   : Load documents → chunk → embed → store in ChromaDB.
    3. query(user_query): Retrieve → build context → call LLM → update history.
    4. chat(user_query) : Thin wrapper returning only the answer string.

    Parameters
    ----------
    config : RAGConfig instance.  If None, default values are used.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._components_built = False
        self._build_components()

    # ------------------------------------------------------------------
    # Component initialisation
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        """Instantiate all pipeline components in dependency order."""
        print("[RAGPipeline] Initialising pipeline components …")

        # Document processing
        self.doc_processor = DocumentProcessor(self.config)

        # Embedding
        self.embedder = EmbedderFactory.create(self.config)

        # Vector store
        self.vector_store = VectorStore(self.config, self.embedder)

        # Retrieval
        self.retriever = Retriever(self.config, self.vector_store, self.embedder)

        # Chat history
        self.history = ChatHistory(
            max_history=self.config.max_history,
            persist_path=self.config.history_persist_path,
        )

        self._components_built = True
        print("[RAGPipeline] All components ready.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: str) -> Dict[str, Any]:
        """
        Load, chunk, embed, and store documents from *source*.

        Parameters
        ----------
        source : Path to a file or directory.

        Returns
        -------
        dict with keys:
            source          : Input path.
            documents_loaded: Number of documents processed.
            total_pages     : Total pages across all documents.
            status          : 'ok' or 'error'.
            error           : Error message (only present on failure).
        """
        try:
            documents = self.doc_processor.process(source)
            if not documents:
                return {
                    "source": source,
                    "documents_loaded": 0,
                    "total_pages": 0,
                    "status": "ok",
                    "message": "No supported documents found.",
                }

            total_pages = sum(d.total_pages for d in documents)
            self.retriever.add_documents(documents)

            stats = {
                "source": source,
                "documents_loaded": len(documents),
                "total_pages": total_pages,
                "collection_stats": self.vector_store.get_collection_stats(),
                "status": "ok",
            }
            print(f"[RAGPipeline] Ingested {len(documents)} document(s), "
                  f"{total_pages} page(s) from: {source}")
            return stats

        except Exception as exc:
            print(f"[RAGPipeline] Ingestion error: {exc}")
            return {"source": source, "status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        use_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a user query.

        Steps
        -----
        1. Retrieve relevant chunks.
        2. Build context string (retrieved docs + optional history).
        3. Call LLM.
        4. Update chat history.

        Parameters
        ----------
        user_query  : The user's natural-language question.
        use_history : Whether to prepend conversation history to the context.

        Returns
        -------
        dict with keys:
            query           : Original query.
            answer          : Generated answer string.
            retrieved_docs  : List of SearchResult dicts.
            context_used    : The full context string sent to the LLM.
            retrieval_mode  : Mode used ('dense', 'sparse', 'hybrid').
            num_retrieved   : Number of retrieved chunks.
            history_included: Whether history was included.
        """
        # 1. Retrieve
        retrieval_result: RetrievalResult = self.retriever.retrieve(user_query)

        # 2. Build context
        context_str = self._build_context(
            retrieval_result,
            self.history if use_history else None,
        )

        # 3. Build LLM messages
        system_msg = (
            "You are a helpful, factual assistant. "
            "Answer the user's question strictly using the provided context. "
            "If the context does not contain enough information, say so clearly."
        )
        user_msg = (
            f"Context:\n{context_str}\n\n"
            f"Question: {user_query}"
        )

        messages: Optional[List[Dict[str, str]]] = None
        if use_history and len(self.history) > 0:
            history_messages = self.history.get_messages_for_llm()
            messages = (
                [{"role": "system", "content": system_msg}]
                + history_messages
                + [{"role": "user", "content": user_msg}]
            )

        # 4. Call LLM
        answer = self._call_llm(system_msg, user_msg, messages=messages)

        # 5. Record turn in history
        self.history.add_turn(
            user_msg=user_query,
            assistant_msg=answer,
            context=context_str,
            audit_status="",
        )

        return {
            "query": user_query,
            "answer": answer,
            "retrieved_docs": [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "score": round(r.score, 4),
                    "source_doc": r.source_doc,
                    "page_num": r.page_num,
                }
                for r in retrieval_result.results
            ],
            "context_used": context_str,
            "retrieval_mode": retrieval_result.retrieval_mode,
            "num_retrieved": retrieval_result.num_retrieved,
            "history_included": use_history,
        }

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def chat(self, user_query: str) -> str:
        """
        Simple chat interface: submit a query, get back the answer string.

        Parameters
        ----------
        user_query : User's question.

        Returns
        -------
        Answer string from the LLM.
        """
        result = self.query(user_query, use_history=True)
        return result["answer"]

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self.history.clear()
        print("[RAGPipeline] Conversation history reset.")

    # ------------------------------------------------------------------
    # Stats and session persistence
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of pipeline state.

        Returns
        -------
        dict with keys:
            collection_stats : ChromaDB collection info.
            history_length   : Number of turns in memory.
            config_summary   : Key configuration values.
        """
        return {
            "collection_stats": self.vector_store.get_collection_stats(),
            "history_length": len(self.history),
            "config_summary": self.config.summary(),
        }

    def save_session(self, path: str) -> None:
        """
        Save the conversation history and pipeline stats to *path* (JSON).

        Parameters
        ----------
        path : Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        session = {
            "stats": self.get_stats(),
            "history": [
                {
                    "turn_id": t.turn_id,
                    "user": t.user_message.content,
                    "assistant": t.assistant_message.content,
                    "audit_status": t.audit_status,
                }
                for t in self.history.get_history()
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(session, fh, indent=2, ensure_ascii=False)
        print(f"[RAGPipeline] Session saved to: {path}")

    def load_session(self, path: str) -> None:
        """
        Restore conversation history from a saved session file.

        Parameters
        ----------
        path : Source file path previously written by `save_session`.
        """
        self.history.load(path)
        print(f"[RAGPipeline] Session loaded from: {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        retrieval_result: RetrievalResult,
        history: Optional[ChatHistory],
    ) -> str:
        """
        Assemble the context string from retrieved documents and
        (optionally) the conversation history.

        Parameters
        ----------
        retrieval_result : Output of Retriever.retrieve().
        history          : ChatHistory instance (None to skip).

        Returns
        -------
        A formatted context string ready for injection into the prompt.
        """
        parts: List[str] = []

        if history and self.config.include_history and len(history) > 0:
            history_text = history.format_history_as_text()
            if history_text:
                parts.append("## Conversation History\n" + history_text)

        if retrieval_result.results:
            doc_parts: List[str] = []
            for i, res in enumerate(retrieval_result.results, start=1):
                source_label = (
                    f"{os.path.basename(res.source_doc)} p.{res.page_num}"
                    if res.source_doc else f"chunk {i}"
                )
                doc_parts.append(
                    f"[{i}] (source: {source_label}, score: {res.score:.3f})\n{res.text}"
                )
            parts.append("## Retrieved Context\n" + "\n\n".join(doc_parts))
        else:
            parts.append("## Retrieved Context\nNo relevant documents found.")

        return "\n\n".join(parts)

    def _call_llm(
        self,
        system_msg: str,
        user_msg: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Call the configured LLM API.

        The payload follows OpenAI-compatible chat-completion format.
        Replace the placeholder block with your preferred provider's SDK or
        requests call.

        Parameters
        ----------
        system_msg : System instruction string.
        user_msg   : Final user message string.
        messages   : Full message list (overrides system_msg / user_msg if provided).

        Returns
        -------
        Generated text from the LLM, or an error string.
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages or [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.config.temperature,
        }

        # here add the api of the model
        # Example Implementation for Requests:
        # try:
        #     response = requests.post(self.config.api_endpoint,
        #                              headers=self.config.headers,
        #                              json=payload)
        #     response.raise_for_status()
        #     return response.json()['choices'][0]['message']['content']
        # except Exception as e:
        #     return f"ERROR: {str(e)}"

        return "LLM_PLACEHOLDER"  # Placeholder for structural integrity
