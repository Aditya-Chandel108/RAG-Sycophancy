"""
chat_history.py
===============
Conversation history management for the RAG pipeline.

Maintains a rolling window of question-answer turns, serialises them for
injection into LLM prompts, and optionally persists to / restores from JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """
    A single message in the conversation.

    Attributes
    ----------
    role      : 'user', 'assistant', or 'system'.
    content   : Text content of the message.
    timestamp : ISO 8601 timestamp (UTC) of when the message was created.
    metadata  : Optional arbitrary key/value pairs (e.g. token count, latency).
    """
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatTurn:
    """
    One complete exchange: a user question paired with the assistant's reply.

    Attributes
    ----------
    turn_id           : Sequential integer ID (1-indexed).
    user_message      : The user's Message.
    assistant_message : The assistant's Message.
    retrieved_context : The document context that was injected into the prompt.
    audit_status      : Result of the sycophancy auditor ('TRUSTED', 'NOISY', '').
    """
    turn_id: int
    user_message: Message
    assistant_message: Message
    retrieved_context: str = ""
    audit_status: str = ""


# ---------------------------------------------------------------------------
# ChatHistory
# ---------------------------------------------------------------------------

class ChatHistory:
    """
    Manages a rolling window of ChatTurn objects.

    Parameters
    ----------
    max_history  : Maximum number of turns to keep in memory.
    persist_path : If provided, automatically save after each new turn.
    """

    def __init__(
        self,
        max_history: int = 10,
        persist_path: Optional[str] = None,
    ):
        self.max_history = max_history
        self.persist_path = persist_path
        self._turns: List[ChatTurn] = []
        self._next_id: int = 1

        if persist_path and os.path.isfile(persist_path):
            try:
                self.load(persist_path)
                print(f"[ChatHistory] Loaded {len(self._turns)} turns from {persist_path}")
            except Exception as exc:
                print(f"[ChatHistory] Could not load history from {persist_path}: {exc}")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        context: str = "",
        audit_status: str = "",
    ) -> ChatTurn:
        """
        Record a new question-answer pair.

        Parameters
        ----------
        user_msg      : The user's raw question.
        assistant_msg : The assistant's answer.
        context       : Retrieved document context used for generation.
        audit_status  : Auditor verdict ('TRUSTED' / 'NOISY' / '').

        Returns
        -------
        The newly created ChatTurn.
        """
        turn = ChatTurn(
            turn_id=self._next_id,
            user_message=Message(role="user", content=user_msg),
            assistant_message=Message(role="assistant", content=assistant_msg),
            retrieved_context=context,
            audit_status=audit_status,
        )
        self._turns.append(turn)
        self._next_id += 1

        # Enforce rolling window
        if len(self._turns) > self.max_history:
            self._turns = self._turns[-self.max_history:]

        if self.persist_path:
            try:
                self.save(self.persist_path)
            except Exception as exc:
                print(f"[ChatHistory] Auto-save failed: {exc}")

        return turn

    def clear(self) -> None:
        """Remove all turns from memory."""
        self._turns.clear()
        self._next_id = 1
        print("[ChatHistory] History cleared.")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_history(self) -> List[ChatTurn]:
        """Return all stored ChatTurn objects (oldest first)."""
        return list(self._turns)

    def get_context_window(self, n: Optional[int] = None) -> List[ChatTurn]:
        """
        Return the last *n* turns (defaults to max_history).

        Parameters
        ----------
        n : Number of most-recent turns to return.
        """
        if n is None:
            n = self.max_history
        return self._turns[-n:] if self._turns else []

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Format the conversation history as a list of role/content dicts
        suitable for passing directly to a chat-completion API.

        Returns
        -------
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        messages: List[Dict[str, str]] = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user_message.content})
            messages.append({"role": "assistant", "content": turn.assistant_message.content})
        return messages

    def format_history_as_text(self) -> str:
        """
        Return the conversation history as a human-readable text block,
        suitable for injection into a prompt as a prefix.

        Example output
        --------------
        [Turn 1]
        User: What is RAG?
        Assistant: RAG stands for Retrieval-Augmented Generation...

        [Turn 2]
        User: ...
        """
        if not self._turns:
            return ""
        lines: List[str] = []
        for turn in self._turns:
            lines.append(f"[Turn {turn.turn_id}]")
            lines.append(f"User: {turn.user_message.content}")
            lines.append(f"Assistant: {turn.assistant_message.content}")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the history to a JSON file.

        Parameters
        ----------
        path : Destination file path.  Parent directories are created if needed.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "next_id": self._next_id,
            "max_history": self.max_history,
            "turns": [self._turn_to_dict(t) for t in self._turns],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """
        Deserialise history from a JSON file previously written by `save()`.

        Parameters
        ----------
        path : Source file path.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._next_id = payload.get("next_id", 1)
        self.max_history = payload.get("max_history", self.max_history)
        self._turns = [self._dict_to_turn(d) for d in payload.get("turns", [])]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of turns currently in history."""
        return len(self._turns)

    def __repr__(self) -> str:
        return (
            f"ChatHistory(turns={len(self._turns)}, "
            f"max_history={self.max_history})"
        )

    # ------------------------------------------------------------------
    # Private serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _turn_to_dict(turn: ChatTurn) -> Dict[str, Any]:
        """Convert a ChatTurn to a JSON-serialisable dict."""
        return {
            "turn_id": turn.turn_id,
            "user_message": asdict(turn.user_message),
            "assistant_message": asdict(turn.assistant_message),
            "retrieved_context": turn.retrieved_context,
            "audit_status": turn.audit_status,
        }

    @staticmethod
    def _dict_to_turn(d: Dict[str, Any]) -> ChatTurn:
        """Reconstruct a ChatTurn from a deserialised dict."""
        return ChatTurn(
            turn_id=d["turn_id"],
            user_message=Message(**d["user_message"]),
            assistant_message=Message(**d["assistant_message"]),
            retrieved_context=d.get("retrieved_context", ""),
            audit_status=d.get("audit_status", ""),
        )
