"""Two-tier memory system.

Short-term: a rolling window of the last N conversation turns kept in RAM.
Long-term:  semantic storage of conversation summaries / key facts in ChromaDB,
            retrieved at query time to provide persistent context across sessions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class MemoryManager:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedder,
        short_term_window: int = 10,
        long_term_threshold: int = 20,
        long_term_top_k: int = 3,
    ):
        self._embedder = embedder
        self.short_term_window = short_term_window
        self.long_term_threshold = long_term_threshold
        self.long_term_top_k = long_term_top_k

        self._short_term: list[Turn] = []
        self._total_turns = 0

        client = chromadb.PersistentClient(
            path=str(Path(persist_dir).resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        self._lt_collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Short-term
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        turn = Turn(role=role, content=content)
        self._short_term.append(turn)
        self._total_turns += 1

        if len(self._short_term) > self.short_term_window:
            evicted = self._short_term.pop(0)
            self._maybe_store_long_term(evicted)

    def get_short_term(self) -> list[Turn]:
        return list(self._short_term)

    def format_history(self) -> str:
        """Return short-term history as a formatted string for LLM context."""
        lines = []
        for t in self._short_term:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Long-term
    # ------------------------------------------------------------------

    def _maybe_store_long_term(self, turn: Turn) -> None:
        """Persist an evicted turn as a long-term memory entry."""
        text = f"{turn.role}: {turn.content}"
        emb = self._embedder.embed(text)
        uid = hashlib.md5(f"{turn.timestamp}{text}".encode()).hexdigest()
        self._lt_collection.upsert(
            ids=[uid],
            embeddings=[emb],
            documents=[text],
            metadatas=[{"role": turn.role, "timestamp": turn.timestamp}],
        )

    def retrieve_long_term(self, query: str) -> list[str]:
        """Return relevant long-term memories for a query."""
        if self._lt_collection.count() == 0:
            return []
        emb = self._embedder.embed(query)
        results = self._lt_collection.query(
            query_embeddings=[emb],
            n_results=min(self.long_term_top_k, self._lt_collection.count()),
            include=["documents"],
        )
        return results["documents"][0]

    def format_long_term(self, query: str) -> str:
        memories = self.retrieve_long_term(query)
        if not memories:
            return ""
        joined = "\n".join(f"- {m}" for m in memories)
        return f"Relevant past context:\n{joined}"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_session(self, path: str) -> None:
        """Dump short-term memory to JSON so a session can be resumed."""
        data = [asdict(t) for t in self._short_term]
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_session(self, path: str) -> None:
        """Restore short-term memory from a previously saved JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._short_term = [Turn(**t) for t in raw]
        self._total_turns = len(self._short_term)
