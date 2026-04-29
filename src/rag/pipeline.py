"""Main RAG pipeline: retrieval -> memory -> LLM generation.

Uses parent-child retrieval: child chunks are searched for precision,
but the full parent section is passed to the LLM so it reads coherent
document context rather than scattered fragments.
"""

from __future__ import annotations

from typing import Generator

from src.db.vector_store import VectorStore
from src.llm.ollama_client import OllamaClient
from src.memory.memory_manager import MemoryManager


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        llm: OllamaClient,
        memory: MemoryManager,
        system_prompt: str,
        top_k: int = 5,
        mmr_lambda: float = 0.5,
        max_context_tokens: int = 4096,
        use_mmr: bool = True,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.memory = memory
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.max_context_tokens = max_context_tokens
        self.use_mmr = use_mmr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self, user_input: str, stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Full RAG turn:
          1. Retrieve relevant parent sections (via child-chunk MMR search).
          2. Inject long-term memories.
          3. Build prompt with short-term history.
          4. Generate answer (streaming or not).
          5. Store turn in memory.

        Returns a string or a generator of string tokens.
        Also returns source_refs as a side-effect via self.last_sources.
        """
        # 1. Retrieve (returns parent-level context + snippet metadata)
        if self.use_mmr:
            hits = self.vector_store.mmr_query(
                user_input, top_k=self.top_k, mmr_lambda=self.mmr_lambda
            )
        else:
            hits = self.vector_store.query(user_input, top_k=self.top_k)

        self.last_sources = hits  # expose for UI

        context = self._build_context(hits)
        lt_memory = self.memory.format_long_term(user_input)
        history = self.memory.format_history()
        messages = self._build_messages(user_input, context, lt_memory, history)

        # 2. Generate
        response = self.llm.chat(messages, stream=stream)

        if stream:
            return self._stream_and_store(user_input, response)

        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", response)
        return response

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_context(self, hits: list[dict]) -> str:
        """Build context string from parent-level hits, with source labels."""
        parts = []
        for i, hit in enumerate(hits, 1):
            src = hit.get("source", "unknown")
            sec = hit.get("section_index", "")
            header = f"[{i}] Document: {src}" + (f", section {sec}" if sec != "" else "")
            parts.append(f"{header}\n{hit['context_text']}")
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self,
        user_input: str,
        context: str,
        lt_memory: str,
        history: str,
    ) -> list[dict]:
        parts: list[str] = []
        if lt_memory:
            parts.append(lt_memory)
        if history:
            parts.append(f"Conversation so far:\n{history}")
        if context:
            parts.append(f"Knowledge base context:\n{context}")
        parts.append(f"Question: {user_input}")

        return [
            {"role": "system", "content": self.system_prompt.strip()},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _stream_and_store(
        self, user_input: str, generator: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        collected: list[str] = []
        for token in generator:
            collected.append(token)
            yield token
        full = "".join(collected)
        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", full)
