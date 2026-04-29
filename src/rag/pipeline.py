"""Main RAG pipeline: wires together retrieval, memory, and LLM generation."""

from __future__ import annotations

from typing import Generator, Optional

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

    def query(self, user_input: str, stream: bool = False):
        """
        Run a full RAG turn:
          1. Retrieve relevant chunks (MMR or plain similarity).
          2. Inject long-term memories.
          3. Build prompt with short-term history.
          4. Generate answer via LLM.
          5. Store the turn in memory.
        """
        # 1. Retrieve
        if self.use_mmr:
            hits = self.vector_store.mmr_query(
                user_input,
                top_k=self.top_k,
                mmr_lambda=self.mmr_lambda,
            )
        else:
            hits = self.vector_store.query(user_input, top_k=self.top_k)

        context = self._build_context(hits)

        # 2. Long-term memory
        lt_memory = self.memory.format_long_term(user_input)

        # 3. Conversation history
        history = self.memory.format_history()

        # 4. Build messages
        messages = self._build_messages(user_input, context, lt_memory, history)

        # 5. Generate
        response = self.llm.chat(messages, stream=stream)

        if stream:
            return self._stream_and_store(user_input, response)

        # Store turn in memory
        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", response)
        return response

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_context(self, hits: list[dict]) -> str:
        parts = []
        for i, hit in enumerate(hits, 1):
            src = hit["metadata"].get("source", "unknown")
            parts.append(f"[{i}] (source: {src})\n{hit['text']}")
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self,
        user_input: str,
        context: str,
        lt_memory: str,
        history: str,
    ) -> list[dict]:
        system = self.system_prompt.strip()

        user_content_parts = []
        if lt_memory:
            user_content_parts.append(lt_memory)
        if history:
            user_content_parts.append(f"Conversation so far:\n{history}")
        if context:
            user_content_parts.append(f"Knowledge base context:\n{context}")
        user_content_parts.append(f"Question: {user_input}")

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

    def _stream_and_store(
        self, user_input: str, generator: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Yield streaming tokens, then persist the full response to memory."""
        collected: list[str] = []
        for token in generator:
            collected.append(token)
            yield token
        full_response = "".join(collected)
        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", full_response)
