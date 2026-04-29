"""Generate embeddings via Ollama.

Default model: jina-embeddings-v4 — 32,768-token context window, top-5 MTEB
retrieval (July 2025). Pull with:
  ollama pull k----n/jina-embeddings-v4-text-retrieval-F16
"""

from __future__ import annotations

import tiktoken
import ollama

# jina-embeddings-v4 supports 32,768 tokens.
# Cap at 30,000 to leave a safe margin against tokenizer differences.
_EMBED_MAX_TOKENS = 30000
_ENC = tiktoken.get_encoding("cl100k_base")


def _truncate(text: str, max_tokens: int = _EMBED_MAX_TOKENS) -> str:
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENC.decode(tokens[:max_tokens])


class Embedder:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        max_tokens: int = _EMBED_MAX_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = ollama.Client(host=base_url)

    def embed(self, text: str) -> list[float]:
        safe_text = _truncate(text, self.max_tokens)
        resp = self._client.embeddings(model=self.model, prompt=safe_text)
        return resp["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return len(self.embed("probe"))
