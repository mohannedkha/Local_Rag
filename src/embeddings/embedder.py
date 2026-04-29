"""Generate embeddings via Ollama.

Default model: bge-m3 — 8192-token context window, strong MTEB retrieval
scores, ideal for large document RAG. Pull with: ollama pull bge-m3
"""

from __future__ import annotations

import tiktoken
import ollama

# bge-m3 supports 8192 tokens. We cap at 8000 to leave a safe margin
# against tokenizer differences between tiktoken and the model's tokenizer.
_EMBED_MAX_TOKENS = 8000
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
