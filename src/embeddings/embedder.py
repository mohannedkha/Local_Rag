"""Generate embeddings via Ollama (mxbai-embed-large or any embed model)."""

from __future__ import annotations

import tiktoken
import ollama

# mxbai-embed-large has a 512-token context limit.
# We truncate conservatively to 480 to stay clear of the boundary
# regardless of tokenizer differences between tiktoken and Ollama.
_EMBED_MAX_TOKENS = 480
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
