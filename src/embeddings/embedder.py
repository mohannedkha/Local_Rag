"""Generate embeddings via Ollama (nomic-embed-text or any embed model)."""

from __future__ import annotations

from typing import Union

import ollama


class Embedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self._client = ollama.Client(host=base_url)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings(model=self.model, prompt=text)
        return resp["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension by probing with a dummy string."""
        return len(self.embed("probe"))
