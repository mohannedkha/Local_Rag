"""Semantic + recursive Markdown chunker.

Strategy:
1. Split on Markdown structural boundaries (headers, blank lines between paragraphs).
2. If a structural chunk exceeds chunk_size tokens, recursively split it at sentence
   boundaries.
3. Apply a sliding overlap window between adjacent chunks.
4. (Semantic mode) Merge consecutive small chunks whose embedding cosine similarity
   exceeds the threshold, so semantically related sentences stay together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken


@dataclass
class Chunk:
    text: str
    source: str                  # originating file name
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class Chunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
        strategy: str = "semantic",
        semantic_threshold: float = 0.5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy
        self.semantic_threshold = semantic_threshold
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(self, md_path: Path) -> list[Chunk]:
        text = md_path.read_text(encoding="utf-8")
        source = md_path.name
        return self._chunk_text(text, source)

    def chunk_text(self, text: str, source: str = "inline") -> list[Chunk]:
        return self._chunk_text(text, source)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _chunk_text(self, text: str, source: str) -> list[Chunk]:
        structural = self._structural_split(text)
        flat = self._enforce_size(structural)
        overlapped = self._apply_overlap(flat)
        filtered = [c for c in overlapped if self._token_count(c) >= self.min_chunk_size]

        if self.strategy == "semantic":
            filtered = self._semantic_merge(filtered)

        return [
            Chunk(text=t, source=source, chunk_index=i)
            for i, t in enumerate(filtered)
        ]

    def _structural_split(self, text: str) -> list[str]:
        """Split on Markdown headers and paragraph double-newlines."""
        header_re = re.compile(r"(?m)^#{1,6} .+$")
        parts: list[str] = []
        last = 0
        for m in header_re.finditer(text):
            before = text[last:m.start()].strip()
            if before:
                parts.extend(self._paragraph_split(before))
            last = m.start()
        tail = text[last:].strip()
        if tail:
            parts.extend(self._paragraph_split(tail))
        return [p for p in parts if p.strip()]

    def _paragraph_split(self, text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    def _enforce_size(self, chunks: list[str]) -> list[str]:
        """Recursively split chunks that exceed chunk_size."""
        result: list[str] = []
        for chunk in chunks:
            if self._token_count(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                result.extend(self._sentence_split(chunk))
        return result

    def _sentence_split(self, text: str) -> list[str]:
        """Split text into sentence-boundary sub-chunks respecting chunk_size."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            t = self._token_count(sent)
            if current_tokens + t > self.chunk_size and current:
                result.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sent)
            current_tokens += t

        if current:
            result.append(" ".join(current))
        return result

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlapping tokens from the previous chunk to each chunk."""
        if self.chunk_overlap == 0 or len(chunks) < 2:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self._enc.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap:]
            overlap_text = self._enc.decode(overlap_tokens)
            result.append(overlap_text + " " + chunks[i])
        return result

    def _semantic_merge(self, chunks: list[str]) -> list[str]:
        """
        Merge consecutive chunks whose embeddings are highly similar.
        Falls back gracefully if sentence-transformers is unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            return chunks

        if len(chunks) <= 1:
            return chunks

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, normalize_embeddings=True)

        merged: list[str] = []
        current = chunks[0]
        current_emb = embeddings[0]

        for i in range(1, len(chunks)):
            sim = float(np.dot(current_emb, embeddings[i]))
            candidate = current + "\n\n" + chunks[i]
            if sim >= self.semantic_threshold and self._token_count(candidate) <= self.chunk_size:
                current = candidate
                # recompute embedding for merged chunk
                current_emb = model.encode([current], normalize_embeddings=True)[0]
            else:
                merged.append(current)
                current = chunks[i]
                current_emb = embeddings[i]

        merged.append(current)
        return merged
