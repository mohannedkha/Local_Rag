"""Parent-Child chunking for document-level RAG retrieval.

Architecture:
  - Parent chunks: large document sections (heading + body, ~1024 tokens).
    These are stored as plain text and returned as LLM context so the model
    reads a coherent section rather than isolated fragments.
  - Child chunks: small, precise sub-chunks of each parent (~256 tokens).
    These are embedded and used for vector search so retrieval is accurate.

Flow:
  PDF/MD -> parent sections -> child chunks
               (stored in               (embedded in
               doc_parents collection)  doc_children collection)

At query time: retrieve child chunks -> look up their parent_id
               -> return parent text as context to LLM.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    source: str           # originating filename
    section_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class ParentChildChunker:
    def __init__(
        self,
        child_chunk_size: int = 256,
        child_chunk_overlap: int = 32,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 64,
        min_chunk_size: int = 80,
        strategy: str = "semantic",
        semantic_threshold: float = 0.5,
    ):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy
        self.semantic_threshold = semantic_threshold
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(
        self, md_path: Path
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        text = md_path.read_text(encoding="utf-8")
        return self.chunk_text(text, source=md_path.name)

    def chunk_text(
        self, text: str, source: str = "inline"
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Return (parent_chunks, child_chunks) for a document."""
        raw_sections = self._split_into_sections(text)
        parent_sections = self._enforce_parent_size(raw_sections)

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        child_global_idx = 0
        for sec_idx, section_text in enumerate(parent_sections):
            pid = self._make_parent_id(source, sec_idx, section_text)
            parent = ParentChunk(
                parent_id=pid,
                text=section_text,
                source=source,
                section_index=sec_idx,
                metadata={"doc_name": source},
            )
            parents.append(parent)

            # Split parent into children
            raw_children = self._split_into_children(section_text)
            for child_text in raw_children:
                if self._token_count(child_text) < self.min_chunk_size:
                    continue
                cid = self._make_child_id(pid, child_global_idx)
                children.append(
                    ChildChunk(
                        child_id=cid,
                        parent_id=pid,
                        text=child_text,
                        source=source,
                        chunk_index=child_global_idx,
                        metadata={"doc_name": source, "section_index": sec_idx},
                    )
                )
                child_global_idx += 1

        if self.strategy == "semantic":
            parents = self._semantic_merge_parents(parents, source)

        return parents, children

    # ------------------------------------------------------------------
    # Section splitting (parent level)
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> list[str]:
        """Split on Markdown H1/H2 headings to preserve document structure."""
        heading_re = re.compile(r"(?m)^#{1,2} .+$")
        sections: list[str] = []
        last = 0
        for m in heading_re.finditer(text):
            before = text[last : m.start()].strip()
            if before:
                sections.append(before)
            last = m.start()
        tail = text[last:].strip()
        if tail:
            sections.append(tail)
        if not sections:
            # No headings — treat whole doc as one section
            sections = [text.strip()]
        return [s for s in sections if s.strip()]

    def _enforce_parent_size(self, sections: list[str]) -> list[str]:
        """Ensure no parent exceeds parent_chunk_size by splitting on paragraphs."""
        result: list[str] = []
        for section in sections:
            if self._token_count(section) <= self.parent_chunk_size:
                result.append(section)
            else:
                result.extend(self._split_large_section(section))
        return result

    def _split_large_section(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            t = self._token_count(para)
            if current_tokens + t > self.parent_chunk_size and current:
                result.append("\n\n".join(current))
                # Overlap: keep last paragraph in the next window
                overlap = current[-1:] if self.parent_chunk_overlap > 0 else []
                current = overlap
                current_tokens = sum(self._token_count(p) for p in current)
            current.append(para)
            current_tokens += t

        if current:
            result.append("\n\n".join(current))
        return result

    # ------------------------------------------------------------------
    # Child splitting
    # ------------------------------------------------------------------

    def _split_into_children(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            t = self._token_count(sent)
            if current_tokens + t > self.child_chunk_size and current:
                result.append(" ".join(current))
                # Overlap window
                overlap_tokens = 0
                overlap: list[str] = []
                for s in reversed(current):
                    overlap_tokens += self._token_count(s)
                    overlap.insert(0, s)
                    if overlap_tokens >= self.child_chunk_overlap:
                        break
                current = overlap
                current_tokens = sum(self._token_count(s) for s in current)
            current.append(sent)
            current_tokens += t

        if current:
            result.append(" ".join(current))
        return result

    # ------------------------------------------------------------------
    # Semantic merging of parents (optional)
    # ------------------------------------------------------------------

    def _semantic_merge_parents(
        self, parents: list[ParentChunk], source: str
    ) -> list[ParentChunk]:
        """Merge consecutive small parent sections that are semantically similar."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            return parents

        if len(parents) <= 1:
            return parents

        texts = [p.text for p in parents]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts, normalize_embeddings=True)

        merged: list[ParentChunk] = []
        current = parents[0]
        current_emb = embs[0]

        for i in range(1, len(parents)):
            sim = float(np.dot(current_emb, embs[i]))
            candidate_text = current.text + "\n\n" + parents[i].text
            candidate_tokens = self._token_count(candidate_text)
            if sim >= self.semantic_threshold and candidate_tokens <= self.parent_chunk_size:
                current = ParentChunk(
                    parent_id=current.parent_id,
                    text=candidate_text,
                    source=source,
                    section_index=current.section_index,
                    metadata=current.metadata,
                )
                current_emb = model.encode([candidate_text], normalize_embeddings=True)[0]
            else:
                merged.append(current)
                current = parents[i]
                current_emb = embs[i]

        merged.append(current)
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text))

    @staticmethod
    def _make_parent_id(source: str, idx: int, text: str) -> str:
        raw = f"{source}::parent::{idx}::{text[:64]}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _make_child_id(parent_id: str, idx: int) -> str:
        return hashlib.md5(f"{parent_id}::child::{idx}".encode()).hexdigest()
