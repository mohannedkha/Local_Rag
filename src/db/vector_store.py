"""ChromaDB-backed vector store with parent-child retrieval.

Two collections:
  doc_children  — small child chunks, embedded for precise vector search.
  doc_parents   — large parent sections, stored as text (no embedding needed).

Query flow:
  1. Embed the query with mxbai-embed-large.
  2. Search doc_children -> top-k child chunk hits.
  3. For each hit, fetch its parent section from doc_parents.
  4. Deduplicate parents and return them as context to the LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.ingestion.parent_chunker import ChildChunk, ParentChunk


class VectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        parent_collection: str,
        embedder,
        distance_metric: str = "cosine",
    ):
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(Path(persist_dir).resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        self._children = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        # Parents are stored as plain text in a separate collection (not embedded).
        self._parents = self._client.get_or_create_collection(
            name=parent_collection,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_document(
        self, parent_chunks: list[ParentChunk], child_chunks: list[ChildChunk]
    ) -> None:
        """Store parent sections and embed child chunks for a document."""
        if parent_chunks:
            self._parents.upsert(
                ids=[p.parent_id for p in parent_chunks],
                documents=[p.text for p in parent_chunks],
                metadatas=[
                    {"source": p.source, "section_index": p.section_index, **p.metadata}
                    for p in parent_chunks
                ],
            )

        if child_chunks:
            texts = [c.text for c in child_chunks]
            embeddings = self._embedder.embed_batch(texts)
            self._children.upsert(
                ids=[c.child_id for c in child_chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source": c.source,
                        "parent_id": c.parent_id,
                        "chunk_index": c.chunk_index,
                        **c.metadata,
                    }
                    for c in child_chunks
                ],
            )

    def delete_source(self, source: str) -> None:
        """Remove all chunks and parents belonging to a source file."""
        self._children.delete(where={"source": source})
        self._parents.delete(where={"source": source})

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top_k results. Each result contains the full *parent* text
        (not the child snippet) so the LLM has rich document context.
        Also returns metadata for display (source, section, matched snippet).
        """
        query_emb = self._embedder.embed(query_text)
        where = {"source": source_filter} if source_filter else None

        results = self._children.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, max(self._children.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._resolve_parents(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )

    def mmr_query(
        self,
        query_text: str,
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
    ) -> list[dict]:
        """MMR retrieval on child chunks, then resolve to parent context."""
        import numpy as np

        if self._children.count() == 0:
            return []

        query_emb = self._embedder.embed(query_text)
        n = min(fetch_k, self._children.count())

        results = self._children.query(
            query_embeddings=[query_emb],
            n_results=n,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        embs = np.array(results["embeddings"][0])

        selected_idx: list[int] = []
        remaining = list(range(len(docs)))

        for _ in range(min(top_k, len(docs))):
            if not remaining:
                break
            if not selected_idx:
                best = max(remaining, key=lambda i: 1 - dists[i])
            else:
                sel_embs = embs[selected_idx]
                scores = []
                for i in remaining:
                    rel = 1 - dists[i]
                    div = float(np.max(embs[i] @ sel_embs.T))
                    scores.append(mmr_lambda * rel - (1 - mmr_lambda) * div)
                best = remaining[int(np.argmax(scores))]
            selected_idx.append(best)
            remaining.remove(best)

        sel_docs = [docs[i] for i in selected_idx]
        sel_metas = [metas[i] for i in selected_idx]
        sel_dists = [dists[i] for i in selected_idx]

        return self._resolve_parents(sel_docs, sel_metas, sel_dists)

    def list_sources(self) -> list[str]:
        """Return distinct source filenames ingested in the store."""
        if self._parents.count() == 0:
            return []
        result = self._parents.get(include=["metadatas"])
        sources = {m["source"] for m in result["metadatas"] if "source" in m}
        return sorted(sources)

    def count_children(self) -> int:
        return self._children.count()

    def count_parents(self) -> int:
        return self._parents.count()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_parents(
        self,
        child_docs: list[str],
        child_metas: list[dict],
        distances: list[float],
    ) -> list[dict]:
        """
        For each child hit, fetch the parent section text and return a result
        dict that contains:
          - context_text: full parent section (what the LLM reads)
          - snippet:      the matched child chunk (shown in the UI as evidence)
          - source:       filename
          - score:        similarity score
          - section_index
        Deduplicate by parent_id so the same section isn't returned twice.
        """
        seen_parents: set[str] = set()
        hits: list[dict] = []

        for child_text, meta, dist in zip(child_docs, child_metas, distances):
            parent_id = meta.get("parent_id", "")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent_text = child_text  # fallback
            if parent_id:
                try:
                    p_result = self._parents.get(
                        ids=[parent_id], include=["documents"]
                    )
                    if p_result["documents"]:
                        parent_text = p_result["documents"][0]
                except Exception:
                    pass

            hits.append(
                {
                    "context_text": parent_text,
                    "snippet": child_text,
                    "source": meta.get("source", "unknown"),
                    "section_index": meta.get("section_index", 0),
                    "score": round(1 - dist, 4),
                    "metadata": meta,
                }
            )

        return hits
