"""Build a fully wired RAGPipeline from config.yaml."""

from __future__ import annotations

import yaml

from src.db.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.ingestion.parent_chunker import ParentChildChunker
from src.llm.ollama_client import OllamaClient
from src.memory.memory_manager import MemoryManager
from src.rag.pipeline import RAGPipeline


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_chunker(cfg: dict) -> ParentChildChunker:
    c = cfg["chunking"]
    return ParentChildChunker(
        child_chunk_size=c["child_chunk_size"],
        child_chunk_overlap=c["child_chunk_overlap"],
        parent_chunk_size=c["parent_chunk_size"],
        parent_chunk_overlap=c["parent_chunk_overlap"],
        min_chunk_size=c["min_chunk_size"],
        strategy=c["strategy"],
        semantic_threshold=c["semantic_threshold"],
    )


def build_pipeline(config_path: str = "config/config.yaml") -> RAGPipeline:
    cfg = load_config(config_path)

    embedder = Embedder(
        model=cfg["ollama"]["embed_model"],
        base_url=cfg["ollama"]["base_url"],
    )
    vector_store = VectorStore(
        persist_dir=cfg["vector_store"]["persist_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
        parent_collection=cfg["vector_store"]["parent_collection"],
        embedder=embedder,
        distance_metric=cfg["vector_store"]["distance_metric"],
    )
    memory = MemoryManager(
        persist_dir=cfg["memory"]["persist_dir"],
        collection_name=cfg["memory"]["conversation_collection"],
        embedder=embedder,
        short_term_window=cfg["memory"]["short_term_window"],
        long_term_threshold=cfg["memory"]["long_term_threshold"],
        long_term_top_k=cfg["memory"]["long_term_top_k"],
    )
    llm = OllamaClient(
        model=cfg["ollama"]["llm_model"],
        base_url=cfg["ollama"]["base_url"],
        temperature=cfg["ollama"]["temperature"],
        context_window=cfg["ollama"]["context_window"],
    )

    return RAGPipeline(
        vector_store=vector_store,
        llm=llm,
        memory=memory,
        system_prompt=cfg["rag"]["system_prompt"],
        top_k=cfg["vector_store"]["top_k"],
        mmr_lambda=cfg["vector_store"]["mmr_lambda"],
        max_context_tokens=cfg["rag"]["max_context_tokens"],
    )
