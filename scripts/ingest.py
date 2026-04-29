#!/usr/bin/env python3
"""Ingest PDFs into the vector store.

Usage:
    python scripts/ingest.py                  # process all PDFs in data/raw/
    python scripts/ingest.py --force          # re-process even if already converted
    python scripts/ingest.py --file report.pdf
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from tqdm import tqdm

from src.embeddings.embedder import Embedder
from src.db.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.pdf_converter import PDFConverter

console = Console()


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store")
    parser.add_argument("--file", help="Single PDF filename inside data/raw/")
    parser.add_argument("--force", action="store_true", help="Re-convert already processed files")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    converter = PDFConverter(
        raw_dir=cfg["ingestion"]["raw_dir"],
        processed_dir=cfg["ingestion"]["processed_dir"],
    )
    chunker = Chunker(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
        min_chunk_size=cfg["chunking"]["min_chunk_size"],
        strategy=cfg["chunking"]["strategy"],
        semantic_threshold=cfg["chunking"]["semantic_threshold"],
    )
    embedder = Embedder(
        model=cfg["ollama"]["embed_model"],
        base_url=cfg["ollama"]["base_url"],
    )
    store = VectorStore(
        persist_dir=cfg["vector_store"]["persist_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
        embedder=embedder,
        distance_metric=cfg["vector_store"]["distance_metric"],
    )

    if args.file:
        raw_path = Path(cfg["ingestion"]["raw_dir"]) / args.file
        if not raw_path.exists():
            console.print(f"[red]File not found:[/red] {raw_path}")
            sys.exit(1)
        md_paths = [converter.convert_file(raw_path, force=args.force)]
    else:
        md_paths = converter.convert_all(force=args.force)

    if not md_paths:
        console.print("[yellow]No files to ingest.[/yellow]")
        return

    total_chunks = 0
    for md_path in tqdm(md_paths, desc="Chunking & embedding"):
        chunks = chunker.chunk_file(md_path)
        store.delete_source(md_path.name)   # remove stale chunks
        store.add_chunks(chunks)
        total_chunks += len(chunks)
        console.print(f"  [green]{md_path.name}[/green] -> {len(chunks)} chunks")

    console.print(f"\n[bold green]Done.[/bold green] {total_chunks} total chunks stored. DB size: {store.count()}")


if __name__ == "__main__":
    main()
