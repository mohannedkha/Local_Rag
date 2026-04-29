#!/usr/bin/env python3
"""Ingest PDFs or Markdown files into the vector store.

Usage:
    python scripts/ingest.py                  # all PDFs/MDs in data/raw/
    python scripts/ingest.py --force          # re-process even if already converted
    python scripts/ingest.py --file report.pdf
    python scripts/ingest.py --file notes.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from tqdm import tqdm

from src.db.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.ingestion.parent_chunker import ParentChildChunker
from src.ingestion.pdf_converter import PDFConverter

console = Console()


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ingest_file(
    file_path: Path,
    chunker: ParentChildChunker,
    store: VectorStore,
    converter: PDFConverter,
    force: bool = False,
) -> int:
    """Ingest a single PDF or Markdown file. Returns number of child chunks added."""
    if file_path.suffix.lower() == ".pdf":
        md_path = converter.convert_file(file_path, force=force)
    elif file_path.suffix.lower() in (".md", ".markdown", ".txt"):
        md_path = file_path
    else:
        console.print(f"[yellow]Skipping unsupported file:[/yellow] {file_path.name}")
        return 0

    parents, children = chunker.chunk_file(md_path)

    # Use the original filename (not the converted .md name) as source key
    source_name = file_path.name
    # But child/parent chunks reference the md filename — normalise to original
    for p in parents:
        p.source = source_name
        p.metadata["doc_name"] = source_name
    for c in children:
        c.source = source_name
        c.metadata["doc_name"] = source_name

    store.delete_source(source_name)
    store.add_document(parents, children)
    return len(children)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store")
    parser.add_argument("--file", help="Single filename inside data/raw/")
    parser.add_argument("--force", action="store_true", help="Re-convert already processed PDFs")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["ingestion"]["raw_dir"])

    converter = PDFConverter(
        raw_dir=cfg["ingestion"]["raw_dir"],
        processed_dir=cfg["ingestion"]["processed_dir"],
    )
    chunker = ParentChildChunker(
        child_chunk_size=cfg["chunking"]["child_chunk_size"],
        child_chunk_overlap=cfg["chunking"]["child_chunk_overlap"],
        parent_chunk_size=cfg["chunking"]["parent_chunk_size"],
        parent_chunk_overlap=cfg["chunking"]["parent_chunk_overlap"],
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
        parent_collection=cfg["vector_store"]["parent_collection"],
        embedder=embedder,
        distance_metric=cfg["vector_store"]["distance_metric"],
    )

    if args.file:
        target = raw_dir / args.file
        if not target.exists():
            console.print(f"[red]File not found:[/red] {target}")
            sys.exit(1)
        files = [target]
    else:
        files = sorted(
            list(raw_dir.glob("*.pdf"))
            + list(raw_dir.glob("*.md"))
            + list(raw_dir.glob("*.markdown"))
            + list(raw_dir.glob("*.txt"))
        )

    if not files:
        console.print(f"[yellow]No files found in {raw_dir}[/yellow]")
        return

    total_children = 0
    for f in tqdm(files, desc="Ingesting"):
        n = ingest_file(f, chunker, store, converter, force=args.force)
        total_children += n
        console.print(f"  [green]{f.name}[/green] -> {n} child chunks")

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{total_children} child chunks | "
        f"{store.count_parents()} parent sections | "
        f"sources: {store.list_sources()}"
    )


if __name__ == "__main__":
    main()
