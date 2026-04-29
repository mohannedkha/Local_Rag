"""Convert PDF files to Markdown using pymupdf4llm."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf4llm
from tqdm import tqdm


class PDFConverter:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _output_path(self, pdf_path: Path) -> Path:
        return self.processed_dir / (pdf_path.stem + ".md")

    def _file_hash(self, path: Path) -> str:
        h = hashlib.md5()
        h.update(path.read_bytes())
        return h.hexdigest()

    def convert_file(self, pdf_path: Path, force: bool = False) -> Path:
        """Convert a single PDF to Markdown. Returns the output path."""
        out_path = self._output_path(pdf_path)
        hash_path = out_path.with_suffix(".md.hash")

        current_hash = self._file_hash(pdf_path)
        if not force and out_path.exists() and hash_path.exists():
            if hash_path.read_text().strip() == current_hash:
                return out_path

        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        out_path.write_text(md_text, encoding="utf-8")
        hash_path.write_text(current_hash, encoding="utf-8")
        return out_path

    def convert_all(self, force: bool = False) -> list[Path]:
        """Convert every PDF in raw_dir. Skips already-converted files."""
        pdfs = sorted(self.raw_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {self.raw_dir}")
            return []

        outputs: list[Path] = []
        for pdf in tqdm(pdfs, desc="Converting PDFs"):
            out = self.convert_file(pdf, force=force)
            outputs.append(out)
            print(f"  {pdf.name} -> {out.name}")

        return outputs
