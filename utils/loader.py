"""Document loading and chunking helpers for the Streamlit RAG app."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


SUPPORTED_TYPES = ["pdf", "txt", "md"]


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """Extract text from a PDF file-like object."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf. Install it with `pip install pypdf`.") from exc

    reader = PdfReader(file_obj)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_text_from_upload(uploaded_file) -> str:
    """Return readable text from a Streamlit uploaded file."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)

    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    if suffix in {".txt", ".md"}:
        return uploaded_file.read().decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_words: int = 160, overlap: int = 35) -> list[tuple[int, str]]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")

    chunks = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append((start, chunk))
        if start + chunk_words >= len(words):
            break
    return chunks
