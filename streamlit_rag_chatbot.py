#!/usr/bin/env python3
"""Streamlit document RAG chatbot with an optional Ollama generation backend."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import streamlit as st

from rag_llm import (
    Chunk,
    build_answer,
    build_prompt,
    build_vectors,
    chunk_document,
    retrieve,
    tokenize,
)


SUPPORTED_TYPES = ["pdf", "txt", "md"]
DEFAULT_MODEL = "llama3.2"
SUMMARY_QUERY = "Summarize the document with the main ideas, key facts, and important conclusions."
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
OLLAMA_FALLBACK_MESSAGE = (
    "Local Ollama is not available in this runtime, so I used the built-in "
    "document retrieval fallback instead."
)


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server cannot answer a request."""


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """Extract text from a PDF upload using whichever PDF reader is installed."""
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


def extract_text(uploaded_file) -> str:
    """Return usable text from a Streamlit upload."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)

    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    if suffix in {".txt", ".md"}:
        return uploaded_file.read().decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix}")


def build_memory_index(text: str, source_name: str, chunk_words: int = 160, overlap: int = 35) -> dict:
    """Build an in-memory TF-IDF chunk index for one uploaded document."""
    chunk_records = []
    tokenized_chunks = []
    for start_word, chunk_text in chunk_document(text, chunk_words=chunk_words, overlap=overlap):
        tokens = tokenize(chunk_text)
        if not tokens:
            continue
        chunk_records.append((start_word, chunk_text))
        tokenized_chunks.append(tokens)

    if not chunk_records:
        raise ValueError("The document did not contain enough readable text to index.")

    vectors, idf = build_vectors(tokenized_chunks)
    chunks = []
    for index, ((start_word, chunk_text), vector) in enumerate(zip(chunk_records, vectors)):
        chunks.append(
            Chunk(
                id=f"upload-chunk-{index}",
                source=source_name,
                start_word=start_word,
                text=chunk_text,
                vector=vector,
            )
        )

    return {
        "version": 1,
        "documents_path": source_name,
        "chunk_words": chunk_words,
        "overlap": overlap,
        "extensions": SUPPORTED_TYPES,
        "idf": idf,
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def build_extractive_summary(text: str, max_sentences: int = 5) -> str:
    """Create a deterministic summary when no user question is provided."""
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text.replace("\n", " ")) if sentence.strip()]
    if not sentences:
        return "I could not find readable text to summarize."

    terms = [token for token in tokenize(text) if len(token) > 3]
    if not terms:
        return " ".join(sentences[:max_sentences])

    frequencies = {}
    for term in terms:
        frequencies[term] = frequencies.get(term, 0) + 1

    ranked = []
    for position, sentence in enumerate(sentences):
        score = sum(frequencies.get(token, 0) for token in tokenize(sentence))
        score = score / max(len(tokenize(sentence)), 1)
        ranked.append((score, -position, sentence))

    selected = sorted(ranked, reverse=True)[:max_sentences]
    selected_positions = sorted((-position, sentence) for _, position, sentence in selected)
    return "\n".join(f"- {sentence}" for _, sentence in selected_positions)


def ask_ollama(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    """Call the local Ollama generate API."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OllamaUnavailable(OLLAMA_FALLBACK_MESSAGE) from exc

    return str(data.get("response", "")).strip()


def answer_question(index: dict, question: str, use_ollama: bool, model: str, top_k: int = 4) -> tuple[str, list[tuple[float, dict]]]:
    """Retrieve context and answer either with Ollama or the extractive fallback."""
    results = retrieve(index, question, top_k=top_k)
    if not use_ollama:
        return build_answer(question, results, max_sentences=5), results

    prompt = build_prompt(question, results)
    return ask_ollama(prompt, model=model), results


def render_sources(results: list[tuple[float, dict]]) -> None:
    if not results:
        return

    with st.expander("Retrieved document context"):
        for rank, (score, chunk) in enumerate(results, start=1):
            st.markdown(f"**[{rank}] {chunk['source']}** | score `{score:.3f}` | word `{chunk['start_word']}`")
            st.write(chunk["text"])


def main() -> None:
    st.set_page_config(page_title="Document RAG Chatbot", page_icon=":mag:", layout="wide")
    st.title("Document RAG Chatbot")

    with st.sidebar:
        st.header("Model")
        use_ollama = st.toggle("Use local Ollama", value=False)
        st.caption("Ollama only works when this app runs on the same machine as the Ollama server.")
        model = st.text_input("Ollama model", value=DEFAULT_MODEL)
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

    uploaded_file = st.file_uploader("Upload a document", type=SUPPORTED_TYPES)
    question = st.text_area(
        "Question",
        placeholder="Ask about the document. Leave blank to generate a summary.",
        height=110,
    )

    if not uploaded_file:
        st.info("Upload a PDF, TXT, or Markdown document to begin.")
        return

    try:
        text = extract_text(uploaded_file)
        index = build_memory_index(text, uploaded_file.name)
    except Exception as exc:
        st.error(str(exc))
        return

    st.caption(f"Extracted {len(text.split()):,} words from `{uploaded_file.name}`.")

    if st.button("Run", type="primary"):
        cleaned_question = question.strip()
        if not cleaned_question:
            if use_ollama:
                prompt = (
                    "Summarize this document clearly. Include the main ideas, key facts, "
                    f"and conclusions.\n\nDocument:\n{text[:12000]}"
                )
                try:
                    st.markdown(ask_ollama(prompt, model=model))
                except OllamaUnavailable as exc:
                    st.info(str(exc))
                    st.markdown(build_extractive_summary(text))
            else:
                st.markdown(build_extractive_summary(text))
            return

        try:
            answer, results = answer_question(index, cleaned_question, use_ollama, model, top_k=top_k)
        except OllamaUnavailable as exc:
            st.info(str(exc))
            answer, results = answer_question(index, cleaned_question, False, model, top_k=top_k)

        st.markdown(answer)
        render_sources(results)

    with st.expander("Extracted text preview"):
        st.text(text[:5000])


if __name__ == "__main__":
    main()
