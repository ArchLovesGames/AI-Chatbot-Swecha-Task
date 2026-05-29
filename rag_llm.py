#!/usr/bin/env python3
"""A small local RAG layer for the tiny LLM project."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    source: str
    start_word: int
    text: str
    vector: dict[str, float]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def read_documents(path: Path, extensions: set[str]) -> list[tuple[Path, str]]:
    if path.is_file():
        return [(path, path.read_text(encoding="utf-8"))]

    documents = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            documents.append((file_path, file_path.read_text(encoding="utf-8")))
    return documents


def chunk_document(text: str, chunk_words: int, overlap: int) -> list[tuple[int, str]]:
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


def term_frequency(tokens: list[str]) -> Counter[str]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return Counter({term: count / total for term, count in counts.items()})


def build_vectors(tokenized_chunks: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    doc_freq = Counter()
    for tokens in tokenized_chunks:
        doc_freq.update(set(tokens))

    total_docs = len(tokenized_chunks)
    idf = {
        term: math.log((total_docs + 1) / (frequency + 1)) + 1.0
        for term, frequency in doc_freq.items()
    }

    vectors = []
    for tokens in tokenized_chunks:
        tf = term_frequency(tokens)
        vector = {term: value * idf[term] for term, value in tf.items()}
        vectors.append(normalize(vector))
    return vectors, idf


def normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {term: value / norm for term, value in vector.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = tokenize(query)
    tf = term_frequency(tokens)
    return normalize({term: value * idf.get(term, 1.0) for term, value in tf.items()})


def save_index(
    documents_path: Path,
    output_path: Path,
    chunk_words: int,
    overlap: int,
    extensions: set[str],
) -> None:
    documents = read_documents(documents_path, extensions)
    if not documents:
        suffixes = ", ".join(sorted(extensions))
        raise ValueError(f"no documents found in {documents_path} with extensions: {suffixes}")

    chunk_records = []
    tokenized_chunks = []
    for file_path, text in documents:
        for start_word, chunk_text in chunk_document(text, chunk_words, overlap):
            tokenized = tokenize(chunk_text)
            if not tokenized:
                continue
            chunk_records.append((file_path, start_word, chunk_text))
            tokenized_chunks.append(tokenized)

    vectors, idf = build_vectors(tokenized_chunks)
    chunks = []
    for index, ((file_path, start_word, chunk_text), vector) in enumerate(zip(chunk_records, vectors)):
        chunks.append(
            Chunk(
                id=f"chunk-{index}",
                source=str(file_path),
                start_word=start_word,
                text=chunk_text,
                vector=vector,
            )
        )

    payload = {
        "version": 1,
        "documents_path": str(documents_path),
        "chunk_words": chunk_words,
        "overlap": overlap,
        "extensions": sorted(extensions),
        "idf": idf,
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"indexed {len(documents)} documents into {len(chunks)} chunks")
    print(f"saved index to {output_path}")


def load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve(index: dict, query: str, top_k: int) -> list[tuple[float, dict]]:
    query_vector = vectorize_query(query, index["idf"])
    scored = []
    for chunk in index["chunks"]:
        score = cosine(query_vector, chunk["vector"])
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def sentence_score(sentence: str, query_terms: set[str]) -> int:
    terms = set(tokenize(sentence))
    return len(terms & query_terms)


def build_answer(query: str, retrieved: list[tuple[float, dict]], max_sentences: int) -> str:
    if not retrieved:
        return "I could not find relevant context in the indexed documents."

    query_terms = set(tokenize(query))
    candidates = []
    for citation_id, (_, chunk) in enumerate(retrieved, start=1):
        for sentence in SENTENCE_RE.split(chunk["text"]):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidates.append((sentence_score(sentence, query_terms), citation_id, sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen_sentences = set()
    for item in candidates:
        score, _, sentence = item
        normalized = re.sub(r"\s+", " ", sentence.lower()).strip()
        if normalized in seen_sentences:
            continue
        if score <= 0 and selected:
            continue
        seen_sentences.add(normalized)
        selected.append(item)
        if len(selected) >= max_sentences:
            break

    lines = ["Answer:"]
    for _, citation_id, sentence in selected:
        lines.append(f"- {sentence} [{citation_id}]")

    lines.append("")
    lines.append("Sources:")
    for citation_id, (score, chunk) in enumerate(retrieved, start=1):
        lines.append(
            f"[{citation_id}] {chunk['source']} "
            f"(chunk={chunk['id']}, score={score:.3f}, start_word={chunk['start_word']})"
        )
    return "\n".join(lines)


def build_prompt(query: str, retrieved: list[tuple[float, dict]]) -> str:
    context = "\n\n".join(
        f"[{index}] Source: {chunk['source']}\n{chunk['text']}"
        for index, (_, chunk) in enumerate(retrieved, start=1)
    )
    return (
        "Use only the context below to answer the question. "
        "If the context is insufficient, say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer with citations like [1], [2]:"
    )


def parse_extensions(raw: str) -> set[str]:
    extensions = set()
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions


def command_index(args: argparse.Namespace) -> None:
    save_index(
        documents_path=Path(args.documents),
        output_path=Path(args.index),
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        extensions=parse_extensions(args.extensions),
    )


def command_retrieve(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index))
    for rank, (score, chunk) in enumerate(retrieve(index, args.query, args.top_k), start=1):
        preview = chunk["text"][: args.preview].replace("\n", " ")
        print(f"{rank}. score={score:.3f} source={chunk['source']} chunk={chunk['id']}")
        print(f"   {preview}")


def command_ask(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index))
    results = retrieve(index, args.query, args.top_k)
    if args.prompt:
        print(build_prompt(args.query, results))
    else:
        print(build_answer(args.query, results, args.max_sentences))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index documents and answer questions with RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="build a retrieval index")
    index.add_argument("--documents", default="docs", help="file or folder to index")
    index.add_argument("--index", default="rag_index.json", help="index JSON path")
    index.add_argument("--chunk-words", type=int, default=140, help="words per chunk")
    index.add_argument("--overlap", type=int, default=30, help="overlapping words between chunks")
    index.add_argument("--extensions", default=".txt,.md", help="comma-separated file extensions")
    index.set_defaults(func=command_index)

    retrieve_cmd = subparsers.add_parser("retrieve", help="show retrieved chunks")
    retrieve_cmd.add_argument("query", help="search query")
    retrieve_cmd.add_argument("--index", default="rag_index.json", help="index JSON path")
    retrieve_cmd.add_argument("--top-k", type=int, default=4, help="chunks to retrieve")
    retrieve_cmd.add_argument("--preview", type=int, default=240, help="preview characters")
    retrieve_cmd.set_defaults(func=command_retrieve)

    ask = subparsers.add_parser("ask", help="answer with retrieved context")
    ask.add_argument("query", help="question to answer")
    ask.add_argument("--index", default="rag_index.json", help="index JSON path")
    ask.add_argument("--top-k", type=int, default=4, help="chunks to retrieve")
    ask.add_argument("--max-sentences", type=int, default=4, help="sentences in the answer")
    ask.add_argument("--prompt", action="store_true", help="print a generation prompt instead of an extractive answer")
    ask.set_defaults(func=command_ask)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
