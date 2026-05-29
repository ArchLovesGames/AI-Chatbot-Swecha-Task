"""Index construction and retrieval helpers for the RAG chatbot."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from utils.embedder import build_tfidf_vectors, cosine, tokenize, vectorize_query
from utils.loader import SUPPORTED_TYPES, chunk_text


@dataclass
class Chunk:
    id: str
    source: str
    start_word: int
    text: str
    vector: dict[str, float]


def build_index(text: str, source_name: str, chunk_words: int = 160, overlap: int = 35) -> dict:
    """Build an in-memory TF-IDF index for a single uploaded document."""
    chunk_records = []
    tokenized_chunks = []
    for start_word, chunk in chunk_text(text, chunk_words=chunk_words, overlap=overlap):
        tokens = tokenize(chunk)
        if not tokens:
            continue
        chunk_records.append((start_word, chunk))
        tokenized_chunks.append(tokens)

    if not chunk_records:
        raise ValueError("The document did not contain enough readable text to index.")

    vectors, idf = build_tfidf_vectors(tokenized_chunks)
    chunks = []
    for index, ((start_word, chunk), vector) in enumerate(zip(chunk_records, vectors)):
        chunks.append(
            Chunk(
                id=f"upload-chunk-{index}",
                source=source_name,
                start_word=start_word,
                text=chunk,
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


def retrieve(index: dict, query: str, top_k: int = 4) -> list[tuple[float, dict]]:
    """Return the highest-scoring chunks for a user query."""
    query_vector = vectorize_query(query, index["idf"])
    scored = []
    for chunk in index["chunks"]:
        score = cosine(query_vector, chunk["vector"])
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]
