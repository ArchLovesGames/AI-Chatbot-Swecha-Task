"""Lightweight TF-IDF embedding helpers for local document retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter


WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> list[str]:
    """Normalize text into simple lowercase word tokens."""
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def term_frequency(tokens: list[str]) -> Counter[str]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return Counter({term: count / total for term, count in counts.items()})


def normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {term: value / norm for term, value in vector.items()}


def build_tfidf_vectors(tokenized_chunks: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Build normalized TF-IDF vectors plus the shared IDF table."""
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
        vectors.append(normalize({term: value * idf[term] for term, value in tf.items()}))
    return vectors, idf


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = tokenize(query)
    tf = term_frequency(tokens)
    return normalize({term: value * idf.get(term, 1.0) for term, value in tf.items()})


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())
