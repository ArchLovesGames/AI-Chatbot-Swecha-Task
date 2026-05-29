#!/usr/bin/env python3
"""Streamlit document RAG chatbot with an optional Ollama generation backend."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st

from rag_llm import build_answer, build_prompt
from utils.embedder import tokenize
from utils.loader import SUPPORTED_TYPES, extract_text_from_upload
from utils.retriever import build_index, retrieve


DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_HOSTED_MODEL = "llama-3.1-8b-instant"
DEFAULT_HOSTED_BASE_URL = "https://api.groq.com/openai/v1"
SUMMARY_QUERY = "Summarize the document with the main ideas, key facts, and important conclusions."
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
OLLAMA_FALLBACK_MESSAGE = (
    "Local Ollama is not available in this runtime, so I used the built-in "
    "document retrieval fallback instead."
)
HOSTED_FALLBACK_MESSAGE = (
    "The hosted LLM is not configured or could not answer, so I used the built-in "
    "document retrieval fallback instead."
)


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server cannot answer a request."""


class HostedLLMUnavailable(RuntimeError):
    """Raised when a hosted OpenAI-compatible model cannot answer a request."""


def read_setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables or Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value

    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        return default
    return str(secret_value) if secret_value else default


def extract_text(uploaded_file) -> str:
    """Backward-compatible wrapper for tests and older imports."""
    return extract_text_from_upload(uploaded_file)


def build_memory_index(text: str, source_name: str, chunk_words: int = 160, overlap: int = 35) -> dict:
    """Backward-compatible wrapper for tests and older imports."""
    return build_index(text, source_name, chunk_words=chunk_words, overlap=overlap)


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


def ask_hosted_llm(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """Call an OpenAI-compatible hosted chat completion endpoint."""
    if not api_key:
        raise HostedLLMUnavailable(HOSTED_FALLBACK_MESSAGE)

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Use only the provided document context and cite sources like [1]."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 900,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HostedLLMUnavailable(f"{HOSTED_FALLBACK_MESSAGE} HTTP status: {exc.code}.") from exc
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise HostedLLMUnavailable(HOSTED_FALLBACK_MESSAGE) from exc

    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HostedLLMUnavailable(HOSTED_FALLBACK_MESSAGE) from exc


def answer_question(
    index: dict,
    question: str,
    use_ollama: bool,
    model: str,
    top_k: int = 4,
) -> tuple[str, list[tuple[float, dict]]]:
    """Retrieve context and answer either with Ollama or the extractive fallback."""
    results = retrieve(index, question, top_k=top_k)
    if not use_ollama:
        return build_answer(question, results, max_sentences=5), results

    prompt = build_prompt(question, results)
    return ask_ollama(prompt, model=model), results


def answer_question_hosted(
    index: dict,
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    top_k: int = 4,
) -> tuple[str, list[tuple[float, dict]]]:
    """Retrieve context and answer with a hosted OpenAI-compatible model."""
    results = retrieve(index, question, top_k=top_k)
    prompt = build_prompt(question, results)
    return ask_hosted_llm(prompt, api_key=api_key, base_url=base_url, model=model), results


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
        engine = st.radio(
            "Answer engine",
            ["Hosted LLM", "Local Ollama", "Extractive fallback"],
            index=0,
        )
        hosted_base_url = DEFAULT_HOSTED_BASE_URL
        hosted_api_key = ""
        hosted_model = DEFAULT_HOSTED_MODEL
        ollama_model = DEFAULT_OLLAMA_MODEL

        if engine == "Hosted LLM":
            st.caption("Works on Streamlit Cloud with a GROQ_API_KEY secret.")
            hosted_api_key = read_setting("GROQ_API_KEY")
            hosted_base_url = read_setting("LLM_BASE_URL", DEFAULT_HOSTED_BASE_URL)
            hosted_model = st.text_input(
                "Hosted model",
                value=read_setting("LLM_MODEL", DEFAULT_HOSTED_MODEL),
            )
            if hosted_api_key:
                st.success("Hosted API key detected.")
            else:
                st.warning("Add GROQ_API_KEY in Streamlit Secrets.")
        elif engine == "Local Ollama":
            st.caption("Only works when this app runs on the same machine as Ollama.")
            ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)

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
            if engine == "Hosted LLM":
                prompt = (
                    "Summarize this document clearly. Include the main ideas, key facts, "
                    f"and conclusions.\n\nDocument:\n{text[:12000]}"
                )
                try:
                    st.markdown(
                        ask_hosted_llm(
                            prompt,
                            api_key=hosted_api_key,
                            base_url=hosted_base_url,
                            model=hosted_model,
                        )
                    )
                except HostedLLMUnavailable as exc:
                    st.info(str(exc))
                    st.markdown(build_extractive_summary(text))
            elif engine == "Local Ollama":
                prompt = (
                    "Summarize this document clearly. Include the main ideas, key facts, "
                    f"and conclusions.\n\nDocument:\n{text[:12000]}"
                )
                try:
                    st.markdown(ask_ollama(prompt, model=ollama_model))
                except OllamaUnavailable as exc:
                    st.info(str(exc))
                    st.markdown(build_extractive_summary(text))
            else:
                st.markdown(build_extractive_summary(text))
            return

        try:
            if engine == "Hosted LLM":
                answer, results = answer_question_hosted(
                    index,
                    cleaned_question,
                    api_key=hosted_api_key,
                    base_url=hosted_base_url,
                    model=hosted_model,
                    top_k=top_k,
                )
            elif engine == "Local Ollama":
                answer, results = answer_question(index, cleaned_question, True, ollama_model, top_k=top_k)
            else:
                answer, results = answer_question(index, cleaned_question, False, "", top_k=top_k)
        except HostedLLMUnavailable as exc:
            st.info(str(exc))
            answer, results = answer_question(index, cleaned_question, False, "", top_k=top_k)
        except OllamaUnavailable as exc:
            st.info(str(exc))
            answer, results = answer_question(index, cleaned_question, False, "", top_k=top_k)

        st.markdown(answer)
        render_sources(results)

    with st.expander("Extracted text preview"):
        st.text(text[:5000])


if __name__ == "__main__":
    main()
