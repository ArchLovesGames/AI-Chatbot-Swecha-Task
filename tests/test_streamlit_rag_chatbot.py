from __future__ import annotations

import json
from io import BytesIO

from streamlit_rag_chatbot import (
    ask_hosted_llm,
    answer_question,
    build_extractive_summary,
    build_memory_index,
    clean_model_response,
    extract_text,
)


class FakeUpload(BytesIO):
    def __init__(self, name: str, text: str):
        super().__init__(text.encode("utf-8"))
        self.name = name


def test_txt_upload_extraction() -> None:
    upload = FakeUpload("policy.txt", "Alpha policy requires encrypted backups.")

    assert extract_text(upload) == "Alpha policy requires encrypted backups."


def test_blank_question_summary_mentions_core_terms() -> None:
    text = (
        "Solar batteries store excess energy from rooftop panels. "
        "The document compares lithium batteries, inverters, and maintenance plans. "
        "Maintenance plans include yearly inspection and firmware updates."
    )

    summary = build_extractive_summary(text, max_sentences=2)

    assert "batteries" in summary.lower()
    assert "maintenance" in summary.lower()


def test_question_retrieves_relevant_answer() -> None:
    text = (
        "The hiring policy says interviews happen every Tuesday. "
        "Expense reports are due on Friday and must include receipts. "
        "The security policy requires multifactor authentication for all staff."
    )
    index = build_memory_index(text, "handbook.txt", chunk_words=20, overlap=4)

    answer, results = answer_question(
        index,
        "What does the security policy require?",
        use_ollama=False,
        model="unused",
        top_k=2,
    )

    assert results
    assert "multifactor authentication" in answer.lower()


def test_hosted_llm_uses_openai_compatible_payload(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Hosted answer"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        captured["headers"] = request.headers
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    answer = ask_hosted_llm(
        "Use context [1].",
        api_key="test-key",
        base_url="https://api.example.com/openai/v1",
        model="test-model",
    )

    assert answer == "Hosted answer"
    assert captured["url"] == "https://api.example.com/openai/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert request_header(captured, "Accept") == "application/json"
    assert request_header(captured, "User-agent") == "Mozilla/5.0 Streamlit-RAG-App/1.0"
    assert captured["body"]["model"] == "test-model"


def request_header(captured: dict, name: str) -> str:
    return captured.get("headers", {}).get(name, "")


def test_clean_model_response_removes_thinking_block() -> None:
    raw = (
        "<think>Okay, let's start by reading through the document carefully. "
        "This private reasoning should not be shown.</think>\n"
        "Summary:\nThe document discusses Swecha branding."
    )

    cleaned = clean_model_response(raw)

    assert "<think>" not in cleaned
    assert "private reasoning" not in cleaned
    assert cleaned.startswith("Summary:")


def test_clean_model_response_removes_unclosed_thinking_block() -> None:
    raw = "<think>Hidden reasoning that never closes"

    assert clean_model_response(raw) == ""
