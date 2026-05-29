from __future__ import annotations

from io import BytesIO

from streamlit_rag_chatbot import (
    answer_question,
    build_extractive_summary,
    build_memory_index,
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
