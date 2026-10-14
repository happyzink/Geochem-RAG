"""Offline tests for the evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from geochem_rag.answering import EvidenceAnswerer
from geochem_rag.domain import AnswerStatus, EvidenceChunk, SearchHit, Source
from geochem_rag.evaluation import (
    load_questions,
    run_mode,
    run_question,
    summarize,
    summarize_human_labels,
    write_results,
    write_review_sheet,
)
from geochem_rag.providers import KIND_NETWORK, ChatResult, ProviderError
from geochem_rag.store import LocalStore

SHA = "12" * 32


@dataclass
class StubChat:
    reply: str
    calls: list = field(default_factory=list)

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatResult(text=self.reply, model="stub", usage={"total_tokens": 1})


class StubRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, question, mode="hybrid", top_k=5):
        return self._hits


class FailingRetriever:
    def retrieve(self, question, mode="hybrid", top_k=5):
        raise ProviderError(KIND_NETWORK, "network down")


def make_store(tmp_path) -> LocalStore:
    store = LocalStore(tmp_path)
    source = Source(
        source_id="src_test",
        title="Test Paper",
        authors=["A. Author"],
        year=2022,
        doi_or_url="https://doi.org/10.1000/test",
        license="CC BY 4.0",
        license_url="https://example.org/license",
        sha256=SHA,
        visibility="public",
        download_date="2026-09-28",
    )
    chunk = EvidenceChunk(
        chunk_id="chk_p0003_c000",
        source_id="src_test",
        pdf_page=3,
        text="Adakite forms by partial melting of thickened lower crust.",
        section="discussion",
        quality="ok",
        chunk_index=0,
        sha256=SHA,
    )
    store.import_document(source, [chunk])
    return store


def hit() -> SearchHit:
    return SearchHit(
        chunk_id="chk_p0003_c000",
        score=0.9,
        rank=1,
        retrievers=["dense", "bm25"],
        source_id="src_test",
        pdf_page=3,
        dense_score=0.9,
        bm25_score=4.0,
    )


def test_run_question_records_recall_and_citations(tmp_path):
    store = make_store(tmp_path)
    chat = StubChat('{"status": "answered", "answer": "加厚下地壳部分熔融[[chk_p0003_c000]]。", "citations": ["chk_p0003_c000"]}')
    answerer = EvidenceAnswerer(store, chat)
    question = {
        "id": "qX",
        "category": "terminology",
        "question": "如何解释埃达克质岩石？",
        "expected_source_ids": ["src_test"],
        "expected_pages": [3],
        "should_refuse": False,
    }
    result = run_question(question, "hybrid", StubRetriever([hit()]), answerer, store, top_k=5)
    assert result.status == AnswerStatus.ANSWERED.value
    assert result.recall_at_5_source is True
    assert result.recall_at_5_page is True
    assert result.citation_details[0]["display"] == "(Author, 2022)"
    assert result.retrieved[0]["retrievers"] == ["dense", "bm25"]


def test_run_question_refusal_correct(tmp_path):
    store = make_store(tmp_path)
    chat = StubChat('{"status": "insufficient_evidence", "answer": "证据不足。", "citations": []}')
    answerer = EvidenceAnswerer(store, chat)
    question = {
        "id": "qY",
        "category": "unanswerable",
        "question": "语料没有的问题？",
        "expected_source_ids": [],
        "expected_pages": [],
        "should_refuse": True,
    }
    result = run_question(question, "dense_only", StubRetriever([hit()]), answerer, store)
    assert result.refused is True
    assert result.refusal_correct is True


def test_run_question_retrieval_error(tmp_path):
    store = make_store(tmp_path)
    answerer = EvidenceAnswerer(store, StubChat("{}"))
    question = {
        "id": "qZ",
        "category": "terminology",
        "question": "q",
        "expected_source_ids": [],
        "expected_pages": [],
        "should_refuse": False,
    }
    result = run_question(question, "dense_only", FailingRetriever(), answerer, store)
    assert result.status == AnswerStatus.ERROR.value
    assert result.reason == "retrieval_provider_error"
    assert result.error["kind"] == KIND_NETWORK


def test_summarize_and_write_results(tmp_path):
    store = make_store(tmp_path)
    answered_chat = StubChat('{"status": "answered", "answer": "a[[chk_p0003_c000]]", "citations": ["chk_p0003_c000"]}')
    refuse_chat = StubChat('{"status": "insufficient_evidence", "answer": "no", "citations": []}')
    questions = [
        {"id": "q1", "category": "terminology", "question": "q1", "expected_source_ids": ["src_test"], "expected_pages": [3], "should_refuse": False},
        {"id": "q2", "category": "unanswerable", "question": "q2", "expected_source_ids": [], "expected_pages": [], "should_refuse": True},
    ]
    results = run_mode(questions[:1], "hybrid", StubRetriever([hit()]), EvidenceAnswerer(store, answered_chat), store)
    results += run_mode(questions[1:], "hybrid", StubRetriever([hit()]), EvidenceAnswerer(store, refuse_chat), store)

    summary = summarize(results, "hybrid")
    assert summary["recall_at_5_page"]["value"] == 1.0
    assert summary["abstention_rate"]["value"] == 1.0
    assert summary["citation_membership_rate"]["value"] == 1.0

    paths = write_results(results, tmp_path / "results", manifest={"question_count": 2})
    assert "summary" in paths
    summary_payload = json.loads((tmp_path / "results" / paths["summary"].split("/")[-1]).read_text(encoding="utf-8"))
    assert summary_payload["human_review"]["status"] == "PENDING_HUMAN_REVIEW"

    review_path = tmp_path / "review_sheet.jsonl"
    write_review_sheet(results, review_path)
    rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_human_labels_are_used_when_present(tmp_path):
    store = make_store(tmp_path)
    chat = StubChat('{"status": "answered", "answer": "a[[chk_p0003_c000]]", "citations": ["chk_p0003_c000"]}')
    question = {"id": "q1", "category": "t", "question": "q1", "expected_source_ids": ["src_test"], "expected_pages": [3], "should_refuse": False}
    results = run_mode([question], "hybrid", StubRetriever([hit()]), EvidenceAnswerer(store, chat), store)
    labels = tmp_path / "human_labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "question_id": "q1",
                "citations": [{"chunk_id": "chk_p0003_c000", "correct": True}],
                "answer_facts": [{"fact": "加厚下地壳", "supported": True}, {"fact": "俯冲洋壳", "supported": False}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = summarize_human_labels(results, labels)
    assert summary["status"] == "REVIEWED"
    assert summary["citation_accuracy"]["value"] == 1.0
    assert summary["answer_supportability"] == {"numerator": 1, "denominator": 2, "value": 0.5}


def test_load_questions_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "q1"}\n\n{"id": "q2"}\n', encoding="utf-8")
    assert [q["id"] for q in load_questions(path)] == ["q1", "q2"]