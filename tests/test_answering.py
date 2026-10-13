"""Offline tests for evidence-constrained answering and citation validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from geochem_rag.answering import EvidenceAnswerer, build_evidence, format_citation
from geochem_rag.domain import AnswerStatus, EvidenceChunk, SearchHit, Source
from geochem_rag.providers import KIND_TIMEOUT, ChatResult, ProviderError
from geochem_rag.store import LocalStore

SHA = "cd" * 32


def make_source() -> Source:
    return Source(
        source_id="src_test",
        title="Test Adakite Paper",
        authors=["A. Author"],
        year=2022,
        doi_or_url="https://doi.org/10.1000/test",
        license="CC BY 4.0",
        license_url="https://example.org/license",
        sha256=SHA,
        visibility="public",
        download_date="2026-09-28",
    )


def make_chunk(chunk_id: str, page: int, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_id="src_test",
        pdf_page=page,
        text=text,
        section="discussion",
        quality="ok",
        chunk_index=0,
        sha256=SHA,
    )


@pytest.fixture()
def store(tmp_path):
    local = LocalStore(tmp_path)
    chunks = [
        make_chunk("chk_p0003_c000", 3, "Adakitic rocks form by partial melting of thickened lower crust."),
        make_chunk("chk_p0007_c001", 7, "Zircon U-Pb ages constrain the magmatic tempo."),
    ]
    local.import_document(make_source(), chunks)
    return local


def make_hit(chunk_id: str, rank: int = 1, page: int = 3, retrievers=("dense", "bm25")) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=1.0 / rank,
        rank=rank,
        retrievers=list(retrievers),
        source_id="src_test",
        pdf_page=page,
    )


@dataclass
class FakeChat:
    reply: str = ""
    error: ProviderError | None = None
    calls: list = field(default_factory=list)

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.reply, model="fake-chat", usage={"total_tokens": 7})


def test_no_hits_returns_insufficient(store):
    answer = EvidenceAnswerer(store, FakeChat()).answer("q", [])
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.citations == []
    assert answer.evidence == []
    assert answer.meta["reason"] == "no_evidence"


def test_missing_chunk_is_skipped(store):
    answer = EvidenceAnswerer(store, FakeChat()).answer("q", [make_hit("chk_does_not_exist")])
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.meta["reason"] == "no_evidence"


def test_answered_maps_citation_to_title_and_page(store):
    chat = FakeChat(reply='{"status": "answered", "answer": "加厚下地壳部分熔融。", "citations": ["chk_p0003_c000"]}')
    answer = EvidenceAnswerer(store, chat).answer("如何解释？", [make_hit("chk_p0003_c000", page=3)])
    assert answer.status == AnswerStatus.ANSWERED.value
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.title == "Test Adakite Paper"
    assert citation.pdf_page == 3
    assert format_citation(citation) == "Test Adakite Paper，PDF 第 3 页"
    assert answer.meta["evidence"][0]["chunk_id"] == "chk_p0003_c000"
    assert answer.meta["evidence"][0]["snippet"]


def test_prompt_contains_evidence_id_and_untrusted_rule(store):
    chat = FakeChat(reply='{"status": "insufficient_evidence", "answer": "", "citations": []}')
    EvidenceAnswerer(store, chat).answer("问题", [make_hit("chk_p0003_c000")])
    system, user = chat.calls[0]
    assert "untrusted" in system["content"]
    assert "[chk_p0003_c000]" in user["content"]
    assert "问题" in user["content"]


def test_fabricated_citation_is_rejected(store):
    chat = FakeChat(reply='{"status": "answered", "answer": "捏造。", "citations": ["chk_fabricated"]}')
    answer = EvidenceAnswerer(store, chat).answer("q", [make_hit("chk_p0003_c000")])
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.citations == []
    assert "chk_fabricated" in answer.meta["rejected_citation_ids"]
    assert "捏造" not in answer.text


def test_answered_without_citation_downgrades(store):
    chat = FakeChat(reply='{"status": "answered", "answer": "没有引用。", "citations": []}')
    answer = EvidenceAnswerer(store, chat).answer("q", [make_hit("chk_p0003_c000")])
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.citations == []
    assert "没有引用" not in answer.text


def test_refusal_does_not_expose_uncited_model_claims(store):
    chat = FakeChat(reply='{"status": "insufficient_evidence", "answer": "未经证实的具体地学结论。", "citations": []}')
    answer = EvidenceAnswerer(store, chat).answer("q", [make_hit("chk_p0003_c000")])
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert "未经证实" not in answer.text


def test_model_refusal_is_insufficient(store):
    chat = FakeChat(reply='{"status": "insufficient_evidence", "answer": "证据不足。", "citations": []}')
    answer = EvidenceAnswerer(store, chat).answer("q", [make_hit("chk_p0003_c000")])
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.text.startswith("证据不足：")


def test_provider_error_becomes_error_status(store):
    chat = FakeChat(error=ProviderError(KIND_TIMEOUT, "request timed out", retryable=True))
    answer = EvidenceAnswerer(store, chat).answer("q", [make_hit("chk_p0003_c000")])
    assert answer.status == AnswerStatus.ERROR.value
    assert "生成失败" in answer.text
    assert answer.citations == []
    assert answer.meta["error"]["kind"] == KIND_TIMEOUT


def test_unparsable_output_becomes_error_status(store):
    chat = FakeChat(reply="I think the answer is adakite.")
    answer = EvidenceAnswerer(store, chat).answer("q", [make_hit("chk_p0003_c000")])
    assert answer.status == AnswerStatus.ERROR.value
    assert answer.meta["reason"] == "unparsable_output"
    assert answer.citations == []


def test_build_evidence_respects_visibility_filter(store):
    private_source = Source(
        source_id="src_priv",
        title="Private Paper",
        authors=["P. Author"],
        year=2021,
        doi_or_url="https://doi.org/10.1000/private",
        license="unknown",
        license_url="n/a",
        sha256="ef" * 32,
        visibility="private",
        download_date="2026-09-28",
    )
    private_chunk = EvidenceChunk(
        chunk_id="chk_priv_c000",
        source_id="src_priv",
        pdf_page=1,
        text="private text",
        section=None,
        quality="ok",
        chunk_index=0,
        sha256="ef" * 32,
    )
    store.import_document(private_source, [private_chunk])
    items = build_evidence(
        [make_hit("chk_priv_c000"), make_hit("chk_p0003_c000")],
        store,
        visibility="public",
    )
    assert [item.chunk_id for item in items] == ["chk_p0003_c000"]

def test_no_evidence_extended_gives_labelled_background(store):
    chat = FakeChat(reply='{"answer": "Generally, adakites are intermediate-felsic rocks."}')
    answer = EvidenceAnswerer(store, chat).answer("什么是埃达克岩？", [], allow_extended=True)
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.citations == []
    assert answer.evidence == []
    assert answer.meta["reason"] == "no_evidence_extended"
    assert answer.meta["extended_explanation"] is not None
    assert "未经本地文献核验" in answer.text


def test_no_evidence_extended_off_does_not_call_model(store):
    chat = FakeChat(reply="{}")
    answer = EvidenceAnswerer(store, chat).answer("什么是埃达克岩？", [], allow_extended=False)
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert answer.meta["reason"] == "no_evidence"
    assert chat.calls == []


def test_no_evidence_extended_provider_error_is_error(store):
    chat = FakeChat(error=ProviderError(KIND_TIMEOUT, "timeout", retryable=True))
    answer = EvidenceAnswerer(store, chat).answer("q", [], allow_extended=True)
    assert answer.status == AnswerStatus.ERROR.value
    assert answer.citations == []
    assert answer.meta["error"]["kind"] == KIND_TIMEOUT
