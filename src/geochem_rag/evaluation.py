"""Fixed-question-set evaluation for dense-only vs hybrid evidence retrieval.

What is measured automatically
- Recall@5 at source level and (when labels exist) at source+PDF-page level.
- Abstention rate on questions that must be refused, and false-answer rate.
- Citation membership rate: share of displayed citations whose chunk id was in
  the evidence actually passed to the model (structural validity, not truth).

What requires a human
- Citation correctness and answer supportability cannot be asserted by this
  script. It emits a review sheet with the citation snippet; once a reviewer
  fills ``eval/human_labels.jsonl`` the corresponding metrics are computed.
  Until then they are reported as PENDING with denominators of 0.

No improvement percentage is produced unless both configurations run on the
same corpus, question set, and model configuration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .answering import EvidenceAnswerer, author_year_citation
from .domain import Answer, AnswerStatus, SearchHit
from .providers import ProviderError
from .retrieval import Retriever
from .store import LocalStore


@dataclass
class QuestionResult:
    question_id: str
    category: str
    question: str
    mode: str
    should_refuse: bool
    status: str
    answer_text: str
    citations: list[dict[str, Any]]
    evidence_ids: list[str]
    retrieved: list[dict[str, Any]]
    recall_at_5_source: bool | None
    recall_at_5_page: bool | None
    refused: bool
    refusal_correct: bool | None
    latency_seconds: float
    reason: str | None = None
    error: dict[str, Any] | None = None
    citation_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "mode": self.mode,
            "should_refuse": self.should_refuse,
            "status": self.status,
            "answer_text": self.answer_text,
            "citations": self.citations,
            "citation_details": self.citation_details,
            "evidence_ids": self.evidence_ids,
            "retrieved": self.retrieved,
            "recall_at_5_source": self.recall_at_5_source,
            "recall_at_5_page": self.recall_at_5_page,
            "refused": self.refused,
            "refusal_correct": self.refusal_correct,
            "latency_seconds": round(self.latency_seconds, 3),
            "reason": self.reason,
            "error": self.error,
        }


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions


def _retrieved_view(hits: Sequence[SearchHit], store: LocalStore) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for hit in hits:
        chunk = store.find_chunk(hit.chunk_id)
        view.append(
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 6),
                "retrievers": hit.retrievers,
                "source_id": hit.source_id,
                "pdf_page": hit.pdf_page,
                "dense_score": hit.dense_score,
                "bm25_score": hit.bm25_score,
                "snippet": chunk.text[:200] if chunk else None,
            }
        )
    return view


def _recall_flags(
    question: Mapping[str, Any], hits: Sequence[SearchHit]
) -> tuple[bool | None, bool | None]:
    expected_sources = set(question.get("expected_source_ids") or [])
    expected_pages = set(question.get("expected_pages") or [])
    if not expected_sources:
        return None, None
    source_hit = any(hit.source_id in expected_sources for hit in hits)
    if not expected_pages:
        return source_hit, None
    page_hit = any(
        hit.source_id in expected_sources and hit.pdf_page in expected_pages
        for hit in hits
    )
    return source_hit, page_hit


def run_question(
    question: Mapping[str, Any],
    mode: str,
    retriever: Retriever,
    answerer: EvidenceAnswerer,
    store: LocalStore,
    *,
    top_k: int = 5,
) -> QuestionResult:
    started = time.time()
    try:
        hits = retriever.retrieve(question["question"], mode=mode, top_k=top_k)
    except ProviderError as exc:
        return QuestionResult(
            question_id=question["id"],
            category=question.get("category", ""),
            question=question["question"],
            mode=mode,
            should_refuse=bool(question.get("should_refuse")),
            status=AnswerStatus.ERROR.value,
            answer_text="检索失败：在线嵌入调用出错，未产生地学结论。",
            citations=[],
            evidence_ids=[],
            retrieved=[],
            recall_at_5_source=None,
            recall_at_5_page=None,
            refused=False,
            refusal_correct=None,
            latency_seconds=time.time() - started,
            reason="retrieval_provider_error",
            error=exc.to_dict(),
        )

    source_recall, page_recall = _recall_flags(question, hits)
    answer: Answer = answerer.answer(question["question"], hits)
    citation_details: list[dict[str, Any]] = []
    for citation in answer.citations:
        evidence_entry = next(
            (item for item in answer.meta.get("evidence", []) if item["chunk_id"] == citation.chunk_id),
            None,
        )
        citation_details.append(
            {
                "chunk_id": citation.chunk_id,
                "display": author_year_citation(citation),
                "snippet": evidence_entry["snippet"] if evidence_entry else None,
            }
        )
    refused = answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    refusal_correct = refused if question.get("should_refuse") else None
    return QuestionResult(
        question_id=question["id"],
        category=question.get("category", ""),
        question=question["question"],
        mode=mode,
        should_refuse=bool(question.get("should_refuse")),
        status=answer.status,
        answer_text=answer.text,
        citations=[c.to_dict() for c in answer.citations],
        citation_details=citation_details,
        evidence_ids=answer.evidence,
        retrieved=_retrieved_view(hits, store),
        recall_at_5_source=source_recall,
        recall_at_5_page=page_recall,
        refused=refused,
        refusal_correct=refusal_correct,
        latency_seconds=time.time() - started,
        reason=answer.meta.get("reason"),
        error=answer.meta.get("error"),
    )


def run_mode(
    questions: Sequence[Mapping[str, Any]],
    mode: str,
    retriever: Retriever,
    answerer: EvidenceAnswerer,
    store: LocalStore,
    *,
    top_k: int = 5,
) -> list[QuestionResult]:
    return [
        run_question(q, mode, retriever, answerer, store, top_k=top_k)
        for q in questions
    ]


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def summarize(results: Sequence[QuestionResult], mode: str) -> dict[str, Any]:
    rows = [r for r in results if r.mode == mode]
    answerable = [r for r in rows if not r.should_refuse]
    refuse = [r for r in rows if r.should_refuse]
    source_cases = [r for r in answerable if r.recall_at_5_source is not None]
    page_cases = [r for r in answerable if r.recall_at_5_page is not None]
    answered = [r for r in rows if r.status == AnswerStatus.ANSWERED.value]

    total_citations = sum(len(r.citations) for r in answered)
    membership_ok = sum(
        1
        for r in answered
        for c in r.citations
        if c["chunk_id"] in set(r.evidence_ids)
    )
    return {
        "mode": mode,
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "refuse_questions": len(refuse),
        "answered": len(answered),
        "errors": sum(1 for r in rows if r.status == AnswerStatus.ERROR.value),
        "recall_at_5_source": {
            "value": _mean(1.0 if r.recall_at_5_source else 0.0 for r in source_cases),
            "numerator": sum(1 for r in source_cases if r.recall_at_5_source),
            "denominator": len(source_cases),
        },
        "recall_at_5_page": {
            "value": _mean(1.0 if r.recall_at_5_page else 0.0 for r in page_cases),
            "numerator": sum(1 for r in page_cases if r.recall_at_5_page),
            "denominator": len(page_cases),
        },
        "abstention_rate": {
            "value": _mean(1.0 if r.refused else 0.0 for r in refuse),
            "numerator": sum(1 for r in refuse if r.refused),
            "denominator": len(refuse),
        },
        "false_answer_rate": {
            "value": _mean(0.0 if r.refused else 1.0 for r in refuse),
            "numerator": sum(1 for r in refuse if not r.refused),
            "denominator": len(refuse),
        },
        "answer_coverage": {
            "value": _mean(1.0 if r.status == AnswerStatus.ANSWERED.value else 0.0 for r in answerable),
            "numerator": sum(1 for r in answerable if r.status == AnswerStatus.ANSWERED.value),
            "denominator": len(answerable),
        },
        "citation_membership_rate": {
            "value": (membership_ok / total_citations) if total_citations else None,
            "numerator": membership_ok,
            "denominator": total_citations,
        },
        "mean_latency_seconds": _mean(r.latency_seconds for r in rows),
    }


def summarize_human_labels(
    results: Sequence[QuestionResult], labels_path: str | Path
) -> dict[str, Any]:
    """Compute citation correctness and supportability once labels exist."""
    path = Path(labels_path)
    if not path.exists():
        return {
            "status": "PENDING_HUMAN_REVIEW",
            "citation_accuracy": {"numerator": 0, "denominator": 0, "value": None},
            "answer_supportability": {"numerator": 0, "denominator": 0, "value": None},
            "note": f"No human labels found at {path.as_posix()}; review sheet emitted instead.",
        }
    labels: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            labels[f"{row['mode']}::{row['question_id']}"] = row

    citation_num = citation_den = 0
    support_num = support_den = 0
    for result in results:
        label = labels.get(f"{result.mode}::{result.question_id}")
        if not label:
            continue
        for decision in label.get("citations", []):
            citation_den += 1
            if decision.get("correct"):
                citation_num += 1
        for fact in label.get("answer_facts", []):
            support_den += 1
            if fact.get("supported"):
                support_num += 1
    return {
        "status": "REVIEWED",
        "citation_accuracy": {
            "numerator": citation_num,
            "denominator": citation_den,
            "value": (citation_num / citation_den) if citation_den else None,
        },
        "answer_supportability": {
            "numerator": support_num,
            "denominator": support_den,
            "value": (support_num / support_den) if support_den else None,
        },
    }


def write_results(
    results: Sequence[QuestionResult],
    results_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths: dict[str, str] = {}
    for mode in sorted({r.mode for r in results}):
        rows = [r.to_dict() for r in results if r.mode == mode]
        path = out / f"results_{mode}_{stamp}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[f"results_{mode}"] = path.as_posix()
    summary = {
        "generated_at_utc": stamp,
        "manifest": dict(manifest),
        "modes": {
            mode: summarize(results, mode)
            for mode in sorted({r.mode for r in results})
        },
        "human_review": summarize_human_labels(
            results, Path(results_dir).parent / "human_labels.jsonl"
        ),
    }
    summary_path = out / f"summary_{stamp}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["summary"] = summary_path.as_posix()
    return paths


def write_review_sheet(results: Sequence[QuestionResult], path: str | Path) -> None:
    """Emit a per-question sheet for human citation/support verification."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "mode": result.mode,
                        "question_id": result.question_id,
                        "question": result.question,
                        "status": result.status,
                        "answer_text": result.answer_text,
                        "citations": result.citation_details,
                        "citations_review": [
                            {"chunk_id": c["chunk_id"], "correct": None}
                            for c in result.citation_details
                        ],
                        "answer_facts_review": [{"fact": "", "supported": None}],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )