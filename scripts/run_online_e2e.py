"""One online run: SiliconFlow embeddings -> DeepSeek chat -> validated citations.

The output record is anonymised: it contains no API key and only short evidence
snippets from the configured local store. Run from the repo root, e.g.

    set -a && . ./.env && set +a
    .venv/Scripts/python.exe scripts/run_online_e2e.py --question-id q11
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geochem_rag.answering import EvidenceAnswerer, author_year_citation  # noqa: E402
from geochem_rag.evaluation import load_questions  # noqa: E402
from geochem_rag.providers import (  # noqa: E402
    chat_provider_from_env,
    embedding_provider_from_env,
)
from geochem_rag.retrieval import Retriever  # noqa: E402
from geochem_rag.store import LocalStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", default="q11")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--store", default="data/processed")
    parser.add_argument("--index-cache", default="data/index/embeddings.json")
    parser.add_argument("--mode", default="hybrid", choices=["dense_only", "bm25_only", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args()

    embedder = embedding_provider_from_env()
    chat_provider = chat_provider_from_env()
    if not embedder.config.api_key:
        print("SILICONFLOW_API_KEY is not set", file=sys.stderr)
        return 2
    if not chat_provider.config.api_key:
        print(f"{chat_provider.config.credential_name} is not set", file=sys.stderr)
        return 2
    store = LocalStore(args.store)
    if not store.count_chunks():
        print("No chunks in store; run the import check first", file=sys.stderr)
        return 2

    question = next(
        (q for q in load_questions(args.questions) if q["id"] == args.question_id), None
    )
    if question is None:
        print(f"question {args.question_id} not found", file=sys.stderr)
        return 2

    retriever = Retriever.from_store(
        store, embedder, cache_path=args.index_cache
    )
    trace = retriever.retrieve_with_trace(
        question["question"], mode=args.mode, top_k=args.top_k
    )
    answerer = EvidenceAnswerer(store, chat_provider, top_k=args.top_k)
    answer = answerer.answer(question["question"], trace.hits)

    record = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "question_id": question["id"],
        "question": question["question"],
        "mode": args.mode,
        "endpoints": {
            "chat": chat_provider.config.base_url,
            "embedding": embedder.config.base_url,
        },
        "models": {
            "chat": chat_provider.config.chat_model,
            "embedding": embedder.config.embedding_model,
        },
        "api_key_present": True,
        "api_key_logged": False,
        "expanded_terms": list(trace.expanded_terms),
        "rrf_k": trace.rrf_k,
        "retrieved": [
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "source_id": hit.source_id,
                "pdf_page": hit.pdf_page,
                "score": round(hit.score, 6),
                "retrievers": hit.retrievers,
            }
            for hit in trace.hits
        ],
        "status": answer.status,
        "answer_text": answer.text,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "display": author_year_citation(c),
                "label": c.label,
                "title": c.title,
                "authors": list(c.authors),
                "year": c.year,
                "pdf_page": c.pdf_page,
            }
            for c in answer.citations
        ],
        "evidence_ids": answer.evidence,
        "rejected_citation_ids": answer.meta.get("rejected_citation_ids", []),
        "usage": answer.meta.get("usage"),
        "error": answer.meta.get("error"),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"online_e2e_{record['generated_at_utc']}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nWROTE {out_path.as_posix()}")
    return 0 if answer.status == "answered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
