"""Run the three fixed demo questions and report statuses.

This script exercises the full pipeline (retrieval + answering) for the three
demo questions documented in docs/DEMO.md. It works in both online and offline
modes; offline mode is used automatically when API keys are missing.

Usage (from repo root):

    # Windows PowerShell
    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py

    # Linux / git-bash
    PYTHONPATH=src .venv/bin/python scripts/run_demo.py

Exit code 0 when all three questions return the expected status pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geochem_rag.answering import EvidenceAnswerer, author_year_citation  # noqa: E402
from geochem_rag.domain import AnswerStatus  # noqa: E402
from geochem_rag.offline import HashingEmbedder, TemplateChatProvider  # noqa: E402
from geochem_rag.providers import (  # noqa: E402
    ProviderError,
    chat_provider_from_env,
    embedding_provider_from_env,
)
from geochem_rag.retrieval import Retriever  # noqa: E402
from geochem_rag.store import LocalStore  # noqa: E402

DEMO_QUESTIONS = [
    {
        "label": "Cross-language (Chinese question about English paper)",
        "question": "这篇英文论文如何解释莫根通埃达克质岩体的成因？请用中文概括。",
        "expected_status": AnswerStatus.ANSWERED.value,
    },
    {
        "label": "Geochemistry abbreviation (MORB)",
        "question": "MORB 是什么缩写？它在岩石成因讨论中通常指示什么源区？",
        # MORB may or may not be in the corpus; both answered and insufficient
        # are acceptable for the demo.
        "expected_status": None,
    },
    {
        "label": "Unanswerable (outside the corpus)",
        "question": "Summarize the conclusions of a paper about kimberlites in Siberia.",
        "expected_status": AnswerStatus.INSUFFICIENT_EVIDENCE.value,
    },
]


def _init_providers():
    """Try online providers; fall back to offline stand-ins."""
    try:
        embedder = embedding_provider_from_env()
        if not embedder.config.api_key:  # type: ignore[attr-defined]
            raise ProviderError("not_configured", "no key")
        chat = chat_provider_from_env()
        if not chat.config.api_key:  # type: ignore[attr-defined]
            raise ProviderError("not_configured", "no key")
        mode = "online"
    except (ProviderError, ValueError, KeyError):
        embedder = HashingEmbedder()
        chat = TemplateChatProvider()
        mode = "offline"
    return embedder, chat, mode


def main() -> int:
    store = LocalStore("data/processed")
    chunk_count = store.count_chunks()
    if chunk_count == 0:
        print("ERROR: No evidence chunks in store. Import PDFs first.", file=sys.stderr)
        return 2

    embedder, chat, mode = _init_providers()
    print(f"=== GeoChem-RAG Demo ({mode} mode, {chunk_count} chunks) ===\n")

    cache_path = (
        "data/index/embeddings.json"
        if mode == "online"
        else "data/index/embeddings_offline.json"
    )
    retriever = Retriever.from_store(store, embedder, cache_path=cache_path)  # type: ignore[arg-type]
    answerer = EvidenceAnswerer(store, chat, top_k=5)  # type: ignore[arg-type]

    all_ok = True
    for idx, demo in enumerate(DEMO_QUESTIONS, start=1):
        print(f"--- Q{idx}: {demo['label']} ---")
        print(f"Question: {demo['question']}")

        trace = retriever.retrieve_with_trace(demo["question"], mode="hybrid", top_k=5)
        answer = answerer.answer(demo["question"], trace.hits)

        print(f"Status: {answer.status}")
        if answer.citations:
            for cit in answer.citations:
                print(f"  Citation: {author_year_citation(cit)}")
        print(f"Answer text: {answer.text[:200]}{'...' if len(answer.text) > 200 else ''}")

        if trace.expanded_terms:
            print(f"Query expansion: +{', '.join(trace.expanded_terms)}")

        expected = demo["expected_status"]
        if expected is not None and answer.status != expected:
            print(f"  WARNING: expected status '{expected}', got '{answer.status}'")
            all_ok = False

        print()

    if all_ok:
        print("=== Demo completed. All status checks passed. ===")
        return 0
    else:
        print("=== Demo completed with warnings (see above). ===")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
