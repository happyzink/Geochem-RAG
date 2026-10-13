"""Offline tests for dense / BM25 / RRF retrieval (no network)."""

from __future__ import annotations

import pytest

from geochem_rag.domain import EvidenceChunk
from geochem_rag.retrieval import (
    BM25Index,
    DenseIndex,
    EmbeddingCache,
    Retriever,
    _rrf_fuse,
    expand_query,
    tokenize,
)

SHA = "ab" * 32
VOCAB = ["adakite", "subduction", "zircon", "isotope"]


def make_chunk(chunk_id: str, text: str, page: int = 1, source_id: str = "src_a") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        pdf_page=page,
        text=text,
        section="introduction",
        quality="ok",
        chunk_index=0,
        sha256=SHA,
    )


class VocabEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(word)) for word in VOCAB]
            vectors.append(vector if any(vector) else [0.001] * len(VOCAB))
        return vectors


def test_tokenize_keeps_latin_words_and_cjk_bigrams():
    tokens = tokenize("The adakitic 埃达克质岩石 (Sr/Y) rocks")
    assert "adakitic" in tokens
    assert "the" not in tokens
    assert "埃达" in tokens
    assert "达克" in tokens
    assert "岩石" in tokens


def test_expand_query_preserves_original_and_lists_added_terms():
    expanded = expand_query("什么是埃达克质岩石的锆石定年？")
    assert expanded.original == "什么是埃达克质岩石的锆石定年？"
    assert expanded.expanded.startswith(expanded.original)
    assert "adakitic" in expanded.added_terms
    assert "zircon" in expanded.added_terms
    assert expanded.expanded != expanded.original

    unchanged = expand_query("plain english question")
    assert unchanged.expanded == unchanged.original
    assert unchanged.added_terms == ()


def test_bm25_ranks_matching_document_first():
    index = BM25Index(
        ["c1", "c2", "c3"],
        ["zircon u-pb dating of adakite", "subduction thermal models", "seismic tomography"],
    )
    ranked = index.search("zircon dating", top_k=3)
    assert ranked[0][0] == "c1"
    assert all(score > 0 for _, score in ranked)


def test_dense_index_cosine_ranking():
    index = DenseIndex(["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]])
    ranked = index.search([1.0, 0.0], top_k=2)
    assert ranked[0] == ("c1", pytest.approx(1.0))


def test_dense_index_rejects_dim_mismatch():
    index = DenseIndex(["c1"], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dim"):
        index.search([1.0, 0.0, 0.0], top_k=1)


def test_rrf_fuse_combines_rankings():
    fused = _rrf_fuse([("c1", 0.9), ("c2", 0.8)], [("c2", 5.0), ("c3", 3.0)], k=60)
    by_id = {cid: (score, paths) for cid, score, paths in fused}
    assert by_id["c2"][1] == ["dense", "bm25"]
    assert by_id["c1"][1] == ["dense"]
    assert by_id["c2"][0] > by_id["c1"][0]


def test_retriever_dense_only_and_hybrid(tmp_path):
    chunks = [
        make_chunk("chk_a", "adakite petrogenesis in the gangdese arc", page=3),
        make_chunk("chk_b", "subduction zone thermal structure", page=5),
        make_chunk("chk_c", "zircon u-pb isotope geochronology", page=7),
    ]
    embedder = VocabEmbedder()
    retriever = Retriever(
        chunks,
        embedder=embedder,
        cache=EmbeddingCache(tmp_path / "emb.json", "VocabEmbedder"),
    )

    dense = retriever.retrieve("adakite", mode="dense_only", top_k=2)
    assert dense[0].chunk_id == "chk_a"
    assert dense[0].retrievers == ["dense"]
    assert dense[0].source_id == "src_a" and dense[0].pdf_page == 3
    assert dense[0].rank == 1

    hybrid_trace = retriever.retrieve_with_trace("adakite", mode="hybrid", top_k=3)
    assert hybrid_trace.mode == "hybrid"
    assert hybrid_trace.dense_ranking and hybrid_trace.bm25_ranking
    assert hybrid_trace.rrf_k == 60
    assert "dense" in hybrid_trace.hits[0].retrievers


def test_retriever_unknown_mode_raises(tmp_path):
    retriever = Retriever(
        [make_chunk("chk_a", "adakite")],
        embedder=VocabEmbedder(),
        cache=EmbeddingCache(tmp_path / "emb.json", "VocabEmbedder"),
    )
    with pytest.raises(ValueError, match="mode"):
        retriever.retrieve("adakite", mode="nope")


def test_embedding_cache_reuses_vectors(tmp_path):
    cache_path = tmp_path / "emb.json"
    chunks = [make_chunk("chk_a", "adakite")]
    first = VocabEmbedder()
    Retriever(chunks, embedder=first, cache=EmbeddingCache(cache_path, "VocabEmbedder"))
    assert first.calls == 1

    second = VocabEmbedder()
    Retriever(chunks, embedder=second, cache=EmbeddingCache(cache_path, "VocabEmbedder"))
    assert second.calls == 0

def test_from_store_filters_non_ok_chunks(tmp_path):
    from geochem_rag.domain import Source
    from geochem_rag.store import LocalStore

    store = LocalStore(tmp_path / "s")
    source = Source(
        source_id="src_a",
        title="T",
        authors=["A"],
        year=2022,
        doi_or_url="https://e.org/x",
        license="CC BY 4.0",
        license_url="https://e.org/l",
        sha256=SHA,
        visibility="public",
        download_date="2026-09-28",
    )
    ok = make_chunk("chk_ok", "adakite petrogenesis in the gangdese arc of southern tibet", page=1)
    bad = EvidenceChunk(
        chunk_id="chk_bad",
        source_id="src_a",
        pdf_page=2,
        text="garbled � garbage",
        section=None,
        quality="garbled",
        chunk_index=1,
        sha256=SHA,
    )
    short = EvidenceChunk(
        chunk_id="chk_short",
        source_id="src_a",
        pdf_page=3,
        text="x",
        section=None,
        quality="ok",
        chunk_index=2,
        sha256=SHA,
    )
    store.import_document(source, [ok, bad, short])
    retriever = Retriever.from_store(
        store, VocabEmbedder(), cache_path=str(tmp_path / "emb.json")
    )
    assert {c.chunk_id for c in retriever.chunks} == {"chk_ok"}