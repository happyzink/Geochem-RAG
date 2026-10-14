"""End-to-end rechunk pipeline checks (task E).

Fresh extraction -> page-local chunking -> SHA/fingerprint-correct store write ->
embedding-cache reuse gated by model + text hash. Synthetic pages only; the real
PDFs are re-processed by scripts/import_all_pdfs.py --manifest against the
allow-list.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pypdf import PdfWriter

from geochem_rag import ingest
from geochem_rag.domain import EvidenceChunk, PageQuality, make_chunk_id
from geochem_rag.ingest import PageRecord, chunking_fingerprint, ingest_pdf
from geochem_rag.retrieval import EmbeddingCache
from geochem_rag.store import LocalStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


index_builder = _load("build_full_index", "scripts/build_full_index.py")


def _blank_pdf(path: Path, page_count: int = 1) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


# Scientific string whose numbers / decimals / signs / symbols must survive the
# whole extract -> chunk path. Values are synthetic test data, not paper data.
SYMBOL_TEXT = (
    "Whole-rock analyses yield SiO2 = 62.5 wt.% and Mg# = 45 with Sr/Y = 55. "
    "Zircon U-Pb ages of 251.0~16 Ma, εNd(t) = -3.5 and 87Sr/86Sr = 0.70512. "
    "Reported error is 0.25 ± 0.05 and the ratio is 1.2e-3 (not ~10 km)."
)
SYMBOL_TOKENS = (
    "62.5", "wt.%", "Mg#", "Sr/Y", "251.0", "~16", "-3.5", "0.70512",
    "0.25", "±", "0.05", "1.2e-3", "~10",
)


def _patch_pages(monkeypatch, pages: list[PageRecord]) -> None:
    monkeypatch.setattr(ingest, "extract_pages", lambda _path: pages)


def test_pipeline_reextracts_page_local_chunks_without_symbol_loss(tmp_path, monkeypatch):
    pdf = _blank_pdf(tmp_path / "sample.pdf", 2)
    pages = [
        PageRecord(1, SYMBOL_TEXT, PageQuality.OK.value, len(SYMBOL_TEXT)),
        PageRecord(2, "Age 230 Ma (Luan et al., 2025).", PageQuality.OK.value, 30),
    ]
    _patch_pages(monkeypatch, pages)
    _source, chunks, _report = ingest_pdf(pdf, title="T", visibility="private", max_chars=1200)

    assert chunks
    assert all(c.pdf_page in (1, 2) for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for token in SYMBOL_TOKENS:
        assert token in joined, token


def test_same_pdf_rerun_stable_then_new_fingerprint_rechunks(tmp_path, monkeypatch):
    pdf = _blank_pdf(tmp_path / "s.pdf", 1)
    long_text = (SYMBOL_TEXT + " ") * 6
    _patch_pages(monkeypatch, [PageRecord(1, long_text, PageQuality.OK.value, len(long_text))])
    store = LocalStore(tmp_path / "store")

    src1, chunks1, rep1 = ingest_pdf(pdf, title="T", visibility="private", max_chars=1200)
    first = store.import_document(src1, chunks1, rep1, chunking_fingerprint=chunking_fingerprint(1200))
    assert first.status == "imported"
    ids1 = [c.chunk_id for c in store.get_chunks()]
    assert ids1 == [c.chunk_id for c in chunks1]

    # Same bytes + same rules -> idempotent, no duplicate chunks.
    src2, chunks2, rep2 = ingest_pdf(pdf, title="T", visibility="private", max_chars=1200)
    again = store.import_document(src2, chunks2, rep2, chunking_fingerprint=chunking_fingerprint(1200))
    assert again.status == "already_exists"
    assert [c.chunk_id for c in store.get_chunks()] == ids1

    # New rules -> rebuild from a *fresh* extraction, not the stale chunk set.
    src3, chunks3, rep3 = ingest_pdf(pdf, title="T", visibility="private", max_chars=200)
    rebuilt = store.import_document(src3, chunks3, rep3, chunking_fingerprint=chunking_fingerprint(200))
    assert rebuilt.status == "rechunked"
    assert len(store.get_chunks()) == len(chunks3)
    assert len(chunks3) > len(chunks1)


def _chunk(text: str, page: int = 1) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=make_chunk_id("a" * 64, page, 0),
        source_id="src_x",
        pdf_page=page,
        text=text,
        section="results",
        quality="ok",
        chunk_index=0,
        sha256="a" * 64,
    )


def test_cache_reuse_requires_model_and_matching_text_hash(tmp_path):
    path = tmp_path / "cache.json"
    chunk = _chunk(SYMBOL_TEXT)

    cache = EmbeddingCache(path, "modelA")
    cache._data[chunk.chunk_id] = {"text_hash": cache._text_hash(chunk.text), "vector": [0.1, 0.2, 0.3]}
    cache._dirty = True
    cache.save()

    # Reopening with the same model keeps the entry; a different model ignores it.
    same = EmbeddingCache(path, "modelA")
    assert chunk.chunk_id in same._data
    other = EmbeddingCache(path, "modelB")
    assert other._data == {}

    # A changed text hash must be treated as missing even under the same model.
    health = index_builder.index_health([_chunk(SYMBOL_TEXT + " extra")], same)
    assert health["covered"] == 0
    assert health["missing"] == [chunk.chunk_id]


def test_index_health_flags_dimension_and_validity(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.json", "m")
    good = _chunk("alpha", 1)
    bad_dim = _chunk("beta", 2)
    non_finite = _chunk("gamma", 3)
    cache._data[good.chunk_id] = {"text_hash": cache._text_hash(good.text), "vector": [1.0, 0.0]}
    cache._data[bad_dim.chunk_id] = {"text_hash": cache._text_hash(bad_dim.text), "vector": [1.0, 0.0, 0.0]}
    cache._data[non_finite.chunk_id] = {
        "text_hash": cache._text_hash(non_finite.text),
        "vector": [float("nan"), 0.0],
    }
    health = index_builder.index_health([good, bad_dim, non_finite], cache)
    assert health["covered"] == 2
    assert health["invalid"] == [non_finite.chunk_id]
    assert health["dims"] == [2, 3]
