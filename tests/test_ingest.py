"""Offline tests for page-local ingest and stable chunk ids."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from geochem_rag.ingest import (
    assess_page_text,
    chunk_page_text,
    chunking_fingerprint,
    extract_pages,
    ingest_pdf,
    sha256_file,
    split_paragraphs,
)
from geochem_rag.domain import make_chunk_id, make_source_id

PAGE1 = (
    "1. Introduction\n\n"
    "Adakitic rocks are often interpreted as products of partial melting of thickened lower crust. "
    "Here we report whole-rock and zircon data from the Mogetong pluton.\n\n"
    "2. Methods\n\n"
    "Zircon U-Pb dating was performed by LA-ICP-MS. Whole-rock major and trace elements "
    "were analyzed by XRF and ICP-MS."
)
PAGE2 = (
    "3. Results\n\n"
    "The pluton yields a crystallization age of ca. 230 Ma. "
    "Sr/Y ratios are elevated and Y concentrations are low, consistent with adakitic affinity."
)


def _make_pdf(path: Path, pages: list[str]) -> Path:
    writer = PdfWriter()
    for _ in pages:
        writer.add_blank_page(width=612, height=792)
    # pypdf cannot easily inject text into blank pages without content streams.
    # Instead we test chunking/assess directly and use a tiny real PDF via reportlab-less
    # fallback: write pages then monkeypatch extract in higher-level tests.
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_assess_empty_and_scanned():
    quality, note = assess_page_text("   \n  ")
    assert quality == "empty"
    quality, note = assess_page_text("Fig.")
    assert quality == "scanned"
    quality, note = assess_page_text(PAGE1)
    assert quality == "ok"
    quality, note = assess_page_text("\x00\x01\x02 crazy " + "�" * 20)
    assert quality == "garbled"


def test_assess_mismapped_cjk_is_garbled():
    # Typical CID/ToUnicode failure: Han text extracted into Indic/SE-Asian blocks.
    broken = "ೆ൭ , ᅵ ҵ ၳ , ֥ ֥[22,26] , ದ [27] Ոၳ ӈ 、ᄯ 、ቔႨ 、หᆘ ഈ , ථջᄝ ն੤ஷሏၛট , ֥ ջྟ 。ݔࢲ࣮ , ൫๭Ֆ Pb༅ 。"
    quality, note = assess_page_text(broken)
    assert quality == "garbled"
    assert note is not None and "encoding" in note
    # Clean Chinese scientific prose stays ok.
    clean = "青藏高原 Pb 同位素地球化学研究表明，印度大陆岩石圈地幔具有独特的同位素组成特征。"
    quality, note = assess_page_text(clean * 3)
    assert quality == "ok"


def test_split_paragraphs_no_blank_line_merge():
    paras = split_paragraphs("a\n\nb\n\n\n\nc")
    assert paras == ["a", "b", "c"]


def test_chunks_never_cross_page_and_ids_stable():
    sha = "c" * 64
    c1 = chunk_page_text(PAGE1, pdf_page=1, source_sha256=sha, max_chars=200)
    c2 = chunk_page_text(PAGE2, pdf_page=2, source_sha256=sha, max_chars=200)
    assert c1 and c2
    for chunk in c1:
        assert chunk.pdf_page == 1
        assert chunk.chunk_id == make_chunk_id(sha, 1, chunk.chunk_index)
    for chunk in c2:
        assert chunk.pdf_page == 2
        assert chunk.chunk_id == make_chunk_id(sha, 2, chunk.chunk_index)

    again = chunk_page_text(PAGE1, pdf_page=1, source_sha256=sha, max_chars=200)
    assert [x.chunk_id for x in again] == [x.chunk_id for x in c1]
    assert [x.text for x in again] == [x.text for x in c1]


def test_chunk_index_zero_based_and_unique():
    sha = "d" * 64
    chunks = chunk_page_text(PAGE1 + "\n\n" + PAGE2, pdf_page=1, source_sha256=sha, max_chars=120)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_page_yields_no_chunks():
    sha = "e" * 64
    assert chunk_page_text("   ", pdf_page=1, source_sha256=sha) == []


def test_oversized_single_paragraph_splits_on_cjk_period():
    sha = "f" * 64
    text = "。 ".join(f"这是第{i}个中文句子，用于验证分块不会跨页" for i in range(20))
    chunks = chunk_page_text(text, pdf_page=3, source_sha256=sha, max_chars=100)
    assert len(chunks) > 1
    assert all(c.pdf_page == 3 for c in chunks)
    again = chunk_page_text(text, pdf_page=3, source_sha256=sha, max_chars=100)
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]


def test_chunking_fingerprint_changes_with_max_chars():
    assert chunking_fingerprint(max_chars=900) != chunking_fingerprint(max_chars=1200)
    assert chunking_fingerprint(max_chars=1200) == chunking_fingerprint(max_chars=1200)


def test_ingest_pdf_blank_pages_reported(tmp_path: Path):
    # Blank pages extract as empty -> reported as failed pages.
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    source, chunks, report = ingest_pdf(
        pdf_path,
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        title="Blank test PDF",
        authors=["Tester"],
        year=2026,
        doi_or_url="https://example.org/blank",
        download_date="2026-09-28",
        visibility="public",
    )
    assert report.page_count == 2
    assert report.failed_pages == [1, 2]
    assert "empty" in report.quality_counts
    assert report.chunk_count == len(chunks) == 0
    assert source.sha256 == sha256_file(pdf_path)
    assert source.source_id == make_source_id(source.sha256)
    assert any("page 1" in w for w in report.warnings)
