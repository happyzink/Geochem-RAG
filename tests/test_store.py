"""Offline tests for LocalStore dedup, replace-on-SHA-change, and delete."""

from __future__ import annotations

from pathlib import Path

import pytest

from geochem_rag.domain import EvidenceChunk, Source, make_chunk_id, make_source_id
from geochem_rag.store import LocalStore

SHA1 = "1" * 64
SHA2 = "2" * 64


def _source(sha: str, title: str = "T", sid: str | None = None) -> Source:
    return Source(
        source_id=sid or make_source_id(sha),
        title=title,
        authors=["A"],
        year=2022,
        doi_or_url="https://example.org/x",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        sha256=sha,
        visibility="public",
        download_date="2026-09-28",
    )


def _chunks(sha: str, pages: list[int], sid: str | None = None) -> list[EvidenceChunk]:
    source_id = sid or make_source_id(sha)
    out = []
    for page in pages:
        out.append(
            EvidenceChunk(
                chunk_id=make_chunk_id(sha, page, 0),
                source_id=source_id,
                pdf_page=page,
                text=f"evidence on page {page}",
                section="results",
                quality="ok",
                chunk_index=0,
                sha256=sha,
            )
        )
    return out


def test_duplicate_import_is_idempotent(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    source = _source(SHA1)
    chunks = _chunks(SHA1, [1, 2, 3])

    first = store.import_document(source, chunks)
    assert first.status == "imported"
    assert first.chunk_count == 3
    assert store.count_chunks() == 3

    second = store.import_document(source, chunks)
    assert second.status == "already_exists"
    assert second.chunk_count == 3
    assert store.count_chunks() == 3
    assert len(store.get_chunks(source.source_id)) == 3


def test_sha_change_invalidates_old_chunks(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    # Stable bibliographic id from the registry; SHA is content identity.
    stable_id = "src_stable_paper"
    src_a = _source(SHA1, title="v1", sid=stable_id)
    store.import_document(src_a, _chunks(SHA1, [1, 2], sid=stable_id))
    assert store.count_chunks() == 2

    # Same source_id, new PDF bytes -> replace and drop old evidence set.
    src_b = _source(SHA2, title="v2", sid=stable_id)
    result = store.import_document(src_b, _chunks(SHA2, [1, 2, 3], sid=stable_id))
    assert result.status == "replaced"
    assert store.count_chunks() == 3
    assert store.get_source(stable_id) is not None
    assert store.get_source(stable_id).sha256 == SHA2
    # Old chunk ids are gone.
    assert store.find_chunk(make_chunk_id(SHA1, 1, 0)) is None
    assert store.find_chunk(make_chunk_id(SHA2, 1, 0)) is not None


def test_same_sha_under_new_id_still_dedupes(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    src_a = _source(SHA1, title="v1")
    store.import_document(src_a, _chunks(SHA1, [1, 2]))
    src_alias = _source(SHA1, title="alias", sid="src_other")
    result = store.import_document(src_alias, _chunks(SHA1, [1, 2]))
    assert result.status == "already_exists"
    assert result.source_id == src_a.source_id
    assert store.count_chunks() == 2


def test_same_file_can_gain_corrected_metadata_without_new_chunks(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    source = _source(SHA1, title="filename-title")
    store.import_document(source, _chunks(SHA1, [1, 2]))
    corrected = _source(SHA1, title="Verified article title")
    result = store.import_document(corrected, _chunks(SHA1, [1, 2]))
    assert result.status == "metadata_updated"
    assert store.get_source(source.source_id).title == "Verified article title"
    assert store.count_chunks() == 2


def test_delete_source_cascades_chunks(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    source = _source(SHA1)
    store.import_document(source, _chunks(SHA1, [1, 2]))
    removed = store.delete_source(source.source_id)
    assert removed == 2
    assert store.count_chunks() == 0
    assert store.list_sources() == []


def test_visibility_filter(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    pub = _source(SHA1, title="public paper")
    priv = Source(
        source_id=make_source_id(SHA2),
        title="private notes",
        authors=["Me"],
        year=2024,
        doi_or_url="local",
        license="private",
        license_url="local",
        sha256=SHA2,
        visibility="private",
        download_date="2026-09-28",
    )
    store.import_document(pub, _chunks(SHA1, [1]))
    store.import_document(priv, _chunks(SHA2, [1]))
    assert len(store.list_sources(visibility="public")) == 1
    assert len(store.list_sources(visibility="private")) == 1
    assert store.list_sources(visibility="public")[0].title == "public paper"


def test_pages_for_source_groups_without_cross_page(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    source = _source(SHA1)
    store.import_document(source, _chunks(SHA1, [1, 1, 2]))
    pages = store.pages_for_source(source.source_id)
    assert set(pages) == {1, 2}
    assert all(c.pdf_page == page for page, cs in pages.items() for c in cs)

def test_chunking_fingerprint_change_rechunks_same_bytes(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    source = _source(SHA1)
    first = store.import_document(source, _chunks(SHA1, [1, 2]), chunking_fingerprint="fp1")
    assert first.status == "imported"
    assert store.count_chunks() == 2

    # Same bytes + same fingerprint -> no re-chunk.
    again = store.import_document(source, _chunks(SHA1, [1, 2]), chunking_fingerprint="fp1")
    assert again.status == "already_exists"
    assert store.count_chunks() == 2

    # Same bytes + new fingerprint -> rebuild the evidence set, not silently skip.
    rebuilt = store.import_document(source, _chunks(SHA1, [1, 2, 3]), chunking_fingerprint="fp2")
    assert rebuilt.status == "rechunked"
    assert store.count_chunks() == 3


def test_no_fingerprint_preserves_legacy_dedup(tmp_path: Path):
    store = LocalStore(tmp_path / "processed")
    source = _source(SHA1)
    store.import_document(source, _chunks(SHA1, [1, 2]))
    again = store.import_document(source, _chunks(SHA1, [1, 2, 3]))
    assert again.status == "already_exists"
    assert store.count_chunks() == 2
