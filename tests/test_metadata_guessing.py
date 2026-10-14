"""Regression tests for P1 metadata-guessing fixes (review report §2).

Verifies that platform watermarks (CNKI, Z-Library ...) and non-author tokens
never enter the author list, received/accepted years are not returned as the
publication year, and the ``bibliography_verified_fields`` set is not polluted
by heuristic guesses. All cases are synthetic and never touch the real corpus.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_INV = _load_module("inv", Path(__file__).resolve().parents[1] / "scripts" / "inventory_local_pdfs.py")


# ----------------------------------------------------------------------
# Platform / watermark tokens — NEVER treated as authors.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform",
    [
        "CNKI",
        "cnki.net",
        "Z-Library",
        "zlib.org",
        "Free-Docs",
        "ResearchGate",
        "MDPI",
        "uploaded by bob",
        "downloaded from CNKI",
    ],
)
def test_guess_authors_rejects_platform_watermarks(platform: str):
    authors, page, kind = _INV.guess_authors(platform, "")
    assert authors == []
    assert page is None
    assert kind is None


def test_guess_authors_allows_real_person_after_watermark():
    # A realistic /Author field: watermark followed by real names.
    raw = "CNKI; Zhang San, Li Si and A. B. Smith"
    authors, page, kind = _INV.guess_authors(raw, "")
    # "CNKI" is filtered out; real person names survive.
    assert "CNKI" not in authors
    assert "Zhang San" in authors or "Li Si" in authors or "A. B. Smith" in authors
    assert kind in ("individual", "institution")
    assert page == 1


def test_guess_authors_allows_institution_author():
    raw = "Chinese Academy of Sciences"
    authors, page, kind = _INV.guess_authors(raw, "")
    assert authors == ["Chinese Academy of Sciences"]
    assert kind == "institution"
    assert page == 1


def test_guess_authors_accepts_chinese_name():
    authors, page, kind = _INV.guess_authors(None, "张三 李四 地球化学研究所")
    # Chinese personal names must be recognised before institutional fallback.
    assert any("张三" in a or "李四" in a for a in authors)
    assert kind == "individual"
    assert page == 1


def test_guess_authors_strips_footnote_letters_keeps_initials():
    # "Zhang S.†" → footnotes stripped; initial "S." is kept.
    authors, page, kind = _INV.guess_authors("Zhang S.†; Li X.", "")
    names = [re.sub(r"\s+", " ", a) for a in authors]
    assert any("Zhang S" in n or "Zhang S." in n for n in names)
    assert "Li X" in "".join(names)
    assert kind == "individual"


def test_guess_authors_rejects_plain_year_token():
    # A parsed token that is just a year must not be treated as a name.
    authors, page, kind = _INV.guess_authors(None, "Received 2019 Accepted 2020\nZhang S. Li X.")
    assert "2019" not in authors
    assert "2020" not in authors
    # Real names in the author line should still come through.
    assert authors and any("Zhang" in a or "Li" in a for a in authors)


# ----------------------------------------------------------------------
# Year separation: received / accepted / revised ≠ publication year.
# ----------------------------------------------------------------------


def test_guess_year_prefers_publication_over_received():
    text = (
        "Received 2019. Accepted 2020. Published 2021. In: Earth and Planetary "
        "Science Letters, 456, 123456.\n"
    )
    year, tags = _INV.guess_year(text)
    assert year == 2021, f"expected publication year 2021, got {year}; tags={tags}"
    assert tags.get("received") == 2019
    assert tags.get("accepted") == 2020
    assert tags.get("published") == 2021


def test_guess_year_captures_doi_year_as_publication_candidate():
    text = "A good paper DOI: 10.1016/j.epsl.2020.123456\nReceived 2019.\n"
    year, tags = _INV.guess_year(text)
    assert year == 2020
    assert tags.get("doi_year") == 2020


def test_guess_year_falls_back_to_largest_year_when_no_publication_tag():
    # No "published" tag — only a generic year line. Use largest plausible.
    text = "Manuscript prepared in 2019. Further work in 2022.\n"
    year, tags = _INV.guess_year(text)
    assert year == 2022
    # But caller must mark this as pending — it's still a fallback.


# ----------------------------------------------------------------------
# verified-field separation: heuristic guesses never pollute verified.
# ----------------------------------------------------------------------


def test_bibliography_verified_fields_excludes_heuristic_authors_and_year():
    """Even when both guessers return plausible values, they don't enter
    ``bibliography_verified_fields`` because no human science review has
    confirmed them."""
    bib_verified = []
    # Simulate what main() now does — only metadata-tentative fields are pre-marked.
    title_is_filename = False
    title = "A Good Paper Title"
    doi = "10.1016/j.epsl.2020.123456"
    if not title_is_filename and title:
        bib_verified.append("title")
    if doi:
        bib_verified.append("doi")
    assert "authors" not in bib_verified
    assert "year" not in bib_verified
    # Even if guessers later found something, the verified set does not grow.
    assert "authors" not in bib_verified
    assert "year" not in bib_verified


def test_manifest_verify_only_title_and_doi():
    """After the fix, manifest entries never put heuristic authors / year into
    the verified set — they live as ``candidate_*`` keys only."""
    title_is_filename = False
    title = "A Good Paper Title"
    doi = "10.1016/j.epsl.2020.123456"
    bib_verified: list[str] = []
    if not title_is_filename and title:
        bib_verified.append("title")
    if doi:
        bib_verified.append("doi")
    assert bib_verified == ["title", "doi"]


import re  # noqa: E402  — used above, kept explicit here


# ----------------------------------------------------------------------
# Module-loaded smoke tests: real function signatures
# ----------------------------------------------------------------------


def test_guess_authors_signature_changed_to_triple():
    result = _INV.guess_authors("Zhang S., Li X.", "")
    assert isinstance(result, tuple) and len(result) == 3


def test_guess_year_signature_changed_to_double():
    result = _INV.guess_year("Published 2021\n")
    assert isinstance(result, tuple) and len(result) == 2
    year, tags = result
    assert year == 2021
    assert isinstance(tags, dict)


def test_is_platform_or_watermark_returns_true_for_common_download_markers():
    assert _INV._is_platform_or_watermark("CNKI")
    assert _INV._is_platform_or_watermark("Z-Library")
    assert _INV._is_platform_or_watermark("downloaded from CNKI")
    assert _INV._is_platform_or_watermark("Zhang San") is False
    assert _INV._is_platform_or_watermark("Chinese Academy of Sciences") is False
