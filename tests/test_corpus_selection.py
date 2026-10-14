"""Content-based geoscience allow-list selection (task D).

Synthetic cases only: a travel book that merely names a mountain must be
excluded, while a machine-learning paper that studies a geoscience problem must
be included. Real manifest counts are produced by scripts/inventory_local_pdfs.py
and are not hard-coded here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inventory = _load("inventory_local_pdfs", "scripts/inventory_local_pdfs.py")
importer = _load("import_all_pdfs", "scripts/import_all_pdfs.py")


# ---------------------------------------------------------------------------
# classify_geoscience: content decides, a mountain name alone never does
# ---------------------------------------------------------------------------


def test_travel_book_with_mountain_name_is_excluded():
    text = (
        "Walking in the Alps. This guide lists day walks, multi-day treks and "
        "hut-to-hut tours, with a chapter on views towards the Himalaya. "
        "Practical information on routes and accommodation."
    )
    decision, _reason, _pages = inventory.classify_geoscience(
        text, "Walking in Austria 101 Routes - Day Walks.pdf"
    )
    assert decision == "exclude"


def test_ml_paper_for_geoscience_is_included():
    text = (
        "We apply machine learning and a neural network to predict the water "
        "content of the mantle and the crust from mid-ocean ridge basalt "
        "geochemistry, using a compiled geochemical dataset."
    )
    decision, _reason, _pages = inventory.classify_geoscience(text, "ml-mantle-water.pdf")
    assert decision == "include"


def test_chinese_geoscience_paper_is_included():
    text = (
        "青藏高原是研究大陆动力学和地壳生长的天然实验室，涉及地幔对流、玄武岩、"
        "花岗岩以及锆石年代学，并讨论造山带与俯冲过程。"
    )
    decision, _reason, _pages = inventory.classify_geoscience(text, "青藏高原地壳生长.pdf")
    assert decision == "include"


def test_generic_long_prose_without_subject_vocabulary_is_excluded():
    text = ("This document is a general introduction to project management. " * 60)
    decision, _reason, _pages = inventory.classify_geoscience(text, "notes.pdf")
    assert decision == "exclude"


def test_short_text_without_subject_vocabulary_is_uncertain():
    text = "A short internal memo with no subject keywords at all."
    decision, _reason, _pages = inventory.classify_geoscience(text, "memo.pdf")
    assert decision == "uncertain"


# ---------------------------------------------------------------------------
# Legacy filename blacklist (used only when no allow-list manifest is given)
# ---------------------------------------------------------------------------


def test_legacy_blacklist_excludes_known_non_geoscience():
    assert importer.is_excluded_non_geoscience("PDF/Psychology_Now_Presents_x.pdf") is True
    assert importer.is_excluded_non_geoscience("PDF/青藏高原造山带研究.pdf") is False


# ---------------------------------------------------------------------------
# Manifest gate: exclude / uncertain / missing never enter the library
# ---------------------------------------------------------------------------


def test_manifest_action_gate():
    manifest = {
        "PDF/a.pdf": {"relative_path": "PDF/a.pdf", "decision": "include", "reason": "地学"},
        "PDF/b.pdf": {"relative_path": "PDF/b.pdf", "decision": "exclude", "reason": "旅游"},
        "PDF/c.pdf": {"relative_path": "PDF/c.pdf", "decision": "uncertain", "reason": "待核"},
    }
    assert importer.manifest_action("PDF/a.pdf", manifest)[0] == "import"
    assert importer.manifest_action("PDF/b.pdf", manifest)[0] == "exclude"
    assert importer.manifest_action("PDF/c.pdf", manifest)[0] == "exclude"
    assert importer.manifest_action("PDF/missing.pdf", manifest)[0] == "exclude"
    # No manifest -> legacy path, resolved by the caller's filename blacklist.
    assert importer.manifest_action("PDF/a.pdf", None)[0] == "legacy"


def test_manifest_signature_stable_and_metadata_sensitive():
    base = {
        "decision": "include",
        "title": "T",
        "title_is_filename": False,
        "authors": ["A"],
        "year": 2020,
        "doi_or_url": None,
    }
    sig = importer.manifest_signature(base, 1200)
    assert importer.manifest_signature(dict(base), 1200) == sig
    assert importer.manifest_signature(dict(base, year=2021), 1200) != sig
    assert importer.manifest_signature(dict(base, authors=["A", "B"]), 1200) != sig
    # A chunking-rule change must also invalidate the SHA-based resume.
    assert importer.manifest_signature(base, 800) != sig
