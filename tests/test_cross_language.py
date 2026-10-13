"""Cross-language retrieval tests: Chinese question → English evidence.

Covers longest-match term expansion, non-synonymous concept separation,
symbol/formula preservation, reverse bridges, and the english_query
trace field — all without network access.
"""

from __future__ import annotations

import pytest

from geochem_rag.retrieval import (
    RetrievalTrace,
    expand_query,
    tokenize,
)


# ---------------------------------------------------------------------------
# Longest match — shorter terms must not fire inside longer ones
# ---------------------------------------------------------------------------


def test_ultramafic_blocks_mafic_inside():
    """超基性 should match as ultramafic, not also trigger 基性→mafic."""
    eq = expand_query("超基性岩的地球化学特征")
    assert "ultramafic" in eq.added_terms
    assert "mafic" not in eq.added_terms
    assert "超基性" in eq.matched_terms
    assert "基性" not in eq.matched_terms


def test_qinghai_tibet_blocks_components():
    """青藏高原 should match as one unit, blocking 青藏 and 高原 inside it."""
    eq = expand_query("青藏高原地壳增厚")
    assert "青藏高原" in eq.matched_terms
    assert "青藏" not in eq.matched_terms
    assert "高原" not in eq.matched_terms


def test_subduction_zone_blocks_subduction():
    """俯冲带 should match, blocking 俯冲 inside it."""
    eq = expand_query("俯冲带流体活动")
    assert "俯冲带" in eq.matched_terms
    assert "俯冲" not in eq.matched_terms


def test_longest_match_multiple_occurrences():
    """A short term outside a longer match should still fire."""
    eq = expand_query("超基性岩与基性岩的对比")
    assert "ultramafic" in eq.added_terms
    assert "mafic" in eq.added_terms
    assert "超基性" in eq.matched_terms
    assert "基性" in eq.matched_terms


# ---------------------------------------------------------------------------
# Non-synonymous concepts must not be conflated
# ---------------------------------------------------------------------------


def test_mantle_source_not_mantle_wedge():
    eq = expand_query("地幔源区特征")
    assert "mantle source" in eq.added_terms
    assert "mantle wedge" not in eq.added_terms


def test_mantle_wedge_separate_entry():
    eq = expand_query("地幔楔的地球化学")
    assert "mantle wedge" in eq.added_terms
    assert "mantle source" not in eq.added_terms


# ---------------------------------------------------------------------------
# Symbol / formula preservation in tokenize
# ---------------------------------------------------------------------------


def test_ratio_preserved_as_single_token():
    tokens = tokenize("Nb/Ta ratio variation")
    assert "nb/ta" in tokens


def test_isotope_ratio_preserved():
    tokens = tokenize("Sr/Y and La/Yb ratios")
    assert "sr/y" in tokens
    assert "la/yb" in tokens


def test_u_pb_dashed_preserved():
    tokens = tokenize("U-Pb zircon dating")
    assert "u-pb" in tokens


def test_abbreviation_case_insensitive():
    tokens1 = tokenize("MORB source")
    tokens2 = tokenize("morb source")
    assert "morb" in tokens1
    assert "morb" in tokens2


def test_chemical_formula_not_split():
    tokens = tokenize("SiO2 and Al2O3 contents")
    assert "sio2" in tokens
    assert "al2o3" in tokens


# ---------------------------------------------------------------------------
# Query expansion preserves original intent
# ---------------------------------------------------------------------------


def test_original_always_preserved():
    q = "弧岩浆中 Nb/Ta 比值的变化意味着什么？"
    eq = expand_query(q)
    assert eq.original == q
    assert eq.expanded.startswith(q)


def test_no_added_terms_for_plain_english():
    eq = expand_query("what is the thermal structure of a subduction zone")
    assert eq.added_terms == ()
    assert eq.expanded == eq.original


def test_added_terms_are_english_for_chinese_question():
    eq = expand_query("埃达克岩的成因机制")
    assert any(t.isascii() for t in eq.added_terms)
    assert "adakite" in eq.added_terms


# ---------------------------------------------------------------------------
# Reverse bridges — English ↔ Chinese
# ---------------------------------------------------------------------------


def test_english_to_chinese_bridge():
    eq = expand_query("adakitic geochemistry")
    assert "埃达克质" in eq.added_terms


def test_tibet_bridge():
    eq = expand_query("tibet crustal thickness")
    assert "青藏" in eq.added_terms or "tibetan plateau" in eq.added_terms
    assert "地壳厚度" in eq.added_terms


# ---------------------------------------------------------------------------
# RetrievalTrace./english_query field
# ---------------------------------------------------------------------------


def test_trace_has_english_query_field():
    trace = RetrievalTrace(
        question="测试",
        mode="hybrid",
        expanded_query="测试",
        expanded_terms=(),
        matched_terms=(),
        rrf_k=60,
        candidate_k=20,
        top_k=5,
    )
    assert trace.english_query is None


def test_trace_to_dict_includes_english_query():
    trace = RetrievalTrace(
        question="测试",
        mode="hybrid",
        expanded_query="测试",
        expanded_terms=(),
        matched_terms=(),
        rrf_k=60,
        candidate_k=20,
        top_k=5,
        english_query="test query",
    )
    d = trace.to_dict()
    assert d["english_query"] == "test query"


# ---------------------------------------------------------------------------
# New terms added in dictionary
# ---------------------------------------------------------------------------


def test_arc_magma_expansion():
    eq = expand_query("弧岩浆的Nb/Ta比值")
    assert "arc magma" in eq.added_terms
    assert "nb/ta" not in eq.added_terms  # already in query


def test_slab_breakoff_expansion():
    eq = expand_query("板片断离与岩浆作用")
    assert "slab breakoff" in eq.added_terms
    assert "slab detachment" in eq.added_terms