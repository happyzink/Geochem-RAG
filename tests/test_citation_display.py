"""Deterministic author--year label tests (task B).

``Luan et al., 2025`` is the style example from the brief and is used here only
in synthetic fixtures; it is never hard-coded into answer output.
"""

from __future__ import annotations

from geochem_rag.citation_display import (
    CitationAuthorInfo,
    CitationLabeller,
    author_core_label,
    format_author_year,
    short_title,
    surname,
)


def info(source_id, authors, year, title="T"):
    return CitationAuthorInfo(source_id=source_id, authors=list(authors), year=year, title=title)


def test_single_english_author() -> None:
    assert format_author_year(info("s1", ["Zhikang Luan"], 2025)) == "(Luan, 2025)"


def test_two_english_authors() -> None:
    assert format_author_year(info("s1", ["Zhikang Luan", "Wei Zhang"], 2025)) == "(Luan & Zhang, 2025)"


def test_three_or_more_english_authors() -> None:
    assert (
        format_author_year(info("s1", ["Zhikang Luan", "Wei Zhang", "Jia Liu"], 2025))
        == "(Luan et al., 2025)"
    )


def test_et_al_keeps_its_period() -> None:
    assert "et al." in format_author_year(info("s1", ["A One", "B Two", "C Three"], 2025))


def test_single_chinese_author() -> None:
    assert format_author_year(info("s1", ["张三"], 2025)) == "（张三，2025）"


def test_two_chinese_authors() -> None:
    assert format_author_year(info("s1", ["张三", "李四"], 2025)) == "（张三、李四，2025）"


def test_three_chinese_authors_collapse_to_deng() -> None:
    assert format_author_year(info("s1", ["张三", "李四", "王五"], 2025)) == "（张三等，2025）"


def test_missing_authors_falls_back_to_short_title() -> None:
    label = format_author_year(info("s1", [], 2025, title="青藏高原巨厚地壳：生长与演化"))
    assert label == "（青藏高原巨厚地壳，作者待核实）"


def test_missing_year_marks_year_only() -> None:
    assert format_author_year(info("s1", ["Zhikang Luan"], None)) == "（Luan，年份待核实）"


def test_missing_both_marks_both() -> None:
    label = format_author_year(info("s1", [], None, title="Some Paper"))
    assert label == "（Some Paper，作者/年份待核实）"


def test_multiple_papers_combined() -> None:
    labeller = CitationLabeller(
        [
            info("s1", ["Zhikang Luan", "A B", "C D"], 2025, "Paper One"),
            info("s2", ["John Smith", "E F", "G H"], 2024, "Paper Two"),
        ]
    )
    assert labeller.label_for_ids(["s1", "s2"]) == "(Luan et al., 2025; Smith et al., 2024)"


def test_same_source_multiple_chunks_merge() -> None:
    labeller = CitationLabeller([info("s1", ["Zhikang Luan", "A B", "C D"], 2025, "Paper")])
    assert labeller.label_for_ids(["s1", "s1"]) == "(Luan et al., 2025)"


def test_same_label_same_year_is_disambiguated() -> None:
    labeller = CitationLabeller(
        [
            info("src_b", ["Zhikang Luan", "A B", "C D"], 2025, "Zeta paper"),
            info("src_a", ["Zhikang Luan", "A B", "C D"], 2025, "Alpha paper"),
        ]
    )
    # Ordering is by (title, source_id): "Alpha paper" -> a, "Zeta paper" -> b.
    assert labeller.label_for("src_a") == "(Luan et al., 2025a)"
    assert labeller.label_for("src_b") == "(Luan et al., 2025b)"


def test_disambiguation_is_stable_across_ordering() -> None:
    a = CitationLabeller(
        [
            info("src_a", ["Zhikang Luan", "A B", "C D"], 2025, "Alpha"),
            info("src_b", ["Zhikang Luan", "A B", "C D"], 2025, "Beta"),
        ]
    )
    b = CitationLabeller(
        [
            info("src_b", ["Zhikang Luan", "A B", "C D"], 2025, "Beta"),
            info("src_a", ["Zhikang Luan", "A B", "C D"], 2025, "Alpha"),
        ]
    )
    assert a.label_for("src_a") == b.label_for("src_a") == "(Luan et al., 2025a)"
    assert a.label_for("src_b") == b.label_for("src_b") == "(Luan et al., 2025b)"


def test_same_author_different_year_not_disambiguated() -> None:
    labeller = CitationLabeller(
        [
            info("s1", ["Zhikang Luan"], 2025, "A"),
            info("s2", ["Zhikang Luan"], 2024, "B"),
        ]
    )
    assert labeller.label_for("s1") == "(Luan, 2025)"
    assert labeller.label_for("s2") == "(Luan, 2024)"


def test_surname_handles_particles_and_comma_form() -> None:
    assert surname("Zhikang Luan") == "Luan"
    assert surname("Luan, Zhikang") == "Luan"
    assert surname("Johannes van der Waals") == "van der Waals"
    assert surname("McDonald") == "McDonald"
    assert surname("张三") == "张三"


def test_author_core_label_returns_none_when_empty() -> None:
    assert author_core_label([]) == (None, False)


def test_short_title_truncates_and_marks() -> None:
    assert short_title("青藏高原巨厚地壳：生长、加厚与演化") == "青藏高原巨厚地壳"
    long = "a" * 40
    assert short_title(long).endswith("…")
