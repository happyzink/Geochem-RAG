"""Regression tests for display-side scientific-text preservation (task C).

These numbers/symbols are synthetic inputs used only to test character handling.
They do not represent any paper's measured values.
"""

from __future__ import annotations

import re

from geochem_rag.scientific_text import (
    contains_unescaped_tilde,
    escape_tilde_for_markdown,
    safe_markdown,
)

SCIENTIFIC_CASES = [
    "0.25 1.05 10.0 -3.5 +2.0 −3.5 1.2e-3 1.2 × 10⁻³",
    "251~16 Ma；0.9~1.9；30~175；约 ~10 km；~~文字~~",
    "50–25 Ma；0.25 ± 0.05；<0.1；>10；≤0.5；≥2；≈3",
    "SiO₂；Fe³⁺/Fe²⁺；Mg#；Sr/Y；La/Yb；Eu/Eu*；εNd(t)；δ¹⁸O；⁸⁷Sr/⁸⁶Sr",
    "5 wt.%；12 ppm；0.6‰；850 °C；1.5 GPa；R² = 0.86",
    r"已转义 \~ 与普通 ~；$10^{-3}$；`Sr/Y`；**重要**；[说明](#evidence)",
]

_DIGIT_RE = re.compile(r"[0-9]")
_DOT_RE = re.compile(r"\.")
_SYMBOL_RE = re.compile(r"[−–±≤≥≈×⁻²³⁺#‰°&]")


def _counts(text: str) -> tuple[int, int, int]:
    return (
        len(_DIGIT_RE.findall(text)),
        len(_DOT_RE.findall(text)),
        len(_SYMBOL_RE.findall(text)),
    )


def test_no_bare_tilde_survives_to_markdown() -> None:
    # After the display-copy pass, no pair-able bare tilde remains: every
    # remaining ``~`` is either escaped (``\~``) or an intentional ``~~``.
    for case in SCIENTIFIC_CASES:
        assert not contains_unescaped_tilde(escape_tilde_for_markdown(case)), case


def test_digits_dots_and_symbols_are_not_changed() -> None:
    # Escaping must never delete or replace digits, dots, sign/relational or
    # isotopic symbols; it may only insert backslashes before bare tildes.
    for case in SCIENTIFIC_CASES:
        assert _counts(escape_tilde_for_markdown(case)) == _counts(case), case


def test_removing_added_escapes_returns_original() -> None:
    # The only change is inserted ``\`` characters; nothing else is rewritten.
    # Normalise the preset ``\~`` on both sides so a pre-escaped tilde is not
    # mistaken for an inserted one.
    for case in SCIENTIFIC_CASES:
        restored = escape_tilde_for_markdown(case).replace("\\~", "~")
        assert restored == case.replace("\\~", "~"), case


def test_range_tilde_is_escaped() -> None:
    assert escape_tilde_for_markdown("251~16 Ma") == "251\\~16 Ma"
    assert escape_tilde_for_markdown("0.9~1.9") == "0.9\\~1.9"
    assert escape_tilde_for_markdown("约 ~10 km") == "约 \\~10 km"


def test_intentional_strikethrough_is_preserved() -> None:
    assert escape_tilde_for_markdown("~~文字~~") == "~~文字~~"


def test_code_span_is_verbatim() -> None:
    text = "比值 `Sr/Y~X` 与 `a~b~c`"
    assert escape_tilde_for_markdown(text) == text


def test_math_span_is_verbatim() -> None:
    assert escape_tilde_for_markdown("$10^{-3}$") == "$10^{-3}$"
    assert escape_tilde_for_markdown("a~b $x~y$ c~d") == "a\\~b $x~y$ c\\~d"


def test_bold_and_link_markers_survive() -> None:
    text = "**重要** 与 [说明](#evidence) 之间 ~x~"
    out = escape_tilde_for_markdown(text)
    assert "**重要**" in out
    assert "[说明](#evidence)" in out
    assert "\\~x\\~" in out


def test_escaping_is_idempotent() -> None:
    for case in SCIENTIFIC_CASES:
        once = escape_tilde_for_markdown(case)
        twice = escape_tilde_for_markdown(once)
        assert once == twice, case


def test_already_escaped_tilde_is_not_double_escaped() -> None:
    assert escape_tilde_for_markdown(r"\~") == r"\~"
    assert escape_tilde_for_markdown(r"a \~ b ~ c") == r"a \~ b \~ c"


def test_no_html_entities_or_control_chars_added() -> None:
    for case in SCIENTIFIC_CASES:
        out = escape_tilde_for_markdown(case)
        assert "&#" not in out
        assert "&tilde" not in out
        assert "\\" in out or "~" not in case


def test_safe_markdown_is_alias() -> None:
    assert safe_markdown("30~175") == escape_tilde_for_markdown("30~175")


def test_contains_unescaped_tilde_detector() -> None:
    assert contains_unescaped_tilde("a~b") is True
    assert contains_unescaped_tilde("~~ok~~") is False
    assert contains_unescaped_tilde("`a~b`") is False
    assert contains_unescaped_tilde("$a~b$") is False
    assert contains_unescaped_tilde(r"a\~b") is False
