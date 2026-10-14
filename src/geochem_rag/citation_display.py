"""Deterministic author--year citation labels for display (task B).

The model still cites evidence by the internal ``chunk_id`` that was actually
passed in this call. After the answerer validates those ids, this module turns
the *verified* source metadata into a stable author--year label. The model never
writes author/year text; a fabricated label can never reach the view.

Display rules (from the task brief):

======================  ==========================================
case                    rendered form
======================  ==========================================
english, one author     ``(Luan, 2025)``
english, two authors    ``(Luan & Zhang, 2025)``
english, 3+ authors     ``(Luan et al., 2025)``
chinese, 1 author       ``（张三，2025）``
chinese, 2 authors      ``（张三、李四，2025）``
chinese, 3+ authors     ``（张三等，2025）``
several papers          ``(Luan et al., 2025; Smith et al., 2024)``
same label + year       ``(Luan et al., 2025a)`` / ``(Luan et al., 2025b)``
missing author or year  ``（来源短标题，作者/年份待核实）``
======================  ==========================================

``et al.`` keeps its period; english labels use half-width brackets with a
space after the comma; chinese labels use full-width brackets. Disambiguation
letters are assigned by a stable key (verified title, then ``source_id``), so a
history replay never points ``2025a`` at a different paper than it did before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "CitationAuthorInfo",
    "CitationLabeller",
    "author_core_label",
    "format_author_year",
    "short_title",
    "surname",
]

# CJK Unified Ideographs (+ Ext-A, compatibility) -> treat as a Chinese name.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# Nobiliary / between-name particles that belong to the family-name part.
_NAME_PARTICLES = {
    "van", "von", "de", "den", "der", "del", "della", "di", "da", "dos", "das",
    "la", "le", "du", "bin", "ibn", "al", "ter", "ten", "op", "af", "zu",
}

# Case-insensitive section anchors used by ``short_title``.
_STOP_PUNCT = ("：", ":", "—", "——", "–", "·", "，", ",")


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def surname(name: str) -> str:
    """Extract a display surname without blindly taking the last token.

    Handles ``Family, Given`` (comma form), ``Given Family`` and multi-part
    family names such as ``van der Waals``. Chinese names are returned whole
    (the label builder lists them in full or collapses to ``X等``).
    """
    raw = (name or "").strip()
    if not raw:
        return raw
    if has_cjk(raw):
        return raw
    if "," in raw:
        return _tidy(raw.split(",", 1)[0])
    parts = raw.split()
    if len(parts) == 1:
        return _tidy(raw)
    lowered = [p.lower().strip(".") for p in parts]
    k = len(parts) - 1
    while k > 0 and lowered[k - 1] in _NAME_PARTICLES:
        k -= 1
    return _tidy(" ".join(parts[k:]))


def _tidy(fragment: str) -> str:
    text = fragment.strip().strip(",;")
    # Drop a trailing lone initial glued on by ``Family, G`` style spills.
    text = re.sub(r"\s+[A-Za-z]\.?$", "", text).strip()
    return text or fragment.strip()


def author_core_label(authors: list[str]) -> tuple[str | None, bool]:
    """Return ``(core, is_cjk)`` or ``(None, False)`` when authors are absent."""
    names = [a.strip() for a in (authors or []) if a and a.strip()]
    if not names:
        return None, False
    if any(has_cjk(a) for a in names):
        if len(names) >= 3:
            core = f"{names[0]}等"
        else:
            core = "、".join(names)
        return core, True
    families = [surname(a) for a in names]
    if len(families) == 1:
        core = families[0]
    elif len(families) == 2:
        core = f"{families[0]} & {families[1]}"
    else:
        core = f"{families[0]} et al."
    return core, False


def short_title(title: str, limit: int = 16) -> str:
    """First meaningful fragment of a title, for the metadata-missing fallback."""
    text = re.sub(r"\s+", " ", (title or "")).strip()
    if not text:
        return "来源标题缺失"
    for sep in _STOP_PUNCT:
        pos = text.find(sep)
        if 0 < pos:
            text = text[:pos]
            break
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


@dataclass(frozen=True)
class CitationAuthorInfo:
    """Verified bibliographic facts for one source, used to build its label."""

    source_id: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    title: str = ""


def format_author_year(info: CitationAuthorInfo, *, suffix: str | None = None) -> str:
    """Render a single source's label, or its metadata-missing fallback."""
    core, is_cjk = author_core_label(info.authors)
    year = info.year
    if core is not None and year is not None:
        year_text = f"{year}{suffix or ''}"
        if is_cjk:
            return f"（{core}，{year_text}）"
        return f"({core}, {year_text})"
    missing: list[str] = []
    if core is None:
        missing.append("作者")
    if year is None:
        missing.append("年份")
    missing_text = "/".join(missing) + "待核实"
    lead = core if core is not None else short_title(info.title)
    return f"（{lead}，{missing_text}）"


class CitationLabeller:
    """Assign stable author--year labels (with a/b disambiguation) to sources."""

    def __init__(self, infos: list[CitationAuthorInfo]) -> None:
        self._infos: dict[str, CitationAuthorInfo] = {i.source_id: i for i in infos}
        self._suffix: dict[str, str] = self._assign_suffixes(infos)

    @staticmethod
    def _assign_suffixes(infos: list[CitationAuthorInfo]) -> dict[str, str]:
        groups: dict[tuple[str, int, bool], list[CitationAuthorInfo]] = {}
        for info in infos:
            core, is_cjk = author_core_label(info.authors)
            if core is None or info.year is None:
                continue
            groups.setdefault((core, info.year, is_cjk), []).append(info)
        suffix: dict[str, str] = {}
        for members in groups.values():
            unique: dict[str, CitationAuthorInfo] = {m.source_id: m for m in members}
            if len(unique) <= 1:
                continue
            ordered = sorted(
                unique.values(), key=lambda i: ((i.title or "").lower(), i.source_id)
            )
            for index, info in enumerate(ordered):
                letter = chr(ord("a") + index) if index < 26 else str(index + 1)
                suffix[info.source_id] = letter
        return suffix

    def suffix_for(self, source_id: str) -> str:
        return self._suffix.get(source_id, "")

    def label_for(self, source_id: str) -> str:
        info = self._infos.get(source_id)
        if info is None:
            return "（来源未知）"
        return format_author_year(info, suffix=self.suffix_for(source_id))

    def label_for_ids(self, source_ids: list[str]) -> str:
        """Combined label for a set of sources, merging duplicates per source."""
        seen: list[str] = []
        for sid in source_ids:
            if sid and sid not in seen:
                seen.append(sid)
        if not seen:
            return ""
        if len(seen) == 1:
            return self.label_for(seen[0])
        cores: list[str] = []
        is_cjk = any(
            (self._infos.get(sid) is not None)
            and author_core_label(self._infos[sid].authors)[1]
            for sid in seen
        )
        for sid in seen:
            info = self._infos.get(sid)
            if info is None:
                cores.append("来源未知")
                continue
            label = self.label_for(sid)
            cores.append(label[1:-1])  # strip the brackets
        if is_cjk:
            return "（" + "；".join(cores) + "）"
        return "(" + "; ".join(cores) + ")"
