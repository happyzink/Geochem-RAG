"""Born-digital PDF import: page-local extraction, quality marks, in-page chunks.

Contract guarantees:
- pdf_page is 1-based PDF page order.
- A chunk never spans two PDF pages.
- chunk_id is stable for the same PDF bytes and the same chunking rules.
- Empty / scanned / garbled pages are reported, never dropped silently.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import logging

from pypdf import PdfReader

from .domain import (
    EvidenceChunk,
    PageQuality,
    Source,
    make_chunk_id,
    make_source_id,
)

# pypdf emits noisy fontTools warnings on CFF fonts; keep the page report readable.
logging.getLogger("pypdf").setLevel(logging.ERROR)

DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 40

_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"^\s*abstract\s*$", re.I)),
    ("introduction", re.compile(r"^\s*(1\.?\s*)?introduction\s*$", re.I)),
    ("methods", re.compile(r"^\s*(\d\.?\s*)?(methods?|materials and methods|analytical methods)\s*$", re.I)),
    ("results", re.compile(r"^\s*(\d\.?\s*)?results?\s*$", re.I)),
    ("discussion", re.compile(r"^\s*(\d\.?\s*)?(discussion|discussions)\s*$", re.I)),
    ("conclusion", re.compile(r"^\s*(\d\.?\s*)?(conclusions?|summary)\s*$", re.I)),
    ("references", re.compile(r"^\s*(\d\.?\s*)?(references|bibliography)\s*$", re.I)),
)

_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*)\.?\s+[A-Z][^\n]{2,80}$")

# Sentence boundary for oversized-paragraph splitting. Includes CJK full stops.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")

# Bump when the chunking algorithm changes materially; drives the fingerprint.
_CHUNKING_RULE_VERSION = 2


@dataclass(frozen=True)
class PageRecord:
    """One PDF page after extraction."""

    pdf_page: int
    text: str
    quality: str
    char_count: int
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestReport:
    """Human-readable import report. Failures are listed, not swallowed."""

    source_id: str
    sha256: str
    local_path: str
    page_count: int
    ok_pages: int
    failed_pages: list[int]
    quality_counts: dict[str, int]
    chunk_count: int
    warnings: list[str] = field(default_factory=list)
    chunking_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunking_fingerprint(
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    overlap_chars: int = 0,
    rule_version: int = _CHUNKING_RULE_VERSION,
) -> str:
    """Stable identifier for the current chunking rules.

    Any chunking parameter or algorithm change must change this value, so a
    re-import under new rules is not silently skipped by SHA-256 dedup.
    """
    payload = f"v{rule_version}:max={max_chars}:overlap={overlap_chars}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _printable_ratio(text: str) -> float:
    """Share of characters that look like real extracted text.

    Control characters and U+FFFD replacement characters count as damage.
    """
    if not text:
        return 0.0
    good = 0
    for ch in text:
        if ch in "\n\r\t ":
            good += 1
        elif ch.isprintable() and ch != "�":
            good += 1
    return good / len(text)


# CID/ToUnicode failures on CJK PDFs often map Han characters into rare
# Indic/Southeast-Asian blocks. Those scripts never appear in these papers.
_MISMAP_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x0DFF),  # Devanagari..Malayalam
    (0x0E00, 0x0EFF),  # Thai/Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x18AF),  # Khmer/Mongolian
    (0xA800, 0xA8DF),
    (0xABC0, 0xABFF),
)


def _mismapped_script_ratio(text: str) -> float:
    """Fraction of non-space characters in unlikely script blocks."""
    counted = 0
    bad = 0
    for ch in text:
        if ch.isspace():
            continue
        counted += 1
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _MISMAP_SCRIPT_RANGES):
            bad += 1
    if counted == 0:
        return 0.0
    return bad / counted


def _looks_scanned(text: str, extracted_chars: int) -> bool:
    compact = re.sub(r"\s+", "", text)
    return extracted_chars < DEFAULT_MIN_CHARS and len(compact) < DEFAULT_MIN_CHARS


def assess_page_text(text: str) -> tuple[str, str | None]:
    """Return (quality, note). Quality is one of ok/empty/scanned/garbled."""
    stripped = text.strip()
    if not stripped:
        return PageQuality.EMPTY.value, "no extractable text"
    if _printable_ratio(stripped) < 0.85:
        return PageQuality.GARBLED.value, "high ratio of non-printable or replacement characters"
    mismapped = _mismapped_script_ratio(stripped)
    if mismapped >= 0.08:
        return (
            PageQuality.GARBLED.value,
            "high ratio of mis-mapped CJK glyphs (broken font encoding)",
        )
    if _looks_scanned(stripped, len(stripped)):
        # Very little text on a page usually means a scanned figure page or image-only page.
        return PageQuality.SCANNED.value, "minimal text; likely scanned or image-only page"
    return PageQuality.OK.value, None


def split_paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs on blank lines / hard line runs."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", normalized)
    paragraphs: list[str] = []
    for part in parts:
        collapsed = re.sub(r"[ \t]+", " ", part).strip()
        collapsed = re.sub(r"\n+", "\n", collapsed)
        if collapsed:
            paragraphs.append(collapsed)
    return paragraphs


def detect_section(paragraphs: Sequence[str], current: str | None) -> str | None:
    """Track the nearest heading-like line as the section label."""
    section = current
    for para in paragraphs:
        first_line = para.split("\n", 1)[0].strip()
        for name, pattern in _SECTION_PATTERNS:
            if pattern.match(first_line):
                section = name
                break
        else:
            if _HEADING_RE.match(first_line) and len(first_line) < 80:
                section = first_line[:80]
    return section


def chunk_page_text(
    page_text: str,
    *,
    pdf_page: int,
    source_sha256: str,
    section: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    source_id: str | None = None,
) -> list[EvidenceChunk]:
    """Deterministically chunk one page. Never crosses the page boundary."""
    if pdf_page < 1:
        raise ValueError("pdf_page must be >= 1")
    if max_chars < 50:
        raise ValueError("max_chars must be >= 50")
    if source_id is None:
        source_id = make_source_id(source_sha256)

    quality, _ = assess_page_text(page_text)
    paragraphs = split_paragraphs(page_text)
    if not paragraphs:
        return []

    section = detect_section(paragraphs, section)
    chunks: list[EvidenceChunk] = []
    buffer: list[str] = []
    buffer_len = 0
    start_cursor = 0
    end_cursor = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, chunk_index, start_cursor
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if not text:
            buffer, buffer_len = [], 0
            return
        chunk = EvidenceChunk(
            chunk_id=make_chunk_id(source_sha256, pdf_page, chunk_index),
            source_id=source_id,
            pdf_page=pdf_page,
            text=text,
            section=section,
            quality=quality,
            chunk_index=chunk_index,
            sha256=source_sha256,
            char_start=start_cursor,
            char_end=end_cursor,
        )
        chunks.append(chunk)
        chunk_index += 1
        buffer, buffer_len = [], 0

    # Track offsets roughly by walking the original page text.
    cursor = 0
    for para in paragraphs:
        para_len = len(para) + 2  # account for join separator
        if buffer and buffer_len + para_len > max_chars:
            flush()
        if not buffer:
            # locate paragraph start in page text for provenance
            found = page_text.find(para[:40] if len(para) > 40 else para, cursor)
            start_cursor = found if found >= 0 else cursor
        buffer.append(para)
        buffer_len += para_len
        end_cursor = start_cursor + sum(len(p) for p in buffer) + 2 * max(len(buffer) - 1, 0)
        cursor = start_cursor + len(para)

        # Oversized single paragraph: split on sentence boundaries.
        if buffer_len >= max_chars * 2 and len(buffer) == 1:
            sentences = _SENTENCE_SPLIT_RE.split(para)
            buffer, buffer_len = [], 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if buffer_len + len(sentence) + 1 > max_chars:
                    flush()
                buffer.append(sentence)
                buffer_len += len(sentence) + 1
            end_cursor = start_cursor + buffer_len
    flush()
    return chunks


def extract_pages(pdf_path: str | Path) -> list[PageRecord]:
    """Extract text page by page. pdf_page is 1-based."""
    reader = PdfReader(str(pdf_path))
    records: list[PageRecord] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - surface as a failed page
            records.append(
                PageRecord(
                    pdf_page=index,
                    text="",
                    quality=PageQuality.EMPTY.value,
                    char_count=0,
                    note=f"extraction error: {exc}",
                )
            )
            continue
        quality, note = assess_page_text(text)
        records.append(
            PageRecord(
                pdf_page=index,
                text=text,
                quality=quality,
                char_count=len(text.strip()),
                note=note,
            )
        )
    return records


def ingest_pdf(
    pdf_path: str | Path,
    *,
    license_name: str | None = None,
    license_url: str | None = None,
    title: str | None = None,
    authors: Sequence[str] | None = None,
    year: int | None = None,
    doi_or_url: str | None = None,
    download_date: str | None = None,
    visibility: str = "private",
    max_chars: int = DEFAULT_MAX_CHARS,
    source_id: str | None = None,
) -> tuple[Source, list[EvidenceChunk], IngestReport]:
    """Extract, chunk, and wrap a PDF into contract objects plus a report.

    source_id is the stable bibliographic id from the registry when provided;
    otherwise it is derived from the PDF SHA-256. chunk_id always embeds SHA so
    a content change yields a new evidence set.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    digest = sha256_file(pdf_path)
    if source_id is None:
        source_id = make_source_id(digest)
    pages = extract_pages(pdf_path)

    source = Source(
        source_id=source_id,
        title=title or pdf_path.stem,
        authors=list(authors or []),
        year=year,
        doi_or_url=doi_or_url,
        license=license_name,
        license_url=license_url,
        sha256=digest,
        visibility=visibility,
        download_date=download_date,
        local_path=str(pdf_path).replace("\\", "/"),
        title_is_filename=title is None,
    )

    all_chunks: list[EvidenceChunk] = []
    failed_pages: list[int] = []
    quality_counts: Counter[str] = Counter()
    warnings: list[str] = []
    current_section: str | None = None

    for page in pages:
        quality_counts[page.quality] += 1
        if page.quality != PageQuality.OK.value:
            failed_pages.append(page.pdf_page)
            if page.note:
                warnings.append(f"page {page.pdf_page}: {page.quality} ({page.note})")
            # Still attempt chunking if some text exists so evidence is not silently lost.
            if page.quality in {PageQuality.EMPTY.value}:
                continue
        if page.text.strip():
            current_section = detect_section(split_paragraphs(page.text), current_section)
            page_chunks = chunk_page_text(
                page.text,
                pdf_page=page.pdf_page,
                source_sha256=digest,
                section=current_section,
                max_chars=max_chars,
                source_id=source_id,
            )
            all_chunks.extend(page_chunks)

    report = IngestReport(
        source_id=source_id,
        sha256=digest,
        local_path=str(pdf_path).replace("\\", "/"),
        page_count=len(pages),
        ok_pages=quality_counts.get(PageQuality.OK.value, 0),
        failed_pages=failed_pages,
        quality_counts=dict(quality_counts),
        chunk_count=len(all_chunks),
        warnings=warnings,
        chunking_fingerprint=chunking_fingerprint(max_chars),
    )
    return source, all_chunks, report


def write_pages_jsonl(pages: Iterable[PageRecord], out_path: str | Path) -> None:
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")
