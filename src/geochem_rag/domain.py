"""Cross-module object contracts for GeoChem-RAG Wave 0.

Field semantics are fixed by docs/PROJECT_BRIEF.md §4. Changing names or
meanings requires a migration note for CODEartsAgent and QoderCN.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Mapping

CONTRACT_VERSION = "0.2.0"


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"


class PageQuality(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    SCANNED = "scanned"
    GARBLED = "garbled"


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_:-]{1,64}$")
_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9_:-]{1,128}$")


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, name: str = "sha256") -> str:
    text = _require_str(value, name).lower()
    if not _SHA256_RE.match(text):
        raise ValueError(f"{name} must be a 64-char lowercase hex digest")
    return text


@dataclass(frozen=True)
class Source:
    """A PDF source. Unknown private bibliographic fields stay absent."""

    source_id: str
    title: str
    authors: list[str]
    year: int | None
    doi_or_url: str | None
    license: str | None
    license_url: str | None
    sha256: str
    visibility: str
    download_date: str | None
    local_path: str | None = None
    third_party_figure_notes: str | None = None
    license_notes: str | None = None
    title_is_filename: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_str(self.source_id, "source_id"))
        if not _SOURCE_ID_RE.match(self.source_id):
            raise ValueError("source_id must match [A-Za-z0-9_:-]{1,64}")
        object.__setattr__(self, "title", _require_str(self.title, "title"))
        if not isinstance(self.title_is_filename, bool):
            raise ValueError("title_is_filename must be a bool")
        if not isinstance(self.authors, list):
            raise ValueError("authors must be a list[str]")
        cleaned_authors = [_require_str(a, "author") for a in self.authors]
        object.__setattr__(self, "authors", cleaned_authors)
        if self.year is not None and (
            not isinstance(self.year, int) or isinstance(self.year, bool) or not (1800 <= self.year <= 2100)
        ):
            raise ValueError("year must be an int in 1800..2100 or None")
        for name in ("doi_or_url", "license", "license_url", "download_date"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_str(value, name))
        object.__setattr__(self, "sha256", _require_sha256(self.sha256))
        try:
            visibility = Visibility(self.visibility)
        except ValueError as exc:
            raise ValueError("visibility must be 'public' or 'private'") from exc
        object.__setattr__(self, "visibility", visibility.value)
        if visibility is Visibility.PUBLIC:
            if self.title_is_filename:
                raise ValueError("public sources require a verified title")
            if not self.authors or self.year is None or not self.doi_or_url or not self.download_date:
                raise ValueError("public sources require complete bibliographic metadata")
            if not self.license:
                raise ValueError("public sources require a license")
            if not self.license_url or not self.license_url.startswith(("http://", "https://")):
                raise ValueError("public sources require an http(s) license_url proof")
        if self.local_path is not None:
            object.__setattr__(self, "local_path", _require_str(self.local_path, "local_path"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Source:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown Source fields: {sorted(unknown)}")
        return cls(**{k: data[k] for k in allowed if k in data})


@dataclass(frozen=True)
class EvidenceChunk:
    """One page-local text chunk. Never spans PDF pages. pdf_page is 1-based."""

    chunk_id: str
    source_id: str
    pdf_page: int
    text: str
    section: str | None
    quality: str
    chunk_index: int
    sha256: str
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _require_str(self.chunk_id, "chunk_id"))
        if not _CHUNK_ID_RE.match(self.chunk_id):
            raise ValueError("chunk_id must match [A-Za-z0-9_:-]{1,128}")
        object.__setattr__(self, "source_id", _require_str(self.source_id, "source_id"))
        if not isinstance(self.pdf_page, int) or self.pdf_page < 1:
            raise ValueError("pdf_page must be an int starting at 1")
        object.__setattr__(self, "text", _require_str(self.text, "text"))
        if self.section is not None:
            object.__setattr__(self, "section", _require_str(self.section, "section"))
        try:
            quality = PageQuality(self.quality)
        except ValueError as exc:
            raise ValueError(
                "quality must be one of ok/empty/scanned/garbled"
            ) from exc
        object.__setattr__(self, "quality", quality.value)
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative int")
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "sha256"))
        if not isinstance(self.char_start, int) or not isinstance(self.char_end, int):
            raise ValueError("char_start/char_end must be ints")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("require 0 <= char_start <= char_end")
        if not isinstance(self.extra, dict):
            raise ValueError("extra must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceChunk:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown EvidenceChunk fields: {sorted(unknown)}")
        return cls(**{k: data[k] for k in allowed if k in data})


@dataclass(frozen=True)
class SearchHit:
    """One retrieval result. retrievers lists ranking paths, e.g. dense/bm25."""

    chunk_id: str
    score: float
    rank: int
    retrievers: list[str]
    source_id: str | None = None
    pdf_page: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _require_str(self.chunk_id, "chunk_id"))
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise ValueError("score must be a number")
        object.__setattr__(self, "score", float(self.score))
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be an int starting at 1")
        if not isinstance(self.retrievers, list) or not self.retrievers:
            raise ValueError("retrievers must be a non-empty list[str]")
        cleaned = [_require_str(r, "retriever") for r in self.retrievers]
        object.__setattr__(self, "retrievers", cleaned)
        if self.pdf_page is not None and (
            not isinstance(self.pdf_page, int) or self.pdf_page < 1
        ):
            raise ValueError("pdf_page, if set, must be an int starting at 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchHit:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown SearchHit fields: {sorted(unknown)}")
        return cls(**{k: data[k] for k in allowed if k in data})


@dataclass(frozen=True)
class Citation:
    """A display citation resolved from an evidence chunk id.

    ``label`` is the deterministic author-year display tag (e.g.
    ``(Luan et al., 2025)``). ``authors``/``year`` carry the verified
    bibliographic values the label is built from; for private sources with
    unknown metadata they stay empty/None and the label falls back to a
    short title with an explicit "待核实" marker. All three default so that
    older serialized citations (chunk_id..pdf_page only) still load.
    """

    chunk_id: str
    source_id: str
    title: str
    pdf_page: int
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    label: str | None = None
    doi_or_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _require_str(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "source_id", _require_str(self.source_id, "source_id"))
        object.__setattr__(self, "title", _require_str(self.title, "title"))
        if not isinstance(self.pdf_page, int) or self.pdf_page < 1:
            raise ValueError("pdf_page must be an int starting at 1")
        if not isinstance(self.authors, list):
            raise ValueError("authors must be a list[str]")
        object.__setattr__(
            self, "authors", [_require_str(a, "author") for a in self.authors]
        )
        if self.year is not None and (
            not isinstance(self.year, int)
            or isinstance(self.year, bool)
            or not (1800 <= self.year <= 2100)
        ):
            raise ValueError("year must be an int in 1800..2100 or None")
        if self.label is not None:
            object.__setattr__(self, "label", _require_str(self.label, "label"))
        if self.doi_or_url is not None:
            object.__setattr__(self, "doi_or_url", _require_str(self.doi_or_url, "doi_or_url"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Citation:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown Citation fields: {sorted(unknown)}")
        return cls(**{k: data[k] for k in allowed if k in data})


@dataclass(frozen=True)
class Answer:
    """Model answer constrained to the current evidence set."""

    status: str
    text: str
    citations: list[Citation]
    evidence: list[str]
    question: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            status = AnswerStatus(self.status)
        except ValueError as exc:
            raise ValueError(
                "status must be answered/insufficient_evidence/error"
            ) from exc
        object.__setattr__(self, "status", status.value)
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.citations, list):
            raise ValueError("citations must be a list[Citation]")
        cleaned_citations: list[Citation] = []
        for item in self.citations:
            if isinstance(item, Citation):
                cleaned_citations.append(item)
            elif isinstance(item, Mapping):
                cleaned_citations.append(Citation.from_dict(item))
            else:
                raise ValueError("citations items must be Citation or mapping")
        object.__setattr__(self, "citations", cleaned_citations)
        if not isinstance(self.evidence, list) or not all(
            isinstance(e, str) and e for e in self.evidence
        ):
            raise ValueError("evidence must be a list[str] of chunk ids")
        object.__setattr__(self, "evidence", list(self.evidence))
        cited_ids = {c.chunk_id for c in cleaned_citations}
        evidence_ids = set(self.evidence)
        if not cited_ids <= evidence_ids:
            raise ValueError("every citation chunk_id must appear in evidence")
        if status is AnswerStatus.ANSWERED and not cleaned_citations:
            raise ValueError("answered results must include at least one citation")
        if status is AnswerStatus.INSUFFICIENT_EVIDENCE and cleaned_citations:
            raise ValueError("insufficient_evidence must not carry citations")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Answer:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown Answer fields: {sorted(unknown)}")
        kwargs = {k: data[k] for k in allowed if k in data}
        return cls(**kwargs)


def make_source_id(sha256: str) -> str:
    """Stable source id derived from the PDF content hash."""
    digest = _require_sha256(sha256)
    return f"src_{digest[:16]}"


def make_chunk_id(sha256: str, pdf_page: int, chunk_index: int) -> str:
    """Stable chunk id for a deterministic page-local chunk.

    Same PDF bytes + same chunking rules => same chunk_id on re-import.
    A SHA-256 change yields new ids and invalidates the previous set.
    """
    digest = _require_sha256(sha256)
    if not isinstance(pdf_page, int) or pdf_page < 1:
        raise ValueError("pdf_page must be an int starting at 1")
    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative int")
    return f"chk_{digest[:16]}_p{pdf_page:04d}_c{chunk_index:03d}"
