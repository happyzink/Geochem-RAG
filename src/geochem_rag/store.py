"""Local persistence with SHA-256 dedup and visibility filtering.

Files live under data/processed/ (gitignored):
- sources.jsonl   one Source per line
- chunks.jsonl    one EvidenceChunk per line
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .domain import EvidenceChunk, Source
from .ingest import IngestReport


@dataclass
class ImportResult:
    status: str  # imported | already_exists | metadata_updated | replaced
    source_id: str
    sha256: str
    chunk_count: int
    report: IngestReport | None = None


class LocalStore:
    def __init__(self, root: str | Path = "data/processed") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources_path = self.root / "sources.jsonl"
        self.chunks_path = self.root / "chunks.jsonl"
        self.config_path = self.root / "chunking.json"

    # --- chunking fingerprint --------------------------------------
    def _read_chunking_config(self) -> dict[str, str]:
        if not self.config_path.exists():
            return {}
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_chunking_config(self, config: dict[str, str]) -> None:
        tmp = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
        tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)

    def _fingerprint_changed(self, source_id: str, fingerprint: str) -> bool:
        return self._read_chunking_config().get(source_id) != fingerprint

    def _record_fingerprint(self, source_id: str, fingerprint: str) -> None:
        config = self._read_chunking_config()
        if config.get(source_id) == fingerprint:
            return
        config[source_id] = fingerprint
        self._write_chunking_config(config)

    # --- IO helpers -------------------------------------------------
    def _read_jsonl(self, path: Path) -> Iterator[dict]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _write_jsonl(self, path: Path, rows: Iterable[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)

    # --- source / chunk access -------------------------------------
    def load_sources(self) -> list[Source]:
        return [Source.from_dict(row) for row in self._read_jsonl(self.sources_path)]

    def load_chunks(self) -> list[EvidenceChunk]:
        return [EvidenceChunk.from_dict(row) for row in self._read_jsonl(self.chunks_path)]

    def get_source(self, source_id: str) -> Source | None:
        for source in self.load_sources():
            if source.source_id == source_id:
                return source
        return None

    def get_source_by_sha(self, sha256: str) -> Source | None:
        digest = sha256.lower()
        for source in self.load_sources():
            if source.sha256 == digest:
                return source
        return None

    def list_sources(self, visibility: str | None = None) -> list[Source]:
        sources = self.load_sources()
        if visibility is None:
            return sources
        return [s for s in sources if s.visibility == visibility]

    def get_chunks(self, source_id: str | None = None) -> list[EvidenceChunk]:
        chunks = self.load_chunks()
        if source_id is None:
            return chunks
        return [c for c in chunks if c.source_id == source_id]

    def find_chunk(self, chunk_id: str) -> EvidenceChunk | None:
        for chunk in self.load_chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def count_chunks(self, source_id: str | None = None) -> int:
        return len(self.get_chunks(source_id))

    # --- write paths ------------------------------------------------
    def upsert_source(self, source: Source) -> None:
        sources = [s for s in self.load_sources() if s.source_id != source.source_id]
        sources.append(source)
        self._write_jsonl(self.sources_path, [s.to_dict() for s in sources])

    def replace_chunks(self, source_id: str, chunks: Sequence[EvidenceChunk]) -> None:
        for chunk in chunks:
            if chunk.source_id != source_id:
                raise ValueError("chunk.source_id must match the target source_id")
        kept = [c for c in self.load_chunks() if c.source_id != source_id]
        self._write_jsonl(
            self.chunks_path,
            [c.to_dict() for c in kept] + [c.to_dict() for c in chunks],
        )

    def import_document(
        self,
        source: Source,
        chunks: Sequence[EvidenceChunk],
        report: IngestReport | None = None,
        *,
        chunking_fingerprint: str | None = None,
    ) -> ImportResult:
        """Import with SHA-256 dedup.

        Same SHA-256 already stored -> status 'already_exists' (no duplicate chunks).
        Same source_id with a different SHA -> old chunks invalidated ('replaced').
        ``chunking_fingerprint`` marks the chunking-rule version; when it differs
        from the stored marker for the same bytes, the evidence set is re-chunked
        ('rechunked') instead of being silently skipped by SHA dedup.
        """
        existing = self.get_source_by_sha(source.sha256)
        if existing is not None:
            # Identical bytes need no new chunks. The same source may still gain
            # corrected bibliographic metadata on a later manual import.
            status = "already_exists"
            if existing.source_id == source.source_id and existing != source:
                self.upsert_source(source)
                status = "metadata_updated"
            if (
                chunking_fingerprint is not None
                and self._fingerprint_changed(existing.source_id, chunking_fingerprint)
            ):
                # Same bytes, new chunking rules: rebuild the evidence set rather
                # than silently keeping stale chunks.
                self.replace_chunks(existing.source_id, chunks)
                self._record_fingerprint(existing.source_id, chunking_fingerprint)
                status = "rechunked"
            return ImportResult(
                status=status,
                source_id=existing.source_id,
                sha256=existing.sha256,
                chunk_count=self.count_chunks(existing.source_id),
                report=report,
            )

        prior = self.get_source(source.source_id)
        status = "replaced" if prior is not None else "imported"
        if prior is not None and prior.sha256 != source.sha256:
            # SHA change: drop the previous evidence set for this source.
            self.delete_source(source.source_id)

        self.upsert_source(source)
        self.replace_chunks(source.source_id, chunks)
        if chunking_fingerprint is not None:
            self._record_fingerprint(source.source_id, chunking_fingerprint)
        return ImportResult(
            status=status,
            source_id=source.source_id,
            sha256=source.sha256,
            chunk_count=len(chunks),
            report=report,
        )

    def delete_source(self, source_id: str) -> int:
        """Remove a source and all of its chunks. Returns chunks removed."""
        before = self.count_chunks(source_id)
        sources = [s for s in self.load_sources() if s.source_id != source_id]
        chunks = [c for c in self.load_chunks() if c.source_id != source_id]
        self._write_jsonl(self.sources_path, [s.to_dict() for s in sources])
        self._write_jsonl(self.chunks_path, [c.to_dict() for c in chunks])
        return before

    def pages_for_source(self, source_id: str) -> dict[int, list[EvidenceChunk]]:
        grouped: dict[int, list[EvidenceChunk]] = defaultdict(list)
        for chunk in self.get_chunks(source_id):
            grouped[chunk.pdf_page].append(chunk)
        return dict(sorted(grouped.items()))
