r"""Batch-import every PDF under PDF/ into a dedicated full-corpus store.

Features
- SHA-256 dedup (same bytes never imported twice)
- Resumable via state file (safe to interrupt and re-run)
- Per-file page quality report (ok / empty / scanned / garbled)
- Explicit failures for encrypted, corrupt, or unreadable files
- All sources are private; unknown bibliographic fields stay empty
- Does not move PDFs; keeps original paths

Default store: data/private/full_corpus/
State file:   data/private/full_corpus/import_state.json

Usage (PowerShell, repo root):
  $env:PYTHONPATH = "src"
  .venv\Scripts\python.exe scripts\import_all_pdfs.py
  .venv\Scripts\python.exe scripts\import_all_pdfs.py --limit 5   # smoke
  .venv\Scripts\python.exe scripts\import_all_pdfs.py --report-only

Allow-list mode (task D/E): pass the geoscience manifest produced by
scripts/inventory_local_pdfs.py; only ``decision == include`` entries are
imported and their verified bibliographic fields are attached to the source.
  .venv\Scripts\python.exe scripts\import_all_pdfs.py \
      --manifest data\private\traecode_20261014\geoscience_manifest.json \
      --store data\private\geoscience_20261014 --max-chars 1200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geochem_rag.domain import PageQuality  # noqa: E402
from geochem_rag.ingest import (  # noqa: E402
    chunking_fingerprint,
    extract_pages,
    ingest_pdf,
    sha256_file,
)
from geochem_rag.store import LocalStore  # noqa: E402

PDF_ROOT = ROOT / "PDF"
STORE_ROOT = ROOT / "data" / "private" / "full_corpus"
STATE_PATH = STORE_ROOT / "import_state.json"
REPORT_PATH = STORE_ROOT / "full_import_report.json"
# Files deliberately kept out of the retrieval corpus per user instruction
# (2026-10): geoscience literature only. Matched against the PDF path.
NON_GEOSCIENCE_PATHS = (
    "Psychology_Now_Presents",
    "Time_Special_Edition",
    "Understanding_Your_Emotions",
    "Walking in Austria",
    "Walks and Treks in the Maritime",
    "Trekking and Climbing in the Indian Himalaya",
    "Science Research Writing",
    "understanding-machine-learning-theory",
)


def is_excluded_non_geoscience(rel_path: str) -> bool:
    lowered = rel_path.lower()
    return any(key.lower() in lowered for key in NON_GEOSCIENCE_PATHS)


def load_state(state_path: Path) -> dict:
    if state_path.is_file():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "by_sha": {},  # sha256 -> result summary
        "failures": [],
    }


def save_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_extract_error(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc).lower()
    if "encrypted" in text or "password" in text or name in {"FileNotDecryptedError", "DependencyError"}:
        return "encrypted"
    if name in {"PdfReadError", "PdfStreamError", "OSError", "ValueError"} or "eof" in text or "crc" in text:
        return "corrupt_or_unreadable"
    return "extract_error"


def manifest_signature(entry: dict | None, max_chars: int) -> str:
    """Signature of the manifest decision + verified metadata for one file.

    When it changes (edited allow-list or corrected bibliographic fields) the
    SHA-based resume must reprocess the file instead of skipping it, so metadata
    updates are never lost just because the bytes are unchanged.
    """
    if entry is None:
        return "legacy"
    payload = json.dumps(
        {
            "decision": entry.get("decision"),
            "title": entry.get("title"),
            "title_is_filename": entry.get("title_is_filename"),
            "authors": entry.get("authors"),
            "year": entry.get("year"),
            "doi_or_url": entry.get("doi_or_url"),
            "chunking": chunking_fingerprint(max_chars),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def manifest_action(rel: str, manifest: dict | None) -> tuple[str, str, str | None]:
    """Classify one relative path against the allow-list.

    Returns ``(action, reason, decision)`` where action is ``import`` or
    ``exclude``. Under ``--manifest`` a file that is absent, ``exclude`` or
    ``uncertain`` never enters the store, BM25 set, vector index or neighbouring
    evidence. With ``manifest is None`` the legacy filename blacklist still
    applies (``legacy`` action is resolved by the caller).
    """
    if manifest is None:
        return "legacy", "", None
    entry = manifest.get(rel)
    if entry is None:
        return "exclude", "文件不在允许清单中（--manifest 模式）", None
    decision = entry.get("decision")
    if decision != "include":
        reason = f"清单判定为 {decision}：{entry.get('reason', '')}".strip("：")
        return "exclude", reason, decision
    return "import", "", decision


def process_one(
    pdf: Path,
    store: LocalStore,
    state: dict,
    state_path: Path,
    max_chars: int = 1200,
    manifest: dict | None = None,
) -> dict:
    rel = pdf.relative_to(ROOT).as_posix()
    entry = manifest.get(rel) if manifest is not None else None

    if manifest is not None:
        # Allow-list mode: the manifest decides eligibility.
        action, reason, decision = manifest_action(rel, manifest)
        if action == "exclude":
            digest = sha256_file(pdf)
            removed = 0
            # Clean up stale store entry from a previous import when the
            # manifest now excludes this file. Keeps the active retrieval
            # set strictly aligned with the manifest.
            existing = next(
                (s for s in store.load_sources() if s.sha256 == digest), None
            )
            if existing is not None:
                removed = store.delete_source(existing.source_id)
            return {
                "path": rel,
                "filename": pdf.name,
                "sha256": digest,
                "status": "excluded",
                "excluded_reason": reason,
                "manifest_decision": decision,
                "chunk_count": removed,
                "removed_from_store": removed > 0,
                "resumed": False,
            }
    elif is_excluded_non_geoscience(rel):
        # Never imported; the exclusion is recorded so later runs skip it too.
        return {
            "path": rel,
            "filename": pdf.name,
            "sha256": sha256_file(pdf),
            "status": "excluded",
            "excluded_reason": "非地学内容，按用户指示不纳入检索语料",
            "chunk_count": 0,
            "resumed": True,
        }
    digest = sha256_file(pdf)
    sig = manifest_signature(entry, max_chars)
    prior = state["by_sha"].get(digest)
    if (
        prior
        and prior.get("status") in {"imported", "skipped_failure"}
        and prior.get("manifest_sig", "legacy") == sig
    ):
        # Older state files did not retain page quality. Recover it from the
        # local PDF without importing/changing its already stored chunks.
        if prior.get("status") == "imported" and "quality_counts" not in prior:
            pages = extract_pages(pdf)
            counts = Counter(page.quality for page in pages)
            prior.update({
                "page_count": len(pages),
                "ok_pages": counts.get(PageQuality.OK.value, 0),
                "quality_counts": dict(counts),
                "empty_pages": counts.get(PageQuality.EMPTY.value, 0),
                "scanned_pages": counts.get(PageQuality.SCANNED.value, 0),
                "garbled_pages": counts.get(PageQuality.GARBLED.value, 0),
                "failed_pages": [page.pdf_page for page in pages if page.quality != PageQuality.OK.value],
            })
            save_state(state, state_path)
        return {
            **prior,
            "path": rel,
            "filename": pdf.name,
            "sha256": digest,
            "bytes": pdf.stat().st_size,
            "visibility": "private",
            "duplicate_of": prior.get("path") if prior.get("path") != rel else None,
            "resumed": True,
        }

    row: dict = {
        "path": rel,
        "filename": pdf.name,
        "sha256": digest,
        "bytes": pdf.stat().st_size,
        "visibility": "private",
    }
    try:
        # Probe open first so encrypted/corrupt files are labelled.
        try:
            pages_probe = extract_pages(pdf)
        except Exception as exc:  # noqa: BLE001
            kind = classify_extract_error(exc)
            row.update(
                {
                    "status": "failed",
                    "failure_kind": kind,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "page_count": None,
                    "ok_pages": 0,
                    "failed_pages": [],
                    "quality_counts": {},
                    "chunk_count": 0,
                    "source_id": None,
                }
            )
            state["failures"].append({"path": rel, "kind": kind, "error": row["error"]})
            state["by_sha"][digest] = {**row, "status": "skipped_failure", "manifest_sig": sig}
            save_state(state, state_path)
            return row

        if entry is not None:
            # Verified bibliographic fields from the allow-list; a filename-only
            # title stays a placeholder rather than masquerading as verified.
            ingest_title = None if entry.get("title_is_filename") else entry.get("title")
            ingest_authors = entry.get("authors") or []
            source, chunks, report = ingest_pdf(
                pdf,
                title=ingest_title,
                authors=ingest_authors,
                year=entry.get("year"),
                doi_or_url=entry.get("doi_or_url"),
                visibility="private",
                max_chars=max_chars,
            )
        else:
            source, chunks, report = ingest_pdf(pdf, visibility="private", max_chars=max_chars)
        result = store.import_document(source, chunks, report, chunking_fingerprint=chunking_fingerprint(max_chars))
        quality_counts = dict(report.quality_counts)
        empty_pages = quality_counts.get(PageQuality.EMPTY.value, 0)
        scanned_pages = quality_counts.get(PageQuality.SCANNED.value, 0)
        garbled_pages = quality_counts.get(PageQuality.GARBLED.value, 0)

        row.update(
            {
                "status": "imported" if result.status in {"imported", "replaced"} else result.status,
                "import_status": result.status,
                "source_id": result.source_id,
                "title": source.title,
                "title_is_filename": source.title_is_filename,
                "page_count": report.page_count,
                "ok_pages": report.ok_pages,
                "failed_pages": report.failed_pages,
                "quality_counts": quality_counts,
                "empty_pages": empty_pages,
                "scanned_pages": scanned_pages,
                "garbled_pages": garbled_pages,
                "chunk_count": result.chunk_count,
                "warnings": report.warnings[:50],
                "duplicate_of": prior.get("path") if prior else None,
            }
        )
        state["by_sha"][digest] = {**row, "status": "imported", "manifest_sig": sig}
        save_state(state, state_path)
        return row
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "status": "failed",
                "failure_kind": classify_extract_error(exc),
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "traceback": traceback.format_exc()[-800:],
            }
        )
        state["failures"].append({"path": rel, "kind": row["failure_kind"], "error": row["error"]})
        state["by_sha"][digest] = {**row, "status": "skipped_failure"}
        save_state(state, state_path)
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Import all PDFs under PDF/ (resumable)")
    parser.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all)")
    parser.add_argument("--report-only", action="store_true", help="print state summary and exit")
    parser.add_argument("--store", default=str(STORE_ROOT), help="store directory")
    parser.add_argument("--reset-state", action="store_true", help="ignore previous state (keep store)")
    parser.add_argument("--max-chars", type=int, default=1200, help="page-local chunk max chars")
    parser.add_argument(
        "--manifest",
        default=None,
        help="allow-list manifest JSON; only decision==include entries are imported",
    )
    args = parser.parse_args()

    manifest: dict | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"missing manifest: {manifest_path}", file=sys.stderr)
            return 2
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {e["relative_path"]: e for e in payload.get("entries", [])}
        print(f"manifest mode: {len(manifest)} entries from {manifest_path}", flush=True)

    store_root = Path(args.store)
    state_path = store_root / "import_state.json"
    report_path = store_root / "full_import_report.json"
    store = LocalStore(store_root)
    if args.reset_state and state_path.is_file():
        state_path.unlink()
    state = load_state(state_path)

    if args.report_only:
        done = sum(1 for v in state["by_sha"].values() if v.get("status") == "imported")
        failed = sum(1 for v in state["by_sha"].values() if v.get("status") == "skipped_failure")
        print(json.dumps({"imported": done, "failed": failed, "unique_sha": len(state["by_sha"]),
                          "chunks_in_store": store.count_chunks(),
                          "sources_in_store": len(store.load_sources())}, ensure_ascii=False, indent=2))
        return 0

    if not PDF_ROOT.is_dir():
        print(f"missing {PDF_ROOT}", file=sys.stderr)
        return 2

    pdfs = sorted(PDF_ROOT.rglob("*.pdf"), key=lambda p: str(p).lower())
    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    rows = []
    for i, pdf in enumerate(pdfs, start=1):
        row = process_one(pdf, store, state, state_path, max_chars=args.max_chars, manifest=manifest)
        rows.append(row)
        status = row.get("status")
        extra = ""
        if status == "failed":
            extra = f" kind={row.get('failure_kind')} err={row.get('error','')[:80]}"
        else:
            extra = (
                f" pages={row.get('page_count')} ok={row.get('ok_pages')} "
                f"chunks={row.get('chunk_count')} q={row.get('quality_counts')}"
            )
        tag = "RESUMED" if row.get("resumed") else status.upper()
        print(f"[{i}/{len(pdfs)}] {tag} {row['path']}{extra}", flush=True)

    # Dedup stats
    unique_sha = len(state["by_sha"])
    dup_files = len(pdfs) - len({r["sha256"] for r in rows}) if not args.limit else None
    unique_imports = {
        row["sha256"]: row for row in rows if row.get("status") == "imported"
    }
    quality_totals = Counter()
    for row in unique_imports.values():
        counts = row.get("quality_counts", {})
        if sum(counts.values()) != row.get("page_count"):
            raise ValueError(f"incomplete page quality counts for {row['sha256']}")
        quality_totals.update(counts)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract_version": "0.2.0",
        "store": args.store,
        "pdf_root": "PDF/",
        "files_seen_this_run": len(pdfs),
        "unique_sha_total": unique_sha,
        "duplicate_file_count_estimate": dup_files,
        "pages_in_unique_imports": sum(row["page_count"] for row in unique_imports.values()),
        "page_quality_counts_unique": dict(quality_totals),
        "sources_in_store": len(store.load_sources()),
        "chunks_in_store": store.count_chunks(),
        "imported_this_run": sum(1 for r in rows if not r.get("resumed") and r.get("status") in {"imported", "already_exists", "replaced", "metadata_updated"}),
        "resumed_this_run": sum(1 for r in rows if r.get("resumed")),
        "failed_this_run": sum(1 for r in rows if r.get("status") == "failed"),
        "excluded_non_geoscience": sum(1 for r in rows if r.get("status") == "excluded"),
        "manifest": args.manifest,
        "manifest_mode": manifest is not None,
        "manifest_included": sum(1 for r in rows if r.get("status") in {"imported", "already_exists", "replaced", "metadata_updated", "rechunked"}),
        "manifest_excluded": sum(
            1 for r in rows if r.get("status") == "excluded" and manifest is not None
        ),
        "manifest_excluded_paths": [
            r["path"] for r in rows if r.get("status") == "excluded" and manifest is not None
        ],
        "files": rows,
        "state_file": str(state_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_state(state, state_path)
    print(f"\nSTORE sources={summary['sources_in_store']} chunks={summary['chunks_in_store']}")
    print(f"REPORT {report_path}")
    print(f"STATE  {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
