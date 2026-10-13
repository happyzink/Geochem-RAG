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
"""
from __future__ import annotations

import argparse
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


def process_one(pdf: Path, store: LocalStore, state: dict, state_path: Path, max_chars: int = 1200) -> dict:
    rel = pdf.relative_to(ROOT).as_posix()
    digest = sha256_file(pdf)
    prior = state["by_sha"].get(digest)
    if prior and prior.get("status") in {"imported", "skipped_failure"}:
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
            state["by_sha"][digest] = {**row, "status": "skipped_failure"}
            save_state(state, state_path)
            return row

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
        state["by_sha"][digest] = {**row, "status": "imported"}
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
    args = parser.parse_args()

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
        row = process_one(pdf, store, state, state_path, max_chars=args.max_chars)
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
