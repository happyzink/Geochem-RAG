r"""Build the full-corpus embedding index (resumable, API-error aware).

Store:  data/private/full_corpus/
Cache:  data/index/embeddings_full.json

Sends chunk texts to SiliconFlow embeddings when keys are present.
Without keys/network the script reports how many vectors are still missing
and exits non-zero — it never pretends the index is complete.

Usage (PowerShell, repo root):
  # load .env first (or use scripts/run_streamlit_online.py's loader)
  $env:PYTHONPATH = "src"
  .venv\Scripts\python.exe scripts/build_full_index.py
  .venv\Scripts\python.exe scripts/build_full_index.py --window 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=str(ROOT / "data" / "private" / "full_corpus"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "index" / "embeddings_full.json"))
    parser.add_argument("--window", type=int, default=32, help="embedding batch size")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    load_env()
    from geochem_rag.providers import ProviderConfig, SiliconFlowClient, ProviderError
    from geochem_rag.retrieval import EmbeddingCache, _model_name
    from geochem_rag.store import LocalStore

    config = ProviderConfig.from_env()
    store = LocalStore(args.store)
    all_chunks = store.get_chunks()
    chunks = [c for c in all_chunks if c.quality == "ok" and len(c.text.strip()) >= 40]
    skipped = len(all_chunks) - len(chunks)
    print(f"store={args.store}")
    print(f"chunks={len(chunks)} (skipped {skipped} non-ok/too-short) sources={len(store.load_sources())}")

    if not chunks:
        print("NO_CHUNKS: run scripts/import_all_pdfs.py first", file=sys.stderr)
        return 2

    # Quality breakdown of what we will embed.
    from collections import Counter

    q = Counter(c.quality for c in chunks)
    print(f"chunk_quality={dict(q)}")

    if not config.api_key:
        print(
            "INCOMPLETE: SILICONFLOW_API_KEY missing. "
            f"{len(chunks)} chunks still need embeddings. "
            "Dense retrieval will not be available until indexing finishes.",
            file=sys.stderr,
        )
        return 3

    client = SiliconFlowClient(config, embedding_batch_size=args.window)
    model = _model_name(client)
    cache = EmbeddingCache(args.cache, model)

    # How many are already cached?
    have = 0
    missing_idx = []
    for i, chunk in enumerate(chunks):
        entry = cache._data.get(chunk.chunk_id)
        if entry and entry.get("text_hash") == cache._text_hash(chunk.text):
            have += 1
        else:
            missing_idx.append(i)
    print(f"cache={args.cache} model={model} already={have} missing={len(missing_idx)}")

    if not missing_idx:
        print(f"INDEX COMPLETE: {have}/{len(chunks)} vectors")
        return 0

    batch = max(1, args.window)
    done = 0
    t0 = time.time()
    for start in range(0, len(missing_idx), batch):
        window = missing_idx[start : start + batch]
        texts = [chunks[i].text for i in window]
        ok = False
        last_err = None
        for attempt in range(1, args.max_retries + 1):
            try:
                vectors = client.embed(texts)
                for i, vec in zip(window, vectors):
                    chunk = chunks[i]
                    cache._data[chunk.chunk_id] = {
                        "text_hash": cache._text_hash(chunk.text),
                        "vector": vec,
                    }
                cache._dirty = True
                cache.save()
                done += len(window)
                ok = True
                break
            except ProviderError as exc:
                last_err = exc
                wait = 2.0 * attempt
                print(f"  batch@{start} attempt {attempt} failed ({exc.kind}); retry in {wait}s")
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"  batch@{start} unexpected {type(exc).__name__}: {exc}")
                break
        if not ok:
            print(
                f"INCOMPLETE: stopped after {done} newly embedded. "
                f"Remaining {len(missing_idx) - done}. Last error: {last_err}",
                file=sys.stderr,
            )
            return 4
        if (done % (batch * 5)) == 0 or start + batch >= len(missing_idx):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            print(f"  progress {done}/{len(missing_idx)} ({rate:.1f} vec/s)", flush=True)

    total_cached = sum(
        1
        for c in chunks
        if c.chunk_id in cache._data and cache._data[c.chunk_id].get("text_hash") == cache._text_hash(c.text)
    )
    print(f"INDEX COMPLETE: {total_cached}/{len(chunks)} vectors model={model}")
    print(f"CACHE {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
