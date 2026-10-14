"""Offline tests for the online-only startup checks.

These verify that the online launcher reports missing keys, missing corpus,
missing/mismatched index and env-vs-.env precedence -- and never silently falls
back to template providers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_streamlit_online.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_streamlit_online", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load()


def _env(**extra) -> dict[str, str]:
    return {"SILICONFLOW_API_KEY": "sk-embed", "DEEPSEEK_API_KEY": "sk-chat", **extra}


def _store_with_chunks(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    (store / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    return store


def test_missing_keys_are_reported():
    errors = mod.check_prerequisites("data/processed", "data/index/embeddings.json", root=ROOT, env={})
    assert any("SILICONFLOW_API_KEY" in e for e in errors)
    assert any("DEEPSEEK_API_KEY" in e for e in errors)


def test_missing_corpus_is_reported(tmp_path: Path):
    errors = mod.check_prerequisites(str(tmp_path / "nonexistent"), "data/index/embeddings.json", root=ROOT, env=_env())
    assert any("无语料" in e for e in errors)


def test_missing_index_is_reported(tmp_path: Path):
    store = _store_with_chunks(tmp_path)
    errors = mod.check_prerequisites(str(store), str(tmp_path / "nope.json"), root=ROOT, env=_env())
    assert any("索引不完整" in e for e in errors)


def test_cache_model_mismatch_is_reported(tmp_path: Path):
    store = _store_with_chunks(tmp_path)
    cache = tmp_path / "emb.json"
    cache.write_text('{"model": "HashingEmbedder", "entries": {}}', encoding="utf-8")
    errors = mod.check_prerequisites(str(store), str(cache), root=ROOT, env=_env())
    assert any("索引模型不匹配" in e for e in errors)


def test_prerequisites_pass_when_ready(tmp_path: Path):
    import hashlib
    import json

    chunk_text = "Adakite forms by partial melting of thickened lower crust."
    text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]

    store = tmp_path / "store"
    store.mkdir()
    chunk = {
        "chunk_id": "chk_p0001_c000",
        "source_id": "src_test",
        "pdf_page": 1,
        "text": chunk_text,
        "section": "intro",
        "quality": "ok",
        "chunk_index": 0,
        "sha256": "ab" * 32,
    }
    (store / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    cache = tmp_path / "emb.json"
    cache.write_text(
        json.dumps(
            {
                "model": "BAAI/bge-m3",
                "entries": {
                    "chk_p0001_c000": {
                        "text_hash": text_hash,
                        "vector": [0.1, 0.2, 0.3],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    errors = mod.check_prerequisites(str(store), str(cache), root=ROOT, env=_env())
    assert errors == []


def test_load_env_does_not_override_environment(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("SILICONFLOW_API_KEY=from_dotenv\nOTHER_FLAG=1\n", encoding="utf-8")
    into = {"SILICONFLOW_API_KEY": "from_environment"}
    loaded = mod.load_env(env_path, into=into)
    assert into["SILICONFLOW_API_KEY"] == "from_environment"  # environment wins
    assert into["OTHER_FLAG"] == "1"
    assert "SILICONFLOW_API_KEY" not in loaded


def test_load_env_missing_file_is_noop(tmp_path: Path):
    loaded = mod.load_env(tmp_path / "does_not_exist.env", into={})
    assert loaded == {}


def test_resolve_cache_selects_full_corpus_file():
    assert "embeddings_full.json" in mod.resolve_cache("data/private/full_corpus", env={})
    assert "embeddings.json" in mod.resolve_cache("data/processed", env={})
    assert "embeddings_geoscience_20261014.json" in mod.resolve_cache(
        "data/private/geoscience_20261014", env={}
    )


def test_resolve_store_prefers_geoscience_when_present(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEOCHEM_STORE", raising=False)
    geo = ROOT / "data" / "private" / "geoscience_20261014"
    full = ROOT / "data" / "private" / "full_corpus"
    resolved = mod.resolve_store(env={})
    if (geo / "chunks.jsonl").is_file():
        assert "geoscience_20261014" in resolved
    elif (full / "chunks.jsonl").is_file():
        assert "full_corpus" in resolved
    else:
        assert resolved == "data/processed"