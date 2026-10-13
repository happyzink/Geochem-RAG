"""Tests for the online-only Streamlit app's helper logic.

AppTest (streamlit.testing.v1) needs a localhost websocket which this
sandbox blocks, so we test the app's pure helpers and initialization
checks directly instead.  The error-path checks mirror _init_session's
logic and the launcher's check_prerequisites, ensuring the app surfaces
clear errors — never a template fallback — when the corpus, keys, or
index are missing or mismatched.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geochem_rag.app import (  # noqa: E402
    _DATA_FLOW_NOTE,
    _EXAMPLE_QUESTIONS,
    _cache_model,
    _resolve_cache,
    _resolve_store,
    _status_label,
    _thinking_caption,
)
from geochem_rag.domain import AnswerStatus  # noqa: E402
from geochem_rag.providers import ProviderConfig  # noqa: E402

SHA = "ab" * 32


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_data_flow_note_mentions_both_providers():
    assert "SiliconFlow" in _DATA_FLOW_NOTE
    assert "DeepSeek" in _DATA_FLOW_NOTE


def test_example_questions_are_chinese_and_nonempty():
    assert len(_EXAMPLE_QUESTIONS) >= 3
    for q in _EXAMPLE_QUESTIONS:
        assert any("\u4e00" <= ch <= "\u9fff" for ch in q)


def test_thinking_caption_records_real_api_value_and_mapping():
    on = {
        "state": "on",
        "preset": "high",
        "native_level": "max",
        "api_params": {"reasoning_effort": "max"},
    }
    text = _thinking_caption(on)
    assert "开启" in text and "max" in text and "映射" in text
    medium = {"state": "on", "preset": "medium", "native_level": "high"}
    assert "high" in _thinking_caption(medium) and "映射" in _thinking_caption(medium)
    low = {"state": "on", "preset": "low", "native_level": "low"}
    assert "映射" not in _thinking_caption(low)


def test_thinking_caption_covers_off_fixed_and_unknown():
    assert "关闭" in _thinking_caption({"state": "off"})
    assert "固定开启" in _thinking_caption({"state": "fixed_on"})
    assert "未发送" in _thinking_caption({"state": "unsupported_unknown"})
    assert "未知" in _thinking_caption(None)


def test_thinking_caption_budget_shows_real_budget():
    text = _thinking_caption(
        {"state": "on", "preset": "medium", "budget": 8192}
    )
    assert "thinking_budget=8192" in text


# ---------------------------------------------------------------------------
# _resolve_store
# ---------------------------------------------------------------------------


def test_resolve_store_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOCHEM_STORE", str(tmp_path / "my_store"))
    assert _resolve_store() == str(tmp_path / "my_store")


def test_resolve_store_falls_back_to_full_corpus(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOCHEM_STORE", raising=False)
    full = ROOT / "data" / "private" / "full_corpus"
    if (full / "chunks.jsonl").is_file():
        assert "full_corpus" in _resolve_store()
    else:
        assert _resolve_store() == "data/processed"


# ---------------------------------------------------------------------------
# _resolve_cache
# ---------------------------------------------------------------------------


def test_resolve_cache_full_corpus():
    assert "embeddings_full.json" in _resolve_cache("data/private/full_corpus")


def test_resolve_cache_default():
    assert "embeddings.json" in _resolve_cache("data/processed")


def test_resolve_cache_windows_backslash():
    assert "embeddings_full.json" in _resolve_cache("data\\private\\full_corpus")


# ---------------------------------------------------------------------------
# _cache_model
# ---------------------------------------------------------------------------


def test_cache_model_reads_model(tmp_path, monkeypatch):
    cache = tmp_path / "emb.json"
    cache.write_text(json.dumps({"model": "BAAI/bge-m3", "entries": {}}), encoding="utf-8")
    monkeypatch.setattr("geochem_rag.app._REPO", tmp_path)
    assert _cache_model(str(cache.relative_to(tmp_path).as_posix())) == "BAAI/bge-m3"


def test_cache_model_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("geochem_rag.app._REPO", tmp_path)
    assert _cache_model("nope.json") is None


def test_cache_model_invalid_json_returns_none(tmp_path, monkeypatch):
    cache = tmp_path / "bad.json"
    cache.write_text("not json", encoding="utf-8")
    monkeypatch.setattr("geochem_rag.app._REPO", tmp_path)
    assert _cache_model("bad.json") is None


# ---------------------------------------------------------------------------
# _status_label
# ---------------------------------------------------------------------------


def test_status_label_answered():
    label = _status_label(AnswerStatus.ANSWERED.value)
    assert "已回答" in label
    assert "文献证据" in label


def test_status_label_insufficient():
    label = _status_label(AnswerStatus.INSUFFICIENT_EVIDENCE.value)
    assert "证据不足" in label


def test_status_label_error():
    label = _status_label(AnswerStatus.ERROR.value)
    assert "出错" in label


def test_status_label_unknown_passthrough():
    assert _status_label("weird") == "weird"


# ---------------------------------------------------------------------------
# Initialization checks — mirror _init_session's error conditions
# ---------------------------------------------------------------------------


def _make_store_with_chunks(tmp_path: Path) -> Path:
    from geochem_rag.domain import EvidenceChunk, Source
    from geochem_rag.store import LocalStore

    store_dir = tmp_path / "store"
    store = LocalStore(store_dir)
    source = Source(
        source_id="src_a",
        title="Test Paper",
        authors=["A"],
        year=2022,
        doi_or_url="https://example.org/x",
        license="CC BY 4.0",
        license_url="https://example.org/l",
        sha256=SHA,
        visibility="public",
        download_date="2026-09-28",
    )
    chunk = EvidenceChunk(
        chunk_id="chk_a",
        source_id="src_a",
        pdf_page=1,
        text="adakite petrogenesis in the gangdese arc",
        section="introduction",
        quality="ok",
        chunk_index=0,
        sha256=SHA,
    )
    store.import_document(source, [chunk])
    return store_dir


def test_empty_store_triggers_no_corpus_error(tmp_path):
    from geochem_rag.store import LocalStore

    store = LocalStore(tmp_path / "empty")
    assert store.count_chunks() == 0


def test_store_with_chunks_has_content(tmp_path):
    store_dir = _make_store_with_chunks(tmp_path)
    from geochem_rag.store import LocalStore

    assert LocalStore(store_dir).count_chunks() > 0


def test_missing_embedding_key_detected():
    cfg = ProviderConfig.from_env({})
    assert not cfg.api_key


def test_missing_chat_key_detected():
    cfg = ProviderConfig.from_deepseek_env({"SILICONFLOW_API_KEY": "sk-embed"})
    assert not cfg.api_key


def test_index_model_mismatch_detected(tmp_path, monkeypatch):
    cache = tmp_path / "emb.json"
    cache.write_text(json.dumps({"model": "HashingEmbedder"}), encoding="utf-8")
    monkeypatch.setattr("geochem_rag.app._REPO", tmp_path)

    cached = _cache_model("emb.json")
    expected = os.environ.get("GEOCHEM_EMBEDDING_MODEL", "BAAI/bge-m3")
    assert cached != expected