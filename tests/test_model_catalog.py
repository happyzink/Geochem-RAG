"""Offline tests for model catalog fetching, parsing and capabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geochem_rag import model_settings as ms
from geochem_rag.providers import (
    KIND_AUTH,
    KIND_BAD_RESPONSE,
    ProviderError,
)


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _opener_recording(url_payloads: dict[str, bytes], seen: list[str]):
    def opener(request, timeout):
        seen.append(request.full_url)
        if request.full_url not in url_payloads:
            raise AssertionError(f"unexpected URL {request.full_url}")
        return FakeResponse(url_payloads[request.full_url])

    return opener


DEEPSEEK_ENV = {"DEEPSEEK_API_KEY": "deepseek-secret"}
SILICONFLOW_ENV = {"SILICONFLOW_API_KEY": "sf-secret"}


# ---------------------------------------------------------------------------
# Parsing and capabilities
# ---------------------------------------------------------------------------


def test_parse_deepseek_catalog_reads_effort_metadata():
    items = [
        {
            "id": "deepseek-flash",
            "object": "model",
            "name": "DeepSeek-V4.1-Flash",
            "context_window": 1048576,
            "input_modalities": ["text", "image"],
            "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"},
        },
        {"id": "legacy-id-only", "object": "model"},
    ]
    models = ms.parse_deepseek_catalog(items)
    assert [m.id for m in models] == ["deepseek-flash", "legacy-id-only"]
    flash, legacy = models
    assert flash.name == "DeepSeek-V4.1-Flash"
    assert flash.thinking.mode == "effort"
    assert flash.thinking.levels == ("low", "high", "max")
    assert flash.thinking.can_disable is True
    assert flash.thinking.source == "catalog"
    assert flash.context_window == 1048576
    assert flash.input_modalities == ("text", "image")
    # Missing metadata must not default to full thinking support.
    assert legacy.thinking.mode == "unknown"
    assert legacy.thinking.levels == ()


def test_parse_deepseek_ignores_unknown_effort_values():
    items = [{"id": "x", "effort": {"supported_levels": ["turbo"]}}]
    models = ms.parse_deepseek_catalog(items)
    assert models[0].thinking.mode == "unknown"


def test_parse_siliconflow_uses_known_rules_and_filters_non_chat():
    items = [
        {"id": "deepseek-ai/DeepSeek-V4-Flash"},
        {"id": "zai-org/GLM-5.3"},
        {"id": "zai-org/GLM-5.3-Flash"},
        {"id": "Qwen/Qwen3-32B"},
        {"id": "deepseek-ai/DeepSeek-R1"},
        {"id": "Qwen/Qwen2.5-7B-Instruct"},
        {"id": "deepseek-ai/DeepSeek-OCR"},
        {"id": "PaddlePaddle/PaddleOCR-VL-1.5"},
        {"id": "Qwen/Qwen3-Omni-30B-A3B-Captioner"},
        {"id": ""},
    ]
    models = ms.parse_siliconflow_catalog(items)
    by_id = {m.id: m for m in models}
    assert "deepseek-ai/DeepSeek-OCR" not in by_id
    assert "PaddlePaddle/PaddleOCR-VL-1.5" not in by_id
    assert "Qwen/Qwen3-Omni-30B-A3B-Captioner" not in by_id
    assert by_id["deepseek-ai/DeepSeek-V4-Flash"].thinking.levels == ("high", "max")
    assert by_id["zai-org/GLM-5.3"].thinking.levels == ("low", "high", "max")
    assert by_id["zai-org/GLM-5.3-Flash"].thinking.mode == "toggle"
    qwen = by_id["Qwen/Qwen3-32B"].thinking
    assert qwen.mode == "budget"
    assert qwen.budget_range == (128, 32768)
    assert by_id["deepseek-ai/DeepSeek-R1"].thinking.mode == "fixed_on"
    assert by_id["Qwen/Qwen2.5-7B-Instruct"].thinking.mode == "unknown"


# ---------------------------------------------------------------------------
# Live fetch through the real client transport with a fake opener
# ---------------------------------------------------------------------------


def test_fetch_deepseek_catalog_uses_models_endpoint_without_subtype():
    seen: list[str] = []
    payload = json.dumps(
        {"object": "list", "data": [{"id": "deepseek-flash", "name": "Flash"}]}
    ).encode("utf-8")
    opener = _opener_recording({"https://api.deepseek.com/models": payload}, seen)
    models = ms.fetch_chat_models("deepseek", env=DEEPSEEK_ENV, opener=opener)
    assert len(seen) == 1
    assert seen[0] == "https://api.deepseek.com/models"
    assert models[0].provider == "deepseek"
    assert models[0].id == "deepseek-flash"


def test_fetch_siliconflow_catalog_uses_chat_subtype():
    seen: list[str] = []
    payload = json.dumps(
        {"object": "list", "data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
    ).encode("utf-8")
    opener = _opener_recording(
        {"https://api.siliconflow.cn/v1/models?sub_type=chat": payload}, seen
    )
    models = ms.fetch_chat_models("siliconflow", env=SILICONFLOW_ENV, opener=opener)
    assert seen == ["https://api.siliconflow.cn/v1/models?sub_type=chat"]
    assert models[0].provider == "siliconflow"


def test_fetch_embedding_catalog_uses_embedding_subtype():
    seen: list[str] = []
    payload = json.dumps({"data": [{"id": "BAAI/bge-m3"}]}).encode("utf-8")
    opener = _opener_recording(
        {"https://api.siliconflow.cn/v1/models?sub_type=embedding": payload}, seen
    )
    models = ms.fetch_embedding_models(env=SILICONFLOW_ENV, opener=opener)
    assert models[0].id == "BAAI/bge-m3"
    assert models[0].thinking.mode == "unknown"


def test_empty_catalog_is_an_error_not_success():
    payload = json.dumps({"data": []}).encode("utf-8")
    opener = _opener_recording(
        {"https://api.deepseek.com/models": payload}, []
    )
    with pytest.raises(ProviderError) as excinfo:
        ms.fetch_chat_models("deepseek", env=DEEPSEEK_ENV, opener=opener)
    assert excinfo.value.kind == KIND_BAD_RESPONSE


def test_invalid_json_catalog_is_bad_response():
    opener = _opener_recording(
        {"https://api.deepseek.com/models": b"not json"}, []
    )
    with pytest.raises(ProviderError) as excinfo:
        ms.fetch_chat_models("deepseek", env=DEEPSEEK_ENV, opener=opener)
    assert excinfo.value.kind == KIND_BAD_RESPONSE


def test_auth_failure_is_classified():
    import urllib.error

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    with pytest.raises(ProviderError) as excinfo:
        ms.fetch_chat_models("deepseek", env=DEEPSEEK_ENV, opener=opener)
    assert excinfo.value.kind == KIND_AUTH
    assert "鉴权失败" in ms.provider_error_zh(excinfo.value)


def test_missing_key_is_not_configured():
    with pytest.raises(ProviderError) as excinfo:
        ms.fetch_chat_models("deepseek", env={}, opener=lambda *a, **k: None)
    assert excinfo.value.kind == "not_configured"
    assert "密钥" in ms.provider_error_zh(excinfo.value)


# ---------------------------------------------------------------------------
# Cache isolation
# ---------------------------------------------------------------------------


def test_fingerprint_isolates_provider_endpoint_and_credential():
    cfg_a = ms.chat_config("deepseek", {"DEEPSEEK_API_KEY": "k1"})
    cfg_b = ms.chat_config("deepseek", {"DEEPSEEK_API_KEY": "k2"})
    f1 = ms.catalog_fingerprint("deepseek", cfg_a)
    f2 = ms.catalog_fingerprint("deepseek", cfg_b)
    f3 = ms.catalog_fingerprint("siliconflow", ms.chat_config("siliconflow", {"SILICONFLOW_API_KEY": "k1"}))
    assert f1 != f2 != f3
    # Never reversible into the key.
    assert "k1" not in f1 and "k2" not in f2
    assert len(f1) == 12


def test_chat_config_uses_independent_keys_and_urls():
    env = {
        "DEEPSEEK_API_KEY": "deepseek-secret",
        "SILICONFLOW_API_KEY": "sf-secret",
    }
    deepseek = ms.chat_config("deepseek", env)
    siliconflow = ms.chat_config("siliconflow", env)
    assert deepseek.api_key == "deepseek-secret"
    assert deepseek.base_url == "https://api.deepseek.com"
    assert deepseek.credential_name == "DEEPSEEK_API_KEY"
    assert siliconflow.api_key == "sf-secret"
    assert siliconflow.base_url == "https://api.siliconflow.cn/v1"
    assert siliconflow.credential_name == "SILICONFLOW_API_KEY"


def test_chat_config_rejects_unknown_provider():
    with pytest.raises(ValueError):
        ms.chat_config("anthropic")
