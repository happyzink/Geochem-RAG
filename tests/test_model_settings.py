"""Offline tests for thinking preference mapping, client binding, persistence."""

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
from geochem_rag.providers import DeepSeekChatClient, ProviderError, SiliconFlowClient

ENV = {
    "DEEPSEEK_API_KEY": "deepseek-secret",
    "SILICONFLOW_API_KEY": "sf-secret",
    "DEEPSEEK_CHAT_MODEL": "deepseek-chat",
    "GEOCHEM_CHAT_MODEL": "Qwen/Qwen2.5-7B-Instruct",
}

DS_CAP = ms.ThinkingCapability(
    mode="effort", can_disable=True, levels=("low", "high", "max"), source="catalog"
)
SF_HIGHMAX = ms.ThinkingCapability(
    mode="effort",
    can_disable=True,
    levels=("high", "max"),
    source="known_rules",
    note="merged",
)
SF_GLM53 = ms.ThinkingCapability(
    mode="effort", can_disable=True, levels=("low", "high", "max"), source="known_rules"
)
SF_BUDGET = ms.ThinkingCapability(
    mode="budget",
    can_disable=True,
    budget_range=(128, 32768),
    budget_presets=dict(ms.SF_BUDGET_PRESETS),
    source="known_rules",
)


# ---------------------------------------------------------------------------
# Preset -> real API value
# ---------------------------------------------------------------------------


def test_three_native_levels_map_low_high_max():
    assert ms.map_effort_preset(("low", "high", "max"), "low") == "low"
    assert ms.map_effort_preset(("low", "high", "max"), "medium") == "high"
    assert ms.map_effort_preset(("low", "high", "max"), "high") == "max"
    assert ms.selectable_presets(DS_CAP) == ("low", "medium", "high")


def test_two_native_levels_disable_low_and_share_medium():
    # (high, max): 低 is not a real level; 中 maps to high, 高 to max.
    assert ms.map_effort_preset(("high", "max"), "low") is None
    assert ms.map_effort_preset(("high", "max"), "medium") == "high"
    assert ms.map_effort_preset(("high", "max"), "high") == "max"
    assert ms.selectable_presets(SF_HIGHMAX) == ("medium", "high")
    label, raw = ms.preset_api_description(SF_HIGHMAX, "medium")
    assert raw == "high"
    assert "映射" in label


def test_preset_description_marks_mapping():
    label, raw = ms.preset_api_description(DS_CAP, "medium")
    assert raw == "high"
    assert "映射" in label
    label_low, raw_low = ms.preset_api_description(DS_CAP, "low")
    assert raw_low == "low"
    assert "映射" not in label_low


# ---------------------------------------------------------------------------
# DeepSeek request bodies
# ---------------------------------------------------------------------------


def test_deepseek_enabled_three_distinct_values():
    expected = {
        "low": "low",
        "medium": "high",
        "high": "max",
    }
    for preset, native in expected.items():
        settings = ms.ChatSettings("deepseek", "deepseek-flash", True, preset)
        body, desc = ms.thinking_request("deepseek", DS_CAP, settings)
        assert body == {"thinking": {"type": "enabled"}, "reasoning_effort": native}
        assert body["thinking"]["type"] == "enabled"
        assert desc["state"] == "on"
        assert desc["api_params"] == body


def test_deepseek_disabled_sends_disabled_only():
    settings = ms.ChatSettings("deepseek", "deepseek-flash", False, "high")
    body, desc = ms.thinking_request("deepseek", DS_CAP, settings)
    assert body == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in body
    assert desc["state"] == "off"


def test_deepseek_fields_are_top_level_not_extra_body():
    settings = ms.ChatSettings("deepseek", "deepseek-flash", True, "high")
    body, _ = ms.thinking_request("deepseek", DS_CAP, settings)
    assert "thinking" in body and isinstance(body["thinking"], dict)
    assert "extra_body" not in body


# ---------------------------------------------------------------------------
# SiliconFlow request bodies
# ---------------------------------------------------------------------------


def test_siliconflow_high_max_does_not_fabricate_medium():
    medium = ms.ChatSettings("siliconflow", "m", True, "medium")
    body, _ = ms.thinking_request("siliconflow", SF_HIGHMAX, medium)
    assert body == {"enable_thinking": True, "reasoning_effort": "high"}
    top = ms.ChatSettings("siliconflow", "m", True, "high")
    body_top, _ = ms.thinking_request("siliconflow", SF_HIGHMAX, top)
    assert body_top == {"enable_thinking": True, "reasoning_effort": "max"}


def test_siliconflow_low_on_highmax_model_is_rejected():
    settings = ms.ChatSettings("siliconflow", "m", True, "low")
    with pytest.raises(ValueError):
        ms.thinking_request("siliconflow", SF_HIGHMAX, settings)


def test_siliconflow_glm53_three_real_levels():
    for preset, native in (("low", "low"), ("medium", "high"), ("high", "max")):
        body, _ = ms.thinking_request(
            "siliconflow", SF_GLM53, ms.ChatSettings("siliconflow", "m", True, preset)
        )
        assert body["reasoning_effort"] == native


def test_siliconflow_toggle_only():
    cap = ms.ThinkingCapability(mode="toggle", can_disable=True, source="known_rules")
    on = ms.ChatSettings("siliconflow", "m", True, "medium")
    body, desc = ms.thinking_request("siliconflow", cap, on)
    assert body == {"enable_thinking": True}
    off = ms.ChatSettings("siliconflow", "m", False, "medium")
    body_off, _ = ms.thinking_request("siliconflow", cap, off)
    assert body_off == {"enable_thinking": False}
    assert ms.selectable_presets(cap) == ()


def test_siliconflow_budget_presets_are_in_range_and_distinct():
    values = set()
    for preset in ("low", "medium", "high"):
        body, desc = ms.thinking_request(
            "siliconflow",
            SF_BUDGET,
            ms.ChatSettings("siliconflow", "m", True, preset),
        )
        assert set(body) == {"enable_thinking", "thinking_budget"}
        assert body["enable_thinking"] is True
        assert 128 <= body["thinking_budget"] <= 32768
        values.add(body["thinking_budget"])
        assert desc["budget"] == body["thinking_budget"]
    assert len(values) == 3


def test_fixed_on_sends_no_switch_or_level():
    settings = ms.ChatSettings("siliconflow", "deepseek-ai/DeepSeek-R1", False, "low")
    body, desc = ms.thinking_request("siliconflow", ms.FIXED_CAPABILITY, settings)
    assert body == {}
    assert desc["state"] == "fixed_on"


def test_unknown_capability_sends_nothing():
    settings = ms.ChatSettings("deepseek", "mystery", True, "high")
    body, desc = ms.thinking_request("deepseek", ms.UNKNOWN_CAPABILITY, settings)
    assert body == {}
    assert desc["state"] == "unsupported_unknown"


def test_deepseek_thinking_body_not_sent_to_siliconflow():
    settings = ms.ChatSettings("siliconflow", "m", True, "high")
    body, _ = ms.thinking_request("siliconflow", SF_GLM53, settings)
    assert "thinking" not in body
    assert body["enable_thinking"] is True


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def test_build_deepseek_client_binds_model_key_and_params():
    settings = ms.ChatSettings("deepseek", "deepseek-flash", True, "high")
    client, desc = ms.build_chat_client(settings, DS_CAP, env=ENV)
    assert isinstance(client, DeepSeekChatClient)
    assert client.config.chat_model == "deepseek-flash"
    assert client.config.api_key == "deepseek-secret"
    assert client.config.base_url == "https://api.deepseek.com"
    assert client.thinking_body["reasoning_effort"] == "max"
    assert client.thinking_active is True
    assert desc["state"] == "on"


def test_build_siliconflow_client_uses_siliconflow_key():
    settings = ms.ChatSettings(
        "siliconflow", "deepseek-ai/DeepSeek-V4-Flash", True, "high"
    )
    client, desc = ms.build_chat_client(settings, SF_HIGHMAX, env=ENV)
    assert isinstance(client, SiliconFlowClient)
    assert client.config.chat_model == "deepseek-ai/DeepSeek-V4-Flash"
    assert client.config.api_key == "sf-secret"
    assert client.config.base_url == "https://api.siliconflow.cn/v1"
    assert client.thinking_body == {"enable_thinking": True, "reasoning_effort": "max"}


def test_unknown_capability_client_sends_no_params():
    settings = ms.ChatSettings("siliconflow", "some/model", True, "high")
    client, desc = ms.build_chat_client(settings, ms.UNKNOWN_CAPABILITY, env=ENV)
    assert client.thinking_body == {}
    assert client.thinking_active is False


def test_fixed_on_client_extends_timeout_but_sends_nothing():
    settings = ms.ChatSettings("siliconflow", "deepseek-ai/DeepSeek-R1", False, "medium")
    client, _ = ms.build_chat_client(settings, ms.FIXED_CAPABILITY, env=ENV)
    assert client.thinking_body == {}
    assert client.thinking_active is True


def test_missing_deepseek_key_still_raises_at_request_time():
    settings = ms.ChatSettings("deepseek", "deepseek-flash", False, "medium")
    client, _ = ms.build_chat_client(settings, DS_CAP, env={"SILICONFLOW_API_KEY": "x"})
    assert client.config.api_key is None
    with pytest.raises(ProviderError):
        client.get_models()


# ---------------------------------------------------------------------------
# Settings file
# ---------------------------------------------------------------------------


def test_settings_roundtrip_and_contents(tmp_path):
    path = tmp_path / "local_settings.json"
    settings = ms.ChatSettings("siliconflow", "zai-org/GLM-5.3", True, "low")
    ms.save_settings_file(path, settings)
    loaded = ms.load_settings_file(path)
    assert loaded == settings
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert raw["version"] == 1
    assert "api" not in text.lower()
    assert "deepseek-secret" not in text and "sf-secret" not in text
    assert set(raw) == {
        "version",
        "chat_provider",
        "chat_model",
        "thinking_enabled",
        "thinking_level",
        "updated_at",
    }


def test_load_settings_missing_or_invalid_returns_none(tmp_path):
    assert ms.load_settings_file(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert ms.load_settings_file(bad) is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"chat_provider": "anthropic", "chat_model": "x"}),
        encoding="utf-8",
    )
    assert ms.load_settings_file(invalid) is None


def test_save_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError):
        ms.save_settings_file(
            tmp_path / "x.json",
            ms.ChatSettings("anthropic", "m", False, "medium"),
        )
    with pytest.raises(ValueError):
        ms.save_settings_file(
            tmp_path / "x.json",
            ms.ChatSettings("deepseek", "", False, "medium"),
        )


def test_resolve_precedence_saved_over_env(tmp_path):
    path = tmp_path / "local_settings.json"
    saved = ms.ChatSettings("deepseek", "deepseek-v4-pro", True, "low")
    ms.save_settings_file(path, saved)
    resolved = ms.resolve_initial_settings(path, ENV)
    assert resolved == saved


def test_resolve_falls_back_to_env_defaults(tmp_path):
    resolved = ms.resolve_initial_settings(tmp_path / "none.json", ENV)
    assert resolved.provider == "deepseek"
    assert resolved.model_id == "deepseek-chat"
    assert resolved.thinking_enabled is False
    sf_env = dict(ENV, GEOCHEM_CHAT_PROVIDER="siliconflow")
    resolved_sf = ms.resolve_initial_settings(tmp_path / "none2.json", sf_env)
    assert resolved_sf.provider == "siliconflow"
    assert resolved_sf.model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_default_settings_ignores_bad_provider_env():
    settings = ms.default_settings({"GEOCHEM_CHAT_PROVIDER": "anthropic"})
    assert settings.provider == "deepseek"
