"""Model catalog, thinking capability rules and persisted chat settings.

Scope (see docs/agents/TraeCode模型选择与思考设置.md):
- Two chat providers only: DeepSeek (official endpoint) and SiliconFlow
  (OpenAI-compatible endpoint). Each uses its own base URL and its own API
  key; keys are always read from environment variables, never stored here.
- Capabilities come first from explicit catalog metadata (DeepSeek returns
  ``effort.supported_levels``). SiliconFlow's catalog gives only ids, so a
  small explicit rule table records models whose controls were confirmed in
  official SiliconFlow documentation checked on 2026-10-12. Everything else
  stays "unknown": the UI disables thinking controls and no unverified
  parameter is sent.
- App presets 低/中/高 are mapped to *real* API values per model. Mapped or
  merged levels are shown as such; we never invent an independent "medium"
  for providers that only accept high/max.

The settings file contains provider/model/thinking preferences only — never
API keys, questions, evidence or call logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .providers import (
    DeepSeekChatClient,
    KIND_AUTH,
    KIND_BAD_REQUEST,
    KIND_BAD_RESPONSE,
    KIND_NETWORK,
    KIND_NOT_CONFIGURED,
    KIND_RATE_LIMIT,
    KIND_SERVER,
    KIND_TIMEOUT,
    ProviderConfig,
    ProviderError,
    SiliconFlowClient,
)

PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_SILICONFLOW = "siliconflow"
CHAT_PROVIDERS = (PROVIDER_DEEPSEEK, PROVIDER_SILICONFLOW)

LEVEL_LOW = "low"
LEVEL_MEDIUM = "medium"
LEVEL_HIGH = "high"
APP_PRESETS = (LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH)

# Native API effort values in ascending order.
NATIVE_ORDER = ("low", "medium", "high", "max")
NATIVE_LABELS = {"low": "低", "medium": "中", "high": "高", "max": "最高"}

# SiliconFlow thinking_budget is documented as 128..32768 for reasoning
# models. Project presets map to concrete, in-range budgets and are labelled
# "预算预设" in the UI; they are never presented as native effort levels.
SF_BUDGET_MIN = 128
SF_BUDGET_MAX = 32768
SF_BUDGET_PRESETS = {"low": 1024, "medium": 8192, "high": 32768}

# --- Explicit, documentation-backed SiliconFlow capability table ----------
# Source: SiliconFlow chat-completions API reference and official model
# announcements checked on 2026-10-12. Only exact catalog ids are listed;
# unknown ids get unknown capabilities instead of name-based guessing.

# reasoning_effort accepting only high/max at the API (low/medium inputs are
# merged into high, so they are not independent levels).
SF_EFFORT_HIGH_MAX: frozenset[str] = frozenset(
    {
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        # The API doc names the "Pro/" id; the catalog currently also lists
        # the non-Pro id. Both spellings are recorded explicitly.
        "Pro/deepseek-ai/DeepSeek-V4",
        "zai-org/GLM-5.2",
        "Pro/zai-org/GLM-5.2",
    }
)
# Official SiliconFlow announcement (2026-09-02): GLM-5.3 deployment accepts
# low/high/max reasoning_effort.
SF_EFFORT_LOW_HIGH_MAX: frozenset[str] = frozenset({"zai-org/GLM-5.3"})
# Model center: GLM-5.3-Flash supports thinking / non-thinking dual mode; no
# graded effort levels documented -> on/off switch only.
SF_TOGGLE_ONLY: frozenset[str] = frozenset({"zai-org/GLM-5.3-Flash"})
# Fixed-reasoning models: thinking cannot be switched off and no effort
# control is documented.
SF_FIXED_ON: frozenset[str] = frozenset(
    {
        "deepseek-ai/DeepSeek-R1",
        "Pro/deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    }
)
# Qwen3 dense models on SiliconFlow accept enable_thinking and
# thinking_budget (128..32768). Exact ids from the 2026-10-12 catalog.
SF_BUDGET_MODELS: frozenset[str] = frozenset(
    {"Qwen/Qwen3-32B", "Qwen/Qwen3-14B", "Qwen/Qwen3-8B"}
)
# Returned under sub_type=chat but not usable for text chat: OCR / image
# captioner models. Kept as an explicit denylist, not name-based filtering.
SF_NON_CHAT_IDS: frozenset[str] = frozenset(
    {
        "PaddlePaddle/PaddleOCR-VL-1.5",
        "deepseek-ai/DeepSeek-OCR",
        "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    }
)

SETTINGS_VERSION = 1


# ---------------------------------------------------------------------------
# Capability and model data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingCapability:
    """What a specific model really supports.

    mode values:
      ``effort``   - switchable + graded native effort levels
      ``budget``   - switchable + graded via thinking_budget presets
      ``toggle``   - switchable, no levels
      ``fixed_on`` - reasoning always on, cannot be disabled
      ``unknown``  - not verified; send nothing
    """

    mode: str = "unknown"
    can_disable: bool = False
    levels: tuple[str, ...] = ()
    budget_range: tuple[int, int] | None = None
    budget_presets: Mapping[str, int] = field(default_factory=dict)
    default_level: str | None = None
    source: str = "unknown"  # catalog | known_rules | unknown
    note: str = ""

    @property
    def known(self) -> bool:
        return self.mode != "unknown"


UNKNOWN_CAPABILITY = ThinkingCapability(mode="unknown", note="能力未知，不发送思考参数")
FIXED_CAPABILITY = ThinkingCapability(
    mode="fixed_on",
    can_disable=False,
    source="known_rules",
    note="该模型思考固定开启，无法关闭",
)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    provider: str
    thinking: ThinkingCapability = UNKNOWN_CAPABILITY
    context_window: int | None = None
    input_modalities: tuple[str, ...] = ()
    owned_by: str = ""

    @property
    def display(self) -> str:
        return f"{self.name}（{self.id}）" if self.name and self.name != self.id else self.id


@dataclass
class CatalogState:
    """Session-scoped catalog, isolated per provider/endpoint/credential."""

    provider: str
    fingerprint: str
    models: list[ModelInfo] = field(default_factory=list)
    fetched_at: str | None = None
    error: str | None = None  # Chinese text of the latest fetch failure
    stale: bool = False
    loaded: bool = False


@dataclass(frozen=True)
class ChatSettings:
    provider: str = PROVIDER_DEEPSEEK
    model_id: str = ""
    thinking_enabled: bool = False
    # App preset, one of low/medium/high. The native API value is derived via
    # the selected model's capability — raw native values are never stored as
    # a cross-provider preference.
    level: str = LEVEL_MEDIUM


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------


def _as_clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _deepseek_capability(item: Mapping[str, Any]) -> ThinkingCapability:
    effort = item.get("effort")
    if not isinstance(effort, dict):
        return ThinkingCapability(
            mode="unknown",
            source="catalog",
            note="目录未提供思考档位元数据，不发送未验证参数",
        )
    raw_levels = effort.get("supported_levels")
    levels = tuple(
        level
        for level in NATIVE_ORDER
        if isinstance(raw_levels, list) and level in raw_levels
    )
    default = effort.get("default_level")
    if default not in NATIVE_ORDER:
        default = None
    if not levels:
        return ThinkingCapability(
            mode="unknown",
            source="catalog",
            note="目录未提供思考档位元数据，不发送未验证参数",
        )
    return ThinkingCapability(
        mode="effort",
        can_disable=True,
        levels=levels,
        default_level=default,
        source="catalog",
    )


def _siliconflow_capability(model_id: str) -> ThinkingCapability:
    if model_id in SF_FIXED_ON:
        return ThinkingCapability(
            mode="fixed_on",
            can_disable=False,
            source="known_rules",
            note="该模型思考固定开启，无法关闭",
        )
    if model_id in SF_EFFORT_HIGH_MAX:
        return ThinkingCapability(
            mode="effort",
            can_disable=True,
            levels=("high", "max"),
            source="known_rules",
            note="SiliconFlow 实际仅接受 high/max；low、medium 会被合并为 high",
        )
    if model_id in SF_EFFORT_LOW_HIGH_MAX:
        return ThinkingCapability(
            mode="effort",
            can_disable=True,
            levels=("low", "high", "max"),
            source="known_rules",
        )
    if model_id in SF_BUDGET_MODELS:
        return ThinkingCapability(
            mode="budget",
            can_disable=True,
            budget_range=(SF_BUDGET_MIN, SF_BUDGET_MAX),
            budget_presets=dict(SF_BUDGET_PRESETS),
            source="known_rules",
            note="档位通过 thinking_budget 预算预设实现（128–32768 tokens）",
        )
    if model_id in SF_TOGGLE_ONLY:
        return ThinkingCapability(
            mode="toggle",
            can_disable=True,
            source="known_rules",
            note="仅支持开启/关闭，不能分级",
        )
    return ThinkingCapability(
        mode="unknown",
        source="unknown",
        note="SiliconFlow 目录不返回能力元数据，该模型未在已核实规则中；不发送未验证思考参数",
    )


def parse_deepseek_catalog(items: Sequence[Mapping[str, Any]]) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for item in items:
        model_id = _as_clean_str(item.get("id"))
        if not model_id:
            continue
        name = _as_clean_str(item.get("name")) or model_id
        context = item.get("context_window")
        modalities_raw = item.get("input_modalities")
        modalities = tuple(
            m for m in modalities_raw if isinstance(m, str)
        ) if isinstance(modalities_raw, list) else ()
        models.append(
            ModelInfo(
                id=model_id,
                name=name,
                provider=PROVIDER_DEEPSEEK,
                thinking=_deepseek_capability(item),
                context_window=context if isinstance(context, int) else None,
                input_modalities=modalities,
                owned_by=_as_clean_str(item.get("owned_by")),
            )
        )
    return models


def parse_siliconflow_catalog(items: Sequence[Mapping[str, Any]]) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for item in items:
        model_id = _as_clean_str(item.get("id"))
        if not model_id or model_id in SF_NON_CHAT_IDS:
            continue
        models.append(
            ModelInfo(
                id=model_id,
                name=model_id,
                provider=PROVIDER_SILICONFLOW,
                thinking=_siliconflow_capability(model_id),
                owned_by=_as_clean_str(item.get("owned_by")),
            )
        )
    return models


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def chat_config(provider: str, env: Mapping[str, str] | None = None) -> ProviderConfig:
    if provider == PROVIDER_DEEPSEEK:
        return ProviderConfig.from_deepseek_env(env)
    if provider == PROVIDER_SILICONFLOW:
        return replace(ProviderConfig.from_env(env), embedding_model="")
    raise ValueError("chat provider must be 'deepseek' or 'siliconflow'")


def catalog_fingerprint(provider: str, config: ProviderConfig) -> str:
    """Short non-reversible identity of provider+endpoint+credential context.

    Used only in-memory to isolate caches across providers/accounts; the raw
    key is never stored, logged or returned.
    """
    raw = f"{provider}|{config.base_url}|{config.api_key or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _bounded_catalog_config(config: ProviderConfig) -> ProviderConfig:
    # Catalog calls are small; cap retries so refresh fails fast and clearly.
    retries = min(config.max_retries, 1)
    timeout = min(config.timeout, 30.0)
    return replace(config, max_retries=retries, timeout=timeout)


def fetch_chat_models(
    provider: str,
    *,
    env: Mapping[str, str] | None = None,
    opener: Any = None,
) -> list[ModelInfo]:
    """Fetch the live chat-model catalog. Raises ProviderError on failure."""
    config = _bounded_catalog_config(chat_config(provider, env))
    if provider == PROVIDER_DEEPSEEK:
        client = DeepSeekChatClient(config, opener=opener)
        raw = client.get_models()
        models = parse_deepseek_catalog(raw)
    elif provider == PROVIDER_SILICONFLOW:
        client = SiliconFlowClient(config, opener=opener)
        raw = client.get_models("chat")
        models = parse_siliconflow_catalog(raw)
    else:
        raise ValueError("chat provider must be 'deepseek' or 'siliconflow'")
    if not models:
        raise ProviderError(KIND_BAD_RESPONSE, "model catalog returned an empty list")
    return models


def fetch_embedding_models(
    *,
    env: Mapping[str, str] | None = None,
    opener: Any = None,
) -> list[ModelInfo]:
    """SiliconFlow embedding catalog (advanced settings; not switched now)."""
    config = _bounded_catalog_config(ProviderConfig.from_env(env))
    client = SiliconFlowClient(config, opener=opener)
    items = client.get_models("embedding")
    models = [
        ModelInfo(
            id=_as_clean_str(item.get("id")),
            name=_as_clean_str(item.get("id")),
            provider=PROVIDER_SILICONFLOW,
            thinking=ThinkingCapability(mode="unknown", note="嵌入模型无思考选项"),
        )
        for item in items
        if _as_clean_str(item.get("id"))
    ]
    if not models:
        raise ProviderError(KIND_BAD_RESPONSE, "model catalog returned an empty list")
    return models


def find_model(models: Sequence[ModelInfo], model_id: str) -> ModelInfo | None:
    for model in models:
        if model.id == model_id:
            return model
    return None


# ---------------------------------------------------------------------------
# Preset -> real API value mapping
# ---------------------------------------------------------------------------


def map_effort_preset(levels: Sequence[str], preset: str) -> str | None:
    """Map an app preset (low/medium/high) to a real native value.

    Returns None when that preset is not valid for the model (UI disables it).
    """
    ordered = [level for level in NATIVE_ORDER if level in levels]
    if preset == LEVEL_HIGH:
        return ordered[-1] if ordered else None
    if preset == LEVEL_LOW:
        return LEVEL_LOW if LEVEL_LOW in ordered else None
    if preset == LEVEL_MEDIUM:
        if len(ordered) >= 3:
            return ordered[1]
        if len(ordered) == 2 and LEVEL_LOW not in ordered:
            # e.g. (high, max): 中 maps to the lower real level (high)
            return ordered[0]
        return None
    return None


def selectable_presets(capability: ThinkingCapability) -> tuple[str, ...]:
    """App presets that produce a distinct, valid API value."""
    if capability.mode == "budget":
        return tuple(
            preset for preset in APP_PRESETS if preset in capability.budget_presets
        )
    if capability.mode != "effort":
        return ()
    return tuple(
        preset for preset in APP_PRESETS if map_effort_preset(capability.levels, preset)
    )


def preset_api_description(
    capability: ThinkingCapability, preset: str
) -> tuple[str, str | None]:
    """Human label and the concrete API value for one preset, if valid."""
    label = NATIVE_LABELS.get(preset, preset)
    if capability.mode == "budget":
        budget = dict(capability.budget_presets).get(preset)
        if budget is None:
            return label, None
        return f"{label}（预算预设 thinking_budget={budget}）", str(budget)
    if capability.mode == "effort":
        raw = map_effort_preset(capability.levels, preset)
        if raw is None:
            return f"{label}（本模型不支持）", None
        text = f"{label}（API: {raw}"
        if raw != preset:
            text += "，应用档位映射，非独立档位"
        text += "）"
        return text, raw
    return label, None


def thinking_request(
    provider: str,
    capability: ThinkingCapability,
    settings: ChatSettings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (top-level API body params, metadata descriptor).

    Raises ValueError when the chosen preset is invalid for the model.
    """
    descriptor: dict[str, Any] = {
        "enabled": bool(settings.thinking_enabled),
        "preset": settings.level,
        "mode": capability.mode,
        "source": capability.source,
        "api_params": {},
        "note": capability.note,
    }
    if capability.mode == "unknown":
        descriptor["state"] = "unsupported_unknown"
        return {}, descriptor
    if capability.mode == "fixed_on":
        descriptor["state"] = "fixed_on"
        descriptor["enabled"] = True
        return {}, descriptor
    if not settings.thinking_enabled:
        descriptor["state"] = "off"
        if provider == PROVIDER_DEEPSEEK:
            body = {"thinking": {"type": "disabled"}}
        else:
            body = {"enable_thinking": False}
        descriptor["api_params"] = body
        return body, descriptor

    if capability.mode == "effort":
        raw = map_effort_preset(capability.levels, settings.level)
        if raw is None:
            raise ValueError(
                f"模型不支持应用档位 {settings.level}（实际档位：{list(capability.levels)}）"
            )
        if provider == PROVIDER_DEEPSEEK:
            body = {"thinking": {"type": "enabled"}, "reasoning_effort": raw}
        elif provider == PROVIDER_SILICONFLOW:
            body = {"enable_thinking": True, "reasoning_effort": raw}
        else:
            raise ValueError("unknown chat provider")
    elif capability.mode == "budget":
        if provider != PROVIDER_SILICONFLOW:
            raise ValueError("thinking_budget is only supported on SiliconFlow")
        budget = dict(capability.budget_presets).get(settings.level)
        if budget is None:
            raise ValueError(f"模型不支持预算档位 {settings.level}")
        lo, hi = capability.budget_range or (SF_BUDGET_MIN, SF_BUDGET_MAX)
        if not lo <= budget <= hi:
            raise ValueError("thinking_budget preset out of documented range")
        body = {"enable_thinking": True, "thinking_budget": budget}
    elif capability.mode == "toggle":
        if provider != PROVIDER_SILICONFLOW:
            raise ValueError("toggle mode rule is only defined for SiliconFlow")
        body = {"enable_thinking": True}
    else:
        descriptor["state"] = "unsupported_unknown"
        return {}, descriptor

    descriptor["state"] = "on"
    descriptor["native_level"] = body.get("reasoning_effort")
    descriptor["budget"] = body.get("thinking_budget")
    descriptor["api_params"] = body
    return body, descriptor


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def build_chat_client(
    settings: ChatSettings,
    capability: ThinkingCapability | None = None,
    *,
    env: Mapping[str, str] | None = None,
    opener: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a provider client bound to model + thinking preference.

    Returns ``(client, thinking_descriptor)``. The key is still read from the
    matching environment variable; only the model id comes from settings.
    """
    cap = capability or UNKNOWN_CAPABILITY
    body, descriptor = thinking_request(settings.provider, cap, settings)
    if settings.provider == PROVIDER_DEEPSEEK:
        config = replace(
            ProviderConfig.from_deepseek_env(env), chat_model=settings.model_id
        )
        client_cls = DeepSeekChatClient
    elif settings.provider == PROVIDER_SILICONFLOW:
        config = replace(
            ProviderConfig.from_env(env),
            chat_model=settings.model_id,
            embedding_model="",
        )
        client_cls = SiliconFlowClient
    else:
        raise ValueError("chat provider must be 'deepseek' or 'siliconflow'")
    active = (
        settings.thinking_enabled and cap.mode in {"effort", "budget", "toggle"}
    ) or cap.mode == "fixed_on"
    client = client_cls(
        config,
        opener=opener,
        thinking_body=body,
        thinking_active=active,
    )
    return client, descriptor


# ---------------------------------------------------------------------------
# Local non-sensitive settings file
# ---------------------------------------------------------------------------


def default_settings(env: Mapping[str, str] | None = None) -> ChatSettings:
    """Initial defaults: provider/key from env, model name from env."""
    source = env if env is not None else os.environ
    provider = source.get("GEOCHEM_CHAT_PROVIDER", PROVIDER_DEEPSEEK).strip().lower()
    if provider not in CHAT_PROVIDERS:
        provider = PROVIDER_DEEPSEEK
    if provider == PROVIDER_DEEPSEEK:
        model = source.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat").strip()
    else:
        model = source.get("GEOCHEM_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()
    return ChatSettings(provider=provider, model_id=model)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_settings_file(path: str | Path, settings: ChatSettings) -> None:
    if settings.provider not in CHAT_PROVIDERS:
        raise ValueError("invalid chat provider")
    if not settings.model_id or settings.level not in APP_PRESETS:
        raise ValueError("invalid model settings")
    payload = {
        "version": SETTINGS_VERSION,
        "chat_provider": settings.provider,
        "chat_model": settings.model_id,
        "thinking_enabled": bool(settings.thinking_enabled),
        "thinking_level": settings.level,
        "updated_at": _utc_now(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".local_settings-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_settings_file(path: str | Path) -> ChatSettings | None:
    """Load validated preferences. Returns None when absent/invalid."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    provider = str(payload.get("chat_provider", "")).strip().lower()
    model_id = str(payload.get("chat_model", "")).strip()
    level = str(payload.get("thinking_level", LEVEL_MEDIUM)).strip().lower()
    if provider not in CHAT_PROVIDERS or not model_id or level not in APP_PRESETS:
        return None
    return ChatSettings(
        provider=provider,
        model_id=model_id,
        thinking_enabled=bool(payload.get("thinking_enabled", False)),
        level=level,
    )


def resolve_initial_settings(
    path: str | Path, env: Mapping[str, str] | None = None
) -> ChatSettings:
    """Saved preference first, otherwise env-based initialization default."""
    return load_settings_file(path) or default_settings(env)


# ---------------------------------------------------------------------------
# Chinese error messages
# ---------------------------------------------------------------------------


def provider_error_zh(exc: ProviderError) -> str:
    head = {
        KIND_NOT_CONFIGURED: "未配置 API 密钥",
        KIND_AUTH: "鉴权失败（API 密钥无效、过期或无权限）",
        KIND_RATE_LIMIT: "触发限流，请稍后再试",
        KIND_TIMEOUT: "请求超时，模型目录服务未在限定时间内响应",
        KIND_NETWORK: "网络连接失败，请检查网络/代理设置",
        KIND_BAD_RESPONSE: "返回内容无效（无法解析或目录为空）",
        KIND_SERVER: "服务商暂时出错（5xx）",
        KIND_BAD_REQUEST: "请求被服务商拒绝",
    }.get(exc.kind, "未知错误")
    detail = str(exc)
    # Upstream detail may contain English explanations; never the key.
    if detail and exc.kind in {KIND_BAD_REQUEST, KIND_SERVER, KIND_BAD_RESPONSE}:
        return f"{head}：{detail}"
    return head
