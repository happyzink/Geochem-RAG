"""Streamlit front-end for GeoChem-RAG (online only).

Top of page: title, short Chinese description, compact online status, and two
buttons — 知识库 (knowledge base) and 设置 (settings). The body is a native chat
interface: a question is answered against the local corpus and every answer
cites the paper title and PDF page. Sources and chunking details are shown only
when the knowledge-base panel is opened; model/retrieval/supplement options are
shown only when the settings panel is opened.

There is no offline/template fallback: missing keys, an empty corpus, or an
incomplete/mismatched index are surfaced as clear errors, never as a fake answer.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]  # src/
_REPO = _SRC.parent  # repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from geochem_rag import model_settings as ms  # noqa: E402
from geochem_rag.answering import (  # noqa: E402
    EvidenceAnswerer,
    EvidenceItem,
    build_evidence,
    format_citation,
)
from geochem_rag.domain import AnswerStatus, Citation, Source  # noqa: E402
from geochem_rag.model_settings import (  # noqa: E402
    APP_PRESETS,
    ChatSettings,
    NATIVE_LABELS,
    PROVIDER_DEEPSEEK,
    PROVIDER_SILICONFLOW,
    UNKNOWN_CAPABILITY,
    find_model,
)
from geochem_rag.providers import (  # noqa: E402
    ProviderError,
    embedding_provider_from_env,
)
from geochem_rag.retrieval import Retriever  # noqa: E402
from geochem_rag.store import LocalStore  # noqa: E402

_PAGE_TITLE = "GeoChem-RAG"
_DATA_FLOW_NOTE = (
    "建索引文本发送至 SiliconFlow 生成向量；提出问题后，问题与命中的证据片段发送至所选对话 provider"
    "（默认 DeepSeek，可切换为 SiliconFlow）生成回答。"
)
_EXAMPLE_QUESTIONS = [
    "什么是埃达克岩（adakite）？",
    "弧岩浆中 Nb/Ta 比值的变化意味着什么？",
    "青藏高原地壳增厚主要发生在什么时期？",
]

_SETTINGS_FILE = _REPO / "data" / "local_settings.json"
_PROVIDER_LABELS = {
    PROVIDER_DEEPSEEK: "DeepSeek（官方接口）",
    PROVIDER_SILICONFLOW: "SiliconFlow（OpenAI 兼容接口）",
}
# Short self-authored probe used by the explicit connection test. Contains no
# private corpus text; model permission is all that is checked.
_TEST_PROMPT = "请只回复两个汉字：正常"


def _resolve_store() -> str:
    if os.environ.get("GEOCHEM_STORE"):
        return os.environ["GEOCHEM_STORE"]
    full = _REPO / "data" / "private" / "full_corpus"
    if (full / "chunks.jsonl").is_file():
        return full.as_posix()
    return "data/processed"


def _resolve_cache(store_root: str) -> str:
    return (
        "data/index/embeddings_full.json"
        if "full_corpus" in store_root.replace("\\", "/")
        else "data/index/embeddings.json"
    )


def _cache_model(path: str) -> str | None:
    p = _REPO / path
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return payload.get("model") if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Model catalog / chat client management
# ---------------------------------------------------------------------------


def _capability_for(settings: ChatSettings, models: list[ms.ModelInfo] | None):
    model = find_model(models or [], settings.model_id)
    return model.thinking if model is not None else UNKNOWN_CAPABILITY


def _refresh_catalog(force: bool) -> None:
    """Fetch the catalog for the active provider once per context, or forced.

    On failure keeps the previous successful list (marked stale) and records a
    Chinese error; never clears the selected model or takes the app offline.
    """
    settings: ChatSettings = st.session_state.chat_settings
    config = ms.chat_config(settings.provider)
    fingerprint = ms.catalog_fingerprint(settings.provider, config)
    current: ms.CatalogState | None = st.session_state.get("catalog")
    if (
        not force
        and current is not None
        and current.fingerprint == fingerprint
        and current.loaded
        and not current.error
    ):
        return
    try:
        models = ms.fetch_chat_models(settings.provider)
    except ProviderError as exc:
        message = ms.provider_error_zh(exc)
        if current is not None and current.fingerprint == fingerprint and current.models:
            st.session_state.catalog = ms.CatalogState(
                provider=settings.provider,
                fingerprint=fingerprint,
                models=current.models,
                fetched_at=current.fetched_at,
                error=message,
                stale=True,
                loaded=True,
            )
        else:
            st.session_state.catalog = ms.CatalogState(
                provider=settings.provider,
                fingerprint=fingerprint,
                models=[],
                error=message,
                stale=True,
                loaded=True,
            )
        return
    except Exception as exc:  # noqa: BLE001 - catalog failure must never crash app
        st.session_state.catalog = ms.CatalogState(
            provider=settings.provider,
            fingerprint=fingerprint,
            models=current.models if current else [],
            fetched_at=current.fetched_at if current else None,
            error=f"模型目录获取失败：{exc}",
            stale=True,
            loaded=True,
        )
        return
    st.session_state.catalog = ms.CatalogState(
        provider=settings.provider,
        fingerprint=fingerprint,
        models=models,
        fetched_at=ms._utc_now(),
        error=None,
        stale=False,
        loaded=True,
    )


def _rebuild_chat(settings: ChatSettings) -> None:
    """Rebuild chat client + answerer from applied settings.

    Retriever, store and chat history are untouched. The API key is still read
    from the matching environment variable; no key is written to the session.
    """
    catalog: ms.CatalogState = st.session_state.catalog
    capability = _capability_for(settings, catalog.models if catalog else [])
    client, descriptor = ms.build_chat_client(settings, capability)
    st.session_state.chat = client
    st.session_state.chat_settings = settings
    st.session_state.model_descriptor = descriptor
    model_meta = {
        "provider": settings.provider,
        "requested_model": settings.model_id,
        "thinking": descriptor,
    }
    st.session_state.answerer = EvidenceAnswerer(
        st.session_state.store,
        client,
        top_k=5,
        model_meta=model_meta,  # type: ignore[arg-type]
    )


def _init_session() -> None:
    if "initialised" in st.session_state:
        return
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("show_kb", False)
    st.session_state.setdefault("show_settings", False)
    st.session_state.setdefault("settings_message", None)

    store_root = _resolve_store()
    store = LocalStore(store_root)
    st.session_state.store = store
    st.session_state.store_root = store_root
    st.session_state.chunk_count = store.count_chunks()
    st.session_state.ready = False
    st.session_state.error = None

    if store.count_chunks() == 0:
        st.session_state.error = (
            "无语料：数据目录为空。请先运行 `scripts/import_all_pdfs.py` 导入 PDF。"
        )
        st.session_state.initialised = True
        return

    # Embeddings stay independent of the chat provider choice.
    try:
        embedder = embedding_provider_from_env()
    except ValueError as exc:
        st.session_state.error = f"配置错误：{exc}"
        st.session_state.initialised = True
        return
    embed_cfg = getattr(embedder, "config", None)
    if not getattr(embed_cfg, "api_key", None):
        st.session_state.error = "缺少 SILICONFLOW_API_KEY（嵌入不可用）。请配置后重启。"
        st.session_state.initialised = True
        return

    # Chat settings: session preference > saved local file > env model default.
    settings = ms.resolve_initial_settings(_SETTINGS_FILE)
    try:
        chat_config = ms.chat_config(settings.provider)
    except ValueError as exc:
        st.session_state.error = f"配置错误：{exc}"
        st.session_state.initialised = True
        return
    if not chat_config.api_key:
        st.session_state.error = (
            f"缺少 {chat_config.credential_name}（对话不可用）。请配置后重启。"
        )
        st.session_state.initialised = True
        return
    st.session_state.chat_settings = settings

    cache_path = _resolve_cache(store_root)
    expected_model = os.environ.get("GEOCHEM_EMBEDDING_MODEL", "BAAI/bge-m3")
    cached_model = _cache_model(cache_path)
    if cached_model != expected_model:
        st.session_state.error = (
            f"索引不完整或模型不匹配：`{cache_path}` 的模型为 `{cached_model}`，"
            f"期望 `{expected_model}`。请先运行 `scripts/build_full_index.py`。"
        )
        st.session_state.initialised = True
        return

    try:
        retriever = Retriever.from_store(store, embedder, cache_path=cache_path)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - surface as a clear error
        st.session_state.error = f"索引加载失败：{exc}"
        st.session_state.initialised = True
        return
    st.session_state.embedder = embedder
    st.session_state.retriever = retriever

    # Fetch the catalog exactly once at startup. No evidence is sent; a catalog
    # failure keeps the app usable with capability marked unknown.
    st.session_state.catalog = ms.CatalogState(
        provider=settings.provider,
        fingerprint=ms.catalog_fingerprint(settings.provider, chat_config),
    )
    _refresh_catalog(force=False)
    _rebuild_chat(settings)

    st.session_state.ready = True
    st.session_state.initialised = True


# ---------------------------------------------------------------------------
# Answer rendering helpers
# ---------------------------------------------------------------------------


def _status_label(status: str) -> str:
    return {
        AnswerStatus.ANSWERED.value: "已回答（有本域文献证据）",
        AnswerStatus.INSUFFICIENT_EVIDENCE.value: "证据不足",
        AnswerStatus.ERROR.value: "出错",
    }.get(status, status)


def _thinking_caption(thinking: dict[str, Any] | None) -> str:
    if not isinstance(thinking, dict):
        return "思考：未知"
    state = thinking.get("state")
    preset = thinking.get("preset")
    preset_label = NATIVE_LABELS.get(preset, preset) if preset else ""
    native = thinking.get("native_level")
    budget = thinking.get("budget")
    if state == "on":
        if native:
            mapped = "（应用档位映射，非独立档位）" if native != preset else ""
            return f"思考：开启 · {preset_label}→API `{native}`{mapped}"
        if budget is not None:
            return f"思考：开启 · {preset_label}→预算预设 `thinking_budget={budget}`"
        return "思考：开启"
    if state == "off":
        return "思考：关闭（已发送关闭参数）"
    if state == "fixed_on":
        return "思考：固定开启（该模型不支持关闭，未发送无效开关）"
    return "思考：未发送控制参数（能力未知，不猜测）"


def _model_caption(meta: dict[str, Any]) -> None:
    provider = _PROVIDER_LABELS.get(meta.get("provider"), meta.get("provider") or "?")
    requested = meta.get("requested_model") or "?"
    actual = meta.get("model")
    actual_text = f" · 实际返回 `{actual}`" if actual and actual != requested else ""
    st.caption(
        f"对话 provider：{provider} · 请求模型 `{requested}`{actual_text} · "
        + _thinking_caption(meta.get("thinking"))
    )


def _render_citation(citation: Citation, evidence_items: list[EvidenceItem], index: int) -> None:
    snippet = ""
    for item in evidence_items:
        if item.chunk_id == citation.chunk_id:
            snippet = item.snippet()
            break
    header = f"**[{index + 1}]** {citation.title} —— PDF 第 {citation.pdf_page} 页"
    with st.expander(header, expanded=False):
        if snippet:
            st.markdown("**证据片段：**")
            st.markdown(f"> {snippet}")
        st.caption(f"chunk `{citation.chunk_id}`")


def _render_answer(answer: Any, evidence_items: list[EvidenceItem]) -> None:
    st.markdown(answer.text)
    _model_caption(answer.meta)

    if answer.status == AnswerStatus.ANSWERED.value:
        unsupported = answer.meta.get("unsupported_aspects") or []
        if unsupported:
            st.warning(
                "**尚无论文证据支持的部分**\n\n" + "\n".join(f"- {item}" for item in unsupported)
            )
        if answer.citations:
            st.markdown("**引用（论文证据）：**")
            for idx, citation in enumerate(answer.citations):
                _render_citation(citation, evidence_items, idx)
        extended = answer.meta.get("extended_explanation")
        if extended:
            with st.expander("通用背景补充（未经本地文献核验）", expanded=False):
                st.markdown(extended)
                st.caption("以上是一般背景知识，不是本域文献结论，不得当作引用来源。")

    elif answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value:
        extended = answer.meta.get("extended_explanation")
        if extended:
            with st.expander("通用背景补充（未经本地文献核验）", expanded=False):
                st.markdown(extended)
                st.caption("以上是一般背景知识，不是本域文献结论，不得当作引用来源。")

    else:  # error
        error = answer.meta.get("error") or {}
        st.caption(f"错误类型：`{error.get('kind', '?')}`")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _render_knowledge_base() -> None:
    st.subheader("知识库")
    chunk_count = st.session_state.chunk_count
    st.caption(f"有效证据块：**{chunk_count}**")
    sources = st.session_state.store.load_sources()
    if not sources:
        st.info("暂无来源。")
        return
    query = st.text_input("按标题/文件名筛选（可选）", key="kb_filter")
    shown = [
        s for s in sources
        if not query or query.lower() in (s.title or "").lower()
    ]
    for source in shown:
        label = source.title
        if source.title_is_filename:
            label = f"[文件名] {label}"
        st.markdown(f"- {label}{('（' + str(source.year) + '）') if source.year else ''} · `{source.visibility}`")


def _save_setting(widget_key: str, setting_key: str) -> None:
    st.session_state[setting_key] = st.session_state[widget_key]


def _model_option_label(model_id: str, catalog: ms.CatalogState, settings: ChatSettings) -> str:
    model = find_model(catalog.models, model_id)
    if model is not None:
        tag = ""
        if model.input_modalities and "image" in model.input_modalities:
            tag = " · 支持图片"
        return f"{model.display}{tag}"
    if model_id == settings.model_id:
        return f"{model_id}（当前选择 · 待验证，不在最新目录中）"
    return model_id


def _apply_settings() -> None:
    provider = st.session_state.get("_set_provider")
    model_id = st.session_state.get("_set_model_id")
    enabled = bool(st.session_state.get("_set_thinking", False))
    level = st.session_state.get("_set_level") or "medium"
    if provider not in (PROVIDER_DEEPSEEK, PROVIDER_SILICONFLOW) or not model_id:
        st.session_state.settings_message = ("error", "设置无效：请选择对话 provider 和模型。")
        return
    if level not in APP_PRESETS:
        level = "medium"

    # Validate the credential for the target provider before switching.
    target_config = ms.chat_config(provider)
    if not target_config.api_key:
        st.session_state.settings_message = (
            "error",
            f"无法应用：环境中缺少 {target_config.credential_name}，请在 .env 配置后重启。",
        )
        return

    provider_changed = provider != st.session_state.chat_settings.provider
    if provider_changed:
        # Catalogs are isolated per provider/endpoint/credential: fetch the new
        # catalog first, never reuse the other provider's list. The selected id
        # comes from the old catalog's dropdown, so it must be re-validated.
        previous_settings: ChatSettings = st.session_state.chat_settings
        previous_catalog: ms.CatalogState = st.session_state.catalog
        st.session_state.chat_settings = ChatSettings(provider=provider, model_id=model_id)
        st.session_state.catalog = ms.CatalogState(
            provider=provider,
            fingerprint=ms.catalog_fingerprint(provider, target_config),
        )
        _refresh_catalog(force=True)
        catalog: ms.CatalogState = st.session_state.catalog
        chosen = find_model(catalog.models, model_id)
        if chosen is None:
            fallback = ms.default_settings(
                {
                    "GEOCHEM_CHAT_PROVIDER": provider,
                    "DEEPSEEK_CHAT_MODEL": os.environ.get("DEEPSEEK_CHAT_MODEL", ""),
                    "GEOCHEM_CHAT_MODEL": os.environ.get("GEOCHEM_CHAT_MODEL", ""),
                }
            )
            chosen = find_model(catalog.models, fallback.model_id)
            if chosen is None and catalog.models:
                chosen = catalog.models[0]
        if chosen is None:
            # Restore the previously applied runtime state so the active chat
            # client and the displayed settings never disagree. Widget keys are
            # already instantiated in this run, so re-seed them on the rerun
            # via the sync flag instead of writing them directly.
            st.session_state.chat_settings = previous_settings
            st.session_state.catalog = previous_catalog
            st.session_state["_set_sync_widgets"] = True
            st.session_state.settings_message = (
                "error",
                "已切换 provider，但模型目录获取失败且无可用模型；请刷新目录后重新选择，本次设置未应用。",
            )
            return
        model_id = chosen.id
        # Re-select thinking against the new model's real capability. Widget
        # values are re-seeded from these applied settings on the rerun.
        new_settings = ChatSettings(
            provider=provider, model_id=model_id, thinking_enabled=False, level="medium"
        )
    else:
        new_settings = ChatSettings(
            provider=provider, model_id=model_id, thinking_enabled=enabled, level=level
        )

    # Validate the thinking preset against the selected model's real capability.
    capability = _capability_for(new_settings, st.session_state.catalog.models)
    if new_settings.thinking_enabled and capability.mode == "effort":
        if ms.map_effort_preset(capability.levels, new_settings.level) is None:
            st.session_state.settings_message = (
                "error",
                f"该模型不支持所选档位（实际支持：{', '.join(capability.levels)}），请重新选择。",
            )
            return

    try:
        _rebuild_chat(new_settings)
    except ProviderError as exc:
        st.session_state.settings_message = (
            "error",
            f"应用失败：{ms.provider_error_zh(exc)}",
        )
        return
    try:
        ms.save_settings_file(_SETTINGS_FILE, new_settings)
    except OSError as exc:
        st.session_state.settings_message = (
            "warning",
            f"已应用到当前会话，但本地偏好保存失败：{exc}",
        )
        st.session_state["_set_sync_widgets"] = True
        return
    st.session_state["_set_sync_widgets"] = True
    st.session_state.settings_message = (
        "success",
        f"已应用：{_PROVIDER_LABELS[provider]} · {model_id} · {_thinking_caption(st.session_state.model_descriptor)}",
    )


def _run_connection_test() -> None:
    settings: ChatSettings = st.session_state.chat_settings
    catalog: ms.CatalogState = st.session_state.catalog
    capability = _capability_for(settings, catalog.models if catalog else [])
    # Keep the probe cheap: turn thinking off whenever the model allows it;
    # fixed-thinking models still run with defaults. No evidence is sent.
    probe_settings = settings
    if capability.can_disable:
        probe_settings = replace(settings, thinking_enabled=False)
    try:
        client, _ = ms.build_chat_client(probe_settings, capability)
        result = client.chat(
            [{"role": "user", "content": _TEST_PROMPT}],
            max_tokens=32,
            json_mode=False,
        )
    except ProviderError as exc:
        st.session_state.settings_message = (
            "error",
            f"连接测试失败：{ms.provider_error_zh(exc)}",
        )
        return
    st.session_state.settings_message = (
        "success",
        f"连接测试成功：服务实际返回模型 `{result.model}`，耗时 {result.latency_seconds:.1f} 秒"
        f"（请求模型 `{settings.model_id}`）。",
    )


def _render_settings() -> None:
    st.subheader("设置")
    settings: ChatSettings = st.session_state.chat_settings
    catalog: ms.CatalogState = st.session_state.catalog

    message = st.session_state.get("settings_message")
    if message is not None:
        kind, text = message
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        else:
            st.error(text)
        st.session_state.settings_message = None

    chat_cfg = getattr(st.session_state.chat, "config", None)
    embed_cfg = getattr(st.session_state.embedder, "config", None)
    if chat_cfg:
        st.caption(f"当前对话 provider：**{_PROVIDER_LABELS.get(settings.provider, settings.provider)}**")
        st.caption(f"接口地址（不含密钥）：`{chat_cfg.base_url}` · API key 从 `{chat_cfg.credential_name}` 读取，不回显")
    if embed_cfg:
        st.caption(
            f"嵌入模型：`{embed_cfg.embedding_model}`（{embed_cfg.base_url}）。"
            "更换嵌入模型需另行建立独立索引，本轮不开放切换。"
        )

    # --- Catalog status + manual refresh ---------------------------------
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if catalog.loaded and catalog.fetched_at:
            status = f"模型目录：**{len(catalog.models)}** 个对话模型 · 更新于 {catalog.fetched_at[11:19]}"
            if catalog.stale:
                status += " · 已过期（显示上次成功结果）"
            st.caption(status)
        elif catalog.loaded and catalog.error:
            st.caption("模型目录获取失败，可点击右侧按钮重试；当前选择已保留。")
        else:
            st.caption("正在获取模型目录……")
    with col_b:
        if st.button("刷新模型列表", key="_set_refresh"):
            _refresh_catalog(force=True)
            _rebuild_chat(st.session_state.chat_settings)
            st.rerun()
    if catalog.error:
        st.error(f"模型目录：{catalog.error}")

    st.markdown("##### 对话模型")

    # Widget keys are independent of persisted config keys. When the panel was
    # hidden Streamlit drops widget state, so seed it from applied settings
    # before creation (never assign to a widget key after instantiation).
    if st.session_state.pop("_set_sync_widgets", False):
        st.session_state["_set_provider"] = settings.provider
        st.session_state["_set_model_id"] = settings.model_id
        st.session_state["_set_thinking"] = settings.thinking_enabled
        st.session_state["_set_level"] = settings.level
    st.session_state.setdefault("_set_provider", settings.provider)
    st.session_state.setdefault("_set_model_id", settings.model_id)
    st.session_state.setdefault(
        "_set_thinking",
        settings.thinking_enabled and settings.model_id in {m.id for m in catalog.models},
    )
    st.session_state.setdefault("_set_level", settings.level)
    st.session_state.setdefault("_set_model_filter", "")

    st.selectbox(
        "对话 provider（切换后需点击下方“保存并应用”）",
        options=(PROVIDER_DEEPSEEK, PROVIDER_SILICONFLOW),
        format_func=lambda value: _PROVIDER_LABELS.get(value, value),
        key="_set_provider",
    )
    pending_provider = st.session_state["_set_provider"] != settings.provider
    if pending_provider:
        st.info("provider 已选择但尚未应用：点击“保存并应用”后将获取该 provider 的模型目录，再选择模型与思考设置。")

    keyword = st.text_input(
        "搜索模型（按名称/ID 过滤）", key="_set_model_filter", disabled=pending_provider
    ).strip().lower()
    ids = [m.id for m in catalog.models]
    if keyword:
        ids = [
            m.id
            for m in catalog.models
            if keyword in m.id.lower() or keyword in (m.name or "").lower()
        ]
    # The currently selected id must always remain a valid selectbox option.
    selected_id = st.session_state["_set_model_id"]
    if selected_id and selected_id not in ids:
        ids = [selected_id, *ids]
    if not ids:
        st.info("目录中暂无可选模型（可使用当前模型，状态为待验证），或点击“刷新模型列表”重试。")
        ids = [selected_id] if selected_id else []

    st.selectbox(
        "对话模型",
        options=ids,
        format_func=lambda mid: _model_option_label(mid, catalog, settings),
        key="_set_model_id",
        disabled=pending_provider,
    )

    # Capability follows the model selected in the widget (applied on save).
    selected_model = find_model(catalog.models, st.session_state["_set_model_id"])
    capability = selected_model.thinking if selected_model is not None else UNKNOWN_CAPABILITY

    st.markdown("##### 思考设置")
    if capability.source == "catalog":
        st.caption("能力来源：模型目录元数据")
    elif capability.source == "known_rules":
        st.caption("能力来源：官方文档/公告已核实规则（2026-10-12）")
    else:
        st.caption("能力来源：未知")
    if capability.note:
        st.caption(capability.note)

    # Seed widget state BEFORE widget creation; never assign to a widget key
    # after the widget has been instantiated in the same run.
    if capability.mode == "fixed_on":
        st.session_state["_set_thinking"] = True
    elif capability.mode == "unknown" and not pending_provider:
        st.session_state["_set_thinking"] = False
    thinking_disabled = capability.mode in {"unknown", "fixed_on"} or pending_provider
    st.toggle(
        "开启思考（真实 API 控制，非提示词模拟）",
        disabled=thinking_disabled,
        key="_set_thinking",
    )

    # Level control
    if capability.mode in {"effort", "budget"}:
        presets = ms.selectable_presets(capability)
        labels = {
            preset: ms.preset_api_description(capability, preset)[0] for preset in presets
        }
        current_level = st.session_state.get("_set_level", "medium")
        if current_level not in presets:
            current_level = presets[len(presets) // 2] if presets else "medium"
            st.session_state["_set_level"] = current_level
        st.selectbox(
            "思考强度（显示应用档位与实际 API 值）",
            options=list(presets),
            format_func=lambda value: labels.get(value, value),
            disabled=not st.session_state.get("_set_thinking", False) or pending_provider,
            key="_set_level",
        )
        if capability.mode == "effort":
            missing = [
                NATIVE_LABELS[p]
                for p in APP_PRESETS
                if ms.map_effort_preset(capability.levels, p) is None
            ]
            if missing:
                st.caption("不可选档位：" + "、".join(missing) + "（本模型不提供该真实档位）")
    else:
        placeholder = {
            "toggle": "该模型仅支持思考开关，不能分级",
            "fixed_on": "思考固定开启，不可关闭，也不发送无效开关",
            "unknown": "能力未知：不发送任何思考控制参数",
            "budget": "",
        }.get(capability.mode, "")
        st.selectbox(
            "思考强度",
            options=["不可用"],
            disabled=True,
            key="_set_level_unavailable",
            help=placeholder,
        )
        st.caption(placeholder)

    col_save, col_test = st.columns(2)
    with col_save:
        if st.button("保存并应用", key="_set_apply", type="primary"):
            _apply_settings()
            st.rerun()
    with col_test:
        if st.button("连接测试（自建短提示）", key="_set_test"):
            _run_connection_test()
            st.rerun()

    st.divider()
    st.caption("偏好（provider/模型/思考）保存在 Git 忽略的本地 data/local_settings.json，不含密钥。")

    # --- Retrieval controls (independent widget keys, immediate effect) --
    defaults = {
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "allow_extended": True,
    }
    for setting_key, default in defaults.items():
        st.session_state.setdefault(setting_key, default)
        st.session_state.setdefault(
            f"_settings_{setting_key}", st.session_state[setting_key]
        )
    st.selectbox(
        "检索方式",
        ["hybrid", "dense_only", "bm25_only"],
        key="_settings_retrieval_mode",
        on_change=_save_setting,
        args=("_settings_retrieval_mode", "retrieval_mode"),
    )
    st.slider(
        "证据块数（top-k）",
        min_value=1,
        max_value=10,
        key="_settings_top_k",
        on_change=_save_setting,
        args=("_settings_top_k", "top_k"),
    )
    st.toggle(
        "允许通用背景补充（未经本地文献核验）",
        key="_settings_allow_extended",
        on_change=_save_setting,
        args=("_settings_allow_extended", "allow_extended"),
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def _run_answer(question: str) -> None:
    retriever: Retriever = st.session_state.retriever
    answerer: EvidenceAnswerer = st.session_state.answerer
    mode = st.session_state.get("retrieval_mode", "hybrid")
    top_k = st.session_state.get("top_k", 5)
    allow_extended = st.session_state.get("allow_extended", True)

    try:
        trace = retriever.retrieve_with_trace(question, mode=mode, top_k=top_k)
    except ProviderError as exc:
        st.session_state.chat_history.append(
            {"role": "assistant", "kind": "error", "text": f"检索失败：{exc.kind}。", "evidence": []}
        )
        return
    answer = answerer.answer(question, trace.hits, top_k=top_k, allow_extended=allow_extended)
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "kind": answer.status,
            "answer": answer,
            "evidence": build_evidence(trace.hits, st.session_state.store, top_k=top_k),
        }
    )


def main() -> None:
    st.set_page_config(page_title=_PAGE_TITLE, layout="wide")
    st.title(_PAGE_TITLE)
    st.caption("岩石地球化学文献问答：每条回答都引用具体论文标题与 PDF 页码。")

    _init_session()

    if not st.session_state.get("ready", False):
        st.error("**无法启动回答功能**\n\n" + str(st.session_state.get("error", "未知错误")))
        st.caption(_DATA_FLOW_NOTE)
        st.stop()

    # Header buttons -----------------------------------------------------
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("知识库", use_container_width=True):
            st.session_state.show_kb = not st.session_state.show_kb
            st.session_state.show_settings = False
    with col2:
        if st.button("设置", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
            st.session_state.show_kb = False
    st.caption(_DATA_FLOW_NOTE)
    st.divider()

    if st.session_state.show_kb:
        _render_knowledge_base()
        st.divider()
    if st.session_state.show_settings:
        _render_settings()
        st.divider()

    # Examples ----------------------------------------------------------
    with st.expander("示例问题", expanded=False):
        for q in _EXAMPLE_QUESTIONS:
            st.markdown(f"- {q}")

    # Chat history ------------------------------------------------------
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["text"])
        else:
            with st.chat_message("assistant"):
                if message.get("kind") == "error":
                    st.error(message["text"])
                else:
                    _render_answer(message["answer"], message.get("evidence", []))

    # Input ------------------------------------------------------------
    question = st.chat_input("向本域文献语料提问…")
    if question:
        st.session_state.chat_history.append({"role": "user", "text": question})
        with st.spinner("检索证据并生成回答…"):
            _run_answer(question)
        st.rerun()


if __name__ == "__main__":
    main()
