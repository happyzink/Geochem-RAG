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

import base64
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
from geochem_rag.scientific_text import safe_markdown  # noqa: E402
from geochem_rag.model_settings import (  # noqa: E402
    APP_PRESETS,
    ChatSettings,
    NATIVE_LABELS,
    PROVIDER_DEEPSEEK,
    PROVIDER_SILICONFLOW,
    UNKNOWN_CAPABILITY,
    find_model,
    load_settings_file,
    preset_api_description,
    save_settings_file,
    selectable_presets,
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
    # The rebuilt geoscience library is the default when present; the old
    # full_corpus store is kept as a rollback snapshot (see the switch report).
    geo = _REPO / "data" / "private" / "geoscience_20261014"
    if (geo / "chunks.jsonl").is_file():
        return geo.as_posix()
    full = _REPO / "data" / "private" / "full_corpus"
    if (full / "chunks.jsonl").is_file():
        return full.as_posix()
    return "data/processed"


def _resolve_cache(store_root: str) -> str:
    if os.environ.get("GEOCHEM_INDEX_CACHE"):
        return os.environ["GEOCHEM_INDEX_CACHE"]
    normalized = store_root.replace("\\", "/")
    if "geoscience_20261014" in normalized:
        return "data/index/embeddings_geoscience_20261014.json"
    if "full_corpus" in normalized:
        return "data/index/embeddings_full.json"
    return "data/index/embeddings.json"


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


def _apply_settings(settings: ChatSettings) -> None:
    """Persist + rebuild chat client when settings change from the composer bar."""
    save_settings_file(_SETTINGS_FILE, settings)
    _rebuild_chat(settings)


def _render_composer() -> None:
    """Chat input + model/thinking selector + PDF upload entry on the bottom bar.

    Model/thinking selectors live outside the ``st.form`` so that changing them
    applies immediately (Streamlit forms defer widget values until submit). The
    question text area + buttons are inside the form so Enter/Ctrl+Enter submits.
    """
    catalog = st.session_state.get("catalog")
    settings: ChatSettings = st.session_state.chat_settings
    capability = _capability_for(settings, catalog.models if catalog else [])

    # --- Model + thinking row (outside form — immediate apply) -------------
    col_model, col_think, _ = st.columns([2.2, 2, 8])
    with col_model:
        model_ids = (
            [m.id for m in catalog.models]
            if catalog and catalog.models
            else []
        )
        if settings.model_id and settings.model_id not in model_ids:
            model_ids = [settings.model_id, *model_ids]
        if not model_ids:
            model_ids = [settings.model_id]
        model_labels = {
            mid: _model_option_label(mid, catalog, settings) for mid in model_ids
        }
        st.selectbox(
            "模型",
            options=model_ids,
            index=(
                model_ids.index(settings.model_id)
                if settings.model_id in model_ids
                else 0
            ),
            key="_composer_model_sel",
            label_visibility="collapsed",
            format_func=lambda mid: model_labels.get(mid, mid),
        )
    with col_think:
        presets = selectable_presets(capability)
        thinking_options: list[str] = ["关闭"] + list(presets)
        if not thinking_options:
            thinking_options = ["关闭"]
        if settings.thinking_enabled and settings.level in presets:
            think_idx = thinking_options.index(settings.level)
        else:
            think_idx = 0
        st.selectbox(
            "思考",
            options=thinking_options,
            index=think_idx,
            key="_composer_think_sel",
            label_visibility="collapsed",
            format_func=(
                lambda v: NATIVE_LABELS.get(v, v) if v != "关闭" else "关闭"
            ),
            disabled=not presets,
        )

    # --- Apply settings when selectboxes changed ---------------------------
    new_model = st.session_state.get("_composer_model_sel")
    if new_model and new_model != settings.model_id:
        new_cap = _capability_for(
            replace(settings, model_id=new_model),
            catalog.models if catalog else [],
        )
        new_settings = replace(
            settings,
            model_id=new_model,
            provider=_provider_for_model(new_model, catalog),
        )
        if not selectable_presets(new_cap) and new_settings.thinking_enabled:
            new_settings = replace(new_settings, thinking_enabled=False)
        _apply_settings(new_settings)
        st.rerun()

    selected_think = st.session_state.get("_composer_think_sel")
    if selected_think == "关闭" and settings.thinking_enabled:
        _apply_settings(replace(settings, thinking_enabled=False))
        st.rerun()
    elif (
        selected_think
        and selected_think != "关闭"
        and (not settings.thinking_enabled or settings.level != selected_think)
    ):
        _apply_settings(replace(settings, thinking_enabled=True, level=selected_think))
        st.rerun()

    # --- Input form ---------------------------------------------------------
    with st.form(key="_composer_form"):
        question = st.text_area(
            "Ask about local geochemistry papers…",
            height=100,
            key="_composer_question",
            label_visibility="collapsed",
            placeholder="Type question here…",
        )
        _c1, _c2, _c3 = st.columns([1, 10, 1])
        with _c1:
            pdf_clicked = st.form_submit_button(
                "+ PDF", use_container_width=True
            )
        with _c3:
            send_clicked = st.form_submit_button(
                "Send", use_container_width=True, type="primary"
            )

    if pdf_clicked:
        st.session_state.show_pdf_upload = not st.session_state.get(
            "show_pdf_upload", False
        )

    if send_clicked and question and question.strip():
        st.session_state.chat_history.append({"role": "user", "text": question.strip()})
        with st.spinner("检索证据并生成回答……"):
            _run_answer(question.strip())
        st.rerun()


def _render_pdf_upload() -> None:
    """Upload → 分类 → 切块预览 → 用户确认入库 → 增量索引。"""
    import importlib.util

    from geochem_rag.ingest import (
        chunking_fingerprint,
        extract_pages,
        ingest_pdf,
        sha256_file,
    )

    # Classify: import from the scripts module to share real rules.
    _inv_path = _REPO / "scripts" / "inventory_local_pdfs.py"
    inv_spec = importlib.util.spec_from_file_location("inventory_local_pdfs", _inv_path)
    inv_mod = importlib.util.module_from_spec(inv_spec)
    assert inv_spec.loader is not None
    inv_spec.loader.exec_module(inv_mod)
    classify_geoscience = inv_mod.classify_geoscience

    store: LocalStore = st.session_state.store
    store_root = Path(st.session_state.store_root)
    upload_dir = _REPO / "PDF" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader(
        "选择 PDF 文件（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key="_pdf_uploader",
    )

    if not uploaded:
        st.info(
            "选择 PDF 后开始处理。已上传文件保存在 PDF/uploads/ 并经地学筛选。"
            "块文本片段会发送至 SiliconFlow 嵌入服务生成向量；提问时相关证据会发送至对话服务。"
        )
        return

    for uf in uploaded:
        fname = uf.name
        with st.status(f"处理 {fname}…", expanded=True):
            st.markdown(f"**{fname}**（{len(uf.getvalue()) / 1024:.1f} KB）")

            # --- Save & dedup ------------------------------------------------
            safe_name = fname.replace(" ", "_")
            pdf_path = upload_dir / safe_name
            counter = 1
            while pdf_path.exists() and pdf_path.read_bytes() != uf.getvalue():
                pdf_path = upload_dir / f"{pdf_path.stem}_{counter}{pdf_path.suffix}"
                counter += 1
            if not pdf_path.exists():
                pdf_path.write_bytes(uf.getvalue())
            digest = sha256_file(pdf_path)

            existing = next((s for s in store.load_sources() if s.sha256 == digest), None)
            if existing is not None:
                st.success(
                    f"已存在相同内容文献：{existing.title}。SHA-256 匹配，跳过重复入库。"
                )
                continue

            # --- Parse & classify --------------------------------------------
            try:
                pages = extract_pages(pdf_path)
            except Exception as exc:
                st.error(f"无法解析 PDF：{exc}")
                continue

            scan_text = "\n".join(p.text for p in pages if p.quality.value in ("ok",))
            decision, reason, _ = classify_geoscience(scan_text, fname)

            if decision == "exclude":
                st.warning(f"**排除（非地学资料）**：{reason}。该文件不会加入知识库。")
                continue

            # --- Preview chunks ----------------------------------------------
            max_chars = 1200
            fp = chunking_fingerprint(max_chars)
            source, chunks, report = ingest_pdf(
                pdf_path, visibility="private", max_chars=max_chars
            )

            ok_pages = [p for p in pages if p.quality.value == "ok"]
            st.markdown(f"**页数**：{len(pages)}，有效文本页：{len(ok_pages)}")
            st.markdown(f"**切块预览**：{len(chunks)} 块，默认块长 {max_chars} 字符")
            if report.failed_pages:
                st.caption(f"异常页（跳过）：{report.failed_pages}")

            with st.expander("前 3 块预览", expanded=False):
                for c in chunks[:3]:
                    st.markdown(
                        f"- PDF 第 {c.pdf_page} 页 · {c.section or '(未知)'} · {len(c.text)} 字符"
                    )
                    st.code(
                        c.text[:400] + ("…" if len(c.text) > 400 else ""),
                        language=None,
                    )

            # --- Confirm ------------------------------------------------------
            if st.button(
                f"入库 {fname}", key=f"_ingest_{digest[:12]}", type="primary"
            ):
                source = replace(source, local_path=str(pdf_path))
                store.import_document(source, chunks, report, chunking_fingerprint=fp)

                cache_path = _resolve_cache_for_store(store_root)
                st.session_state._pdf_ingest_cache = str(cache_path)

                with st.spinner("增量建索引（仅嵌入新增块）……"):
                    try:
                        _run_build_index(store_root, cache_path)
                        st.success(
                            f"{fname} 已入库并索引完成。刷新页面后即可提问。"
                        )
                        st.session_state.chunk_count = store.count_chunks()
                    except SystemExit as exc:
                        rc = exc.code if isinstance(exc.code, int) else 1
                        st.error(
                            f"索引失败（退出码 {rc}）。块已写入 store 但向量未生成。"
                            f"请检查 API key 或运行 scripts/build_full_index.py 诊断。"
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"索引异常：{type(exc).__name__}: {exc}")

    st.divider()
    st.caption(
        "以上处理仅在本地执行。块文本片段会发送至 SiliconFlow 嵌入服务生成向量；"
        "提问时相关证据会发送至对话服务。"
    )


def _resolve_cache_for_store(store_root: Path) -> Path:
    """Mirror scripts/run_streamlit_online.resolve_cache for the current store."""
    norm = str(store_root).replace("\\", "/")
    if "geoscience_20261014" in norm:
        return _REPO / "data" / "index" / "embeddings_geoscience_20261014.json"
    if "full_corpus" in norm:
        return _REPO / "data" / "index" / "embeddings_full.json"
    return _REPO / "data" / "index" / "embeddings.json"


def _run_build_index(store_root: Path, cache_path: Path) -> None:
    """Invoke build_full_index with argv set for this store + cache."""
    import importlib.util
    import sys

    _bfi_path = _REPO / "scripts" / "build_full_index.py"
    bfi_spec = importlib.util.spec_from_file_location("build_full_index", _bfi_path)
    bfi_mod = importlib.util.module_from_spec(bfi_spec)
    assert bfi_spec.loader is not None
    bfi_spec.loader.exec_module(bfi_mod)

    old_argv = sys.argv
    sys.argv = [
        "build_full_index.py",
        "--store",
        str(store_root),
        "--cache",
        str(cache_path),
    ]
    try:
        rc = bfi_mod.main()
        if rc != 0:
            raise SystemExit(rc)
    finally:
        sys.argv = old_argv


def _provider_for_model(model_id: str, catalog) -> str:
    """Best-effort provider derivation from catalog; defaults to current."""
    if catalog and catalog.models:
        for m in catalog.models:
            if m.id == model_id:
                return getattr(m, "provider", PROVIDER_DEEPSEEK) or PROVIDER_DEEPSEEK
    return PROVIDER_DEEPSEEK


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


def _source_local_path(source_id: str, evidence_items: list[EvidenceItem]) -> str | None:
    for item in evidence_items:
        if item.source_id == source_id and item.local_path:
            return item.local_path
    # Fallback: look up the source by source_id, then scan PDF/ for a file
    # whose sha256 matches. Covers older imports that did not store local_path.
    store = st.session_state.store
    source = next((s for s in store.load_sources() if s.source_id == source_id), None)
    if source is None:
        return None
    pdf_root = _REPO / "PDF"
    if not pdf_root.is_dir():
        return None
    for pdf in pdf_root.rglob("*.pdf"):
        try:
            if source.sha256 == _sha256_of(pdf):
                return str(pdf)
        except OSError:
            continue
    return None


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _pdf_preview_iframe(path: str, page: int | None = None) -> str:
    """Return an iframe HTML snippet that embeds the local PDF via data URI.

    The PDF bytes are read once and base64-encoded; ``#page=N`` targets the
    first cited page. No filesystem path is exposed to the browser — the PDF
    content is embedded inline so Streamlit's server serves the content without
    exposing the host path.
    """
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        data = base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return ""
    anchor = f"#page={page}" if page else ""
    return (
        f'<iframe src="data:application/pdf;base64,{data}{anchor}" '
        f'width="100%" height="680" '
        f'style="border:1px solid #ccc;border-radius:4px;"></iframe>'
    )


def _evidence_snippet(chunk_id: str, evidence_items: list[EvidenceItem]) -> str:
    for item in evidence_items:
        if item.chunk_id == chunk_id:
            return item.snippet()
    return ""


def _render_citations(citations: list[Citation], evidence_items: list[EvidenceItem]) -> None:
    """Author--year reference list, de-duplicated by source, page evidence kept."""
    st.markdown("**引用（论文证据，作者—年份）：**")
    order: list[str] = []
    groups: dict[str, list[Citation]] = {}
    for citation in citations:
        if citation.source_id not in groups:
            groups[citation.source_id] = []
            order.append(citation.source_id)
        groups[citation.source_id].append(citation)

    for source_id in order:
        items = groups[source_id]
        first = items[0]
        label = first.label or format_citation(first)
        pages = sorted({c.pdf_page for c in items})
        header = safe_markdown(f"{label} · {first.title} —— PDF 第 {pages[0]} 页起")
        with st.expander(header, expanded=False):
            authors = "、".join(first.authors) if first.authors else "作者未知（未核实）"
            st.markdown(safe_markdown(f"- 作者：{authors}"))
            st.markdown(
                safe_markdown(f"- 年份：{first.year if first.year else '年份未知（未核实）'}")
            )
            st.markdown(safe_markdown(f"- 标题：{first.title}"))
            if first.doi_or_url:
                st.markdown(safe_markdown(f"- DOI/URL：{first.doi_or_url}"))
            for citation in items:
                st.markdown(f"**支持页：PDF 第 {citation.pdf_page} 页**")
                snippet = _evidence_snippet(citation.chunk_id, evidence_items)
                if snippet:
                    # Literal display so digits, signs and symbols are never
                    # re-parsed as markdown (a lone ``~`` becomes strikethrough).
                    st.code(snippet, language=None)
                st.caption(f"证据块 `{citation.chunk_id}`")
            local_path = _source_local_path(source_id, evidence_items)

            if local_path:
                try:
                    pdf_bytes = Path(local_path).read_bytes()
                    fname = Path(local_path).name
                    st.markdown(f"**PDF 文件**（共 {len(pdf_bytes)//1024} KB，打开至第 {pages[0]} 页）：")
                    st.download_button(
                        f"下载 {fname}",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        key=f"_dl_{source_id}_{pages[0]}",
                        use_container_width=True,
                    )
                except OSError:
                    st.caption(f"PDF 文件不可访问：{local_path}")


def _render_answer(answer: Any, evidence_items: list[EvidenceItem]) -> None:
    st.markdown(safe_markdown(answer.text))
    _model_caption(answer.meta)

    if answer.status == AnswerStatus.ANSWERED.value:
        unsupported = answer.meta.get("unsupported_aspects") or []
        if unsupported:
            st.warning(
                safe_markdown(
                    "**尚无论文证据支持的部分**\n\n"
                    + "\n".join(f"- {item}" for item in unsupported)
                )
            )
        if answer.citations:
            _render_citations(answer.citations, evidence_items)
        extended = answer.meta.get("extended_explanation")
        if extended:
            with st.expander("通用背景补充（未经本地文献核验）", expanded=False):
                st.markdown(safe_markdown(extended))
                st.caption("以上是一般背景知识，不是本域文献结论，不得当作引用来源。")

    elif answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value:
        extended = answer.meta.get("extended_explanation")
        if extended:
            with st.expander("通用背景补充（未经本地文献核验）", expanded=False):
                st.markdown(safe_markdown(extended))
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
        st.markdown(
            safe_markdown(f"- {label}{('（' + str(source.year) + '）') if source.year else ''}")
            + f" · `{source.visibility}`"
        )


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
                st.markdown(safe_markdown(message["text"]))
        else:
            with st.chat_message("assistant"):
                if message.get("kind") == "error":
                    st.error(message["text"])
                else:
                    _render_answer(message["answer"], message.get("evidence", []))

    # PDF upload panel (collapsible, triggered by the + button) -----------
    if st.session_state.get("show_pdf_upload", False):
        with st.expander("添加 PDF 到知识库", expanded=True):
            _render_pdf_upload()
        st.divider()

    # Input ----------------------------------------------------------------
    _render_composer()


if __name__ == "__main__":
    main()
