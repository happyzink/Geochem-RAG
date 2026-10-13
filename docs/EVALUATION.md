# GeoChem-RAG 评估与接口说明（CODEartsAgent）

状态：**部分完成**。检索、受约束回答、引用校验、拒答、评估脚手架与离线自检已实现并通过 104 项离线测试。双 provider 集成已验证：**嵌入**用 SiliconFlow（`BAAI/bge-m3`，密钥 `SILICONFLOW_API_KEY`），**对话**用 DeepSeek 官方 API（`deepseek-chat`，密钥 `DEEPSEEK_API_KEY`），两套密钥/模型/base URL 独立读取。真实「问题→检索→回答→引用」全链路已于 2026-09-28 跑通（见 §3、§7.1、`eval/results/online_e2e_20260928T112233Z.json`）。dense/hybrid 同条件对照因运行环境沙箱间歇屏蔽 API 域名而未完成批量跑，命令与输出 schema 见 §7，网络恢复后在沙箱外执行即可补齐。**本文未含任何未经复现的提升百分比。**

## 1. 范围与职责

负责 `providers.py`、`retrieval.py`、`answering.py`、`evaluation.py`、`offline.py`、`.env.example`、对应测试与本文。不修改 PDF 抽取（`ingest.py`/`store.py`）与 Streamlit 页面。第一版不做表格定位、OCR、地化计算与代码执行。

## 2. 配置与安全

- **双 provider，独立密钥**：对话密钥只从 `DEEPSEEK_API_KEY` 读取（`ProviderConfig.from_deepseek_env`），嵌入密钥只从 `SILICONFLOW_API_KEY` 读取（`ProviderConfig.from_env`）。两套密钥/模型/base URL 互不耦合，由 `chat_provider_from_env()` 与 `embedding_provider_from_env()` 分别构造。`.env` 已被 `.gitignore` 忽略，`.env.example` 不含真实密钥。
- 模型 ID 与 base URL 从配置读取，不写死。`scripts/list_models.py` 可重新核对 SiliconFlow 当前可用模型；DeepSeek 官方模型为 `deepseek-chat`/`deepseek-reasoner`。
- 错误分类为 `ProviderError(kind)`：`not_configured/auth/bad_request/rate_limit/server/network/timeout/bad_response`；429/5xx/超时/网络错误可重试，401/400/422 不重试。错误信息不含密钥（测试覆盖）。
- 文档/模型输出视为不可信输入：系统提示要求忽略证据块内指令；全程不 `eval`/`exec` 模型输出。
- 可选 `GEOCHEM_DISABLE_PROXY=1` 绕过系统代理（本机代理 127.0.0.1:10808 不可达时使用）。
- 数据流：私有文献仅在本地入库；调用在线 API 回答问题时，相关证据片段会传给对话 provider（DeepSeek），嵌入 provider（SiliconFlow）只接收检索文本。不得把整本私有教材上传到在线解析服务。

## 3. 模型核实（2026-09-28，真实 API）

通过真实 POST 调用核实，选用免费可用模型：

| 用途 | 平台 | 模型 ID | base URL | 实测 |
| --- | --- | --- | --- | --- |
| 对话 | DeepSeek 官方 | `deepseek-chat` | `https://api.deepseek.com` | POST `/chat/completions` 返回 `ok`；e2e 中 q11 回答 458 completion tokens |
| 嵌入 | SiliconFlow | `BAAI/bge-m3` | `https://api.siliconflow.cn/v1` | POST `/embeddings` 返回 1024 维；全语料 262 块嵌入成功 |
| 重排（备选，第一版未启用） | SiliconFlow | `BAAI/bge-reranker-v2-m3` | `https://api.siliconflow.cn/v1` | 平台在列 |

错误路径已实测：错误模型 ID → HTTP 400 `Model does not exist...`，被映射为 `ProviderError(kind=bad_request)`。完整匿名化记录见 `eval/results/online_verification_record.json`（gitignored）。

### 3.1 真实端到端验证（q11，hybrid）

`scripts/_e2e_retry.py` 于 2026-09-28T11:22:33Z 跑通：SiliconFlow 嵌入 262 块（dim=1024）→ hybrid 检索 Top5 → DeepSeek `deepseek-chat` 受约束回答。结果 `status=answered`，4 条引用全部绑定本次证据块（`citations ⊆ evidence`），`rejected_citation_ids=[]`，无虚构引用。回答用中文概括了莫根通埃达克质岩体由加厚下地壳部分熔融形成并有岩浆混合贡献的成因。记录见 `eval/results/online_e2e_20260928T112233Z.json`（gitignored，不含密钥或私有全文）。

## 4. 模块接口签名

```python
# providers.py —— 双 provider 工厂（密钥/模型/base URL 独立）
chat_provider_from_env(env=None) -> ChatProvider          # 默认 DeepSeek，读 DEEPSEEK_API_KEY
embedding_provider_from_env(env=None) -> EmbeddingProvider # 默认 SiliconFlow，读 SILICONFLOW_API_KEY

ProviderConfig.from_env(env=None) -> ProviderConfig        # SiliconFlow：读 SILICONFLOW_API_KEY
ProviderConfig.from_deepseek_env(env=None) -> ProviderConfig  # DeepSeek：读 DEEPSEEK_API_KEY
config.require_key() -> str                                # 缺失抛 ProviderError(not_configured)

SiliconFlowClient(config).chat(messages, *, temperature=0.0, max_tokens=None, json_mode=False) -> ChatResult
SiliconFlowClient(config).embed(texts) -> list[list[float]]  # 内部按 32 批分片
DeepSeekChatClient(config).chat(...) -> ChatResult          # 复用 SiliconFlowClient 的 HTTP 传输
DeepSeekChatClient(config).embed(...) -> raises ProviderError(bad_request)  # 不提供嵌入

# retrieval.py
Retriever.from_store(store, embedder, *, visibility=None, cache_path=..., rrf_k=60, candidate_k=20) -> Retriever
retriever.retrieve(question, mode="hybrid", top_k=5) -> list[SearchHit]          # mode ∈ {dense_only, bm25_only, hybrid}
retriever.retrieve_with_trace(question, mode, top_k) -> RetrievalTrace           # 含 expanded_query/dense_ranking/bm25_ranking/rrf_k
expand_query(question) -> ExpandedQuery(original, expanded, added_terms, matched_terms)  # 不覆盖原查询

# answering.py
EvidenceAnswerer(store, chat_provider, *, top_k=5, max_evidence_chars=900, visibility=None)
  .answer(question, hits, *, top_k=5, allow_extended=True) -> Answer
    # allow_extended=True: 空证据时调用 LLM 生成简短通用背景（标注"未经本地文献核验"），存 meta.extended_explanation
    # status 仍为 insufficient_evidence，citations 为空
    # 有证据时：模型若报告无法用证据支持的部分，存 meta.unsupported_aspects
build_evidence(hits, store, *, top_k=5, visibility=None) -> list[EvidenceItem]
format_citation(citation) -> str   # "论文标题，PDF 第 N 页"

# evaluation.py
load_questions(path) -> list[dict]
run_mode(questions, mode, retriever, answerer, store, *, top_k=5) -> list[QuestionResult]
summarize(results, mode) -> dict                 # Recall@5、拒答率、引用合法性等
write_results(results, results_dir, *, manifest=...) -> dict[paths]
write_review_sheet(results, path) -> None        # 人工标注模板
```

返回样例（`Answer`，引用仅来自本次证据）：

```json
{
  "status": "answered",
  "text": "该论文认为莫根通埃达克质岩体由加厚下地壳部分熔融形成……",
  "citations": [{"chunk_id": "chk_6ee572d18962b49b_p0001_c002", "source_id": "src_feart_2022_845763", "title": "Petrogenesis ... Mogetong Adakitic Pluton ...", "pdf_page": 1}],
  "evidence": ["chk_6ee572d18962b49b_p0001_c002", "chk_6ee572d18962b49b_p0001_c003", "chk_6ee572d18962b49b_p0010_c001", "chk_6ee572d18962b49b_p0013_c000", "chk_6ee572d18962b49b_p0014_c004"],
  "meta": {"model": "deepseek-chat", "usage": {"prompt_tokens": 1840, "completion_tokens": 458, "total_tokens": 2298}, "rejected_citation_ids": [], "evidence": [{"chunk_id": "...", "title": "...", "pdf_page": 1, "snippet": "..."}]}
}
```

`status` 取值 `answered / insufficient_evidence / error`；`answered` 必有 ≥1 条引用且 `citations ⊆ evidence`；`insufficient_evidence` 不带引用（可有 `meta.extended_explanation` 通用背景补充，标注"未经本地文献核验"，不构成引用来源）；`error` 不呈现为地学结论。`SearchHit` 字段：`chunk_id, score, rank, retrievers, source_id, pdf_page, dense_score, bm25_score`。

### 4.1 通用背景补充（allow_extended）

当检索无证据（`hits=[]`）且 `allow_extended=True` 时，`answering.py` 用独立的系统提示（`_NO_EVIDENCE_SYSTEM_PROMPT`）调用对话 provider 生成一段简短通用背景解释，存入 `meta.extended_explanation`。该解释：

- 在界面中显示于折叠区"通用背景补充（未经本地文献核验）"，明确标注不是本域文献结论；
- 不产生引用，`status` 仍为 `insufficient_evidence`；
- 不添加原文没有的结论、年代、数值或 DOI；
- provider 出错时 `status=error`，不伪装为成功回答。

`allow_extended=False` 时不调用模型，直接返回拒答文本。有证据时 `allow_extended` 不影响主回答流程；模型若报告部分问题无法用证据支持，存入 `meta.unsupported_aspects`，界面以 warning 展示。

## 5. 检索策略对比（特点/场景/优缺点）

| 维度 | dense-only | hybrid（BM25 + RRF） |
| --- | --- | --- |
| 检索特点 | 语义向量余弦，跨语言能力强 | 语义 + 词面，RRF(`score=Σ 1/(k+rank)`, k=60) 融合两路排名 |
| 适用场景 | 中英跨语言提问、术语改写、概念相近 | 含专有名词/缩写/精确词面的提问，或 dense 召回偏移时 |
| 优点 | 不依赖词面命中 | 精确词面兜底，对缩写/人名/页内关键词更稳 |
| 缺点 | 罕见词面可能漏召 | 多一路索引与融合参数；词面噪声可能引入无关块 |
| 本实现 | bge-m3 1024 维（SiliconFlow），纯 Python 余弦索引（见 §9） | BM25（k1=1.5, b=0.75）+ RRF；查询扩展仅作用于 BM25，保留原查询 |

RRF 参数、扩展词、各路排名均写入 `RetrievalTrace` 与评估 manifest，可追溯。

## 6. 指标定义

- **Recall@5（source）**：可回答题中，Top5 命中任一期望 `source_id` 的题数 / 有期望 source 的题数。
- **Recall@5（page）**：当题集填写了 `expected_pages` 时，Top5 命中 `(source_id, pdf_page)` 对的题数 / 有期望页的题数（当前仅 q05/q08/q16 有页标注）。
- **拒答率**：应拒答题中返回 `insufficient_evidence` 的题数 / 应拒答题总数。
- **误答率**：应拒答题中返回 `answered` 的题数 / 应拒答题总数。
- **引用合法性（citation_membership_rate）**：所显示引用的 `chunk_id` 属于本次证据集的比例（结构校验，非真值）。
- **引用正确率 / 人工支持性**：需人工逐题标注，脚本输出 `review_sheet.jsonl` 模板；填好 `eval/human_labels.jsonl` 后由 `summarize_human_labels` 计算。**未标注前记为 PENDING，不计入任何提升宣称。**

## 7. 同条件对照结果

### 7.1 真实端到端验证（已完成，q11）

```bash
set -a && . ./.env && set +a
.venv/Scripts/python.exe scripts/_e2e_retry.py
```

结果（`eval/results/online_e2e_20260928T112233Z.json`）：`deepseek-chat` + `BAAI/bge-m3`，hybrid 模式，`status=answered`，4 条引用全部来自本次证据，`rejected_citation_ids=[]`，usage total_tokens=2298。回答正确概括了莫根通埃达克质岩体成因（加厚下地壳部分熔融 + 岩浆混合，板片断离背景）。

### 7.2 真实模型批量对照（PENDING）

命令（沙箱外或代理可用时执行；密钥来自 `.env`）：

```bash
set -a && . ./.env && set +a
.venv/Scripts/python.exe scripts/build_index.py
.venv/Scripts/python.exe scripts/run_evaluation.py --modes dense_only hybrid
```

输出：`eval/results/results_<mode>_<stamp>.jsonl`（逐题，含证据 ID/页码/片段/命中排名）、`summary_<stamp>.json`（manifest + 各模式指标 + 人工复核状态）、`review_sheet.jsonl`。manifest 记录语料 SHA、题集 SHA、模型 ID（`deepseek-chat` + `BAAI/bge-m3`）、RRF/top_k 与时间，保证同条件可复跑。

### 7.3 离线自检（harness self-test，**非质量结果**）

用确定性占位 provider（`HashingEmbedder` 128 维 + `TemplateChatProvider`，见 `offline.py`）在无网络下跑完全流程，验证脚手架与失败检测：

```bash
.venv/Scripts/python.exe scripts/run_evaluation.py --offline --results eval/results_offline
```

`eval/results_offline/summary_20260928T100054Z.json`（题集 SHA `33f87339967fadb5`，语料 262 块）：

| 指标 | dense_only | hybrid |
| --- | --- | --- |
| Recall@5 source | 22/24 = 0.917 | 23/24 = 0.958 |
| Recall@5 page | 0/3 = 0.00 | 1/3 = 0.333 |
| 拒答率（应拒答 12 题） | 0/12 = 0.00 | 0/12 = 0.00 |
| 误答率 | 12/12 = 1.00 | 12/12 = 1.00 |
| 引用合法性 | 80/80 = 1.00 | 80/80 = 1.00 |
| 平均延迟 | 0.172 s | 0.162 s |

**解读**：占位 chat 永远引用首个证据、从不拒答，因此误答率 1.00、拒答率 0.00——这正是脚手架应检测到的失败模式，说明拒答/误答指标有效。Recall 数字由哈希嵌入产生，**不代表真实检索质量**，仅证明流程贯通与同条件对照表能正确生成。引用合法性 1.00 说明引用校验生效。

## 8. 失败案例与误差分类

误差分为抽取 / 检索 / 生成 / 引用映射四类。离线自检暴露的典型失败（见 `eval/results_offline/results_*.jsonl` 中 `should_refuse=true` 且 `status=answered` 的行）：

- **生成类**：占位模型对不可问题仍作答（q31–q40），未触发拒答。真实模型下需复核是否真的拒答。
- **引用映射类**：`answering.py` 对模型虚构的 `chunk_id` 计入 `meta.rejected_citation_ids` 并降级为 `insufficient_evidence`（单测 `test_fabricated_citation_is_rejected` 覆盖）。
- **检索类**：跨语言中文提问在 dense-only 下 Recall@5 page=0（占位嵌入无语义），hybrid 下页级召回 1/3。
- **API 类**：`test_provider_error_becomes_error_status`、`test_run_question_retrieval_error` 覆盖超时/网络/400 → `error` 状态，不伪装为地学结论。

## 9. 已知限制

1. **向量库**：计划用 Chroma，但本环境 PyPI 不可达、`chromadb` 未安装；`DenseIndex` 改为纯 Python 余弦索引，接口与 `search(query_vector, top_k)` 一致，可后续替换为 Chroma 而不动调用方。
2. **中文分词**：无 jieba；`tokenize` 对 CJK 取字符单字 + 二元组，对 Latin 取词/数字。跨语言主要靠 dense（bge-m3）承担。
3. **BM25 扩展**：地学缩写/术语词典仅作用于 BM25 查询，保留原查询；词典在 `retrieval._GEO_EXPANSIONS`，可审阅。
4. **人工指标**：引用正确率与支持性需人工标注，当前 PENDING。
5. **网络**：本会话沙箱间歇屏蔽 `api.siliconflow.cn` 与 `api.deepseek.com`（WinError 10051 / `hit restricted`），真实全链路在窗口打开时跑通（§7.1），但批量对照待在网络稳定可达环境补跑（§7.2）。
6. **语料**：当前 4 篇 Frontiers 公开样本（CC BY 4.0）用于打通流程；正式语料由用户放入 `PDF/` 并经许可审查后重新导入。

## 10. 复现命令

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\run_evaluation.py --offline --results eval\results_offline
# 真实 API（需两套密钥与网络）：
$env:SILICONFLOW_API_KEY = "<your-siliconflow-key>"
$env:DEEPSEEK_API_KEY = "<your-deepseek-key>"
.\.venv\Scripts\python.exe scripts\build_index.py
.\.venv\Scripts\python.exe scripts\run_evaluation.py --modes dense_only hybrid
```

Linux / git-bash：

```bash
.venv/Scripts/python.exe -m pytest tests -q          # 或 python -m pytest
.venv/Scripts/python.exe scripts/run_evaluation.py --offline --results eval/results_offline
set -a && . ./.env && set +a
.venv/Scripts/python.exe scripts/build_index.py
.venv/Scripts/python.exe scripts/run_online_e2e.py --question-id q11 --mode hybrid
.venv/Scripts/python.exe scripts/run_evaluation.py --modes dense_only hybrid
```

## 11. 人工复核流程

1. 跑 §7.2 生成 `eval/results/review_sheet.jsonl`（每题含答案、引用显示串与证据片段）。
2. 逐题填 `citations[*].correct` 与 `answer_facts[*].supported`，保存为 `eval/human_labels.jsonl`。
3. 重跑 `run_evaluation.py`，`summary_*.json.human_review` 即给出引用正确率与支持性（分子/分母公开）。