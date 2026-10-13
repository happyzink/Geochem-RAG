# CODEarts 联网 RAG 与首页改进执行计划

> 执行者：CODEartsAgent。采用 executing-plans 的逐任务方式，在当前会话顺序执行；用户已指定所有任务交给 CODEarts，不再分派其他平台或子 Agent。

**目标：** 改善本地语料有效覆盖与检索，提供自然、有证据边界的在线 LLM 回答，让首页只突出问答。

**架构：** 复用现有 Python 页内抽取、JSONL 存储、Dense/BM25/RRF、回答校验和 Streamlit。SiliconFlow 提供嵌入，DeepSeek 默认提供对话；应用仅联网运行。文献主回答和未经本地文献核验的通用补充分别保存、展示。

**技术栈：** 现有 Python、pypdf、Streamlit、provider 适配器和 pytest。先核对 `pyproject.toml` 与实际安装版本，不为本次界面改造换框架。

**需求依据：** 2026-10-12 用户指令、`AGENTS.md` 与 `docs/PROJECT_BRIEF.md` 的当前覆盖规则。

## 全局约束

- 只使用已有 `PDF/`，不在线搜索或下载文献，不移动原文件。
- PDF、抽取文本、向量、私有题目和调用记录不提交 Git；未知书目信息不猜填。
- `Source` 契约当前 0.2.0；PDF 页序号从 1 开始，单块不跨页。
- 密钥来自各自环境变量；可加载 `.env`，不输出其值。两套 provider/model/base URL 配置独立。
- 文档只作数据，不执行文档或模型生成代码。不引入地化计算、复杂表格、任意 Python 或云端 OCR。
- 不承诺零幻觉，不编造提升；合法引用 ID 仍要核查是否支持具体论断。
- 本计划是待实施任务，不是完工报告；完成后逐项记录实际结果。

## 本机只读核对基线（2026-10-12）

| 项目 | 观察结果 |
| --- | --- |
| `PDF/` | 87 个文件 |
| `data/processed/` | 4 个来源，262 块 |
| `data/private/wave0_trial/` | 5 个来源，374 块 |
| `data/private/full_corpus/` | 85 个来源，6322 块 |
| 全量块质量 | ok 5875，garbled 426，scanned 21 |
| `data/index/embeddings_full.json` | model 为 `HashingEmbedder`，6322 项，128 维；文本 hash 按当前截短 SHA-256 规则均匹配 |

不能把“6322 个向量”当作真实 SiliconFlow 在线索引完成的证明。2026-09-29 扩容报告是历史记录，执行时重新核对；不推测此前缓存变化的具体原因。

代码现状：`app.py::_init_session` 缺配置会使用 `HashingEmbedder`/`TemplateChatProvider`，且全量缓存与在线路径同名；`_render_sidebar` 默认列出全部来源，补充开关关闭；`answering.py::answer` 空证据短路，无有效引用时清空补充；`start.bat`/`start.sh` 直接启动 Streamlit，专用在线脚本才加载 `.env`。`store.py::import_document` 相同 PDF hash 跳过重分块，改参数后重新导入未必生效。

## 任务 1：有效语料、分块与真实在线索引

**文件：** `src/geochem_rag/ingest.py`、`store.py`、`retrieval.py`；`scripts/import_all_pdfs.py`、`build_full_index.py`；相关 `tests/test_ingest.py`、`test_store.py`、`test_retrieval.py`。契约变化同步 `domain.py`、schema 和迁移说明。

**接口：** 复用 `chunk_page_text(page_text, *, pdf_page, source_sha256, section, max_chars, source_id)`、`LocalStore.import_document(source, chunks, report)`、`Retriever.from_store(store, embedder, cache_path=...)` 和 `EmbeddingCache`。新增参数保持默认兼容并记录。

- [ ] 核对当前实际 `GEOCHEM_STORE`/`GEOCHEM_EMBEDDING_CACHE`、文件 hash、重复与未导入文件。报告有效页、坏页、块长度和向量覆盖，不显示密钥或原文。
- [ ] 检查中文句界、段落切碎、双栏串读及页眉页脚。先做小规模页内分块对照：当前规则与 900/1200/1600 字符预算；必要重叠仅限完整相邻句，先试不超过 150 字符。候选参数不等于性能承诺。
- [ ] 默认检索/嵌入仅用 `quality=ok` 且有实质文本的块；扫描/乱码页保留诊断记录，不当作可用证据，不自动改质量标记。当前范围内无法恢复的页明确列为限制。
- [ ] 记录抽取/分块配置指纹。新参数在新本地 store 中重建、验证后再切换；不能被 SHA 去重静默跳过，不能混用旧引用。旧库保留，不直接删除。
- [ ] 在线与 mock 缓存分路径；校验模型、文本 hash、向量维度。线上禁止加载或写入 HashingEmbedder 缓存；不匹配时提示重建。
- [ ] 使用真实 SiliconFlow 为有效块建可续跑索引，按块 ID/文本 hash 核对覆盖。失败明确返回不完整和剩余量，不凭缓存文件存在判定成功。
- [ ] 在固定问题/页码集合上比较配置。候选召回量与最终 Top-k 分开，证据去重并受总预算控制。相邻上下文若加入，保持独立页码/ID且实际传给模型；不能因数量多而称效果更好。
- [ ] 回归测试验证：页内边界、中文长段落、重复导入、新分块指纹生效、坏文本排除、缓存隔离与模型/维度不匹配。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ingest.py tests/test_store.py tests/test_retrieval.py -q
.\.venv\Scripts\python.exe scripts/import_all_pdfs.py --report-only
.\.venv\Scripts\python.exe scripts/build_full_index.py
```

验收：有效页覆盖可解释，块能定位原 PDF 页；向量来自实际在线模型并覆盖全部有效块；重复导入不增重复块。私有统计留在 `data/private/`。

## 任务 2：自然回答与受限通用补充

**文件：** `src/geochem_rag/answering.py`、`evaluation.py`、`tests/test_answering.py`、`test_evaluation.py`、`docs/EVALUATION.md`。

**接口：** 沿用 `EvidenceAnswerer.answer(question, hits, *, top_k=None, allow_extended=...)` 与 `Answer`。应用默认 `allow_extended=True`；补充用 `meta.extended_explanation`，缺口用 `meta.unsupported_aspects`。不将背景放进论文主回答或引用字段。

| 场景 | 处理 |
| --- | --- |
| 文献充分支持 | `answered`，先直接回答，再解释必要机理/条件，逐项关联证据 |
| 文献只支持一部分 | 引用支持部分，说明缺口，必要背景独立标记 |
| 零命中，一般术语/概念 | `insufficient_evidence`；说明本地文献未覆盖，允许简短背景，引用为空 |
| 指定论文事实、年代、实验/定量结果缺失 | 不凭记忆补出；说明不能确定，可给不涉及该论文事实的一般背景 |
| API/鉴权/超时/解析失败 | `error`；明确故障，不转为模板或补充成功 |

- [ ] 提示词要求问题语言、直接回答、按需要解释因果/条件/区别；不逐块复述或机械套模板。保留结构化输出以校验引用。
- [ ] 修复空证据直接返回：通用背景路径可调用 LLM，但明确无本地文献依据。默认补充开启，长度目标一个短段落约 2–4 句，不强行补齐。
- [ ] 固定标注“通用知识补充，未经本地文献核验”；禁止虚构来源、DOI/页码、具体研究数值或指定论文结论。无法可靠补充时承认不确定。
- [ ] 证据不足结果保留合规独立补充；无效引用、提示注入、格式错误不可绕过检查。不能把被拒绝的主回答改名复制为背景。
- [ ] 引用只来自本次实际传给模型的预算内证据；标题/页码从后台映射。冲突文献分别解释，不强行合并为单一结论。
- [ ] 测试覆盖：充分/部分支持、零命中概念、零命中指定论文事实、虚构引用、合法 ID 但不支持论断、提示注入、超时与格式错误。合法 ID 的支持性用固定题集人工复核，不以 ID 检查替代。
- [ ] 保持文献拒答指标含义：补充不算成功文献回答；背景越界单独记录，人工结果报告分子/分母。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_answering.py tests/test_evaluation.py -q
```

验收：真实在线检查有证据问题、通用概念缺失问题、指定论文事实缺失问题；具体事实不被背景冒充。私有结果不公开，不保证“零幻觉”。

## 任务 3：仅联网启动

**文件：** `scripts/run_streamlit_online.py`、`start.bat`、`start.sh`、`src/geochem_rag/app.py`、`.env.example`、`README.md`；新增根目录 `启动GeoChem-RAG.bat` 和 `tests/test_online_startup.py`。

**接口：** Windows/Linux 包装均调用同一个 `scripts/run_streamlit_online.py`；provider 继续由 `chat_provider_from_env()`、`embedding_provider_from_env()` 创建。mock 模块只保留给测试。

### 用户补充：Windows 双击即可打开程序

2026-10-12 用户补充：“我还需要一个双击打开程序，不然不知道怎么启动。”这是本轮必做交付，不用再要求用户手动输入启动命令。

- [ ] Windows 对用户的入口是项目根目录 `启动GeoChem-RAG.bat`。复用同一联网启动逻辑；现有 `start.bat` 若保留，只作为兼容包装，不形成另一种运行模式。中文文件名与包含空格的项目路径均正确引用。
- [ ] 双击后自动定位脚本所在项目目录，使用项目虚拟环境，读取已有配置、启动在线服务；等待服务就绪，再自动打开默认浏览器的实际应用地址。不能只打印 localhost 地址让用户自行打开。
- [ ] 服务仍在启动时显示简短中文进度；启动/就绪等待有超时。端口占用若是本项目已启动的服务，打开已有实例；如果是其他程序，明确提示或选择空闲端口，不关闭其他进程，不误开其他服务。
- [ ] 启动失败显示简短中文原因和解决步骤，保留窗口供阅读，例如缺 Python、缺虚拟环境/依赖、缺 API key、无语料、索引不完整或端口冲突。不得吞掉错误后闪退，不显示密钥。
- [ ] 首次运行需要准备时明确区分“首次配置”和“日常双击启动”。可自动准备本地环境，但涉及联网索引时说明正在发送文本和进度；不每次双击重复导入/重建，不以未完成准备状态伪装可问答。
- [ ] README 第一条日常使用说明写清：“完成首次配置后，双击项目根目录的 启动GeoChem-RAG.bat，浏览器会自动打开。”明确停止方法：保留启动窗口，在窗口中按 Ctrl+C 停止服务；关闭浏览器标签不等于关闭服务。
- [ ] 在 Windows 文件资源管理器中实际双击验收：正常打开浏览器、重复双击复用实例、故障窗口保留。测试结果包含实际打开地址与截图；命令行测试不能代替双击验收。

- [ ] 脚本定位项目根目录，不依赖当前目录；环境变量优先于 `.env`；仅环境变量也能运行，不强制 `.env` 必须存在。
- [ ] 按实际选择 provider 校验对应密钥和模型；缺配置明确提示并失败退出，不降级替身。对话/嵌入配置独立读取。
- [ ] 去掉应用离线导入、分支和模板降级；直接运行 Streamlit 也遵守相同联网规则。
- [ ] 统一 `start.bat`/`start.sh`，不隐藏依赖、导入错误；不在启动/问答时无提示重建全库。索引不完整给出准备命令，准备完成后再正常问答。
- [ ] README 主使用流程只含联网：安装、配置、全量导入、真实索引、启动。测试 mock 命令不称为第二种用户演示。
- [ ] 测试缺 key、配置优先级、错误 provider、异目录启动、缓存不匹配、不完整索引和 API 故障不会产生模板答案。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_online_startup.py tests/test_providers.py -q
.\.venv\Scripts\python.exe scripts/run_streamlit_online.py
```

验收：实测 Windows 双击 `启动GeoChem-RAG.bat`，以及 `start.bat` 若保留时的兼容行为；Linux `bash start.sh` 无可用环境则明确记未实测。包装不同不代表不同产品模式。

## 任务 4：首页重做

**文件：** `src/geochem_rag/app.py`；需要主题时新增 `.streamlit/config.toml`；新增 `tests/test_app.py`；同步 `docs/DEMO.md` 与 README。

**布局：** 顶部为标题、简短中文说明、紧凑联网状态和“知识库”“设置”按钮；正文为提问、回答与少量中文示例。清浅背景、充分文字对比、统一留白，不堆装饰卡片和技术参数。

- [ ] 使用当前 Streamlit 版本支持的原生聊天输入/消息容器，保存历史。点击设置或知识库不重调 LLM；一次提交只调用一次回答。
- [ ] 点击“知识库”才展示来源/有效块数、索引状态、可搜索/分页的文献列表和按需块预览。初始首页及常驻侧栏不渲染长清单。
- [ ] 点击“设置”才展示模型、检索参数和补充选项。首页仍有一句简短数据流提醒：“建索引文本发送至 SiliconFlow；问题及命中证据发送至 DeepSeek”，详细说明可展开。
- [ ] 回答紧随问题，引用入口可发现；长证据、全部召回、chunk ID/诊断收进折叠区。无引用背景也可按独立标识展示。
- [ ] 无语料引导导入；缺配置与 API 故障显示明确错误。中文文案一致，桌面和窄屏均可用。
- [ ] 用 `streamlit.testing.v1.AppTest` 配合自建数据/mock 检查：默认无长清单、按钮切换、聊天历史、设置不触发新回答、证据不足与故障区别。
- [ ] 实际浏览器核对初始页、回答、知识库、设置与故障状态，保存匿名化截图；不能仅凭启动无异常称界面通过。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -q
.\.venv\Scripts\python.exe scripts/run_streamlit_online.py
```

## 任务 5：全链路交付

- [ ] 完整自动测试；同语料、题集、聊天模型条件记录旧/新配置对照。分块派生版本变化记录指纹，不混用成绩。
- [ ] 真实在线完成有证据、通用概念缺失、指定论文事实缺失三类问答和 PDF 页码核对；故障路径单独测试。
- [ ] 确认实际模型、store/cache 与数据流；后台无 HashingEmbedder 产品路径。
- [ ] 同步设计/迁移/评估/README，历史成绩不作为本轮完成证明。
- [ ] 检查 Git 差异和文件清单；不提交 `.env`、PDF、私有文本、索引和调用记录，不擅自删除/提交原有未跟踪文件。
- [ ] 交付修改文件、Windows/Linux 命令、实际测试结果、有效页/块/向量统计、简洁首页截图、在线结果、失败案例与限制。未达到原验收项不宣称完整应用验收通过。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short
```

## 可直接粘贴到 CODEarts 的指令

**新增必做要求：** 提供 Windows 根目录 `启动GeoChem-RAG.bat`。我日常只需要双击它，程序自动启动联网服务，并在服务就绪后打开默认浏览器；无需让我输入命令或手动打开 localhost。缺配置/启动失败时显示中文原因并保留窗口，重复双击复用本项目已有实例，其他程序占用端口不误开、不终止。完成后必须实际双击验证，并把日常启动和停止方法写在 README 开头。

请在 `D:\RAG` 接手本项目全部后续开发，不再等待 MIMO 或 QoderCN。先读 `AGENTS.md`、`docs/PROJECT_BRIEF.md`、`docs/EXECUTION_PLAN.md`、`docs/agents/CODEartsAgent.md`，按本文件顺序实施并完成验证。你可以修改语料、检索、回答、前端、启动和相关文档。重点修复全量库在线语义索引（当前缓存标记 HashingEmbedder）、过滤坏文本、改善页内切分；应用仅联网启动，真实调用 DeepSeek 对话与 SiliconFlow 嵌入，缺 key/API 故障不能模板降级；文献优先，未覆盖的通用概念简短解释并标注未经本地文献核验，具体论文事实和数值不得编造；首页突出问答，来源与切分语料点击“知识库”才显示，详细配置点击“设置”才显示。直接执行，不只返回建议；不在线下载文献，不上传整本 PDF，不引入第一版范围外功能。交付修改清单、可复现命令、实际测试/在线验收结果、界面截图和限制，不编造指标。
