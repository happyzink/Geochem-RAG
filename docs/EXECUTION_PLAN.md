# GeoChem-RAG 第一版执行计划

> 最新执行方覆盖规则（2026-10-12）：全部未完成任务由 TraeCode 接手，见 `docs/agents/TraeCode接手任务.md`。以下 CODEarts 指令作为历史交接，不能要求它继续实施；保留有效改动后逐项补齐验收。

状态（2026-10-12）：导入、检索、回答与界面代码已存在；以下 Wave 清单保留为原始验收记录，未逐项重新认定完成。本轮全部改进由 CODEartsAgent 顺序实施，不再向 MIMO/QoderCN 分配任务。

## 当前执行入口与覆盖规则

执行 `docs/agents/CODEartsAgent.md` 和 `docs/superpowers/plans/2026-10-12-codearts-online-rag-upgrade.md`：先核对有效语料与真实在线索引，再改进回答，最后统一联网启动、简化首页并验收。以下旧的协作分工、仅 SiliconFlow 对话、固定 Top 5 输入、无证据一律不允许背景解释、离线应用入口和“尚非 Git 仓库”描述不再作为当前指令。

当前对话默认 DeepSeek、嵌入默认 SiliconFlow；单一联网应用入口，不自动降级模板；默认允许明确标识的简短通用背景解释，具体论文结论仍以证据为限。保留检索/引用指标口径，背景解释边界单独检查，不把无引用背景算成功文献回答。原始 Wave 的隐私、页码、可复现性与最终复核要求继续适用。

用户补充的 Windows 双击启动也是必做验收：交付根目录 `启动GeoChem-RAG.bat`，自动启动同一联网服务、等待就绪并打开默认浏览器；出错窗口保留。日常使用无需命令行，必须实际从文件资源管理器双击验证，详见本轮计划任务 3。

**目标**：用用户已有的本地岩石地化 PDF 构建文献问答演示，回答有逐页证据，评估能暴露检索、引用与拒答错误。Agent 不在线下载文献。

**架构**：Python 负责 PDF 抽取、页内分块、Chroma 向量索引、BM25 检索、RRF 融合和回答校验；SiliconFlow 提供 Embedding 与对话 API；Streamlit 展示结果。每个模块通过 `docs/PROJECT_BRIEF.md` 中的对象字段连接。

**范围约束**：第一版不做 OCR、复杂表格、数值计算和代码执行；用户现有资料从 `PDF/` 或 `data/private/` 导入，默认不提交 Git；所有成绩在实测后填写。

## 建议目录与责任

```text
GeoChem-RAG/
├── AGENTS.md
├── README.md
├── PDF/                            # 用户已有本地语料；整个目录 Git 忽略
├── pyproject.toml                  # MIMO 首建，后续共同维护依赖
├── .env.example                    # CODEartsAgent 建立，无真实密钥
├── data/
│   ├── public/                     # 已有验证文件；默认 Git 忽略
│   ├── private/                    # 用户稍后复制；Git 忽略
│   └── sources.json                # MIMO：当前来源清单；本地来源另作登记
├── src/geochem_rag/
│   ├── domain.py                    # MIMO：对象类型及校验
│   ├── ingest.py                    # MIMO：PDF 逐页解析与分块
│   ├── store.py                     # MIMO：本地持久化与去重
│   ├── providers.py                 # CODEartsAgent：SiliconFlow 适配
│   ├── retrieval.py                 # CODEartsAgent：dense / BM25 / RRF
│   ├── answering.py                 # CODEartsAgent：证据约束回答与引用校验
│   ├── evaluation.py                # CODEartsAgent：指标计算与题目运行
│   └── app.py                       # QoderCN：Streamlit UI
├── eval/
│   ├── questions.jsonl             # MIMO 起草，用户地学审核
│   └── results/                    # 生成文件，不提交私有原文
├── tests/                           # 每个 Agent 测自己的模块
└── docs/                            # 设计、任务、评估报告
```

根目录不是 Git 仓库。开始开发时由用户或协调者决定是否执行 `git init`；公开仓库发布前做一次文件清单与密钥扫描。`PDF/`、`data/private/`、`data/public/` 均默认忽略。公开演示只使用用户明确确认可再分发的本地文件或自建测试样例；没有合格文件时，README 指导用户添加自己的本地 PDF，不写下载脚本。

## Wave 0：资料与契约（MIMO，随后协调者审核）

- [ ] 清点 `PDF/` 中用户已有的 PDF，先选 3–5 篇与岩石地球化学直接相关、可提取文本的文件作为本地试验子集。为每篇记录文件名、SHA-256、可确认的标题/作者/年份及 `private` 可见性；未知书目信息留空并标记待核，不在线查找或补造。
- [x] 将现有 `Source` 0.1.0 契约迁移到本地资料场景：私有来源允许作者、年份、DOI、许可证、下载日期缺失；标题可明确标注为“文件名暂代”，不能伪造作者/年份；公开来源仍严格要求许可证明。schema 样例、CLI 和测试已更新；迁移说明见 `docs/CONTRACT_MIGRATION_0.2.md`。
- [ ] 在 Git 忽略的 `data/local_sources.json` 登记本地试验子集；由私有语料派生的评估题放到 Git 忽略的 `eval/private/`，不要覆盖既有公开样例题集。
- [ ] 保留既有 `data/sources.json` 和 `data/public/` 的历史验证结果，但不再下载新文献；这些文件默认不作为公开仓库样本。公开材料须由用户从本地文件中明确指定并确认再分发条件。
- [ ] 在 `src/geochem_rag/domain.py` 定义 `Source`、`EvidenceChunk`、`SearchHit`、`Answer`，字段与 `PROJECT_BRIEF.md` 一致；用测试固定 PDF 页码从 1 开始、`chunk_id` 稳定且不同页不混合。
- [ ] 在 `src/geochem_rag/ingest.py` 实现 born-digital PDF 逐页文本抽取、质量标记与页内分块；OCR 和表格抽取失败要显式列在入库报告中。
- [ ] 在 `src/geochem_rag/store.py` 实现来源、块的本地保存与 SHA-256 去重。删除或更新来源时清理对应块；公开和私有来源可过滤。
- [ ] 基于本地试验子集在 `eval/private/questions.jsonl` 起草 40 道题与标准页码；地学答案要点由用户复核后锁定评估集版本。保留原有 `eval/questions.jsonl` 作为历史验证题集，不覆盖它；私有题目和答案默认不公开。

**Wave 0 验收**：同一 PDF 重复导入不重复；每个返回块能打开相应 PDF 页并核对原文；扫描页有明确失败/警告；测试不依赖在线 API。

## Wave 1：检索与受约束回答（CODEartsAgent）

- [ ] `providers.py` 从环境读取 `SILICONFLOW_API_KEY`、base URL、聊天/向量模型 ID；实现超时、限流重试和错误传递。开始实施时查询平台当前可用模型，记录实际模型 ID；不把价格或“免费”假定写成固定事实。
- [ ] `retrieval.py` 建立 Chroma dense 检索与 BM25 检索，通过 RRF 融合。保留 `dense_only` 模式以做同集对照；输出每个 hit 的来源、分数和排序路径。
- [ ] `answering.py` 仅把 Top 5 证据和问题交给模型。提示词使用证据 ID；解析结果时校验每个 ID 都来自本次证据。空证据、无支持的回答或无效引用按规则返回 `insufficient_evidence` / `error`，不可伪造页码。
- [ ] 用假 provider 做确定性测试：英文来源中文问题、同义词/缩写、不同 PDF 页、来源冲突、空检索结果、模型返回虚构引用、API 超时。

**Wave 1 验收**：从 CLI 或 Python API 完成“问题 → 回答 → 来源/页码/短证据”；无效来源 ID 不进入答案；API 错误不显示成正常地学结论；所有单元测试无网络可运行。

## Wave 2：固定评估与迭代（CODEartsAgent，MIMO 协助地学核查）

- [ ] `evaluation.py` 从锁定的题集运行 dense-only 与 hybrid 两种配置；记录语料 SHA、题集版本、模型 ID、时间、Top 5 命中、回答、引用、拒答和费用/延迟。
- [ ] 计算 Recall@5、引用正确率、应拒答题拒答率；人工逐题标注回答支持性并报告分子与分母。将误差分为抽取、检索、生成、引用映射四类。
- [ ] 生成 `docs/EVALUATION.md`，包含指标定义、同条件对照表、逐题结果文件位置、失败案例、实际限制。只有在 baseline 存在且有必要时才试 reranker 或 query rewriting；不要为了功能列表添加模块。
- [ ] 可选 Ragas 辅助分数单独一栏，注明评估模型、语言适配和版本，不与人工引用正确率混为一谈。

**Wave 2 验收**：在同一版本的本地 PDF 与题集上可复跑；结果文件可按题查到证据 ID；私有原文与由其派生的题目不公开；报告不含未经测试的提升宣称。对外可复现演示另用用户确认可公开的本地材料或自建测试样例。

## Wave 3：公开演示与包装（QoderCN）

- [ ] `app.py` 显示答案状态、各引用的标题/PDF 页/短证据，允许打开公开原文；展示检索配置和可见的私有资料上传/在线 API 数据流说明。
- [ ] 将 README 从规划版改为准确的安装、配置、放入本地 PDF、索引、启动、运行评估步骤；为 Windows PowerShell 和 Linux shell 各给一组验证过的命令。不要要求在线下载样本。
- [ ] 录制固定三题的短 GIF/视频并绘制架构图；只在真实端到端运行之后放入 README。
- [ ] 验证全新环境的 Quick Start、无 key 错误提示、公开/私有来源隔离、引用页码跳转及 `.gitignore`。发布前检查 Git 提交清单，确保无私有 PDF、密钥、索引或私有检索日志。

**Wave 3 验收**：审阅者按 README 使用自带或经用户确认可公开的本地样本跑通；界面可从回答追到 PDF 页；演示画面与实际程序一致。

## 合并与复核门槛

1. Wave 0 的 `domain.py` 字段和样本数据提交给其他 Agent 后，才并行开展 Wave 1 与 Wave 3 的界面骨架。
2. CODEartsAgent 修改对象契约时先写迁移说明，MIMO 和 QoderCN 才调整各自代码。
3. 每个 Wave 保存运行命令与结果，协调者按对应验收项复核。未达到量化目标时记录结果与失败原因，不改写阈值掩盖问题。
4. 最终简历描述只取 `docs/EVALUATION.md` 中有脚本、题集和语料版本支持的数字。
