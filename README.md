# GeoChem-RAG

岩石地球化学文献证据检索与问答系统。从本地 PDF 提取证据，返回带有可核查引用（论文标题、PDF 页码、证据片段）的回答。

Evidence-grounded literature Q&A for rock geochemistry. Retrieves evidence from local PDFs and returns answers with verifiable citations (paper title, PDF page number, evidence snippet).

## 项目状态 / Project Status

**已实现功能：**

- PDF 逐页文本抽取、质量标记与页内分块（`ingest.py`）
- 本地 JSONL 存储与 SHA-256 去重（`store.py`）
- SiliconFlow / DeepSeek API 适配（`providers.py`）
- Dense + BM25 + RRF 混合检索，地学缩写词典扩展（`retrieval.py`）
- 证据约束回答与引用校验（`answering.py`）
- 固定题集评估脚本与指标计算（`evaluation.py`）
- Streamlit 界面：知识库/设置按钮、聊天界面、引用卡片与隐私说明（`app.py`）
- 仅联网运行，缺密钥/语料/索引时明确报错，不降级到模板回答
- Windows 根目录双击 `启动GeoChem-RAG.bat` 启动并自动打开浏览器
- 设置中获取/刷新模型目录、选择对话模型并保存偏好；思考控制按模型实际支持能力显示（`model_settings.py`）
- 本地 `PDF/` 全量续跑导入与独立索引脚本（私有语料留在本机）

**尚未完成：**

- 真实模型下的全链路评估（需要网络与 API 密钥）
- 人工标注的引用正确率与支持性指标
- 复杂表格单元格定位、OCR、图像理解
- 地化计算（Mg#、氧逸度等）

第一版聚焦一条完整流程：从 `PDF/` 导入用户已有的本地地学文献 → 提问 → 返回有 PDF 页码和原文证据的回答 → 对检索、引用和拒答进行可复现评估。`PDF/` 是私有语料目录，不随 GitHub 仓库发布。项目不需要 Agent 在线搜索或下载文献。

## 快速开始 / Quick Start

### 环境要求

- Python 3.11 或更高版本
- Windows PowerShell 或 Linux shell
- （可选）SiliconFlow API 密钥（用于嵌入）和 DeepSeek API 密钥（用于对话生成）

### 安装步骤

**Windows PowerShell：**

```powershell
# 克隆或下载项目后，进入项目目录
cd 路径\到\GeoChem-RAG

# 创建虚拟环境（推荐）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -e .

# 安装开发依赖（可选，用于运行测试）
pip install -e ".[dev]"
```

**Linux / git-bash：**

```bash
cd /path/to/RAG

# 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

### 配置 API 密钥

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# SiliconFlow API 密钥（用于嵌入，必填）
SILICONFLOW_API_KEY=your-siliconflow-key-here

# DeepSeek API 密钥（用于对话生成，推荐）
DEEPSEEK_API_KEY=your-deepseek-key-here

# 其他配置保持默认即可
GEOCHEM_CHAT_PROVIDER=deepseek
GEOCHEM_EMBEDDING_PROVIDER=siliconflow
```

**注意：** `.env` 文件已被 `.gitignore` 忽略，不会被提交到 Git。应用仅联网运行，缺少密钥时会显示明确的配置错误，不会降级到模板回答。

### 导入本地 PDF

把你的 PDF 文献放在 `PDF/` 目录下，然后运行导入命令：

**Windows PowerShell：**

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m geochem_rag.import_pdf --pdf "PDF\你的论文.pdf"
```

**Linux / git-bash：**

```bash
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
.venv/bin/python -m geochem_rag.import_pdf --pdf "PDF/你的论文.pdf"
```

命令返回 `imported`、`already_exists`、`metadata_updated` 或 `replaced`，并报告总页数、成功页、失败页和证据块数。重复导入同一文件不会重复建块。

如果已知书目信息，可以添加参数：

```powershell
.\.venv\Scripts\python.exe -m geochem_rag.import_pdf --pdf "PDF\paper.pdf" --title "Paper Title" --author "Author One" --author "Author Two" --year 2024 --doi "10.xxxx/xxxxx"
```

默认按私有文献导入；标题暂用文件名，未知作者、年份、DOI、许可证保持空值。

**导入整个 `PDF/`：** 在项目根目录运行下列命令。它按 SHA-256 去重，续跑时跳过已入库内容；报告写入 `data/private/full_corpus/full_import_report.json`。

```powershell
.\.venv\Scripts\python.exe scripts\import_all_pdfs.py
.\.venv\Scripts\python.exe scripts\import_all_pdfs.py --report-only
```

导入结果默认在 `data/private/full_corpus/`。原始 PDF、抽取文本及报告均保持私有，不提交到 GitHub。部分扫描页和乱码页只能标记质量，当前没有 OCR。

### 构建嵌入索引

如果你配置了 API 密钥，需要构建嵌入索引以启用 dense 检索：

**Windows PowerShell：**

```powershell
# 加载 .env 中的环境变量
$env:SILICONFLOW_API_KEY = (Get-Content .env | Select-String "SILICONFLOW_API_KEY=" | ForEach-Object { $_ -replace "SILICONFLOW_API_KEY=", "" })

.\.venv\Scripts\python.exe scripts\build_index.py
```

**Linux / git-bash：**

```bash
set -a && source .env && set +a
.venv/bin/python scripts/build_index.py
```

输出示例：`indexed chunks=262 dim=1024 model=BAAI/bge-m3`

嵌入向量会缓存在 `data/index/embeddings.json`（已 gitignore）。相同文本不会重复嵌入。

若使用上面的全量库，改运行 `scripts/build_full_index.py`；脚本读取 `.env`，向 SiliconFlow 逐块发送抽取文本，并将向量缓存在 `data/index/embeddings_full.json`。中断后可续跑；未完成时以非零退出码报告，不能当作索引完成。

```powershell
.\.venv\Scripts\python.exe scripts\build_full_index.py
```

### 启动应用

**停止方法：** 在运行终端按 `Ctrl+C`，或直接关闭终端窗口。

**Windows 双击启动（推荐）：** 完成首次配置（`.env` + 导入 PDF + 构建索引）后，双击项目根目录的 `启动GeoChem-RAG.bat`。脚本自动加载 `.env`、校验配置、启动 Streamlit 并打开浏览器。重复双击会复用已运行的实例。

**命令行启动：**

```powershell
# Windows
.\.venv\Scripts\python.exe scripts\run_streamlit_online.py
```

```bash
# Linux / git-bash
.venv/bin/python scripts/run_streamlit_online.py
```

浏览器会自动打开 `http://localhost:8501`。界面包括：

- 聊天输入框与示例问题折叠区
- **知识库**按钮：来源列表、有效证据块数、按标题筛选
- **设置**按钮：检索方式（hybrid/dense_only/bm25_only）、top-k、通用背景补充开关
- 回答下方可展开的引用卡片：论文标题、PDF 页码、证据片段
- 通用背景补充折叠区（标注"未经本地文献核验"）

缺少密钥、语料或索引时，界面显示明确的错误提示，不产生模板回答。

`GEOCHEM_STORE` 和 `GEOCHEM_EMBEDDING_CACHE` 环境变量可覆盖默认的语料和缓存路径。

### 运行固定演示

三个固定演示问题记录在 `docs/DEMO.md`：

1. 跨语言：英文论文的中文提问
2. 地学缩写：MORB 的含义
3. 不可回答：语料外的问题

**Windows PowerShell：**

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\run_demo.py
```

**Linux / git-bash：**

```bash
PYTHONPATH=src .venv/bin/python scripts/run_demo.py
```

脚本会打印每个问题的状态、引用和答案摘要。

### 运行评估

评估使用固定题集（`eval/questions.jsonl`），在 dense-only 和 hybrid 两种模式下运行：

**Windows PowerShell：**

```powershell
# 加载环境变量
$env:SILICONFLOW_API_KEY = ...
$env:DEEPSEEK_API_KEY = ...

.\.venv\Scripts\python.exe scripts\run_evaluation.py --modes dense_only hybrid
```

**Linux / git-bash：**

```bash
set -a && source .env && set +a
.venv/bin/python scripts/run_evaluation.py --modes dense_only hybrid
```

输出文件（在 `eval/results/` 下，已 gitignore）：

- `results_<mode>_<stamp>.jsonl`：逐题结果，含证据 ID、页码、片段
- `summary_<stamp>.json`：指标汇总与 manifest
- `review_sheet.jsonl`：人工标注模板


## 评估结果 / Evaluation Results

详细的评估报告见 `docs/EVALUATION.md`。主要发现：

**离线自检（占位符模型，非质量结果）：**

| 指标 | dense_only | hybrid |
| --- | --- | --- |
| Recall@5 source | 22/24 = 0.917 | 23/24 = 0.958 |
| 拒答率（应拒答 12 题） | 0/12 = 0.00 | 0/12 = 0.00 |
| 误答率 | 12/12 = 1.00 | 12/12 = 1.00 |
| 引用合法性 | 80/80 = 1.00 | 80/80 = 1.00 |

**解读：** 占位符 chat 模型永远引用首个证据、从不拒答，因此误答率 1.00、拒答率 0.00——这正是脚手架应检测到的失败模式。Recall 数字由哈希嵌入产生，**不代表真实检索质量**。引用合法性 1.00 说明引用校验生效。

**真实模型评估：** 因运行环境沙箱在会话中途屏蔽 `api.siliconflow.cn`，真实全链路与 dense/hybrid 同条件对照未完成。网络恢复后在沙箱外执行上述评估命令即可补齐。**本文未含任何未经复现的提升百分比。**

人工标注的引用正确率与支持性指标待用户复核后填写。

## 项目结构 / Project Structure

```
GeoChem-RAG/
├── AGENTS.md                    # Agent 工作约定
├── README.md                    # 本文件
├── pyproject.toml               # 依赖与构建配置
├── .env.example                 # 环境变量模板
├── .gitignore                   # Git 忽略规则
├── 启动GeoChem-RAG.bat          # Windows 双击启动入口
├── start.bat / start.sh         # 兼容包装启动脚本
│
├── PDF/                         # 用户本地语料（gitignore）
├── data/
│   ├── public/                  # 公开样本 PDF（gitignore）
│   ├── private/                 # 私有语料（gitignore）
│   ├── processed/               # 解析后的来源与块（gitignore）
│   └── index/                   # 嵌入缓存（gitignore）
│
├── src/geochem_rag/
│   ├── domain.py                # 数据契约（Source, EvidenceChunk, SearchHit, Answer）
│   ├── ingest.py                # PDF 逐页解析与中英文句界分块
│   ├── store.py                 # 本地持久化、去重与 chunking fingerprint
│   ├── providers.py             # SiliconFlow / DeepSeek API 适配
│   ├── retrieval.py             # Dense + BM25 + RRF 检索 + 坏文本过滤
│   ├── answering.py             # 证据约束回答、引用校验与通用背景补充
│   ├── evaluation.py            # 评估脚本与指标计算
│   ├── offline.py               # 占位符提供程序（仅测试用）
│   └── app.py                   # Streamlit 界面（仅联网）
│
├── scripts/
│   ├── build_index.py           # 构建嵌入索引
│   ├── import_all_pdfs.py       # 全量私有 PDF 可续跑导入
│   ├── build_full_index.py      # 全量库可续跑嵌入索引
│   ├── run_streamlit_online.py  # 联网启动编排（校验 + 启动 + 开浏览器）
│   ├── run_evaluation.py        # 运行评估
│   ├── run_demo.py              # 固定三题演示
│   ├── run_online_e2e.py        # 单题在线端到端测试
│   └── run_import_check.py      # 导入验证脚本
│
├── tests/                       # 自动测试（2026-10-12 复跑 175 项通过）
├── eval/
│   ├── questions.jsonl          # 固定题集（40 题）
│   └── results/                 # 评估结果（gitignore）
│
└── docs/
    ├── PROJECT_BRIEF.md         # 产品与技术设计
    ├── EXECUTION_PLAN.md        # 执行计划与验收标准
    ├── EVALUATION.md            # 评估报告
    ├── DEMO.md                  # 演示指南
    ├── CONTRACT_MIGRATION_0.2.md # Source 契约迁移说明
    └── agents/                  # Agent 任务书
```

## 隐私与数据流 / Privacy & Data Flow

- **本地数据：** PDF 原文、抽取文本、嵌入向量都保存在本地。`PDF/`、`data/private/`、`data/processed/`、`data/index/` 均被 `.gitignore` 忽略。
- **在线 API：** 建索引时，抽取出的文本按证据块发送到 SiliconFlow 嵌入 API；问答时，问题与检索命中的证据块发送到 DeepSeek 对话 API（或你配置的对话 provider）。不上传原始 PDF 文件。
- **API 密钥：** 只从环境变量读取，不显示在界面中，不记录在日志中。
- **公开仓库：** 不包含私有 PDF、API 密钥、嵌入索引或私有检索日志。

Streamlit 界面底部有数据流说明（建索引文本发送至 SiliconFlow，问题与证据片段发送至 DeepSeek）。

## 文档链接 / Documentation

- [产品与技术设计](docs/PROJECT_BRIEF.md)
- [执行计划与验收标准](docs/EXECUTION_PLAN.md)
- [评估报告](docs/EVALUATION.md)
- [演示指南](docs/DEMO.md)
- [Source 契约迁移说明](docs/CONTRACT_MIGRATION_0.2.md)
- [Agent 任务书](docs/agents/)
- [全量语料扩容交付报告（公开版）](docs/FULL_CORPUS_EXPANSION_REPORT.md)

## 已知限制 / Known Limitations

1. **向量库：** 计划用 Chroma，但当前环境 PyPI 不可达；`DenseIndex` 使用纯 Python 余弦索引，接口与 `search(query_vector, top_k)` 一致，可后续替换为 Chroma。
2. **中文分词：** 无 jieba；对 CJK 取字符单字 + 二元组，对 Latin 取词/数字。跨语言主要靠 dense（bge-m3）承担。
3. **人工指标：** 引用正确率与支持性需人工标注，当前 PENDING。
4. **真实评估：** 因网络限制，真实模型下的全链路评估未完成。
5. **语料：** 历史 4 篇 Frontiers 样本用于原型验证；本机可导入 `PDF/` 全量私有语料。GitHub 克隆者没有这些 PDF 和索引，不能直接复现本机的全量结果。公开演示语料仍需用户明确确认可再分发。

## 测试 / Testing

运行所有单元测试：

```bash
.venv/bin/python -m pytest tests -q
```

运行上述命令验证当前测试，覆盖：

- `domain.py`：数据契约校验
- `ingest.py`：PDF 抽取、中英文句界分块与 chunking fingerprint
- `store.py`：持久化、去重与 fingerprint 变更检测
- `providers.py`：API 适配与错误处理
- `retrieval.py`：检索、查询扩展与坏文本过滤
- `answering.py`：引用校验、拒答与零证据通用背景
- `evaluation.py`：指标计算
- `offline.py`：占位符提供程序（仅测试用）
- `app.py`：界面辅助函数与初始化校验
- `online_startup`：联网启动校验（缺 key/语料/索引检测）

## 许可证 / License

本项目代码部分遵循项目根目录的许可证（如有）。公开样本 PDF（`data/public/`）来自 Frontiers 期刊，采用 CC BY 4.0 许可证。用户添加的私有文献版权归原创作者所有。

## 贡献 / Contributing

本项目为简历展示与 GitHub 开源项目。如需报告问题或建议，请创建 Issue。

---

**最后更新：** 2026-10-12。CODEartsAgent 统一开发：仅联网运行、首页简化（知识库/设置按钮）、双击启动、分块改进与坏文本过滤。
