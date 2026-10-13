# GeoChem-RAG 演示指南

状态：CODEartsAgent 在线版（2026-10-12）。应用仅联网运行，无离线降级。

## 界面布局

首页突出问答：

1. **标题与简短说明**：页面顶部显示"GeoChem-RAG"和一句话描述。
2. **数据流披露**：用 caption 说明"建索引文本发送至 SiliconFlow 生成向量；提出问题后，问题与命中的证据片段发送至 DeepSeek 生成回答。"
3. **知识库 / 设置按钮**：两个按钮切换面板，互斥显示。
   - **知识库**：有效证据块数、来源列表（标题、年份、可见性）、按标题筛选。
   - **设置**：对话/嵌入模型信息、检索方式（hybrid/dense_only/bm25_only）、top-k 滑块、允许通用背景补充开关。
4. **示例问题**：折叠区，列出三个中文示例问题。
5. **聊天界面**：`st.chat_input` 输入框 + `st.chat_message` 历史记录。
6. **回答渲染**：回答文本 → 尚无论文证据支持的部分（warning）→ 引用卡片（论文标题 + PDF 页码 + 证据片段）→ 通用背景补充（折叠，标注"未经本地文献核验"）。

默认不显示来源列表和详细配置，保持首页简洁。

## 推荐演示问题

### 1. 跨语言：中文提问检索英文文献

**问题：** "什么是埃达克岩（adakite）？"

**测试点：** 中文问题 → 术语词典扩展（埃达克岩 → adakite）→ 从英文原文检索证据 → 中文回答 + 英文引用片段。

**预期：**
- 状态：`answered`（已回答，有本域文献证据）
- 至少一条引用，含论文标题与 PDF 页码
- 回答用中文概括，引用展开后显示英文证据片段

### 2. 地化比值与岩浆过程

**问题：** "弧岩浆中 Nb/Ta 比值的变化意味着什么？"

**测试点：** 化学式/比值保护（Nb/Ta 不被拆分）、地球化学机理检索。

**预期：**
- 若语料覆盖：`answered`，引用含 Nb/Ta 讨论的证据
- 若语料未覆盖：`insufficient_evidence`，不编造回答

### 3. 构造演化时间约束

**问题：** "青藏高原地壳增厚主要发生在什么时期？"

**测试点：** 中文地名/构造术语检索，时间约束回答。

**预期：**
- 若语料覆盖：`answered`，回答含具体时代/年龄约束
- 若语料未覆盖：`insufficient_evidence`

### 4. 语料外问题（拒答测试）

**问题：** "Summarize the conclusions of a paper about kimberlites in Siberia."

**测试点：** 语料中无西伯利亚金伯利岩论文，系统必须拒答，不编造引用。

**预期：**
- 状态：`insufficient_evidence`
- 无引用
- 回答说明本地语料未覆盖该主题

## 启动方式

完成首次配置（`.env` 填入两套密钥 + 导入 PDF + 构建索引）后：

**Windows 双击启动：** 双击项目根目录的 `启动GeoChem-RAG.bat`。脚本自动加载 `.env`、校验配置、启动 Streamlit 并打开浏览器。重复双击会复用已运行的实例。

**命令行启动：**

```powershell
# Windows
.\.venv\Scripts\python.exe scripts\run_streamlit_online.py
```

```bash
# Linux / git-bash
.venv/bin/python scripts/run_streamlit_online.py
```

**停止：** 在运行终端按 `Ctrl+C`，或直接关闭终端窗口。

## 演示录制清单

1. 启动应用，确认首页只显示标题、说明、知识库/设置按钮和聊天输入框。
2. 点击"知识库"按钮，展示来源列表和有效证据块数。再次点击收起。
3. 提问示例 1，展开至少一条引用，展示论文标题、PDF 页码和证据片段。
4. 提问示例 4（语料外），展示 `insufficient_evidence` 状态和无引用。
5. 点击"设置"按钮，展示检索方式、top-k 和通用背景补充开关。

## 架构图

```
PDF/ (用户本地文件，gitignore)
  |
  v
[ingest.py] — 逐页文本抽取、中英文句界分块、质量标记
  |
  v
[store.py] — JSONL 持久化、SHA-256 去重、chunking fingerprint
  |
  +---> [retrieval.py]
  |       |
  |       +-- DenseIndex (bge-m3 余弦相似度，SiliconFlow 嵌入)
  |       +-- BM25Index (词法 + 地学缩写/术语扩展)
  |       +-- RRF 融合 (k=60)
  |       +-- from_store 过滤 garbled/scanned/过短块
  |       |
  |       v
  |     Top-k SearchHits
  |       |
  |       v
  +---> [answering.py]
          |
          +-- build_evidence() 解析 chunk_id → EvidenceItem
          +-- 系统提示约束 + 证据块 + 问题 → DeepSeek 对话
          +-- 引用校验：每个 chunk_id 必须在证据中
          +-- 空证据 + allow_extended → 通用背景补充（标注未经核验）
          |
          v
        Answer (status, text, citations, meta)
          |
          v
        [app.py] Streamlit 界面（仅联网）
          |
          +-- 知识库按钮 → 来源列表
          +-- 设置按钮 → 检索/模型/补充配置
          +-- 聊天界面 → 回答 + 引用卡片 + 通用背景补充
```
