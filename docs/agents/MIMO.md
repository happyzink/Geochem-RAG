# MIMO 执行任务书

> 2026-10-12：归档为历史交接资料，后续工作全部转交 CODEartsAgent，不再向 MIMO 派新任务。以下旧范围限制不约束 CODEartsAgent。当前入口：`docs/agents/CODEartsAgent.md`。

状态：此前的 Wave 0 初版已交付；用户现要求只用本地文献。继续工作时先读根目录 `AGENTS.md`、`docs/PROJECT_BRIEF.md` 与 `docs/EXECUTION_PLAN.md`。

## 负责范围

负责 Wave 0 的本地资料适配：`PDF/` 语料清点与私有来源清单、现有 `Source` 契约迁移、导入命令适配、证据元数据与私有评估题初稿。主要文件：`data/local_sources.json`（Git 忽略）、`src/geochem_rag/domain.py`、`import_pdf.py`、相应 `tests/`、`docs/schemas/contract_samples.json`、`eval/private/`（Git 忽略）。现有 PDF 抽取/去重模块可复用。请勿修改 `retrieval.py`、`answering.py` 或 `app.py`。

## 实施要求

1. 只从用户已有的 `PDF/` 选 3–5 篇地学论文做本地试验，不在线搜索、下载或抓取新文献。`PDF/` 整体视为私有；不要把其中的文件复制进公开目录或提交仓库。用户以后另放在 `data/private/` 的文件也适用相同规则。
2. 既有 `data/public/` 中的历史验证样本保留原状，不再下载新样本，不把它们自动视作最终公开演示语料。只有用户明确指定本地文件并确认可再分发时，才准备公开样本。
3. 当前 `Source` 0.1.0 强制要求作者、年份、DOI/URL、许可证及下载日期，无法正确表示元数据未知的本地文件。迁移契约：私有来源允许这些字段缺失，标题可标注为文件名暂代；公开来源仍要求许可证明。不要填入虚构作者、年份或网址。更新 schema 样例、导入 CLI、测试和契约版本，并通知 CODEartsAgent/QoderCN。
4. 抽取保证 `pdf_page` 是从 1 开始的 PDF 页序号；块不跨页；`chunk_id` 对同一 PDF 内容重复运行保持一致。记录文本为空、乱码或疑似扫描页，入库报告展示这些页。
5. 本地清单放 `data/local_sources.json`；私有文献派生的题目放 `eval/private/`。评估题覆盖术语、跨中英、跨文献和不可回答/冲突场景；题目草案交用户审核，不以 Agent 的猜测直接锁定科学标准答案。

## 验收与交接

给出一条可重复的导入命令、每篇成功/失败页统计、重复导入前后块数、至少三条 `chunk_id → PDF 页面 → 原文` 核对记录以及离线测试结果。交接时声明数据契约版本和已知 PDF 解析限制。
