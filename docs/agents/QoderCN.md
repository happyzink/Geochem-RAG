# QoderCN 执行任务书

> 2026-10-12：归档为历史交接资料，后续前端、启动与包装工作全部转交 CODEartsAgent，不再向 QoderCN 派新任务。以下旧范围限制不约束 CODEartsAgent。当前入口：`docs/agents/CODEartsAgent.md`。

状态：按用户后续开发指令执行。先读根目录 `AGENTS.md`、设计、执行计划和 `docs/CONTRACT_MIGRATION_0.2.md`。可在后端开发时准备 UI 布局；最终接线等 CODEartsAgent 的接口稳定。不得自行在线下载文献。

## 负责范围

负责 Wave 3：Streamlit 页面、公开仓库 README、演示录屏/GIF、架构展示、跨平台 Quick Start 与端到端验收。主要文件：`src/geochem_rag/app.py`、`README.md`、`docs/DEMO.md`、相关 UI 测试或烟雾测试。不要修改检索排序、PDF 分块或评估公式。

## 实施要求

1. 页面优先显示引用的可核查性：每个结论旁给出处、PDF 页、短证据；可以直接打开公开原文页。证据不足或 API 出错显示不同状态。
2. 明确提示本地私有资料不会随公开仓库发布，但在线问答会把检索到的片段发送给 SiliconFlow。公开与私有来源要可区分，避免在演示截图中泄露私有文件路径或文本。
3. README 的 Quick Start 从全新环境实测，分别给出 PowerShell 与 POSIX shell 命令；说明如何放入用户自己的本地 PDF，不提供在线下载文献步骤。只引用可公开的真实评估结果和许可信息。不存在的功能、成绩、GIF 或 Docker 命令不得写成已完成。
4. 录屏固定演示跨语言问题、缩写问题、无证据问题，并至少一次展开 PDF 页与证据。架构图与实际流程一致。

## 验收与交接

提交全新环境复现记录、三道固定演示题的实际结果、无密钥和 API 故障时的界面截图/说明、公开发布前文件清单检查结果。README 所有指标必须指向同一版本的评估报告。
