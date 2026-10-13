#!/usr/bin/env bash
# GeoChem-RAG 仅联网启动（与 Windows 双击入口同一逻辑）
# 用法：bash start.sh

set -e
cd "$(dirname "$0")"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    echo "[错误] 未找到虚拟环境 .venv，请先完成首次配置："
    echo "  python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

exec "$PYTHON" scripts/run_streamlit_online.py "$@"