@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==============================================
echo   GeoChem-RAG  联网启动
echo ==============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv。
    echo 请先完成首次配置（命令行执行）：
    echo    python -m venv .venv
    echo    .venv\Scripts\pip install -e .
    echo 然后把 .env.example 复制为 .env 并填入 API key。
    echo.
    pause
    exit /b 1
)

echo [1/2] 检查配置、语料与索引...
.venv\Scripts\python.exe scripts\run_streamlit_online.py
if errorlevel 1 (
    echo.
    echo [启动失败] 请按上面的中文提示完成准备后，再次双击本文件。
    pause
    exit /b 1
)

echo.
echo 服务已停止。关闭本窗口即可。
endlocal