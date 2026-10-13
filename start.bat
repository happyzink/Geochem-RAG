@echo off
chcp 65001 >nul
REM 兼容包装：与「启动GeoChem-RAG.bat」完全相同的仅联网启动。
call "%~dp0启动GeoChem-RAG.bat"
exit /b %errorlevel%