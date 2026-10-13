r"""Load .env and start Streamlit in online mode, opening the browser when ready.

Behaviour
- Environment variables take precedence over ``.env``; ``.env`` fills gaps only.
- Chat (DeepSeek) and embedding (SiliconFlow) keys are checked independently;
  a missing key is a hard error — there is no offline/template fallback.
- Corpus and index are validated: no corpus, or a HashingEmbedder/incorrect-model
  cache, is reported as a clear preparation step rather than a pretend-answer app.
- If the app is already running on the target port it is reused (browser opened),
  otherwise Streamlit is started and the default browser opens once ready.

Usage (PowerShell, repo root):
    .venv\Scripts\python.exe scripts\run_streamlit_online.py
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
APP = ROOT / "src" / "geochem_rag" / "app.py"
DEFAULT_PORT = 8501

REQUIRED_CHAT_KEY = "DEEPSEEK_API_KEY"
REQUIRED_EMBED_KEY = "SILICONFLOW_API_KEY"


def load_env(path: Path, into: dict[str, str] | None = None) -> dict[str, str]:
    """Load .env entries into the target dict, without overwriting existing values."""
    target = os.environ if into is None else into
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in target:
            target[key] = value
            loaded[key] = value
    return loaded


def resolve_store(root: Path = ROOT, env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    if env.get("GEOCHEM_STORE"):
        return env["GEOCHEM_STORE"]
    full = root / "data" / "private" / "full_corpus"
    if (full / "chunks.jsonl").is_file():
        return full.as_posix()
    return "data/processed"


def resolve_cache(store: str) -> str:
    if "full_corpus" in store.replace("\\", "/"):
        return "data/index/embeddings_full.json"
    return "data/index/embeddings.json"


def check_prerequisites(
    store: str,
    cache: str,
    root: Path = ROOT,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Return a list of human-readable error messages; empty means ready."""
    env = os.environ if env is None else env
    errors: list[str] = []

    if not env.get(REQUIRED_EMBED_KEY, "").strip():
        errors.append("缺少 SILICONFLOW_API_KEY（嵌入不可用）")
    if not env.get(REQUIRED_CHAT_KEY, "").strip():
        errors.append("缺少 DEEPSEEK_API_KEY（对话不可用）")

    store_path = Path(store)
    chunks_path = store_path / "chunks.jsonl"
    if not chunks_path.is_file():
        errors.append(f"无语料：未找到 {chunks_path}，请先运行 scripts/import_all_pdfs.py")
    else:
        try:
            chunk_count = sum(1 for _ in open(chunks_path, encoding="utf-8"))
        except OSError:
            chunk_count = 0
        if chunk_count == 0:
            errors.append(f"无语料：{chunks_path} 为空，请先运行 scripts/import_all_pdfs.py")

    cache_path = Path(root) / cache if not Path(cache).is_absolute() else Path(cache)
    expected_model = env.get("GEOCHEM_EMBEDDING_MODEL", "BAAI/bge-m3")
    if not cache_path.is_file():
        errors.append(f"索引不完整：未找到 {cache}，请先运行 scripts/build_full_index.py")
    else:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            payload = {}
        model = payload.get("model") if isinstance(payload, dict) else None
        if model != expected_model:
            errors.append(
                f"索引模型不匹配：{cache} 的 model={model!r}，期望 {expected_model!r}。"
                "请删除该缓存后重新运行 scripts/build_full_index.py"
            )

    return errors


def _http_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:  # noqa: BLE001
        return False


def _open_when_ready(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _http_ready(url, timeout=1.0):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start GeoChem-RAG in online mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
    args = parser.parse_args(argv)

    load_env(ENV_PATH)
    os.environ.setdefault("GEOCHEM_STORE", resolve_store())
    os.environ.setdefault("GEOCHEM_CHAT_PROVIDER", "deepseek")
    os.environ.setdefault("GEOCHEM_EMBEDDING_PROVIDER", "siliconflow")

    store = os.environ["GEOCHEM_STORE"]
    cache = resolve_cache(store)
    errors = check_prerequisites(store, cache)
    if errors:
        print("无法启动（在线模式，无模板降级）：", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        print("\n常见准备步骤（首次配置）：", file=sys.stderr)
        print("  1. 复制 .env.example 为 .env 并填入两套 API key", file=sys.stderr)
        print("  2. .venv\\Scripts\\python.exe scripts\\import_all_pdfs.py", file=sys.stderr)
        print("  3. .venv\\Scripts\\python.exe scripts\\build_full_index.py", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}"
    if _http_ready(url, timeout=1.0):
        print(f"服务已经在 {url} 运行，直接复用它。")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    print(f"Chat 提供方: DeepSeek ({os.environ.get('DEEPSEEK_CHAT_MODEL', 'deepseek-chat')})")
    print(f"嵌入提供方: SiliconFlow ({os.environ.get('GEOCHEM_EMBEDDING_MODEL', 'BAAI/bge-m3')})")
    print(f"语料 store: {store}")
    print(f"索引缓存: {cache}")
    print(f"启动中…… 就绪后将自动打开浏览器 {url}")

    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(APP),
        "--server.headless=true",
        f"--server.port={args.port}",
        "--browser.gatherUsageStats=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())