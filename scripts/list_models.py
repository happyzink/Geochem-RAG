"""List currently available models for the configured chat provider.

Usage:
    python scripts/list_models.py                 # current GEOCHEM_CHAT_PROVIDER
    python scripts/list_models.py deepseek        # DeepSeek official endpoint
    python scripts/list_models.py siliconflow     # SiliconFlow chat catalog
    python scripts/list_models.py siliconflow --embedding

Uses only the matching environment variable (DEEPSEEK_API_KEY or
SILICONFLOW_API_KEY). Never prints the key. Everyday users use the
"刷新模型列表" button in the app; this CLI is for troubleshooting.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geochem_rag import model_settings as ms
from geochem_rag.providers import ProviderError


def _load_env() -> None:
    """Fill missing variables from repo .env; real environment wins."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(description="List available provider models")
    parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        choices=("deepseek", "siliconflow"),
        help="chat provider (defaults to GEOCHEM_CHAT_PROVIDER or deepseek)",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="list SiliconFlow embedding models instead of chat models",
    )
    args = parser.parse_args(argv)

    if args.embedding:
        try:
            models = ms.fetch_embedding_models()
        except ProviderError as exc:
            print(f"embedding: ERROR {exc.kind}: {exc}", file=sys.stderr)
            return 1
        print(f"== siliconflow embedding ({len(models)}) ==")
        for model in models:
            print(model.id)
        return 0

    provider = args.provider or os.environ.get(
        "GEOCHEM_CHAT_PROVIDER", "deepseek"
    ).strip().lower()
    config = ms.chat_config(provider)
    if not config.api_key:
        print(f"{config.credential_name} is not set", file=sys.stderr)
        return 2
    try:
        models = ms.fetch_chat_models(provider)
    except ProviderError as exc:
        print(f"{provider}: ERROR {exc.kind}: {exc}", file=sys.stderr)
        return 1
    print(f"== {provider} chat ({len(models)}) ==")
    for model in models:
        cap = model.thinking
        if cap.mode == "effort":
            extra = f"  thinking={'/'.join(cap.levels)} ({cap.source})"
        elif cap.mode == "budget":
            extra = f"  thinking=budget {cap.budget_range} ({cap.source})"
        elif cap.mode == "toggle":
            extra = f"  thinking=toggle ({cap.source})"
        elif cap.mode == "fixed_on":
            extra = "  thinking=fixed_on"
        else:
            extra = "  thinking=unknown"
        print(f"{model.display}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
