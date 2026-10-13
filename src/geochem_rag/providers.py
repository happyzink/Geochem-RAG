"""SiliconFlow (OpenAI-compatible) adapters for chat and embeddings.

Design rules (see AGENTS.md and docs/agents/CODEartsAgent.md):
- The API key is read from the ``SILICONFLOW_API_KEY`` environment variable only.
  It is never accepted as a literal default and never logged.
- Model ids and base URL come from configuration, never hard-coded assumptions.
- Failures are classified and surfaced as ``ProviderError`` so callers can show
  an explicit error state instead of a fake geochemistry answer.
- No model-generated code is executed anywhere in this module.

Only the Python standard library is used so the module runs in an offline
environment where PyPI packages cannot be installed.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

# Error kinds used by callers to build an explicit, non-geological status.
KIND_NOT_CONFIGURED = "not_configured"
KIND_AUTH = "auth"
KIND_BAD_REQUEST = "bad_request"
KIND_RATE_LIMIT = "rate_limit"
KIND_SERVER = "server"
KIND_NETWORK = "network"
KIND_TIMEOUT = "timeout"
KIND_BAD_RESPONSE = "bad_response"


class ProviderError(RuntimeError):
    """A classified upstream failure. Never contains the API key."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retryable = retryable
        self.attempts = attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": str(self),
            "status": self.status,
            "retryable": self.retryable,
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class ProviderConfig:
    """Runtime configuration resolved from environment variables."""

    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    chat_model: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    timeout: float = 60.0
    max_retries: int = 2
    backoff_seconds: float = 1.5
    disable_proxy: bool = False
    credential_name: str = "SILICONFLOW_API_KEY"
    # Reasoning/thinking calls can take substantially longer than normal
    # chat calls; bounded separately so the app never waits indefinitely.
    thinking_timeout: float = 180.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ProviderConfig":
        env = env if env is not None else os.environ

        def pick(name: str, default: str) -> str:
            value = env.get(name)
            return value.strip() if value and value.strip() else default

        def pick_float(name: str, default: float) -> float:
            raw = env.get(name)
            if not raw or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def pick_int(name: str, default: int) -> int:
            raw = env.get(name)
            if not raw or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        key = env.get("SILICONFLOW_API_KEY")
        disable_proxy = pick("GEOCHEM_DISABLE_PROXY", "").lower() in {"1", "true", "yes", "on"}
        return cls(
            api_key=key.strip() if key and key.strip() else None,
            base_url=pick("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            chat_model=pick("GEOCHEM_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            embedding_model=pick("GEOCHEM_EMBEDDING_MODEL", "BAAI/bge-m3"),
            timeout=pick_float("GEOCHEM_API_TIMEOUT", 60.0),
            max_retries=pick_int("GEOCHEM_API_MAX_RETRIES", 2),
            backoff_seconds=pick_float("GEOCHEM_API_BACKOFF", 1.5),
            disable_proxy=disable_proxy,
            thinking_timeout=pick_float("GEOCHEM_THINKING_TIMEOUT", 180.0),
        )

    def require_key(self) -> str:
        if not self.api_key:
            raise ProviderError(
                KIND_NOT_CONFIGURED,
                f"{self.credential_name} is not set; cannot call the online API",
            )
        return self.api_key

    @classmethod
    def from_deepseek_env(cls, env: Mapping[str, str] | None = None) -> "ProviderConfig":
        """Use the official DeepSeek endpoint for chat, independently of embeddings."""
        source = env if env is not None else os.environ
        adapted = dict(source)
        adapted["SILICONFLOW_API_KEY"] = source.get("DEEPSEEK_API_KEY", "")
        adapted["SILICONFLOW_BASE_URL"] = source.get("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL
        adapted["GEOCHEM_CHAT_MODEL"] = source.get("DEEPSEEK_CHAT_MODEL") or DEEPSEEK_CHAT_MODEL
        return replace(
            cls.from_env(adapted),
            embedding_model="",
            credential_name="DEEPSEEK_API_KEY",
        )


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    latency_seconds: float = 0.0
    # Chain-of-thought / reasoning content returned by thinking models.
    # Kept separate from ``text``: it is never parsed as the structured answer
    # and never used as citation evidence.
    reasoning: str | None = None


@runtime_checkable
class ChatProvider(Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class _Response:
    """Thin wrapper so tests can fake urlopen return values."""

    __slots__ = ("_raw",)

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw.read()


class SiliconFlowClient:
    """Synchronous SiliconFlow client using urllib (no third-party deps)."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        embedding_batch_size: int = 32,
        thinking_body: Mapping[str, Any] | None = None,
        thinking_active: bool = False,
    ) -> None:
        self.config = config or ProviderConfig.from_env()
        if opener is not None:
            self._opener = opener
        elif self.config.disable_proxy:
            self._opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({})
            ).open
        else:
            self._opener = urllib.request.urlopen
        self._sleep = sleeper or time.sleep
        self.embedding_batch_size = embedding_batch_size
        # Provider-specific, top-level reasoning controls already mapped for
        # the bound model (e.g. DeepSeek ``thinking``/``reasoning_effort`` or
        # SiliconFlow ``enable_thinking``/``thinking_budget``). Never contains
        # model ids, messages, secrets or private text.
        self.thinking_body: dict[str, Any] = dict(thinking_body or {})
        self.thinking_active = thinking_active

    # --- low level -------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        key = self.config.require_key()
        url = f"{self.config.base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        attempt = 0
        last_error: ProviderError | None = None
        while attempt <= self.config.max_retries:
            attempt += 1
            try:
                request = urllib.request.Request(
                    url, data=body, headers=headers, method=method
                )
                with self._opener(request, timeout=timeout or self.config.timeout) as response:
                    raw = response.read()
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ProviderError(
                        KIND_BAD_RESPONSE,
                        "upstream returned a non-JSON body",
                        attempts=attempt,
                    ) from exc
                if not isinstance(data, dict):
                    raise ProviderError(
                        KIND_BAD_RESPONSE,
                        "upstream returned an unexpected JSON shape",
                        attempts=attempt,
                    )
                return data
            except ProviderError as exc:
                exc.attempts = attempt
                if not exc.retryable:
                    raise
                last_error = exc
            except urllib.error.HTTPError as exc:
                kind, retryable = _classify_status(exc.code)
                detail = _read_error_detail(exc)
                message = f"upstream HTTP {exc.code}"
                if detail:
                    message = f"{message}: {detail}"
                last_error = ProviderError(
                    kind,
                    message,
                    status=exc.code,
                    retryable=retryable,
                    attempts=attempt,
                )
                if not retryable:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, socket.timeout):
                    last_error = ProviderError(
                        KIND_TIMEOUT,
                        "request timed out",
                        retryable=True,
                        attempts=attempt,
                    )
                else:
                    last_error = ProviderError(
                        KIND_NETWORK,
                        f"network error: {reason}",
                        retryable=True,
                        attempts=attempt,
                    )
            except socket.timeout:
                last_error = ProviderError(
                    KIND_TIMEOUT, "request timed out", retryable=True, attempts=attempt
                )
            except (TimeoutError, ConnectionError) as exc:
                last_error = ProviderError(
                    KIND_NETWORK,
                    f"network error: {exc}",
                    retryable=True,
                    attempts=attempt,
                )
            if attempt <= self.config.max_retries:
                self._sleep(self.config.backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _get(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        return self._request("GET", path, timeout=timeout)

    # --- public API ------------------------------------------------
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Model-specific reasoning controls are merged at the top level of the
        # JSON body (NOT nested under an extra_body layer, which is an SDK-only
        # concept). Core keys are never overridable through thinking_body.
        for key, value in self.thinking_body.items():
            if key not in ("model", "messages", "temperature", "response_format"):
                payload[key] = value
        started = time.time()
        timeout = (
            self.config.thinking_timeout if self.thinking_active else self.config.timeout
        )
        data = self._request(
            "POST", "/chat/completions", payload, timeout=timeout
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                KIND_BAD_RESPONSE, "chat response has no choices"
            )
        message = choices[0].get("message") or {}
        text = message.get("content")
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = None
        if not isinstance(text, str) or not text.strip():
            # A thinking model may finish a turn having produced only
            # reasoning_content. That is never a valid structured answer.
            if reasoning is not None:
                raise ProviderError(
                    KIND_BAD_RESPONSE,
                    "chat response contained only reasoning content, no answer",
                )
            raise ProviderError(
                KIND_BAD_RESPONSE, "chat response has no text content"
            )
        return ChatResult(
            text=text,
            model=data.get("model", self.config.chat_model),
            usage=dict(data.get("usage") or {}),
            finish_reason=choices[0].get("finish_reason"),
            latency_seconds=time.time() - started,
            reasoning=reasoning,
        )

    def get_models(self, sub_type: str = "chat") -> list[dict[str, Any]]:
        """Fetch the raw model catalog. SiliconFlow groups by ``sub_type``."""
        data = self._get(f"/models?sub_type={sub_type}")
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderError(KIND_BAD_RESPONSE, "model catalog has unexpected shape")
        return items

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch = max(1, self.embedding_batch_size)
        for start in range(0, len(texts), batch):
            window = list(texts[start : start + batch])
            data = self._post(
                "/embeddings",
                {"model": self.config.embedding_model, "input": window},
            )
            items = data.get("data")
            if not isinstance(items, list) or len(items) != len(window):
                raise ProviderError(
                    KIND_BAD_RESPONSE,
                    "embedding response count does not match request",
                )
            ordered = sorted(items, key=lambda item: item.get("index", 0))
            for item in ordered:
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise ProviderError(
                        KIND_BAD_RESPONSE, "embedding vector missing"
                    )
                vectors.append([float(x) for x in vector])
        return vectors


class DeepSeekChatClient(SiliconFlowClient):
    """Chat-only adapter for DeepSeek's OpenAI-compatible API.

    Reuses the tested HTTP transport. Embeddings remain a separate provider.
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        thinking_body: Mapping[str, Any] | None = None,
        thinking_active: bool = False,
    ) -> None:
        super().__init__(
            config or ProviderConfig.from_deepseek_env(),
            opener=opener,
            sleeper=sleeper,
            thinking_body=thinking_body,
            thinking_active=thinking_active,
        )

    def get_models(self, sub_type: str = "chat") -> list[dict[str, Any]]:
        """DeepSeek's catalog endpoint takes no ``sub_type`` parameter."""
        data = self._get("/models")
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderError(KIND_BAD_RESPONSE, "model catalog has unexpected shape")
        return items

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ProviderError(
            KIND_BAD_REQUEST,
            "DeepSeekChatClient does not provide embeddings",
        )


def chat_provider_from_env(env: Mapping[str, str] | None = None) -> ChatProvider:
    """Select the chat provider without coupling it to the embedding backend."""
    source = env if env is not None else os.environ
    name = source.get("GEOCHEM_CHAT_PROVIDER", "deepseek").strip().lower()
    if name == "deepseek":
        return DeepSeekChatClient(ProviderConfig.from_deepseek_env(source))
    if name == "siliconflow":
        return SiliconFlowClient(ProviderConfig.from_env(source))
    raise ValueError("GEOCHEM_CHAT_PROVIDER must be deepseek or siliconflow")


def embedding_provider_from_env(env: Mapping[str, str] | None = None) -> EmbeddingProvider:
    """Use SiliconFlow embeddings, independently of the chat provider."""
    source = env if env is not None else os.environ
    name = source.get("GEOCHEM_EMBEDDING_PROVIDER", "siliconflow").strip().lower()
    if name != "siliconflow":
        raise ValueError("GEOCHEM_EMBEDDING_PROVIDER must be siliconflow")
    return SiliconFlowClient(ProviderConfig.from_env(source))


def _classify_status(status: int) -> tuple[str, bool]:
    if status in (401, 403):
        return KIND_AUTH, False
    if status == 429:
        return KIND_RATE_LIMIT, True
    if status in (400, 404, 422):
        return KIND_BAD_REQUEST, False
    if 500 <= status <= 599:
        return KIND_SERVER, True
    return KIND_BAD_REQUEST, False


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    """Pull a short upstream error message without leaking the API key."""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 - best effort only
        return ""
    if not raw:
        return ""
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return raw.decode("utf-8", errors="replace")[:200]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message[:300]
        for key in ("message", "detail", "msg"):
            value = payload.get(key)
            if isinstance(value, str):
                return value[:300]
    return json.dumps(payload, ensure_ascii=False)[:200]


def list_available_models(
    config: ProviderConfig | None = None, sub_type: str = "chat"
) -> list[str]:
    """Query the platform for currently available model ids.

    Used at implementation time to confirm model ids instead of hard-coding a
    possibly retired name. Requires a valid key.
    """
    config = config or ProviderConfig.from_env()
    key = config.require_key()
    url = f"{config.base_url}/models?sub_type={sub_type}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        kind, _ = _classify_status(exc.code)
        raise ProviderError(kind, f"upstream HTTP {exc.code}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(KIND_NETWORK, f"network error: {exc.reason}") from exc
    return [item.get("id", "") for item in data.get("data", []) if item.get("id")]
