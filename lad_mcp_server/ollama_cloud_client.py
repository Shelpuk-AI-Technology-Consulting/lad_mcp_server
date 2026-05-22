from __future__ import annotations

import asyncio
import atexit
import json
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lad_mcp_server.openrouter_client import OpenRouterCallResult


_OLLAMA_PREFIX_RE = re.compile(r"^(?:ollama_cloud|ollama)/", re.IGNORECASE)
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"


class OllamaCloudClientError(RuntimeError):
    pass


def is_ollama_model(model: str) -> bool:
    if not isinstance(model, str):
        return False
    return _OLLAMA_PREFIX_RE.match(model.strip()) is not None


def normalize_ollama_model_name(model: str) -> str:
    if not isinstance(model, str):
        return model
    normalized = _OLLAMA_PREFIX_RE.sub("", model.strip(), count=1)
    return normalized if normalized else model.strip()


def _flatten_content(content: str | list | None) -> str:
    """Flatten CC content (string or list of content blocks) to a plain string for Ollama."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _cc_tool_calls_to_ollama(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert CC tool_calls (arguments as JSON string) to Ollama format (arguments as dict)."""
    result: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        result.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "arguments": args,
            },
        })
    return result


def translate_messages_to_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style messages to Ollama native format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            result.append({
                "role": "tool",
                "tool_name": msg.get("name", ""),
                "content": _flatten_content(msg.get("content", "")),
            })
        else:
            translated: dict[str, Any] = {"role": role}
            content = msg.get("content")
            if content is not None:
                translated["content"] = _flatten_content(content)
            if msg.get("tool_calls"):
                translated["tool_calls"] = _cc_tool_calls_to_ollama(msg["tool_calls"])
            if msg.get("thinking"):
                translated["thinking"] = msg["thinking"]
            result.append(translated)
    return result


def translate_ollama_response(parsed: dict[str, Any]) -> OpenRouterCallResult:
    """Translate an Ollama /api/chat response into OpenRouterCallResult.

    Converts Ollama tool call arguments (dict) back to JSON strings for OpenAI/CC format.
    Includes index field to disambiguate parallel tool calls.
    """
    try:
        message = parsed.get("message", {})
        content = message.get("content") or None
        raw_tool_calls = message.get("tool_calls")

        tool_calls: list[dict[str, Any]] = []
        if raw_tool_calls:
            for i, tc in enumerate(raw_tool_calls):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls.append({
                    "index": tc.get("function", {}).get("index", i),
                    "id": tc.get("id", f"call_{i:024x}"),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    },
                })
    except Exception:
        content = None
        tool_calls = []

    return OpenRouterCallResult(
        content=content,
        tool_calls=tool_calls,
        raw=parsed,
        reasoning_content=None,
    )


class OllamaCloudClient:
    def __init__(
        self,
        *,
        api_key: str,
        max_concurrent_requests: int,
        base_url: str = OLLAMA_CLOUD_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_concurrent_requests = max_concurrent_requests
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._semaphore_init_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self._closed = False
        atexit.register(self.close)

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._semaphore
        if sem is not None and self._semaphore_loop is loop:
            return sem
        with self._semaphore_init_lock:
            sem = self._semaphore
            if sem is not None and self._semaphore_loop is loop:
                return sem
            self._semaphore = asyncio.Semaphore(self._max_concurrent_requests)
            self._semaphore_loop = loop
            return self._semaphore

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout_seconds: int,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> OpenRouterCallResult:
        ollama_messages = translate_messages_to_ollama(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": max_output_tokens},
        }
        if tools is not None:
            body["tools"] = tools
        if extra_body:
            body.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "claude-code/1.0",
        }

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def _do_request() -> dict[str, Any]:
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                raise OllamaCloudClientError(
                    f"Ollama Cloud endpoint HTTP {getattr(exc, 'code', 'error')}: {body_text[:300]}"
                ) from exc
            except Exception as exc:
                raise OllamaCloudClientError(f"Ollama Cloud endpoint request failed: {exc}") from exc
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OllamaCloudClientError("Ollama Cloud endpoint response was not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise OllamaCloudClientError("Ollama Cloud endpoint response JSON was not an object")
            if "error" in parsed:
                err = parsed["error"]
                if isinstance(err, dict):
                    err = err.get("message", str(err))
                raise OllamaCloudClientError(f"Ollama Cloud endpoint error: {err}")
            return parsed

        async with self._get_semaphore():
            loop = asyncio.get_running_loop()
            try:
                parsed = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, _do_request),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise OllamaCloudClientError(
                    f"Ollama Cloud endpoint request timed out after {timeout_seconds}s"
                ) from exc

        return translate_ollama_response(parsed)
