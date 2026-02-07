from __future__ import annotations

import asyncio
import atexit
import json
import threading
import urllib.request
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class OpenRouterClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterCallResult:
    """
    Normalized view of a chat completion response.

    We normalize only what this project needs: assistant content + tool calls (if any).
    """

    content: str | None
    tool_calls: list[dict[str, Any]]
    raw: Any


def _normalize_tool_calls(tool_calls_obj: Any) -> list[dict[str, Any]]:
    if tool_calls_obj is None:
        return []
    if isinstance(tool_calls_obj, list):
        # Could already be list[dict], or list of typed objects.
        normalized: list[dict[str, Any]] = []
        for tc in tool_calls_obj:
            if isinstance(tc, dict):
                normalized.append(tc)
            else:
                # Best-effort attribute extraction.
                normalized.append(
                    {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", None),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                        },
                    }
                )
        return normalized
    return []


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        http_referer: str | None,
        x_title: str | None,
        max_concurrent_requests: int,
    ) -> None:
        self._api_key = api_key
        self._default_headers: dict[str, str] = {}
        if http_referer:
            self._default_headers["HTTP-Referer"] = http_referer
        if x_title:
            self._default_headers["X-Title"] = x_title
        # NOTE: asyncio synchronization primitives can be event-loop bound (notably in newer Python versions).
        # This client is typically constructed outside of an active event loop (e.g., at FastMCP app startup),
        # so we must initialize loop-bound primitives lazily when `chat_completion()` runs.
        self._max_concurrent_requests = max_concurrent_requests
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._semaphore_init_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self._closed = False
        atexit.register(self.close)

        self._client = None
        self._client_lock = threading.Lock()

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
        """
        Best-effort cleanup for background resources (ThreadPoolExecutor).

        The MCP server typically runs as a long-lived process; without closing, the executor can
        leak threads across reloads/tests. `atexit` also calls this method.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python <3.9 compatibility (cancel_futures not supported).
            self._executor.shutdown(wait=False)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import AsyncOpenAI
            except Exception as exc:  # pragma: no cover
                # Fall back to stdlib HTTP client when `openai` isn't installed. This is primarily for
                # environments where installing packages is unavailable.
                self._client = "stdlib"
                return self._client

            self._client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self._api_key,
                default_headers=self._default_headers or None,
            )
            return self._client

    async def _chat_completion_stdlib(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout_seconds: int,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
    ) -> OpenRouterCallResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self._default_headers)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if extra_body:
            body.update(extra_body)

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def _do_request() -> dict[str, Any]:
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
            except Exception as exc:
                raise OpenRouterClientError(f"OpenRouter request failed: {exc}") from exc
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OpenRouterClientError("OpenRouter response was not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise OpenRouterClientError("OpenRouter response JSON was not an object")
            if "error" in parsed:
                raise OpenRouterClientError(f"OpenRouter error: {parsed.get('error')}")
            return parsed

        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(self._executor, _do_request)

        try:
            choice0 = (parsed.get("choices") or [])[0]
            msg = choice0.get("message") or {}
            content = msg.get("content")
            tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
        except Exception:
            content = None
            tool_calls = []

        return OpenRouterCallResult(content=content, tool_calls=tool_calls, raw=parsed)

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
        """
        Call OpenRouter via OpenAI-compatible chat completions.
        """
        client = self._get_client()

        async with self._get_semaphore():
            if client == "stdlib":
                return await self._chat_completion_stdlib(
                    model=model,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    extra_body=extra_body,
                )

            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        max_tokens=max_output_tokens,
                        extra_body=extra_body,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise OpenRouterClientError(f"OpenRouter request timed out after {timeout_seconds}s") from exc
            except Exception as exc:
                raise OpenRouterClientError(f"OpenRouter request failed: {exc}") from exc

        # Normalize
        try:
            choice0 = response.choices[0]
            msg = choice0.message
            content = getattr(msg, "content", None)
            tool_calls = _normalize_tool_calls(getattr(msg, "tool_calls", None))
        except Exception:
            content = None
            tool_calls = []

        return OpenRouterCallResult(content=content, tool_calls=tool_calls, raw=response)
