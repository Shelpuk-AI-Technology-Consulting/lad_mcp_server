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

from lad_mcp_server.openrouter_client import OpenRouterCallResult, extract_reasoning_content


_ZAI_PREFIX_RE = re.compile(r"^z-?ai/", re.IGNORECASE)
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


class ZaiCodingClientError(RuntimeError):
    pass


def is_zai_model(model: str) -> bool:
    if not isinstance(model, str):
        return False
    return _ZAI_PREFIX_RE.match(model.strip()) is not None


def normalize_zai_model_name(model: str) -> str:
    if not isinstance(model, str):
        return model
    normalized = _ZAI_PREFIX_RE.sub("", model.strip(), count=1)
    return normalized if normalized else model.strip()


def _normalize_tool_calls(tool_calls_obj: Any) -> list[dict[str, Any]]:
    if tool_calls_obj is None:
        return []
    if isinstance(tool_calls_obj, list):
        normalized: list[dict[str, Any]] = []
        for tc in tool_calls_obj:
            if isinstance(tc, dict):
                normalized.append(tc)
            else:
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


class ZaiCodingClient:
    def __init__(
        self,
        *,
        api_key: str,
        max_concurrent_requests: int,
        base_url: str = ZAI_CODING_BASE_URL,
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
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import AsyncOpenAI
            except Exception:
                self._client = "stdlib"
                return self._client

            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={"User-Agent": "claude-code/1.0"},
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
            "User-Agent": "claude-code/1.0",
        }

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
            f"{self._base_url}/chat/completions",
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
                raise ZaiCodingClientError(
                    f"Z.AI Coding endpoint HTTP {getattr(exc, 'code', 'error')}: {body_text[:300]}"
                ) from exc
            except Exception as exc:
                raise ZaiCodingClientError(f"Z.AI Coding endpoint request failed: {exc}") from exc
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ZaiCodingClientError("Z.AI Coding endpoint response was not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ZaiCodingClientError("Z.AI Coding endpoint response JSON was not an object")
            if "error" in parsed:
                raise ZaiCodingClientError(f"Z.AI Coding endpoint error: {parsed.get('error')}")
            return parsed

        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(self._executor, _do_request)

        try:
            choice0 = (parsed.get("choices") or [])[0]
            msg = choice0.get("message") or {}
            content = msg.get("content")
            tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
            reasoning_content = extract_reasoning_content(msg)
        except Exception:
            content = None
            tool_calls = []
            reasoning_content = None

        return OpenRouterCallResult(
            content=content,
            tool_calls=tool_calls,
            raw=parsed,
            reasoning_content=reasoning_content,
        )

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

            # Z.AI Coding Plan supports Preserved Thinking — we deliberately do NOT strip
            # `reasoning_content` from outgoing assistant messages. Pass `timeout=` to the SDK
            # so its idle read-timeout matches our outer wait_for budget; reasoning models like
            # GLM-5 pause between hidden reasoning tokens and would otherwise trip a spurious
            # early APITimeoutError.
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        max_tokens=max_output_tokens,
                        extra_body=extra_body,
                        timeout=timeout_seconds,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise ZaiCodingClientError(f"Z.AI Coding endpoint request timed out after {timeout_seconds}s") from exc
            except Exception as exc:
                raise ZaiCodingClientError(f"Z.AI Coding endpoint request failed: {exc}") from exc

        try:
            choice0 = response.choices[0]
            msg = choice0.message
            content = getattr(msg, "content", None)
            tool_calls = _normalize_tool_calls(getattr(msg, "tool_calls", None))
            reasoning_content = extract_reasoning_content(msg)
        except Exception:
            content = None
            tool_calls = []
            reasoning_content = None

        return OpenRouterCallResult(
            content=content,
            tool_calls=tool_calls,
            raw=response,
            reasoning_content=reasoning_content,
        )
