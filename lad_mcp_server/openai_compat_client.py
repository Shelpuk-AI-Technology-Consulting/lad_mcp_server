"""Direct provider for any OpenAI-compatible endpoint.

Covers LiteLLM, vLLM, and internal corporate gateways: anything serving
``POST {base_url}/chat/completions`` with the OpenAI schema. Unlike the other direct
clients, the base URL comes from configuration rather than a fixed constant — that
is the whole point of this one.

Two things differ from a copy of :mod:`lad_mcp_server.deepseek_client`, and both are
deliberate:

* **The API key is optional.** A keyless local gateway is squarely within "any
  OpenAI-compatible endpoint", so no ``Authorization`` header is sent when none is
  configured.
* **``reasoning_content`` is stripped.** Z.AI and DeepSeek want it echoed back across
  tool-call rounds, but an arbitrary gateway may 4xx on an unknown field, so the safe
  default is to drop it. The cost: a gateway fronting DeepSeek or GLM loses that
  round-tripping.

**Never construct an OpenAI SDK client without a credential.** ``AsyncOpenAI`` has no
safe "no key" value: ``None`` sends the ambient ``OPENAI_API_KEY`` — which for this
client means handing an operator's personal key to *their* configured base URL — and
``""`` raises from ``openai`` 2.34.0 onward. See :meth:`OpenAiCompatClient._get_client`
for the measurements; a keyless client takes the stdlib path, which omits the header.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lad_mcp_server.openrouter_client import (
    OpenRouterCallResult,
    extract_reasoning_content,
    strip_reasoning_content,
)

# `litellm/` is accepted alongside the neutral name so the originally requested setup
# stays discoverable. Dual prefixes have precedent in `ollama_cloud_client.py`.
_OPENAI_COMPAT_PREFIX_RE = re.compile(r"^(?:openai_compat|litellm)/", re.IGNORECASE)


# Guards the environment swap below. Module-level because the swap is process-wide:
# two clients constructing at once must not restore each other's snapshot.
_AMBIENT_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def _ambient_openai_settings_suppressed() -> Any:
    """Remove every ``OPENAI_*`` variable for the duration of the block.

    The OpenAI SDK reads several settings from the environment when the matching
    constructor argument is omitted — the API key, the organization and project ids,
    and (from 2.53) arbitrary request headers. Every one of those would be sent to
    this client's *user-supplied* base URL. Naming them individually would be a
    denylist against a dependency with no upper bound, so the whole namespace goes.

    Yields:
        ``None``, with the variables removed; they are restored on exit.
    """
    with _AMBIENT_ENV_LOCK:
        saved = {key: value for key, value in os.environ.items() if key.startswith("OPENAI_")}
        for key in saved:
            del os.environ[key]
        try:
            yield
        finally:
            os.environ.update(saved)


class OpenAiCompatClientError(RuntimeError):
    """Raised when a call to the configured OpenAI-compatible endpoint fails."""


def is_openai_compat_model(model: str) -> bool:
    """Report whether a model name is routed to an OpenAI-compatible endpoint.

    Args:
        model: The configured reviewer model name.

    Returns:
        ``True`` when the name carries the ``openai_compat/`` or ``litellm/`` prefix.
    """
    if not isinstance(model, str):
        return False
    return _OPENAI_COMPAT_PREFIX_RE.match(model.strip()) is not None


def normalize_openai_compat_model_name(model: str) -> str:
    """Strip the routing prefix, leaving the name the gateway knows.

    Returns the original string when stripping would leave nothing, so a bare
    ``litellm/`` never becomes an empty model name — an empty name would be a third
    state that routing and budgeting could disagree about. Matches the convention in
    the other direct clients.

    Args:
        model: The configured reviewer model name.

    Returns:
        The model name with its prefix removed.
    """
    if not isinstance(model, str):
        return model
    normalized = _OPENAI_COMPAT_PREFIX_RE.sub("", model.strip(), count=1)
    return normalized if normalized else model.strip()


def _normalize_tool_calls(tool_calls_obj: Any) -> list[dict[str, Any]]:
    """Normalise SDK or raw tool calls into plain dictionaries.

    Args:
        tool_calls_obj: Whatever the endpoint returned for ``tool_calls``.

    Returns:
        A list of tool-call dictionaries; empty when there were none.
    """
    if not isinstance(tool_calls_obj, list):
        return []
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


class OpenAiCompatClient:
    """Chat-completions client for a user-supplied OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        max_concurrent_requests: int,
    ) -> None:
        """Configure the client.

        Args:
            base_url: The endpoint root, e.g. ``https://litellm.internal/v1``.
            api_key: Bearer token, or ``None`` for a gateway that needs no auth.
            max_concurrent_requests: Cap on in-flight requests.
        """
        self._base_url = base_url.rstrip("/")
        # Normalised to "" rather than kept as None: see the module docstring — the
        # SDK treats None as "read OPENAI_API_KEY from the environment".
        self._api_key = api_key or ""
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
        """Return a concurrency semaphore bound to the running loop.

        Returns:
            The semaphore for the current event loop.
        """
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
        """Shut down the request thread pool. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - Python < 3.9
            self._executor.shutdown(wait=False)

    def _get_client(self) -> Any:
        """Return the OpenAI SDK client, or the string ``"stdlib"`` as a fallback.

        A keyless gateway is supported, but the SDK refuses to be a party to it: no
        value of ``api_key`` works across the versions this project accepts. Measured
        on ``AsyncOpenAI(base_url=..., api_key=X)``:

        =========  =============================  =============================
        ``X``      ``openai`` 2.16                ``openai`` 2.53
        =========  =============================  =============================
        ``None``   sends ambient OPENAI_API_KEY   sends ambient OPENAI_API_KEY
        ``""``     no ``Authorization`` header    raises ``OpenAIError``
        =========  =============================  =============================

        ``None`` would send an operator's personal OpenAI key to a user-supplied base
        URL, and ``""`` stopped working in 2.34. Since ``pyproject.toml`` bounds
        ``openai`` only from below, the only version-independent answer is not to
        construct an SDK client at all without a credential — the stdlib path below
        simply omits the header.

        The key is not the only ambient setting the SDK reads. Measured on both 2.16
        and 2.53, ``OPENAI_ORG_ID`` and ``OPENAI_PROJECT_ID`` become
        ``openai-organization`` / ``openai-project`` request headers, and 2.53 also
        merges arbitrary headers from ``OPENAI_CUSTOM_HEADERS``. All of that would go
        to the operator's gateway. Suppressing the three known names would be a
        denylist against an unbounded dependency — the same mistake as pinning
        ``api_key=""`` — so construction happens with every ``OPENAI_*`` variable
        removed, which also covers whatever a future release adds.

        Returns:
            An ``AsyncOpenAI`` instance, or ``"stdlib"`` when the SDK is absent or no
            credential is configured.
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            if not self._api_key:
                self._client = "stdlib"
                return self._client
            try:
                from openai import AsyncOpenAI
            except Exception:  # pragma: no cover - exercised only without the SDK
                self._client = "stdlib"
                return self._client

            with _ambient_openai_settings_suppressed():
                self._client = AsyncOpenAI(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    default_headers={"User-Agent": "claude-code/1.0"},
                )
            return self._client

    def _headers(self) -> dict[str, str]:
        """Build request headers, omitting auth entirely when no key is configured.

        Returns:
            Headers for the stdlib request path.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "claude-code/1.0",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
        """Call the endpoint using only the standard library.

        Args:
            model: Model name as the gateway knows it.
            messages: Chat messages.
            timeout_seconds: Request timeout.
            max_output_tokens: Output cap.
            tools: Tool schemas, if any.
            tool_choice: Tool choice directive, if any.
            extra_body: Additional request fields.

        Returns:
            The normalised call result.

        Raises:
            OpenAiCompatClientError: On transport, status or decoding failure.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": strip_reasoning_content(messages),
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
            headers=self._headers(),
            method="POST",
        )

        def _do_request() -> dict[str, Any]:
            """Perform the blocking HTTP call.

            Returns:
                The decoded JSON response.

            Raises:
                OpenAiCompatClientError: On any transport or decoding failure.
            """
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                raise OpenAiCompatClientError(
                    f"OpenAI-compatible endpoint HTTP {getattr(exc, 'code', 'error')}: {detail[:300]}"
                ) from exc
            except Exception as exc:
                raise OpenAiCompatClientError(f"OpenAI-compatible endpoint request failed: {exc}") from exc
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OpenAiCompatClientError("OpenAI-compatible endpoint response was not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise OpenAiCompatClientError("OpenAI-compatible endpoint response JSON was not an object")
            if "error" in parsed:
                raise OpenAiCompatClientError(f"OpenAI-compatible endpoint error: {parsed.get('error')}")
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
        """Call the configured endpoint's chat-completions API.

        Args:
            model: Model name as the gateway knows it.
            messages: Chat messages.
            timeout_seconds: Request timeout.
            max_output_tokens: Output cap.
            tools: Tool schemas, if any.
            tool_choice: Tool choice directive, if any.
            extra_body: Additional request fields.

        Returns:
            The normalised call result.

        Raises:
            OpenAiCompatClientError: On timeout or any request failure.
        """
        # Wrapped so construction failures arrive as this client's error type. Left
        # bare, an SDK error would escape the provider-fallback handling and be shown
        # with OpenRouter troubleshooting advice.
        try:
            client = self._get_client()
        except Exception as exc:
            raise OpenAiCompatClientError(
                f"OpenAI-compatible endpoint client could not be created: {exc}"
            ) from exc

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

            # An arbitrary gateway may reject an unknown `reasoning_content` field, and
            # the SDK forwards unknown message keys verbatim rather than dropping them.
            sanitized_messages = strip_reasoning_content(messages)
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=sanitized_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        max_tokens=max_output_tokens,
                        extra_body=extra_body,
                        timeout=timeout_seconds,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise OpenAiCompatClientError(
                    f"OpenAI-compatible endpoint request timed out after {timeout_seconds}s"
                ) from exc
            except Exception as exc:
                raise OpenAiCompatClientError(f"OpenAI-compatible endpoint request failed: {exc}") from exc

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
