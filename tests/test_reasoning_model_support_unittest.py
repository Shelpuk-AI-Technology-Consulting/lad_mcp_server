from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.kimi_code_client import KimiCodeClient
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.openrouter_client import OpenRouterClient
from lad_mcp_server.review_service import ReviewService
from lad_mcp_server.zai_coding_client import ZaiCodingClient


# ---------------------------------------------------------------------------
# Fakes for the AsyncOpenAI SDK shape
# ---------------------------------------------------------------------------


class _FakeFn:
    def __init__(self, name: str | None, arguments: str | None) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, *, id: str, name: str | None = None, arguments: str | None = None) -> None:
        self.id = id
        self.type = "function"
        self.function = _FakeFn(name, arguments)


class _FakeMessage:
    """Pydantic-like message object with a `reasoning_content` attribute."""

    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content


class _FakeMessageModelExtra:
    """SDK Pydantic-style message where `reasoning_content` lives only in `model_extra`."""

    def __init__(self, *, content: str | None = None, reasoning_content: str | None = None) -> None:
        self.content = content
        self.tool_calls = []
        self.model_extra = {"reasoning_content": reasoning_content} if reasoning_content else {}


class _FakeChoice:
    def __init__(self, message) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, message) -> None:
        self.choices = [_FakeChoice(message)]


class _RecordingSDKClient:
    """Stand-in for openai.AsyncOpenAI — records `.create()` kwargs and returns scripted responses."""

    def __init__(self, response_message) -> None:
        self._response_message = response_message
        self.calls: list[dict[str, Any]] = []
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent) -> None:
            self.completions = parent._Completions(parent)

    class _Completions:
        def __init__(self, parent) -> None:
            self._parent = parent

        async def create(self, **kwargs):
            self._parent.calls.append(kwargs)
            return _FakeResponse(self._parent._response_message)


def _install_sdk_client(client_instance, fake_sdk_client) -> None:
    """Bypass the lazy `_get_client` initialization to inject the fake."""
    client_instance._client = fake_sdk_client


# ---------------------------------------------------------------------------
# R1: timeout forwarding
# ---------------------------------------------------------------------------


class TestTimeoutForwarding(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_openrouter_client_forwards_timeout_to_sdk(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = OpenRouterClient(
            api_key="test", http_referer=None, x_title=None, max_concurrent_requests=2
        )
        _install_sdk_client(client, sdk)
        self._run(client.chat_completion(
            model="any/model",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=295,
            max_output_tokens=100,
        ))
        self.assertEqual(sdk.calls[0].get("timeout"), 295)

    def test_zai_coding_client_forwards_timeout_to_sdk(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = ZaiCodingClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        self._run(client.chat_completion(
            model="glm-5",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=295,
            max_output_tokens=100,
        ))
        self.assertEqual(sdk.calls[0].get("timeout"), 295)

    def test_kimi_code_client_forwards_timeout_to_sdk(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = KimiCodeClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        self._run(client.chat_completion(
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=295,
            max_output_tokens=100,
        ))
        self.assertEqual(sdk.calls[0].get("timeout"), 295)


# ---------------------------------------------------------------------------
# R2: capture reasoning_content
# ---------------------------------------------------------------------------


class TestCaptureReasoningContent(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_openrouter_client_captures_reasoning_content(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="visible", reasoning_content="R1"))
        client = OpenRouterClient(
            api_key="test", http_referer=None, x_title=None, max_concurrent_requests=2
        )
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="any/model",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertEqual(result.reasoning_content, "R1")

    def test_zai_coding_client_captures_reasoning_content(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="visible", reasoning_content="R2"))
        client = ZaiCodingClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="glm-5",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertEqual(result.reasoning_content, "R2")

    def test_kimi_code_client_captures_reasoning_content(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="visible", reasoning_content="R3"))
        client = KimiCodeClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertEqual(result.reasoning_content, "R3")

    def test_openrouter_client_captures_reasoning_content_via_model_extra(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessageModelExtra(content="visible", reasoning_content="R4"))
        client = OpenRouterClient(
            api_key="test", http_referer=None, x_title=None, max_concurrent_requests=2
        )
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="any/model",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertEqual(result.reasoning_content, "R4")

    def test_no_reasoning_content_yields_none(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="just visible"))
        client = OpenRouterClient(
            api_key="test", http_referer=None, x_title=None, max_concurrent_requests=2
        )
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="any/model",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertIsNone(result.reasoning_content)


# ---------------------------------------------------------------------------
# R5: reasoning_content stripping for non-Z.AI; pass-through for Z.AI
# ---------------------------------------------------------------------------


class TestReasoningContentStripping(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def _make_messages_with_reasoning(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                "reasoning_content": "PRIOR_REASONING",
            },
            {"role": "tool", "tool_call_id": "t1", "name": "x", "content": "{}"},
        ]

    def test_openrouter_client_strips_reasoning_content_from_messages(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = OpenRouterClient(
            api_key="test", http_referer=None, x_title=None, max_concurrent_requests=2
        )
        _install_sdk_client(client, sdk)
        messages = self._make_messages_with_reasoning()
        self._run(client.chat_completion(
            model="any/model",
            messages=messages,
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        sent = sdk.calls[0]["messages"]
        for m in sent:
            self.assertNotIn("reasoning_content", m)
        # Original input is not mutated.
        self.assertEqual(messages[2]["reasoning_content"], "PRIOR_REASONING")

    def test_kimi_code_client_strips_reasoning_content_from_messages(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = KimiCodeClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        messages = self._make_messages_with_reasoning()
        self._run(client.chat_completion(
            model="kimi-for-coding",
            messages=messages,
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        sent = sdk.calls[0]["messages"]
        for m in sent:
            self.assertNotIn("reasoning_content", m)

    def test_zai_coding_client_passes_reasoning_content_through(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = ZaiCodingClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        messages = self._make_messages_with_reasoning()
        self._run(client.chat_completion(
            model="glm-5",
            messages=messages,
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        sent = sdk.calls[0]["messages"]
        # Z.AI is the one provider that wants reasoning_content preserved.
        assistant_messages = [m for m in sent if m.get("role") == "assistant"]
        self.assertTrue(any("reasoning_content" in m for m in assistant_messages))


# ---------------------------------------------------------------------------
# R3: _tool_loop threads reasoning_content into the next assistant turn
# ---------------------------------------------------------------------------


def _build_settings() -> Settings:
    return Settings(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="z-ai/glm-5",
        openrouter_secondary_reviewer_model="0",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        openrouter_reviewer_timeout_seconds=10,
        openrouter_tool_call_timeout_seconds=15,
        openrouter_max_concurrent_requests=2,
        openrouter_fixed_output_tokens=1000,
        openrouter_context_overhead_tokens=2000,
        openrouter_model_metadata_ttl_seconds=3600,
        openrouter_max_input_chars=10000,
        openrouter_include_reasoning=False,
        lad_serena_max_tool_calls=8,
        lad_serena_tool_timeout_seconds=2,
        lad_serena_max_tool_result_chars=12000,
        lad_serena_max_total_chars=50000,
        lad_serena_max_dir_entries=100,
        lad_serena_max_search_results=20,
        zai_coding_plan_key="zai-test-key",
        intermittent_review_calls=0,
    )


class _SerenaCtxStub:
    def __init__(self) -> None:
        self.activated_project: str | None = "."
        self.used_tools: set[str] = set()
        self.used_memories: set[str] = set()
        self.used_paths: set[str] = set()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}}}
        ]

    def call_tool(self, name: str, arguments_json: str) -> str:
        return "{}"


class _ModelsStub:
    def get_model(self, model_id: str) -> ModelMetadata:
        return ModelMetadata(
            model_id=model_id,
            context_length=50000,
            supported_parameters=("tools", "tool_choice", "max_tokens"),
            provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
        )


def _tool_call_response(call_id: str, *, reasoning_content: str | None = None) -> Any:
    obj_attrs = {
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
        ],
        "raw": {},
    }
    if reasoning_content is not None:
        obj_attrs["reasoning_content"] = reasoning_content
    else:
        obj_attrs["reasoning_content"] = None
    return type("R", (), obj_attrs)()


def _final_response(content: str = "## Summary\nOK") -> Any:
    return type("R", (), {"content": content, "tool_calls": [], "raw": {}, "reasoning_content": None})()


class _RecordingZaiClient:
    """Records the `messages` argument it receives, sequenced via scripted responses."""

    def __init__(self, scripted_responses: list[Any]) -> None:
        self._scripted = list(scripted_responses)
        self._idx = 0
        self.calls: list[list[dict[str, Any]]] = []

    async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens,
                              tools=None, tool_choice=None, extra_body=None):
        self.calls.append([dict(m) for m in messages])
        idx = min(self._idx, len(self._scripted) - 1)
        self._idx += 1
        return self._scripted[idx]


class TestToolLoopReasoningContent(unittest.TestCase):
    def test_tool_loop_resends_reasoning_content_to_zai(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena").mkdir()
                zai = _RecordingZaiClient([
                    _tool_call_response("t1", reasoning_content="REASONING_FROM_TURN_1"),
                    _final_response("## Summary\nDone"),
                ])
                service = ReviewService(
                    repo_root=repo,
                    settings=_build_settings(),
                    openrouter_client=mock.Mock(),
                    models_client=_ModelsStub(),
                    zai_client=zai,
                )
                await service._tool_loop(
                    model="z-ai/glm-5",
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "u"},
                    ],
                    tools=[
                        {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}}}
                    ],
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtxStub(),
                    extra_body=None,
                    reviewer_timeout_seconds=10,
                    max_output_tokens=100,
                    max_tool_calls=4,
                    tool_timeout_seconds=2,
                    use_zai_direct=True,
                    direct_model_name="glm-5",
                )
                # Two main calls recorded (turn 1 tool_call, turn 2 final).
                self.assertEqual(len(zai.calls), 2)
                turn2_messages = zai.calls[1]
                # The assistant message in turn 2's payload should carry the prior reasoning.
                assistant_msgs = [m for m in turn2_messages if m.get("role") == "assistant"]
                self.assertTrue(any(m.get("reasoning_content") == "REASONING_FROM_TURN_1" for m in assistant_msgs))

        asyncio.run(scenario())

    def test_tool_loop_omits_reasoning_content_when_not_returned(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena").mkdir()
                zai = _RecordingZaiClient([
                    _tool_call_response("t1", reasoning_content=None),
                    _final_response("## Summary\nDone"),
                ])
                service = ReviewService(
                    repo_root=repo,
                    settings=_build_settings(),
                    openrouter_client=mock.Mock(),
                    models_client=_ModelsStub(),
                    zai_client=zai,
                )
                await service._tool_loop(
                    model="z-ai/glm-5",
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "u"},
                    ],
                    tools=[
                        {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}}}
                    ],
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtxStub(),
                    extra_body=None,
                    reviewer_timeout_seconds=10,
                    max_output_tokens=100,
                    max_tool_calls=4,
                    tool_timeout_seconds=2,
                    use_zai_direct=True,
                    direct_model_name="glm-5",
                )
                turn2_messages = zai.calls[1]
                assistant_msgs = [m for m in turn2_messages if m.get("role") == "assistant"]
                for m in assistant_msgs:
                    self.assertNotIn("reasoning_content", m)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
