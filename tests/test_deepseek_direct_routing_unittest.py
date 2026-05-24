from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    is_deepseek_model,
    normalize_deepseek_model_name,
)
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.openrouter_client import OpenRouterCallResult
from lad_mcp_server.review_service import ReviewService


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
    client_instance._client = fake_sdk_client


# ---------------------------------------------------------------------------
# R1: Model identification
# ---------------------------------------------------------------------------


class TestIsDeepSeekModel(unittest.TestCase):
    def test_deepseek_prefix(self) -> None:
        self.assertTrue(is_deepseek_model("deepseek/deepseek-v4-pro"))

    def test_deepseek_r1(self) -> None:
        self.assertTrue(is_deepseek_model("deepseek/deepseek-r1"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(is_deepseek_model("DeepSeek/deepseek-v4-pro"))

    def test_non_deepseek(self) -> None:
        self.assertFalse(is_deepseek_model("google/gemini-2.5-pro"))

    def test_non_string(self) -> None:
        self.assertFalse(is_deepseek_model(123))  # type: ignore[arg-type]

    def test_none(self) -> None:
        self.assertFalse(is_deepseek_model(None))  # type: ignore[arg-type]

    def test_empty_string(self) -> None:
        self.assertFalse(is_deepseek_model(""))

    def test_deepseek_without_slash(self) -> None:
        self.assertFalse(is_deepseek_model("deepseek-v4-pro"))

    def test_whitespace_handling(self) -> None:
        self.assertTrue(is_deepseek_model("  deepseek/deepseek-v4-pro  "))


# ---------------------------------------------------------------------------
# R2: Model name normalization
# ---------------------------------------------------------------------------


class TestNormalizeDeepSeekModelName(unittest.TestCase):
    def test_strips_prefix(self) -> None:
        self.assertEqual(normalize_deepseek_model_name("deepseek/deepseek-v4-pro"), "deepseek-v4-pro")

    def test_strips_r1_prefix(self) -> None:
        self.assertEqual(normalize_deepseek_model_name("deepseek/deepseek-r1"), "deepseek-r1")

    def test_case_insensitive(self) -> None:
        self.assertEqual(normalize_deepseek_model_name("DeepSeek/deepseek-v4-pro"), "deepseek-v4-pro")

    def test_no_prefix_returns_as_is(self) -> None:
        self.assertEqual(normalize_deepseek_model_name("deepseek-v4-pro"), "deepseek-v4-pro")

    def test_non_string_passthrough(self) -> None:
        self.assertIsNone(normalize_deepseek_model_name(None))  # type: ignore[arg-type]

    def test_whitespace_trimmed(self) -> None:
        self.assertEqual(normalize_deepseek_model_name("  deepseek/deepseek-v4-pro  "), "deepseek-v4-pro")


# ---------------------------------------------------------------------------
# R3: DeepSeek client — timeout forwarding
# ---------------------------------------------------------------------------


class TestDeepSeekClientTimeoutForwarding(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_forwards_timeout_to_sdk(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = DeepSeekClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        self._run(client.chat_completion(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=295,
            max_output_tokens=100,
        ))
        self.assertEqual(sdk.calls[0].get("timeout"), 295)


# ---------------------------------------------------------------------------
# R3: DeepSeek client — reasoning_content capture
# ---------------------------------------------------------------------------


class TestDeepSeekClientReasoningContent(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_captures_reasoning_content(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="visible", reasoning_content="REASONING"))
        client = DeepSeekClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertEqual(result.reasoning_content, "REASONING")

    def test_no_reasoning_content_yields_none(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="just visible"))
        client = DeepSeekClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        result = self._run(client.chat_completion(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        self.assertIsNone(result.reasoning_content)


# ---------------------------------------------------------------------------
# R3: DeepSeek client — reasoning_content NOT stripped (pass-through like Z.AI)
# ---------------------------------------------------------------------------


class TestDeepSeekClientReasoningContentPassThrough(unittest.TestCase):
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

    def test_deepseek_client_passes_reasoning_content_through(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = DeepSeekClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        messages = self._make_messages_with_reasoning()
        self._run(client.chat_completion(
            model="deepseek-v4-pro",
            messages=messages,
            timeout_seconds=10,
            max_output_tokens=100,
        ))
        sent = sdk.calls[0]["messages"]
        assistant_messages = [m for m in sent if m.get("role") == "assistant"]
        self.assertTrue(any("reasoning_content" in m for m in assistant_messages))


# ---------------------------------------------------------------------------
# R5: Config — DEEPSEEK_API_KEY parsing
# ---------------------------------------------------------------------------


class TestDeepSeekConfigParsing(unittest.TestCase):
    def test_key_present(self) -> None:
        with mock.patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "test",
            "DEEPSEEK_API_KEY": "dsk-test-key",
        }, clear=False):
            s = Settings.from_env()
            self.assertEqual(s.deepseek_api_key, "dsk-test-key")

    def test_key_absent(self) -> None:
        env = {"OPENROUTER_API_KEY": "test"}
        with mock.patch.dict("os.environ", env, clear=False):
            # Remove if present
            mock.patch.dict("os.environ", {}, clear=False).__enter__()
            import os
            os.environ.pop("DEEPSEEK_API_KEY", None)
            s = Settings.from_env()
            self.assertIsNone(s.deepseek_api_key)


# ---------------------------------------------------------------------------
# R6: Routing — _prepare_reviewer_config detects DeepSeek
# ---------------------------------------------------------------------------


def _build_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="deepseek/deepseek-v4-pro",
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
        zai_coding_plan_key=None,
        kimi_code_api_key=None,
        deepseek_api_key="dsk-test-key",
        intermittent_review_calls=0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestPrepareReviewerConfigDeepSeek(unittest.TestCase):
    def test_detects_deepseek_model_with_key(self) -> None:
        settings = _build_settings(deepseek_api_key="dsk-test-key")
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            cfg = service._prepare_reviewer_config("deepseek/deepseek-v4-pro", repo_root=Path(td))
            self.assertTrue(cfg.use_deepseek_direct)
            self.assertEqual(cfg.direct_deepseek_model_name, "deepseek-v4-pro")

    def test_no_direct_without_key(self) -> None:
        settings = _build_settings(deepseek_api_key=None)
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            # Without key, should go to OpenRouter — need model metadata mock
            mock_models = mock.Mock()
            mock_models.get_model.return_value = ModelMetadata(
                model_id="deepseek/deepseek-v4-pro",
                context_length=50000,
                supported_parameters=("tools", "tool_choice", "max_tokens"),
                provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
            )
            service._models = mock_models
            cfg = service._prepare_reviewer_config("deepseek/deepseek-v4-pro", repo_root=Path(td))
            self.assertFalse(cfg.use_deepseek_direct)
            self.assertIsNone(cfg.direct_deepseek_model_name)

    def test_non_deepseek_model_ignored(self) -> None:
        settings = _build_settings(deepseek_api_key="dsk-test-key")
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            mock_models = mock.Mock()
            mock_models.get_model.return_value = ModelMetadata(
                model_id="google/gemini-2.5-pro",
                context_length=50000,
                supported_parameters=("tools", "tool_choice", "max_tokens"),
                provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
            )
            service._models = mock_models
            cfg = service._prepare_reviewer_config("google/gemini-2.5-pro", repo_root=Path(td))
            self.assertFalse(cfg.use_deepseek_direct)
            self.assertIsNone(cfg.direct_deepseek_model_name)


# ---------------------------------------------------------------------------
# R7: Fallback — _call_model_with_provider_fallback
# ---------------------------------------------------------------------------


class TestDeepSeekProviderFallback(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_falls_back_to_openrouter_on_deepseek_failure(self) -> None:
        settings = _build_settings(deepseek_api_key="dsk-test-key")
        with tempfile.TemporaryDirectory() as td:
            # DeepSeek client that raises
            failing_deepseek = mock.Mock()
            failing_deepseek.chat_completion = mock.AsyncMock(
                side_effect=DeepSeekClientError("DeepSeek API error")
            )
            # OpenRouter client that succeeds
            ok_or = mock.Mock()
            ok_or.chat_completion = mock.AsyncMock(
                return_value=OpenRouterCallResult(content="OK", tool_calls=[], raw={})
            )
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=ok_or,
                models_client=mock.Mock(),
                deepseek_client=failing_deepseek,
            )
            provider_used: list[str] = ["deepseek"]
            provider_notes: list[str] = []
            result = self._run(service._call_model_with_provider_fallback(
                model="deepseek/deepseek-v4-pro",
                direct_model_name="deepseek-v4-pro",
                use_zai_direct=False,
                direct_kimi_model_name=None,
                use_kimi_direct=False,
                use_deepseek_direct=True,
                direct_deepseek_model_name="deepseek-v4-pro",
                messages=[{"role": "user", "content": "hi"}],
                timeout_seconds=10,
                max_output_tokens=100,
                tools=None,
                preferred_tool_choice=None,
                extra_body=None,
                provider_used=provider_used,
                provider_notes=provider_notes,
            ))
            self.assertEqual(result.content, "OK")
            self.assertEqual(provider_used, ["openrouter"])
            self.assertTrue(any("DeepSeek" in n for n in provider_notes))

    def test_deepseek_direct_succeeds(self) -> None:
        settings = _build_settings(deepseek_api_key="dsk-test-key")
        with tempfile.TemporaryDirectory() as td:
            ok_deepseek = mock.Mock()
            ok_deepseek.chat_completion = mock.AsyncMock(
                return_value=OpenRouterCallResult(
                    content="DeepSeek review",
                    tool_calls=[],
                    raw={},
                    reasoning_content="thoughts",
                )
            )
            or_client = mock.Mock()
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=or_client,
                models_client=mock.Mock(),
                deepseek_client=ok_deepseek,
            )
            provider_used: list[str] = ["openrouter"]
            provider_notes: list[str] = []
            result = self._run(service._call_model_with_provider_fallback(
                model="deepseek/deepseek-v4-pro",
                direct_model_name="deepseek-v4-pro",
                use_zai_direct=False,
                direct_kimi_model_name=None,
                use_kimi_direct=False,
                use_deepseek_direct=True,
                direct_deepseek_model_name="deepseek-v4-pro",
                messages=[{"role": "user", "content": "hi"}],
                timeout_seconds=10,
                max_output_tokens=100,
                tools=None,
                preferred_tool_choice=None,
                extra_body=None,
                provider_used=provider_used,
                provider_notes=provider_notes,
            ))
            self.assertEqual(result.content, "DeepSeek review")
            self.assertEqual(result.reasoning_content, "thoughts")
            self.assertEqual(provider_used, ["deepseek"])
            # OpenRouter should NOT have been called
            or_client.chat_completion.assert_not_called()


# ---------------------------------------------------------------------------
# R4: Thinking mode — extra_body injection
# ---------------------------------------------------------------------------


class TestDeepSeekThinkingMode(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_extra_body_includes_thinking_enabled(self) -> None:
        sdk = _RecordingSDKClient(_FakeMessage(content="ok"))
        client = DeepSeekClient(api_key="test", max_concurrent_requests=2)
        _install_sdk_client(client, sdk)
        self._run(client.chat_completion(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10,
            max_output_tokens=100,
            extra_body={"thinking": {"type": "enabled"}},
        ))
        self.assertEqual(sdk.calls[0].get("extra_body"), {"thinking": {"type": "enabled"}})


class TestOpenRouterOnlyKeysFiltering(unittest.TestCase):
    """Verify that _call_model_with_provider_fallback strips OpenRouter-only
    keys from extra_body before forwarding to direct providers."""

    def _build_service(self, *, deepseek_key: str = "test-ds-key") -> tuple[ReviewService, _RecordingSDKClient]:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings(
                openrouter_api_key="test-or",
                openrouter_primary_reviewer_model="deepseek/deepseek-v4-flash",
                openrouter_secondary_reviewer_model="0",
                openrouter_http_referer=None,
                openrouter_x_title=None,
                openrouter_reviewer_timeout_seconds=30,
                openrouter_tool_call_timeout_seconds=30,
                openrouter_max_concurrent_requests=2,
                openrouter_fixed_output_tokens=1000,
                openrouter_context_overhead_tokens=2000,
                openrouter_model_metadata_ttl_seconds=3600,
                openrouter_max_input_chars=10000,
                openrouter_include_reasoning=False,
                lad_serena_max_tool_calls=4,
                lad_serena_tool_timeout_seconds=2,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
                deepseek_api_key=deepseek_key,
            )
            ds_client = DeepSeekClient(api_key=deepseek_key, max_concurrent_requests=2)
            sdk = _RecordingSDKClient(_FakeMessage(content="## Summary\nReview complete with findings."))
            _install_sdk_client(ds_client, sdk)
            models_mock = mock.Mock()
            models_mock.get_model.return_value = ModelMetadata(
                model_id="deepseek/deepseek-v4-flash",
                context_length=50000,
                supported_parameters=("tools", "tool_choice", "max_tokens"),
                provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
            )
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=models_mock,
                deepseek_client=ds_client,
            )
            return service, sdk

    def test_include_reasoning_stripped_from_direct_call(self) -> None:
        service, sdk = self._build_service()
        provider_used = ["openrouter"]
        provider_notes: list[str] = []
        asyncio.run(service._call_model_with_provider_fallback(
            model="deepseek/deepseek-v4-flash",
            direct_model_name=None,
            use_zai_direct=False,
            direct_kimi_model_name=None,
            use_kimi_direct=False,
            direct_deepseek_model_name="deepseek-v4-flash",
            use_deepseek_direct=True,
            messages=[{"role": "user", "content": "review this"}],
            timeout_seconds=10,
            max_output_tokens=100,
            tools=None,
            preferred_tool_choice=None,
            extra_body={"include_reasoning": True, "max_completion_tokens": 2000},
            provider_used=provider_used,
            provider_notes=provider_notes,
        ))
        self.assertEqual(provider_used[0], "deepseek")
        sent_extra_body = sdk.calls[0].get("extra_body")
        self.assertNotIn("include_reasoning", sent_extra_body or {})
        self.assertNotIn("max_completion_tokens", sent_extra_body or {})

    def test_non_openrouter_keys_preserved(self) -> None:
        service, sdk = self._build_service()
        provider_used = ["openrouter"]
        provider_notes: list[str] = []
        asyncio.run(service._call_model_with_provider_fallback(
            model="deepseek/deepseek-v4-flash",
            direct_model_name=None,
            use_zai_direct=False,
            direct_kimi_model_name=None,
            use_kimi_direct=False,
            direct_deepseek_model_name="deepseek-v4-flash",
            use_deepseek_direct=True,
            messages=[{"role": "user", "content": "review this"}],
            timeout_seconds=10,
            max_output_tokens=100,
            tools=None,
            preferred_tool_choice=None,
            extra_body={"thinking": {"type": "enabled"}},
            provider_used=provider_used,
            provider_notes=provider_notes,
        ))
        self.assertEqual(provider_used[0], "deepseek")
        sent_extra_body = sdk.calls[0].get("extra_body")
        self.assertEqual(sent_extra_body, {"thinking": {"type": "enabled"}})

    def test_none_extra_body_stays_none(self) -> None:
        service, sdk = self._build_service()
        provider_used = ["openrouter"]
        provider_notes: list[str] = []
        asyncio.run(service._call_model_with_provider_fallback(
            model="deepseek/deepseek-v4-flash",
            direct_model_name=None,
            use_zai_direct=False,
            direct_kimi_model_name=None,
            use_kimi_direct=False,
            direct_deepseek_model_name="deepseek-v4-flash",
            use_deepseek_direct=True,
            messages=[{"role": "user", "content": "review this"}],
            timeout_seconds=10,
            max_output_tokens=100,
            tools=None,
            preferred_tool_choice=None,
            extra_body=None,
            provider_used=provider_used,
            provider_notes=provider_notes,
        ))
        self.assertEqual(provider_used[0], "deepseek")


if __name__ == "__main__":
    unittest.main()
