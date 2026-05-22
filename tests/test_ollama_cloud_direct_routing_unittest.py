from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.ollama_cloud_client import (
    OllamaCloudClientError,
    is_ollama_model,
    normalize_ollama_model_name,
    translate_messages_to_ollama,
    translate_ollama_response,
)
from lad_mcp_server.review_service import ReviewService


# ---------------------------------------------------------------------------
# R1: Model identification
# ---------------------------------------------------------------------------


class TestIsOllamaModel(unittest.TestCase):
    def test_ollama_prefix(self) -> None:
        self.assertTrue(is_ollama_model("ollama/gpt-oss:120b"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(is_ollama_model("Ollama/gpt-oss:120b"))

    def test_non_ollama(self) -> None:
        self.assertFalse(is_ollama_model("deepseek/deepseek-v4-pro"))

    def test_none(self) -> None:
        self.assertFalse(is_ollama_model(None))  # type: ignore[arg-type]

    def test_empty(self) -> None:
        self.assertFalse(is_ollama_model(""))

    def test_whitespace(self) -> None:
        self.assertTrue(is_ollama_model("  ollama/gpt-oss  "))

    def test_no_prefix(self) -> None:
        self.assertFalse(is_ollama_model("gpt-oss:120b"))


# ---------------------------------------------------------------------------
# R2: Model name normalization
# ---------------------------------------------------------------------------


class TestNormalizeOllamaModelName(unittest.TestCase):
    def test_strips_prefix(self) -> None:
        self.assertEqual(normalize_ollama_model_name("ollama/gpt-oss:120b"), "gpt-oss:120b")

    def test_case_insensitive(self) -> None:
        self.assertEqual(normalize_ollama_model_name("Ollama/gpt-oss:120b"), "gpt-oss:120b")

    def test_no_prefix(self) -> None:
        self.assertEqual(normalize_ollama_model_name("gpt-oss:120b"), "gpt-oss:120b")

    def test_none_passthrough(self) -> None:
        self.assertIsNone(normalize_ollama_model_name(None))  # type: ignore[arg-type]

    def test_whitespace(self) -> None:
        self.assertEqual(normalize_ollama_model_name("  ollama/gpt-oss:120b  "), "gpt-oss:120b")


# ---------------------------------------------------------------------------
# R3: Message translation to Ollama format
# ---------------------------------------------------------------------------


class TestTranslateMessagesToOllama(unittest.TestCase):
    def test_system_user_assistant(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = translate_messages_to_ollama(messages)
        self.assertEqual(result[0], {"role": "system", "content": "sys"})
        self.assertEqual(result[1], {"role": "user", "content": "hello"})
        self.assertEqual(result[2], {"role": "assistant", "content": "hi"})

    def test_tool_message_translated(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "t1", "name": "read_file", "content": "file contents"},
        ]
        result = translate_messages_to_ollama(messages)
        self.assertEqual(result[0]["role"], "tool")
        self.assertEqual(result[0]["tool_name"], "read_file")
        self.assertEqual(result[0]["content"], "file contents")
        self.assertNotIn("tool_call_id", result[0])

    def test_assistant_with_tool_calls(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            },
        ]
        result = translate_messages_to_ollama(messages)
        self.assertIn("tool_calls", result[0])


# ---------------------------------------------------------------------------
# R4: Response translation from Ollama format
# ---------------------------------------------------------------------------


class TestTranslateOllamaResponse(unittest.TestCase):
    def test_basic_response(self) -> None:
        ollama_resp = {
            "message": {"role": "assistant", "content": "Looks good."},
            "done_reason": "stop",
            "model": "gpt-oss:120b",
        }
        result = translate_ollama_response(ollama_resp)
        self.assertEqual(result.content, "Looks good.")
        self.assertEqual(result.tool_calls, [])
        self.assertIsNone(result.reasoning_content)

    def test_tool_calls_response(self) -> None:
        ollama_resp = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "src/main.py"},
                        }
                    }
                ],
            },
            "done_reason": "stop",
        }
        result = translate_ollama_response(ollama_resp)
        self.assertIsNone(result.content)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "read_file")
        # Arguments should be a JSON string
        args = result.tool_calls[0]["function"]["arguments"]
        self.assertIsInstance(args, str)
        parsed = json.loads(args)
        self.assertEqual(parsed["path"], "src/main.py")

    def test_empty_content(self) -> None:
        ollama_resp = {
            "message": {"role": "assistant", "content": ""},
            "done_reason": "stop",
        }
        result = translate_ollama_response(ollama_resp)
        self.assertIsNone(result.content)


# ---------------------------------------------------------------------------
# R5: Config parsing
# ---------------------------------------------------------------------------


class TestOllamaConfigParsing(unittest.TestCase):
    def test_key_present(self) -> None:
        with mock.patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "test",
            "OLLAMA_API_KEY": "oll-test-key",
        }, clear=False):
            s = Settings.from_env()
            self.assertEqual(s.ollama_api_key, "oll-test-key")

    def test_key_absent(self) -> None:
        import os
        os.environ.pop("OLLAMA_API_KEY", None)
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}, clear=False):
            s = Settings.from_env()
            self.assertIsNone(s.ollama_api_key)


# ---------------------------------------------------------------------------
# R6: Routing — _prepare_reviewer_config detects Ollama
# ---------------------------------------------------------------------------


def _build_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="ollama/gpt-oss:120b",
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
        deepseek_api_key=None,
        ollama_api_key="oll-test-key",
        intermittent_review_calls=0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestPrepareReviewerConfigOllama(unittest.TestCase):
    def test_detects_ollama_model_with_key(self) -> None:
        settings = _build_settings(ollama_api_key="oll-test-key")
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            cfg = service._prepare_reviewer_config("ollama/gpt-oss:120b", repo_root=Path(td))
            self.assertTrue(cfg.use_ollama_direct)
            self.assertEqual(cfg.direct_ollama_model_name, "gpt-oss:120b")

    def test_no_direct_without_key(self) -> None:
        settings = _build_settings(ollama_api_key=None)
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            mock_models = mock.Mock()
            mock_models.get_model.return_value = ModelMetadata(
                model_id="ollama/gpt-oss:120b",
                context_length=50000,
                supported_parameters=("tools", "tool_choice", "max_tokens"),
                provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
            )
            service._models = mock_models
            cfg = service._prepare_reviewer_config("ollama/gpt-oss:120b", repo_root=Path(td))
            self.assertFalse(cfg.use_ollama_direct)
            self.assertIsNone(cfg.direct_ollama_model_name)


# ---------------------------------------------------------------------------
# R7: Fallback — _call_model_with_provider_fallback
# ---------------------------------------------------------------------------


class TestOllamaProviderFallback(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_falls_back_to_openrouter_on_ollama_failure(self) -> None:
        from lad_mcp_server.openrouter_client import OpenRouterCallResult
        settings = _build_settings(ollama_api_key="oll-test-key")
        with tempfile.TemporaryDirectory() as td:
            failing_ollama = mock.Mock()
            failing_ollama.chat_completion = mock.AsyncMock(
                side_effect=OllamaCloudClientError("Ollama API error")
            )
            ok_or = mock.Mock()
            ok_or.chat_completion = mock.AsyncMock(
                return_value=OpenRouterCallResult(content="OK", tool_calls=[], raw={})
            )
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=ok_or,
                models_client=mock.Mock(),
                ollama_client=failing_ollama,
            )
            provider_used: list[str] = ["ollama"]
            provider_notes: list[str] = []
            result = self._run(service._call_model_with_provider_fallback(
                model="ollama/gpt-oss:120b",
                direct_model_name=None,
                use_zai_direct=False,
                direct_kimi_model_name=None,
                use_kimi_direct=False,
                direct_deepseek_model_name=None,
                use_deepseek_direct=False,
                direct_ollama_model_name="gpt-oss:120b",
                use_ollama_direct=True,
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
            self.assertTrue(any("Ollama" in n for n in provider_notes))

    def test_ollama_direct_succeeds(self) -> None:
        from lad_mcp_server.openrouter_client import OpenRouterCallResult
        settings = _build_settings(ollama_api_key="oll-test-key")
        with tempfile.TemporaryDirectory() as td:
            ok_ollama = mock.Mock()
            ok_ollama.chat_completion = mock.AsyncMock(
                return_value=OpenRouterCallResult(content="Ollama review", tool_calls=[], raw={})
            )
            or_client = mock.Mock()
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=or_client,
                models_client=mock.Mock(),
                ollama_client=ok_ollama,
            )
            provider_used: list[str] = ["openrouter"]
            provider_notes: list[str] = []
            result = self._run(service._call_model_with_provider_fallback(
                model="ollama/gpt-oss:120b",
                direct_model_name=None,
                use_zai_direct=False,
                direct_kimi_model_name=None,
                use_kimi_direct=False,
                direct_deepseek_model_name=None,
                use_deepseek_direct=False,
                direct_ollama_model_name="gpt-oss:120b",
                use_ollama_direct=True,
                messages=[{"role": "user", "content": "hi"}],
                timeout_seconds=10,
                max_output_tokens=100,
                tools=None,
                preferred_tool_choice=None,
                extra_body=None,
                provider_used=provider_used,
                provider_notes=provider_notes,
            ))
            self.assertEqual(result.content, "Ollama review")
            self.assertEqual(provider_used, ["ollama"])
            or_client.chat_completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
