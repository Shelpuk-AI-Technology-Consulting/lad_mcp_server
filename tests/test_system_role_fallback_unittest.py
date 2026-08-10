"""
Tests for system-role message fallback when providers reject role: "system".

R2: When a provider (e.g., Minimax) rejects system-role messages, Lad must:
- Detect the error on first call
- Cache the fallback mode for the model
- Convert all system-role messages to user-role messages for that model
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.openrouter_client import OpenRouterClientError
from lad_mcp_server.review_service import ReviewService


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="minimax/minimax-m2.7",
        openrouter_secondary_reviewer_model="0",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        openrouter_reviewer_timeout_seconds=5,
        openrouter_tool_call_timeout_seconds=10,
        openrouter_max_concurrent_requests=2,
        openrouter_fixed_output_tokens=1000,
        openrouter_context_overhead_tokens=2000,
        openrouter_model_metadata_ttl_seconds=3600,
        openrouter_max_input_chars=10000,
        openrouter_include_reasoning=False,
        lad_serena_max_tool_calls=5,
        lad_serena_tool_timeout_seconds=1,
        lad_serena_max_tool_result_chars=12000,
        lad_serena_max_total_chars=50000,
        lad_serena_max_dir_entries=100,
        lad_serena_max_search_results=20,
        zai_coding_plan_key=None,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class _ModelsStub:
    def __init__(self, models: dict[str, ModelMetadata]):
        self._models = models

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._models[model_id]


SYSTEM_ROLE_ERROR = (
    "OpenRouter request failed: Error code: 400 - "
    "{'error': {'message': 'Provider returned error', 'code': 400, "
    "'metadata': {'raw': '{\"type\":\"error\",\"error\":{"
    "\"type\":\"bad_request_error\","
    "\"message\":\"invalid params, chat content has invalid message role: system (2013)\"}}}', "
    "'provider_name': 'Minimax'}}}"
)


class _SerenaCtx:
    def __init__(self) -> None:
        self.activated_project: str | None = None
        self.used_tools: set[str] = set()
        self.used_memories: set[str] = set()
        self.used_paths: set[str] = set()

    def tool_schemas(self):
        return [
            {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}},
        ]

    def call_tool(self, name: str, arguments_json: str) -> str:
        if name == "activate_project":
            self.activated_project = "."
        return json.dumps({"entries": []})


class TestSystemRoleFallback(unittest.TestCase):
    def test_system_role_error_is_detected_as_retryable(self) -> None:
        """_is_retryable_system_role_error should detect Minimax-style system-role rejection."""
        service = ReviewService(
            repo_root=Path(tempfile.mkdtemp()),
            settings=_make_settings(),
            openrouter_client=_NoOpClient(),
            models_client=_ModelsStub({
                "minimax/minimax-m2.7": ModelMetadata(
                    model_id="minimax/minimax-m2.7",
                    context_length=50000,
                    supported_parameters=("tools",),
                    provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                ),
            }),
        )
        exc = OpenRouterClientError(SYSTEM_ROLE_ERROR)
        self.assertTrue(service._is_retryable_system_role_error(exc))

    def test_non_system_role_error_is_not_retryable(self) -> None:
        service = ReviewService(
            repo_root=Path(tempfile.mkdtemp()),
            settings=_make_settings(),
            openrouter_client=_NoOpClient(),
            models_client=_ModelsStub({
                "minimax/minimax-m2.7": ModelMetadata(
                    model_id="minimax/minimax-m2.7",
                    context_length=50000,
                    supported_parameters=("tools",),
                    provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                ),
            }),
        )
        exc = OpenRouterClientError("OpenRouter request failed: Error code: 401 - unauthorized")
        self.assertFalse(service._is_retryable_system_role_error(exc))

    def test_fallback_converts_system_to_user_in_messages(self) -> None:
        """When system-role fallback is active, _adapt_messages_for_model converts system to user."""
        service = ReviewService(
            repo_root=Path(tempfile.mkdtemp()),
            settings=_make_settings(),
            openrouter_client=_NoOpClient(),
            models_client=_ModelsStub({
                "minimax/minimax-m2.7": ModelMetadata(
                    model_id="minimax/minimax-m2.7",
                    context_length=50000,
                    supported_parameters=("tools",),
                    provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                ),
            }),
        )
        model = "minimax/minimax-m2.7"
        service._remember_system_role_fallback(model)

        messages = [
            {"role": "system", "content": "You are a reviewer."},
            {"role": "user", "content": "Review this."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "name": "list_dir", "content": "{}"},
            {"role": "system", "content": "Preflight check warning."},
        ]

        adapted = service._adapt_messages_for_model(model, messages)

        self.assertEqual(adapted[0]["role"], "user")
        self.assertEqual(adapted[0]["content"], "You are a reviewer.")
        self.assertEqual(adapted[1]["role"], "user")
        self.assertEqual(adapted[2]["role"], "assistant")
        self.assertEqual(adapted[3]["role"], "tool")
        self.assertEqual(adapted[4]["role"], "user")
        self.assertEqual(adapted[4]["content"], "Preflight check warning.")

    def test_fallback_not_active_passes_messages_through(self) -> None:
        """When no fallback is cached, messages pass through unchanged."""
        service = ReviewService(
            repo_root=Path(tempfile.mkdtemp()),
            settings=_make_settings(),
            openrouter_client=_NoOpClient(),
            models_client=_ModelsStub({
                "minimax/minimax-m2.7": ModelMetadata(
                    model_id="minimax/minimax-m2.7",
                    context_length=50000,
                    supported_parameters=("tools",),
                    provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                ),
            }),
        )
        messages = [
            {"role": "system", "content": "You are a reviewer."},
            {"role": "user", "content": "Review this."},
        ]
        adapted = service._adapt_messages_for_model("minimax/minimax-m2.7", messages)
        self.assertEqual(adapted[0]["role"], "system")
        self.assertEqual(adapted[1]["role"], "user")

    def test_tool_loop_retries_on_system_role_error_and_caches_fallback(self) -> None:
        """
        Integration test: the tool loop should retry with user-role messages
        after hitting a system-role error, and cache the fallback.
        """
        model = "minimax/minimax-m2.7"
        models = _ModelsStub({
            model: ModelMetadata(
                model_id=model,
                context_length=50000,
                supported_parameters=("tools",),
                provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
            ),
        })
        settings = _make_settings(openrouter_primary_reviewer_model=model)

        class _SystemRoleErrorThenSuccessClient:
            def __init__(self) -> None:
                self.calls = 0
                self.all_messages_had_no_system = True

            async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens, tools=None, tool_choice=None, extra_body=None):
                self.calls += 1
                if self.calls == 1:
                    raise OpenRouterClientError(SYSTEM_ROLE_ERROR)

                # Verify all messages use user/assistant/tool (no system)
                for msg in messages:
                    if msg["role"] == "system":
                        self.all_messages_had_no_system = False

                # Return final content
                return type("R", (), {"content": "## Summary\nOK", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _SystemRoleErrorThenSuccessClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            serena = _SerenaCtx()

            out = asyncio.run(
                service._tool_loop(
                    model=model,
                    messages=[{"role": "system", "content": "review"}, {"role": "user", "content": "code"}],
                    tools=serena.tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=serena,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=100,
                    max_tool_calls=2,
                    tool_timeout_seconds=1,
                )
            )
            self.assertIn("OK", out)
            self.assertTrue(client.all_messages_had_no_system)
            self.assertTrue(service._is_system_role_fallback_active(model))

    def test_cached_fallback_is_used_on_subsequent_calls(self) -> None:
        """Once cached, the fallback should be active immediately on next call."""
        model = "minimax/minimax-m2.7"
        models = _ModelsStub({
            model: ModelMetadata(
                model_id=model,
                context_length=50000,
                supported_parameters=("tools",),
                provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
            ),
        })
        settings = _make_settings(openrouter_primary_reviewer_model=model)

        class _VerifyNoSystemClient:
            def __init__(self) -> None:
                self.calls = 0
                self.all_had_no_system = True

            async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens, tools=None, tool_choice=None, extra_body=None):
                self.calls += 1
                for msg in messages:
                    if msg["role"] == "system":
                        self.all_had_no_system = False
                return type("R", (), {"content": "## Summary\nDone", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _VerifyNoSystemClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)

            # Pre-cache the fallback
            service._remember_system_role_fallback(model)

            out = asyncio.run(
                service._tool_loop(
                    model=model,
                    messages=[{"role": "system", "content": "review"}, {"role": "user", "content": "code"}],
                    tools=None,
                    tool_choice_supported=False,
                    serena_ctx=None,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=100,
                    max_tool_calls=2,
                    tool_timeout_seconds=1,
                )
            )
            self.assertIn("Done", out)
            self.assertTrue(client.all_had_no_system)


class _NoOpClient:
    async def chat_completion(self, **kwargs):
        return type("R", (), {"content": "", "tool_calls": [], "raw": {}})()


if __name__ == "__main__":
    unittest.main()
