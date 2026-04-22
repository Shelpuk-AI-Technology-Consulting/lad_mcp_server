import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.openrouter_client import OpenRouterClientError
from lad_mcp_server.prompts import force_finalize_system_message
from lad_mcp_server.review_service import ReviewService


class _ModelsStub:
    def __init__(self, models: dict[str, ModelMetadata]):
        self._models = models

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._models[model_id]


class _OpenRouterClientStub:
    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._lock_loop = None
        self._calls: dict[str, int] = {}

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens, tools=None, tool_choice=None, extra_body=None):
        # Minimal tool-loop simulator:
        # - honor forced tool_choice preflight (activate_project, read_project_overview, read_baseline_memories)
        # - then return a final content response.
        async with self._get_lock():
            idx = self._calls.get(model, 0)
            self._calls[model] = idx + 1

        forced_name = None
        if isinstance(tool_choice, dict):
            forced_name = (tool_choice.get("function") or {}).get("name")

        if tools and forced_name == "activate_project":
            return type(
                "R",
                (),
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{idx}",
                            "type": "function",
                            "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"},
                        }
                    ],
                    "raw": {},
                },
            )()

        if tools and forced_name == "read_project_overview":
            return type(
                "R",
                (),
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{idx}",
                            "type": "function",
                            "function": {"name": "read_project_overview", "arguments": "{}"},
                        }
                    ],
                    "raw": {},
                },
            )()

        if tools and forced_name == "read_baseline_memories":
            return type(
                "R",
                (),
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{idx}",
                            "type": "function",
                            "function": {"name": "read_baseline_memories", "arguments": "{}"},
                        }
                    ],
                    "raw": {},
                },
            )()

        return type("R", (), {"content": "## Summary\nOK", "tool_calls": [], "raw": {}})()


class TestReviewServiceToolLoop(unittest.TestCase):
    def test_both_reviewers_use_serena_tools_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena" / "memories").mkdir(parents=True)
            (repo / ".serena" / "memories" / "project_overview.md").write_text("First line\nSecond line\n", encoding="utf-8")

            primary = "moonshotai/kimi-k2-thinking"
            secondary = "z-ai/glm-4.7"

            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice", "max_tokens"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                    secondary: ModelMetadata(
                        model_id=secondary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice", "max_tokens"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )

            # Create settings without relying on env.
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
                openrouter_secondary_reviewer_model=secondary,
                openrouter_http_referer=None,
                openrouter_x_title=None,
                openrouter_reviewer_timeout_seconds=5,
                openrouter_tool_call_timeout_seconds=10,
                openrouter_max_concurrent_requests=4,
                openrouter_fixed_output_tokens=1000,
                openrouter_context_overhead_tokens=2000,
                openrouter_model_metadata_ttl_seconds=3600,
                openrouter_max_input_chars=10000,
                openrouter_include_reasoning=False,
                lad_serena_max_tool_calls=8,
                lad_serena_tool_timeout_seconds=5,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )

            service = ReviewService(
                repo_root=repo,
                settings=settings,
                openrouter_client=_OpenRouterClientStub(),
                models_client=models,
            )

            out = asyncio.run(
                service.system_design_review(
                    proposal="This is a valid proposal with enough length.",
                    constraints=None,
                    context=None,
                )
            )
            # Disclosure marker for both sections
            self.assertIn("## Primary Reviewer", out)
            self.assertIn("## Secondary Reviewer", out)
            self.assertIn("Serena tools used: yes", out)

    def test_tool_call_timeout_is_reported(self) -> None:
        class _SlowSerenaContext:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return []

            def call_tool(self, name: str, arguments_json: str) -> str:
                import time

                time.sleep(0.2)
                return "{}"

        class _ToolCallOnceClient:
            async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens, tools=None, tool_choice=None, extra_body=None):
                # one tool call then stop
                if tools is not None and not any(m.get("role") == "tool" for m in messages):
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}],
                            "raw": {},
                        },
                    )()
                tool_msgs = [m for m in messages if m.get("role") == "tool"]
                echoed = tool_msgs[-1]["content"] if tool_msgs else ""
                return type("R", (), {"content": echoed, "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            primary = "moonshotai/kimi-k2-thinking"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
                openrouter_secondary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=2,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            service = ReviewService(
                repo_root=repo,
                settings=settings,
                openrouter_client=_ToolCallOnceClient(),
                models_client=models,
            )
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=[{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}}}],
                    tool_choice_supported=False,
                    serena_ctx=_SlowSerenaContext(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=2,
                    tool_timeout_seconds=0.01,
                )
            )
            self.assertIn("timed out", out)

    def test_tool_choice_fallback_retries_and_is_remembered(self) -> None:
        class _SerenaCtx:
            activated_project = None
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return "{}"

        class _FallbackClient:
            def __init__(self) -> None:
                self.tool_choices: list[object] = []

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.tool_choices.append(tool_choice)
                if isinstance(tool_choice, dict):
                    raise OpenRouterClientError(
                        "OpenRouter request failed: Error code: 400 - {'error': {'message': 'Tool choice must be auto', 'code': 400}}"
                    )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "z-ai/glm-5"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=2,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _FallbackClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            serena_ctx = _SerenaCtx()
            tools = serena_ctx.tool_schemas()
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]

            out1 = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=list(messages),
                    tools=tools,
                    tool_choice_supported=True,
                    serena_ctx=serena_ctx,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=2,
                    tool_timeout_seconds=1,
                )
            )
            out2 = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=list(messages),
                    tools=tools,
                    tool_choice_supported=True,
                    serena_ctx=serena_ctx,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=2,
                    tool_timeout_seconds=1,
                )
            )

            self.assertEqual(out1, "ok")
            self.assertEqual(out2, "ok")
            self.assertEqual(len(client.tool_choices), 3)
            self.assertIsInstance(client.tool_choices[0], dict)
            self.assertEqual(client.tool_choices[1], "auto")
            self.assertEqual(client.tool_choices[2], "auto")

    def test_tool_choice_fallback_cache_expiry_restores_forced_preflight(self) -> None:
        class _SerenaCtx:
            activated_project = None
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return "{}"

        class _FallbackClient:
            def __init__(self) -> None:
                self.tool_choices: list[object] = []

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.tool_choices.append(tool_choice)
                if isinstance(tool_choice, dict):
                    raise OpenRouterClientError(
                        "OpenRouter request failed: Error code: 400 - {'error': {'message': 'Tool choice must be auto', 'code': 400}}"
                    )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "z-ai/glm-5"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=2,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _FallbackClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            serena_ctx = _SerenaCtx()
            tools = serena_ctx.tool_schemas()
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]

            out1 = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=list(messages),
                    tools=tools,
                    tool_choice_supported=True,
                    serena_ctx=serena_ctx,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=2,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out1, "ok")
            service._tool_choice_fallback_until_by_model[primary] = 0.0
            out2 = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=list(messages),
                    tools=tools,
                    tool_choice_supported=True,
                    serena_ctx=serena_ctx,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=2,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out2, "ok")
            self.assertEqual(len(client.tool_choices), 4)
            self.assertIsInstance(client.tool_choices[0], dict)
            self.assertEqual(client.tool_choices[1], "auto")
            self.assertIsInstance(client.tool_choices[2], dict)
            self.assertEqual(client.tool_choices[3], "auto")

    def test_unrelated_openrouter_error_does_not_trigger_fallback_retry(self) -> None:
        class _SerenaCtx:
            activated_project = None
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return "{}"

        class _FailingClient:
            def __init__(self) -> None:
                self.tool_choices: list[object] = []

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.tool_choices.append(tool_choice)
                raise OpenRouterClientError("OpenRouter request failed: Error code: 401 - unauthorized")

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "z-ai/glm-5"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=2,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _FailingClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            serena_ctx = _SerenaCtx()
            tools = serena_ctx.tool_schemas()
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]

            with self.assertRaises(OpenRouterClientError):
                asyncio.run(
                    service._tool_loop(
                        model=primary,
                        messages=list(messages),
                        tools=tools,
                        tool_choice_supported=True,
                        serena_ctx=serena_ctx,
                        extra_body=None,
                        reviewer_timeout_seconds=5,
                        max_output_tokens=10,
                        max_tool_calls=2,
                        tool_timeout_seconds=1,
                    )
                )
            self.assertEqual(len(client.tool_choices), 1)
            self.assertIsInstance(client.tool_choices[0], dict)

    def test_tool_call_protocol_includes_assistant_tool_calls_message(self) -> None:
        class _SerenaCtx:
            def __init__(self) -> None:
                self.activated_project: str | None = None
                self.used_tools: set[str] = set()
                self.used_memories: set[str] = set()
                self.used_paths: set[str] = set()

            def tool_schemas(self):
                return [
                    {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "read_project_overview", "parameters": {"type": "object"}}},
                ]

            def call_tool(self, name: str, arguments_json: str) -> str:
                if name == "activate_project":
                    self.activated_project = "."
                return "{}"

        class _StrictProtocolClient:
            def __init__(self) -> None:
                self.calls = 0
                self.expected_tool_call_id = "call_function_abc_1"

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": self.expected_tool_call_id,
                                    "type": "function",
                                    "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()

                saw_assistant_call = False
                saw_tool_result = False
                for msg in messages:
                    if msg.get("role") == "assistant":
                        tool_calls = msg.get("tool_calls") or []
                        for tc in tool_calls:
                            if tc.get("id") == self.expected_tool_call_id:
                                saw_assistant_call = True
                    if msg.get("role") == "tool" and msg.get("tool_call_id") == self.expected_tool_call_id:
                        saw_tool_result = True

                if not (saw_assistant_call and saw_tool_result):
                    raise OpenRouterClientError(
                        "OpenRouter request failed: Error code: 400 - {'error': {'message': 'Provider returned error', 'metadata': {'raw': '{\"error\":{\"message\":\"invalid params, tool result\\'s tool id(call_function_abc_1) not found (2013)\"}}'}}}"
                    )

                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=3,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _StrictProtocolClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            serena_ctx = _SerenaCtx()
            tools = serena_ctx.tool_schemas()

            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=tools,
                    tool_choice_supported=True,
                    serena_ctx=serena_ctx,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=3,
                    tool_timeout_seconds=1,
                )
            )

            self.assertEqual(out, "ok")

    def test_tool_call_protocol_limits_assistant_calls_to_executed_subset(self) -> None:
        class _SerenaCtx:
            def __init__(self) -> None:
                self.activated_project = "."
                self.used_tools: set[str] = set()
                self.used_memories: set[str] = set()
                self.used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return "{}"

        class _SubsetStrictClient:
            def __init__(self) -> None:
                self.calls = 0
                self.expected_id = "call_a"
                self.unexpected_id = "call_b"

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": self.expected_id,
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                },
                                {
                                    "id": self.unexpected_id,
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                },
                            ],
                            "raw": {},
                        },
                    )()

                latest_assistant_tool_ids: list[str] = []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        latest_assistant_tool_ids = [tc.get("id") for tc in (msg.get("tool_calls") or [])]
                        break
                tool_result_ids = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]

                if latest_assistant_tool_ids != [self.expected_id] or tool_result_ids != [self.expected_id]:
                    raise OpenRouterClientError(
                        "OpenRouter request failed: Error code: 400 - {'error': {'message': 'Provider returned error', 'metadata': {'raw': '{\"error\":{\"message\":\"invalid params, tool call and result not match (2013)\"}}'}}}"
                    )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=1,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _SubsetStrictClient()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            serena_ctx = _SerenaCtx()
            tools = serena_ctx.tool_schemas()

            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=tools,
                    tool_choice_supported=False,
                    serena_ctx=serena_ctx,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=1,
                    tool_timeout_seconds=1,
                )
            )

            self.assertEqual(out, "ok")

    def test_empty_tool_output_adds_system_hint_and_summary(self) -> None:
        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return ""

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_hint = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()

                self.saw_hint = any(
                    msg.get("role") == "system" and "Tool budget/state failed" in str(msg.get("content", ""))
                    for msg in messages
                )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=3,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtx(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=3,
                    tool_timeout_seconds=1,
                )
            )
            self.assertTrue(client.saw_hint)
            self.assertIn("## Tooling Degradation Summary", out)

    def test_consecutive_degraded_tool_outputs_disable_tools_and_force_finalize(self) -> None:
        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return ""

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.third_call_tools_none = False
                self.saw_force_finalize = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls in (1, 2):
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{self.calls}",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()

                self.third_call_tools_none = tools is None
                self.saw_force_finalize = any(
                    msg.get("role") == "system" and msg.get("content") == force_finalize_system_message()
                    for msg in messages
                )
                return type("R", (), {"content": "done", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtx(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=5,
                    tool_timeout_seconds=1,
                )
            )
            self.assertTrue(client.third_call_tools_none)
            self.assertTrue(client.saw_force_finalize)
            self.assertIn("## Tooling Degradation Summary", out)
            self.assertIn("Consecutive guard triggered: yes", out)

    def test_invalid_json_tool_output_counted_as_degradation(self) -> None:
        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return "not-json"

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_hint = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()

                self.saw_hint = any(
                    msg.get("role") == "system" and "Tool budget/state failed" in str(msg.get("content", ""))
                    for msg in messages
                )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=3,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtx(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=3,
                    tool_timeout_seconds=1,
                )
            )
            self.assertTrue(client.saw_hint)
            self.assertIn("## Tooling Degradation Summary", out)

    def test_preflight_forces_read_baseline_memories_after_overview(self) -> None:
        class _SerenaCtx:
            activated_project = None
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [
                    {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "read_project_overview", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "read_baseline_memories", "parameters": {"type": "object"}}},
                ]

            def call_tool(self, name: str, arguments_json: str) -> str:
                if name == "activate_project":
                    self.activated_project = "."
                if name == "read_baseline_memories":
                    result = {
                        "required": ["project_overview.md", "research_summary.md"],
                        "present": ["project_overview.md"],
                        "loaded": [{"name": "project_overview.md", "content": "overview"}],
                        "missing": ["research_summary.md"],
                    }
                    return json.dumps(
                        {
                            "tool_status": "ok",
                            "tool_name": "read_baseline_memories",
                            "tool_budget": {
                                "max_total_chars": 10000,
                                "used_chars": 0,
                                "remaining_chars": 10000,
                                "emitted_chars_this_call": 1,
                            },
                            "tool_result_json": json.dumps(result),
                        }
                    )
                return "{}"

        class _Client:
            def __init__(self) -> None:
                self.tool_choice_names: list[str] = []
                self.calls = 0

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                forced_name = None
                if isinstance(tool_choice, dict):
                    forced_name = (tool_choice.get("function") or {}).get("name")
                    if forced_name:
                        self.tool_choice_names.append(forced_name)

                if forced_name == "activate_project":
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"}}],
                            "raw": {},
                        },
                    )()
                if forced_name == "read_project_overview":
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "read_project_overview", "arguments": "{}"}}],
                            "raw": {},
                        },
                    )()
                if forced_name == "read_baseline_memories":
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [{"id": "t3", "type": "function", "function": {"name": "read_baseline_memories", "arguments": "{}"}}],
                            "raw": {},
                        },
                    )()
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=6,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            serena = _SerenaCtx()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=serena.tool_schemas(),
                    tool_choice_supported=True,
                    serena_ctx=serena,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=6,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out, "ok")
            self.assertEqual(client.tool_choice_names[:3], ["activate_project", "read_project_overview", "read_baseline_memories"])

    def test_tool_loop_emits_post_preflight_validation_message_with_missing_memories(self) -> None:
        class _SerenaCtx:
            activated_project = None
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [
                    {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "read_project_overview", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "read_baseline_memories", "parameters": {"type": "object"}}},
                ]

            def call_tool(self, name: str, arguments_json: str) -> str:
                if name == "activate_project":
                    self.activated_project = "."
                    return "{}"
                if name == "read_project_overview":
                    return "{}"
                if name == "read_baseline_memories":
                    result = {
                        "required": ["project_overview.md", "research_summary.md"],
                        "present": ["project_overview.md"],
                        "loaded": [{"name": "project_overview.md", "content": "overview"}],
                        "missing": ["research_summary.md"],
                    }
                    return json.dumps(
                        {
                            "tool_status": "ok",
                            "tool_name": "read_baseline_memories",
                            "tool_budget": {
                                "max_total_chars": 10000,
                                "used_chars": 0,
                                "remaining_chars": 10000,
                                "emitted_chars_this_call": 1,
                            },
                            "tool_result_json": json.dumps(result),
                        }
                    )
                return "{}"

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_validation_message = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                forced_name = None
                if isinstance(tool_choice, dict):
                    forced_name = (tool_choice.get("function") or {}).get("name")
                if forced_name == "activate_project":
                    return type("R", (), {"content": None, "tool_calls": [{"id": "a1", "type": "function", "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"}}], "raw": {}})()
                if forced_name == "read_project_overview":
                    return type("R", (), {"content": None, "tool_calls": [{"id": "a2", "type": "function", "function": {"name": "read_project_overview", "arguments": "{}"}}], "raw": {}})()
                if forced_name == "read_baseline_memories":
                    return type("R", (), {"content": None, "tool_calls": [{"id": "a3", "type": "function", "function": {"name": "read_baseline_memories", "arguments": "{}"}}], "raw": {}})()

                self.saw_validation_message = any(
                    msg.get("role") == "system"
                    and "missing required preflight memories" in str(msg.get("content", "")).lower()
                    and "research_summary.md" in str(msg.get("content", ""))
                    for msg in messages
                )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools", "tool_choice"),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=6,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            serena = _SerenaCtx()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=serena.tool_schemas(),
                    tool_choice_supported=True,
                    serena_ctx=serena,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=6,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out, "ok")
            self.assertTrue(client.saw_validation_message)

    def test_tool_loop_warns_on_deeper_exploration_when_memory_checklist_skipped(self) -> None:
        class _SerenaCtx:
            activated_project = None
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [
                    {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "read_memory", "parameters": {"type": "object"}}},
                ]

            def call_tool(self, name: str, arguments_json: str) -> str:
                if name == "activate_project":
                    self.activated_project = "."
                return "{}"

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_skipped_warning = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                # No forced tool-choice support in this test; simulate model skipping memory preflight.
                if self.calls == 1:
                    return type("R", (), {"content": None, "tool_calls": [{"id": "x1", "type": "function", "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"}}], "raw": {}})()
                if self.calls == 2:
                    return type("R", (), {"content": None, "tool_calls": [{"id": "x2", "type": "function", "function": {"name": "list_dir", "arguments": "{\"path\": \".\"}"}}], "raw": {}})()

                self.saw_skipped_warning = any(
                    msg.get("role") == "system"
                    and "skipped required preflight memories" in str(msg.get("content", "")).lower()
                    for msg in messages
                )
                return type("R", (), {"content": "ok", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=6,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            serena = _SerenaCtx()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=serena.tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=serena,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=6,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out, "ok")
            self.assertTrue(client.saw_skipped_warning)

    def test_tool_loop_retries_once_when_post_tool_response_is_empty_then_recovers(self) -> None:
        class _SerenaCtx:
            def __init__(self) -> None:
                self.activated_project = None
                self.used_tools: set[str] = set()
                self.used_memories: set[str] = set()
                self.used_paths: set[str] = set()

            def tool_schemas(self):
                return [
                    {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
                ]

            def call_tool(self, name: str, arguments_json: str) -> str:
                if name == "activate_project":
                    self.activated_project = "."
                return "{}"

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_retry_hint = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "e1",
                                    "type": "function",
                                    "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                if self.calls == 2:
                    return type("R", (), {"content": "", "tool_calls": [], "raw": {}})()
                self.saw_retry_hint = any(
                    msg.get("role") == "system" and "empty response" in str(msg.get("content", "")).lower()
                    for msg in messages
                )
                return type("R", (), {"content": "final-review", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=6,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            serena = _SerenaCtx()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=serena.tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=serena,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=6,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out, "final-review")
            self.assertEqual(client.calls, 3)
            self.assertTrue(client.saw_retry_hint)

    def test_tool_loop_retries_once_only_when_post_tool_response_stays_empty(self) -> None:
        class _SerenaCtx:
            def __init__(self) -> None:
                self.activated_project = None
                self.used_tools: set[str] = set()
                self.used_memories: set[str] = set()
                self.used_paths: set[str] = set()

            def tool_schemas(self):
                return [
                    {"type": "function", "function": {"name": "activate_project", "parameters": {"type": "object"}}},
                ]

            def call_tool(self, name: str, arguments_json: str) -> str:
                if name == "activate_project":
                    self.activated_project = "."
                return "{}"

        class _Client:
            def __init__(self) -> None:
                self.calls = 0

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "e1",
                                    "type": "function",
                                    "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                return type("R", (), {"content": "", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
                lad_serena_max_tool_calls=6,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )
            client = _Client()
            serena = _SerenaCtx()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=serena.tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=serena,
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=6,
                    tool_timeout_seconds=1,
                )
            )
            self.assertEqual(out, "")
            self.assertEqual(client.calls, 3)

    def test_hallucinated_tool_call_after_budget_exhaustion_is_retried(self) -> None:
        """After tools are disabled (budget exhausted), model hallucinates a tool call.
        The loop should recover by injecting a hint and retrying once."""

        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return ""

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_hallucination_hint = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    # First call: real tool call (uses the only budget slot).
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                if self.calls == 2:
                    # Second call: tools still in list, but remaining=0 so executable_tool_calls
                    # will be empty. Then remaining<=0 sets tools=None.
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                if self.calls == 3:
                    # Third call: tools=None. Model hallucinates a tool call.
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_hallucinated",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                # Fourth call: after recovery hint, model provides final review.
                self.saw_hallucination_hint = any(
                    msg.get("role") == "system"
                    and "tools are no longer available" in str(msg.get("content", ""))
                    for msg in messages
                )
                return type("R", (), {"content": "final-review", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtx(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=1,  # budget=1 so it exhausts after the first tool call
                    tool_timeout_seconds=1,
                )
            )
            self.assertIn("final-review", out)
            self.assertNotIn("no tools were available", out)
            self.assertEqual(client.calls, 4)
            self.assertTrue(client.saw_hallucination_hint)

    def test_hallucinated_tool_call_after_degradation_guard_is_retried(self) -> None:
        """After tools are disabled (consecutive degradation guard), model hallucinates.
        The loop should recover by injecting a hint and retrying once."""

        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return ""  # empty output → degraded

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_hallucination_hint = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls in (1, 2):
                    # First two calls: real tool calls with degraded (empty) output.
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{self.calls}",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                if self.calls == 3:
                    # Third call: tools=None (degradation guard triggered), model hallucinates.
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_hallucinated",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                # Fourth call: after recovery hint, model provides final review.
                self.saw_hallucination_hint = any(
                    msg.get("role") == "system"
                    and "tools are no longer available" in str(msg.get("content", ""))
                    for msg in messages
                )
                return type("R", (), {"content": "final-review", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtx(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=5,
                    tool_timeout_seconds=1,
                )
            )
            self.assertIn("final-review", out)
            self.assertNotIn("no tools were available", out)
            self.assertEqual(client.calls, 4)
            self.assertTrue(client.saw_hallucination_hint)

    def test_hallucinated_tool_call_retry_is_one_shot(self) -> None:
        """If model hallucinates tool calls twice after tools are disabled, the second
        hallucination returns content with degradation summary (not infinite loop)."""

        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return ""

        class _Client:
            def __init__(self) -> None:
                self.calls = 0

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    # First call: real tool call (uses the only budget slot).
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                if self.calls == 2:
                    # Second call: tools still in list, but remaining=0 so executable_tool_calls
                    # will be empty. Then remaining<=0 sets tools=None.
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                # All subsequent calls (tools=None): model keeps hallucinating tool calls.
                return type(
                    "R",
                    (),
                    {
                        "content": "partial-review",
                        "tool_calls": [
                            {
                                "id": f"call_hall_{self.calls}",
                                "type": "function",
                                "function": {"name": "list_dir", "arguments": "{}"},
                            }
                        ],
                        "raw": {},
                    },
                )()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtx(),
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=1,  # budget=1 so it exhausts after first tool call
                    tool_timeout_seconds=1,
                )
            )
            # Call 3 (first hallucination with tools=None) is retried,
            # call 4 (second hallucination) causes early return with degradation summary.
            self.assertIn("partial-review", out)
            self.assertNotIn("no tools were available", out)
            self.assertEqual(client.calls, 4)

    def test_hallucinated_tool_call_when_serena_ctx_is_none(self) -> None:
        """When serena_ctx is None (Serena not activated), model hallucinates a tool call.
        The loop should recover by injecting a hint and retrying once."""

        class _SerenaCtx:
            activated_project = "."
            used_tools: set[str] = set()
            used_memories: set[str] = set()
            used_paths: set[str] = set()

            def tool_schemas(self):
                return [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}}]

            def call_tool(self, name: str, arguments_json: str) -> str:
                return ""

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.saw_hallucination_hint = False

            async def chat_completion(
                self,
                *,
                model,
                messages,
                timeout_seconds,
                max_output_tokens,
                tools=None,
                tool_choice=None,
                extra_body=None,
            ):
                self.calls += 1
                if self.calls == 1:
                    # First call: model tries a tool call despite no serena_ctx.
                    return type(
                        "R",
                        (),
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "list_dir", "arguments": "{}"},
                                }
                            ],
                            "raw": {},
                        },
                    )()
                # Second call: after recovery hint, model provides final review.
                self.saw_hallucination_hint = any(
                    msg.get("role") == "system"
                    and "tools are no longer available" in str(msg.get("content", ""))
                    for msg in messages
                )
                return type("R", (), {"content": "final-review", "tool_calls": [], "raw": {}})()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            primary = "minimax/minimax-m2.7"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=("tools",),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )
            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
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
            )
            client = _Client()
            service = ReviewService(repo_root=repo, settings=settings, openrouter_client=client, models_client=models)
            out = asyncio.run(
                service._tool_loop(
                    model=primary,
                    messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tools=_SerenaCtx().tool_schemas(),
                    tool_choice_supported=False,
                    serena_ctx=None,  # serena_ctx is None — tools were never enabled
                    extra_body=None,
                    reviewer_timeout_seconds=5,
                    max_output_tokens=10,
                    max_tool_calls=5,
                    tool_timeout_seconds=1,
                )
            )
            self.assertIn("final-review", out)
            self.assertNotIn("no tools were available", out)
            self.assertEqual(client.calls, 2)
            self.assertTrue(client.saw_hallucination_hint)


if __name__ == "__main__":
    unittest.main()
