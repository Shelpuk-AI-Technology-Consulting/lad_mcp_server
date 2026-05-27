import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.review_service import ReviewService


class _ModelsStub:
    def __init__(self, models: dict[str, ModelMetadata]):
        self._models = models

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._models[model_id]


class TestSettingsTimeoutDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self._env_snapshot = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_timeout_defaults_are_derived(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test"
        os.environ.pop("OPENROUTER_REVIEWER_TIMEOUT_SECONDS", None)
        os.environ.pop("OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS", None)
        os.environ.pop("LAD_SERENA_MAX_TOTAL_CHARS", None)

        s = Settings.from_env()
        self.assertEqual(s.openrouter_reviewer_timeout_seconds, 300)
        self.assertEqual(s.openrouter_tool_call_timeout_seconds, 360)
        self.assertEqual(s.lad_serena_max_total_chars, 100000)

    def test_tool_call_timeout_cannot_be_smaller_than_reviewer_timeout(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test"
        os.environ["OPENROUTER_REVIEWER_TIMEOUT_SECONDS"] = "300"
        os.environ["OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS"] = "240"

        with self.assertRaises(ValueError) as ctx:
            Settings.from_env()
        self.assertIn("OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS", str(ctx.exception))
        self.assertIn("OPENROUTER_REVIEWER_TIMEOUT_SECONDS", str(ctx.exception))

    def test_derived_tool_call_timeout_tracks_reviewer_timeout(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test"
        os.environ["OPENROUTER_REVIEWER_TIMEOUT_SECONDS"] = "500"
        os.environ.pop("OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS", None)

        s = Settings.from_env()
        self.assertEqual(s.openrouter_tool_call_timeout_seconds, 560)

    def test_reviewer_model_defaults_match_documented_pair(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test"
        os.environ.pop("OPENROUTER_PRIMARY_REVIEWER_MODEL", None)
        os.environ.pop("OPENROUTER_SECONDARY_REVIEWER_MODEL", None)

        s = Settings.from_env()
        self.assertEqual(s.openrouter_primary_reviewer_model, "moonshotai/kimi-k2.5")
        self.assertEqual(s.openrouter_secondary_reviewer_model, "minimax/minimax-m2.7")


class TestTimeoutMessages(unittest.TestCase):
    def test_reviewer_timeout_is_actionable_without_exploration_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            primary = "moonshotai/kimi-k2-thinking"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=(),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )

            class _SlowClient:
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
                    await asyncio.sleep(timeout_seconds + 1)
                    return type("R", (), {"content": "never", "tool_calls": [], "raw": {}})()

            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
                openrouter_secondary_reviewer_model="0",
                openrouter_http_referer=None,
                openrouter_x_title=None,
                openrouter_reviewer_timeout_seconds=1,
                openrouter_tool_call_timeout_seconds=10,
                openrouter_max_concurrent_requests=1,
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
                openrouter_client=_SlowClient(),
                models_client=models,
            )

            out = asyncio.run(service.code_review(code="x", context=None, paths=None))
            self.assertIn("Reviewer timed out after 1s", out)

    def test_reviewer_timeout_with_serena_data_includes_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena" / "memories").mkdir(parents=True)
            (repo / ".serena" / "memories" / "project_overview.md").write_text("overview", encoding="utf-8")
            (repo / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")

            primary = "moonshotai/kimi-k2-thinking"
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

            class _SlowAfterExplorationClient:
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
                    forced_name = None
                    if isinstance(tool_choice, dict):
                        forced_name = (tool_choice.get("function") or {}).get("name")
                    if forced_name == "activate_project":
                        return type("R", (), {"content": None, "tool_calls": [{"id": "a1", "type": "function", "function": {"name": "activate_project", "arguments": "{\"project\": \".\"}"}}], "raw": {}})()
                    if forced_name == "read_project_overview":
                        return type("R", (), {"content": None, "tool_calls": [{"id": "a2", "type": "function", "function": {"name": "read_project_overview", "arguments": "{}"}}], "raw": {}})()
                    if forced_name == "read_baseline_memories":
                        return type("R", (), {"content": None, "tool_calls": [{"id": "a3", "type": "function", "function": {"name": "read_baseline_memories", "arguments": "{}"}}], "raw": {}})()
                    await asyncio.sleep(timeout_seconds + 1)
                    return type("R", (), {"content": "never", "tool_calls": [], "raw": {}})()

            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
                openrouter_secondary_reviewer_model="0",
                openrouter_http_referer=None,
                openrouter_x_title=None,
                openrouter_reviewer_timeout_seconds=1,
                openrouter_tool_call_timeout_seconds=10,
                openrouter_max_concurrent_requests=1,
                openrouter_fixed_output_tokens=1000,
                openrouter_context_overhead_tokens=2000,
                openrouter_model_metadata_ttl_seconds=3600,
                openrouter_max_input_chars=10000,
                openrouter_include_reasoning=False,
                lad_serena_max_tool_calls=8,
                lad_serena_tool_timeout_seconds=1,
                lad_serena_max_tool_result_chars=12000,
                lad_serena_max_total_chars=50000,
                lad_serena_max_dir_entries=100,
                lad_serena_max_search_results=20,
            )

            service = ReviewService(
                repo_root=repo,
                settings=settings,
                openrouter_client=_SlowAfterExplorationClient(),
                models_client=models,
            )

            out = asyncio.run(service.code_review(code="x", context=None, paths=None))
            self.assertIn("## Partial Review Context", out)
            self.assertIn("Reviewer timed out after 1s", out)
            self.assertIn("Exploration completed before timeout", out)
            self.assertIn("read_baseline_memories", out)
            self.assertIn("project_overview.md", out)
            self.assertIn("## Key Findings", out)
            self.assertIn("## Recommendations", out)
            self.assertIn("## Questions / Unknowns", out)
            self.assertIn("This is not a complete review", out)

    def test_empty_exception_message_includes_type_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            primary = "moonshotai/kimi-k2-thinking"
            models = _ModelsStub(
                {
                    primary: ModelMetadata(
                        model_id=primary,
                        context_length=50000,
                        supported_parameters=(),
                        provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                    ),
                }
            )

            class _FailingClient:
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
                    raise Exception()

            settings = Settings(
                openrouter_api_key="test",
                openrouter_primary_reviewer_model=primary,
                openrouter_secondary_reviewer_model="0",
                openrouter_http_referer=None,
                openrouter_x_title=None,
                openrouter_reviewer_timeout_seconds=5,
                openrouter_tool_call_timeout_seconds=10,
                openrouter_max_concurrent_requests=1,
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
                openrouter_client=_FailingClient(),
                models_client=models,
            )

            out = asyncio.run(service.code_review(code="x", context=None, paths=None))
            self.assertIn("Exception", out)


if __name__ == "__main__":
    unittest.main()
