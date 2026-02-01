import asyncio
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


class _OpenRouterClientStub:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._calls: dict[str, int] = {}

    async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens, tools=None, tool_choice=None, extra_body=None):
        # Minimal tool-loop simulator:
        # - honor forced tool_choice preflight (activate_project, read_project_overview)
        # - then return a final content response.
        async with self._lock:
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
                    model=None,
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


if __name__ == "__main__":
    unittest.main()
