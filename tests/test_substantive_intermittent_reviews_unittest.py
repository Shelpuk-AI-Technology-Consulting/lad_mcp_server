from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.review_service import (
    ReviewService,
    _is_substantive_review_content,
    _build_tool_trace_summary,
)
from lad_mcp_server.token_budget import TokenBudget


def _build_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="test/model",
        openrouter_secondary_reviewer_model="0",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        openrouter_reviewer_timeout_seconds=300,
        openrouter_tool_call_timeout_seconds=360,
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
        intermittent_review_calls=2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# R1: Proportional intermittent timeout
# ---------------------------------------------------------------------------


class TestProportionalIntermittentTimeout(unittest.TestCase):
    def test_default_reviewer_timeout_gives_37(self) -> None:
        settings = _build_settings(openrouter_reviewer_timeout_seconds=300)
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            self.assertEqual(service._intermittent_timeout, 37)  # min(45, max(20, 300//8))

    def test_short_reviewer_timeout_gives_floor_20(self) -> None:
        settings = _build_settings(openrouter_reviewer_timeout_seconds=60)
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            self.assertEqual(service._intermittent_timeout, 20)  # max(20, 60//8=7) = 20

    def test_long_reviewer_timeout_gives_cap_45(self) -> None:
        settings = _build_settings(openrouter_reviewer_timeout_seconds=600)
        with tempfile.TemporaryDirectory() as td:
            service = ReviewService(
                repo_root=Path(td),
                settings=settings,
                openrouter_client=mock.Mock(),
                models_client=mock.Mock(),
            )
            self.assertEqual(service._intermittent_timeout, 45)  # min(45, max(20, 600//8=75))


# ---------------------------------------------------------------------------
# R2: Content quality validation
# ---------------------------------------------------------------------------


class TestIsSubstantiveReviewContent(unittest.TestCase):
    def test_real_review_is_substantive(self) -> None:
        content = (
            "## Summary\n"
            "The code has a potential null pointer issue in the handler.\n\n"
            "## Key Findings\n"
            "- Missing null check on line 42\n"
        )
        self.assertTrue(_is_substantive_review_content(content))

    def test_placeholder_only_is_not_substantive(self) -> None:
        content = (
            "## Summary\n"
            "*(No Summary provided by reviewer)*\n\n"
            "## Key Findings\n"
            "*(No Key Findings provided by reviewer)*\n\n"
            "## Recommendations\n"
            "*(No Recommendations provided by reviewer)*\n\n"
            "## Questions / Unknowns\n"
            "*(No Questions / Unknowns provided by reviewer)*\n"
        )
        self.assertFalse(_is_substantive_review_content(content))

    def test_mixed_placeholder_and_real_is_substantive(self) -> None:
        content = (
            "## Summary\n"
            "*(No Summary provided by reviewer)*\n\n"
            "## Key Findings\n"
            "- The authentication module lacks CSRF protection on POST endpoints.\n"
        )
        self.assertTrue(_is_substantive_review_content(content))

    def test_empty_string_is_not_substantive(self) -> None:
        self.assertFalse(_is_substantive_review_content(""))

    def test_whitespace_only_is_not_substantive(self) -> None:
        self.assertFalse(_is_substantive_review_content("   \n  \n  "))

    def test_short_but_real_prose_is_substantive(self) -> None:
        content = "## Summary\nLooks good to me.\n"
        self.assertTrue(_is_substantive_review_content(content))

    def test_bare_section_headers_only_not_substantive(self) -> None:
        content = "## Summary\n\n## Key Findings\n\n## Recommendations\n"
        self.assertFalse(_is_substantive_review_content(content))


# ---------------------------------------------------------------------------
# R3: Tool-trace fallback summary
# ---------------------------------------------------------------------------


class TestBuildToolTraceSummary(unittest.TestCase):
    def test_generates_structured_summary(self) -> None:
        result = _build_tool_trace_summary(
            model="test/model",
            timeout_seconds=240,
            tool_calls_made=13,
            tools_invoked={"read_file", "search_for_pattern"},
            memories_used={"project_overview.md"},
            paths_used={"src/main.py", "src/utils.py"},
        )
        self.assertIn("test/model", result)
        self.assertIn("240", result)
        self.assertIn("13", result)
        self.assertIn("read_file", result)
        self.assertIn("src/main.py", result)
        self.assertIn("project_overview.md", result)
        self.assertIn("tool-exploration trace", result)

    def test_handles_empty_collections(self) -> None:
        result = _build_tool_trace_summary(
            model="test/model",
            timeout_seconds=120,
            tool_calls_made=0,
            tools_invoked=set(),
            memories_used=set(),
            paths_used=set(),
        )
        self.assertIn("test/model", result)
        self.assertIn("0", result)
        self.assertIn("none", result.lower())

    def test_result_is_valid_markdown(self) -> None:
        result = _build_tool_trace_summary(
            model="test/model",
            timeout_seconds=240,
            tool_calls_made=5,
            tools_invoked={"read_file"},
            memories_used=set(),
            paths_used={"src/main.py"},
        )
        self.assertTrue(result.startswith("## Summary"))
        self.assertIn("## Files Explored", result)
        self.assertIn("## Tools Used", result)


# ---------------------------------------------------------------------------
# R4: Timeout branch returns tool-trace when no snapshot
# ---------------------------------------------------------------------------


class TestTimeoutBranchToolTraceFallback(unittest.TestCase):
    def test_timeout_returns_tool_trace_when_no_snapshot(self) -> None:
        """When reviewer times out and intermittent snapshot is empty,
        the timeout branch should return a tool-trace summary instead of
        the generic error stub."""
        settings = _build_settings(
            openrouter_reviewer_timeout_seconds=1,
            openrouter_tool_call_timeout_seconds=2,
            intermittent_review_calls=100,  # effectively disable side-calls for this test
        )
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / ".serena" / "memories").mkdir()

            # Mock client that always times out
            slow_client = mock.Mock()
            async def _timeout_completion(**kwargs):
                await asyncio.sleep(10)  # exceed the 1s reviewer timeout
            slow_client.chat_completion = _timeout_completion

            service = ReviewService(
                repo_root=repo,
                settings=settings,
                openrouter_client=slow_client,
                models_client=mock.Mock(),
            )
            serena_ctx = mock.Mock()
            serena_ctx.activated_project = "."
            serena_ctx.used_tools = {"read_file", "search_for_pattern"}
            serena_ctx.used_memories = {"project_overview.md"}
            serena_ctx.used_paths = {"src/main.py"}
            serena_ctx.tool_schemas.return_value = [
                {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
            ]

            async def scenario():
                return await service._run_single_reviewer(
                    cfg=mock.Mock(
                        model="test/model",
                        budget=TokenBudget(
                            effective_context_length=50000,
                            effective_output_budget=1000,
                            overhead_tokens=2000,
                        ),
                        supported_parameters=("tools", "tool_choice", "max_tokens"),
                        tool_calling_supported=True,
                        tool_choice_supported=False,
                        serena_ctx=serena_ctx,
                        serena_disabled_reason=None,
                        use_zai_direct=False,
                        direct_model_name=None,
                        use_kimi_direct=False,
                        direct_kimi_model_name=None,
                        use_deepseek_direct=False,
                        direct_deepseek_model_name=None,
                    ),
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "You are a reviewer.",
                    build_user_prompt=lambda tool_calling_enabled, redacted: "Review this code.",
                    redacted_inputs={"code": "def foo(): pass"},
                    requested_paths=None,
                    file_context_builder=mock.Mock(),
                )

            outcome = asyncio.run(scenario())
            # Should NOT be the generic error stub
            self.assertTrue(outcome.ok)
            self.assertTrue(outcome.is_intermittent)
            self.assertIn("tool-exploration trace", outcome.markdown)
            self.assertIsNotNone(outcome.provider_note)


if __name__ == "__main__":
    unittest.main()
