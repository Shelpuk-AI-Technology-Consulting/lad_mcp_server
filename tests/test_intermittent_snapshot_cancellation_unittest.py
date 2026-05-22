from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.openrouter_client import OpenRouterCallResult
from lad_mcp_server.review_service import IntermittentReviewState, ReviewService
from lad_mcp_server.token_budget import TokenBudget


def _build_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="test/model",
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
        ollama_api_key=None,
        intermittent_review_calls=5,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestIntermittentDispatchCancellation(unittest.TestCase):
    def test_new_dispatch_does_not_cancel_in_flight_task(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                service = ReviewService(
                    repo_root=Path(td),
                    settings=_build_settings(),
                    openrouter_client=mock.Mock(),
                    models_client=mock.Mock(),
                )
                state = IntermittentReviewState()
                never_done = asyncio.create_task(asyncio.sleep(3600))
                state.in_flight_task = never_done

                service._dispatch_intermittent_review(
                    state=state,
                    model="test/model",
                    messages=[{"role": "user", "content": "new"}],
                    use_zai_direct=False,
                    direct_model_name=None,
                    use_kimi_direct=False,
                    direct_kimi_model_name=None,
                    use_deepseek_direct=False,
                    direct_deepseek_model_name=None,
                    use_ollama_direct=False,
                    direct_ollama_model_name=None,
                    extra_body=None,
                    max_output_tokens=100,
                )

                self.assertIs(state.in_flight_task, never_done)
                self.assertFalse(never_done.cancelled())
                self.assertIsNotNone(state.queued_snapshot)
                never_done.cancel()
                try:
                    await never_done
                except asyncio.CancelledError:
                    pass

        asyncio.run(scenario())

    def test_queued_snapshot_runs_after_current_task_completes(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                calls: list[list[dict]] = []
                openrouter = mock.Mock()

                async def _complete(**kwargs):
                    calls.append(kwargs["messages"])
                    await asyncio.sleep(0.01)
                    return OpenRouterCallResult(
                        content="## Summary\nQueued snapshot contains useful review details.",
                        tool_calls=[],
                        raw={},
                    )

                openrouter.chat_completion = _complete
                service = ReviewService(
                    repo_root=Path(td),
                    settings=_build_settings(),
                    openrouter_client=openrouter,
                    models_client=mock.Mock(),
                )
                state = IntermittentReviewState()

                service._dispatch_intermittent_review(
                    state=state,
                    model="test/model",
                    messages=[{"role": "user", "content": "first"}],
                    use_zai_direct=False,
                    direct_model_name=None,
                    use_kimi_direct=False,
                    direct_kimi_model_name=None,
                    use_deepseek_direct=False,
                    direct_deepseek_model_name=None,
                    use_ollama_direct=False,
                    direct_ollama_model_name=None,
                    extra_body=None,
                    max_output_tokens=100,
                )
                service._dispatch_intermittent_review(
                    state=state,
                    model="test/model",
                    messages=[{"role": "user", "content": "second"}],
                    use_zai_direct=False,
                    direct_model_name=None,
                    use_kimi_direct=False,
                    direct_kimi_model_name=None,
                    use_deepseek_direct=False,
                    direct_deepseek_model_name=None,
                    use_ollama_direct=False,
                    direct_ollama_model_name=None,
                    extra_body=None,
                    max_output_tokens=100,
                )

                await asyncio.wait_for(state.in_flight_task, timeout=1)
                if state.in_flight_task is not None:
                    await asyncio.wait_for(state.in_flight_task, timeout=1)

                self.assertGreaterEqual(len(calls), 2)
                self.assertEqual(calls[-1][0]["content"], "second")
                self.assertIn("Queued snapshot", state.latest_markdown or "")

        asyncio.run(scenario())


class TestTimeoutWaitsForInFlightSnapshot(unittest.TestCase):
    def test_timeout_waits_for_in_flight_snapshot_and_returns_it(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                settings = _build_settings(
                    openrouter_reviewer_timeout_seconds=1,
                    openrouter_tool_call_timeout_seconds=2,
                    intermittent_review_calls=5,
                )
                slow_openrouter = mock.Mock()

                async def _main_call(**kwargs):
                    await asyncio.sleep(10)

                slow_openrouter.chat_completion = _main_call
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=slow_openrouter,
                    models_client=mock.Mock(),
                )
                state = IntermittentReviewState()

                async def _finish_snapshot():
                    await asyncio.sleep(0.05)
                    state.latest_markdown = "## Summary\nSnapshot finished during timeout grace period."
                    state.snapshot_tool_call_index = 5

                state.in_flight_task = asyncio.create_task(_finish_snapshot())

                return await service._run_single_reviewer(
                    cfg=mock.Mock(
                        model="test/model",
                        budget=TokenBudget(
                            effective_context_length=50000,
                            effective_output_budget=1000,
                            overhead_tokens=2000,
                        ),
                        supported_parameters=("max_tokens",),
                        tool_calling_supported=False,
                        tool_choice_supported=False,
                        serena_ctx=None,
                        serena_disabled_reason="no serena",
                        use_zai_direct=False,
                        direct_model_name=None,
                        use_kimi_direct=False,
                        direct_kimi_model_name=None,
                        use_deepseek_direct=False,
                        direct_deepseek_model_name=None,
                        use_ollama_direct=False,
                        direct_ollama_model_name=None,
                    ),
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "sys",
                    build_user_prompt=lambda tool_calling_enabled, redacted: "user",
                    redacted_inputs={},
                    requested_paths=None,
                    file_context_builder=mock.Mock(),
                    intermittent_state_override=state,
                )

        outcome = asyncio.run(scenario())
        self.assertTrue(outcome.ok)
        self.assertTrue(outcome.is_intermittent)
        self.assertIn("Snapshot finished", outcome.markdown)


if __name__ == "__main__":
    unittest.main()
