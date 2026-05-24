from __future__ import annotations

import asyncio
import time
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.openrouter_client import OpenRouterCallResult, OpenRouterClientError
from lad_mcp_server.token_budget import TokenBudget
from lad_mcp_server.review_service import (
    INTERMITTENT_REVIEW_TIMEOUT_SECONDS,
    EXPLORATION_DIGEST_MAX_SNIPPET_CHARS,
    ExplorationDigest,
    IntermittentReviewState,
    ReviewerOutcome,
    ReviewService,
    _build_tool_trace_summary,
    _render_digest_snapshot,
    _select_best_interim_markdown,
    _update_exploration_digest,
)


# ---------------------------------------------------------------------------
# Settings parsing
# ---------------------------------------------------------------------------


class TestSettingsIntermittentReviewCalls(unittest.TestCase):
    def _required_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {"OPENROUTER_API_KEY": "test-key"}
        if extra:
            env.update(extra)
        return env

    def test_settings_intermittent_review_calls_default_is_2(self) -> None:
        with mock.patch.dict(os.environ, self._required_env(), clear=True):
            s = Settings.from_env()
        self.assertEqual(s.intermittent_review_calls, 2)

    def test_settings_intermittent_max_output_tokens_default_is_1500(self) -> None:
        with mock.patch.dict(os.environ, self._required_env(), clear=True):
            s = Settings.from_env()
        self.assertEqual(s.intermittent_max_output_tokens, 1500)

    def test_settings_intermittent_max_output_tokens_explicit_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            self._required_env({"INTERMITTENT_MAX_OUTPUT_TOKENS": "2000"}),
            clear=True,
        ):
            s = Settings.from_env()
        self.assertEqual(s.intermittent_max_output_tokens, 2000)

    def test_settings_intermittent_max_output_tokens_zero_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            self._required_env({"INTERMITTENT_MAX_OUTPUT_TOKENS": "0"}),
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_settings_intermittent_review_calls_explicit_zero(self) -> None:
        with mock.patch.dict(
            os.environ,
            self._required_env({"INTERMITTENT_REVIEW_CALLS": "0"}),
            clear=True,
        ):
            s = Settings.from_env()
        self.assertEqual(s.intermittent_review_calls, 0)

    def test_settings_intermittent_review_calls_negative_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            self._required_env({"INTERMITTENT_REVIEW_CALLS": "-1"}),
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_settings_intermittent_review_calls_non_integer_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            self._required_env({"INTERMITTENT_REVIEW_CALLS": "abc"}),
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()


# ---------------------------------------------------------------------------
# Phase 3: ExplorationDigest
# ---------------------------------------------------------------------------


class TestExplorationDigest(unittest.TestCase):
    def test_defaults_are_empty_collections_and_zero_counters(self) -> None:
        digest = ExplorationDigest()
        self.assertEqual(digest.files_read, [])
        self.assertEqual(digest.symbols_found, [])
        self.assertEqual(digest.search_matches, [])
        self.assertEqual(digest.llm_findings, [])
        self.assertEqual(digest.llm_recommendations, [])
        self.assertEqual(digest.llm_open_questions, [])
        self.assertEqual(digest.tools_invoked, set())
        self.assertEqual(digest.memories_used, set())
        self.assertEqual(digest.paths_visited, set())
        self.assertEqual(digest.degraded_outputs, 0)
        self.assertEqual(digest.total_tool_calls, 0)

    def test_state_initializes_empty_digest(self) -> None:
        state = IntermittentReviewState()
        self.assertIsInstance(state.digest, ExplorationDigest)
        self.assertEqual(state.digest.total_tool_calls, 0)
        self.assertEqual(state.digest.files_read, [])

    def test_update_records_read_file_path(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(
            digest,
            "read_file",
            '{"path": "src/main.py"}',
            '{"path": "src/main.py", "content": "secret file contents"}',
            False,
        )
        self.assertEqual(digest.total_tool_calls, 1)
        self.assertEqual(digest.tools_invoked, {"read_file"})
        self.assertEqual(digest.files_read, ["src/main.py"])
        self.assertEqual(digest.paths_visited, {"src/main.py"})
        self.assertNotIn("secret file contents", repr(digest))

    def test_update_records_read_memory_name(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(
            digest,
            "read_memory",
            '{"name": "project_overview"}',
            '{"name": "project_overview.md", "content": "memory contents"}',
            False,
        )
        self.assertEqual(digest.memories_used, {"project_overview.md"})
        self.assertNotIn("memory contents", repr(digest))

    def test_update_records_degraded_result(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(digest, "read_file", "{}", '{"error": "boom"}', True)
        self.assertEqual(digest.total_tool_calls, 1)
        self.assertEqual(digest.degraded_outputs, 1)

    def test_update_records_bounded_search_match_snippets(self) -> None:
        digest = ExplorationDigest()
        huge_match = "src/main.py:10:" + ("x" * (EXPLORATION_DIGEST_MAX_SNIPPET_CHARS * 3))
        _update_exploration_digest(
            digest,
            "search_for_pattern",
            '{"pattern": "x", "path": "src"}',
            '{"matches": ["' + huge_match + '"]}',
            False,
        )
        self.assertEqual(digest.total_tool_calls, 1)
        self.assertEqual(digest.paths_visited, {"src"})
        self.assertEqual(len(digest.search_matches), 1)
        self.assertLessEqual(len(digest.search_matches[0]), EXPLORATION_DIGEST_MAX_SNIPPET_CHARS + 1)

    def test_render_digest_snapshot_includes_required_sections_and_status(self) -> None:
        state = IntermittentReviewState()
        digest = state.digest
        digest.total_tool_calls = 3
        digest.files_read.append("src/main.py")
        digest.memories_used.add("project_overview.md")
        digest.tools_invoked.update({"read_file", "read_memory"})
        digest.degraded_outputs = 1
        state.last_status = "timeout"
        state.last_error = "side-call timed out after 37s"

        result = _render_digest_snapshot(
            model="test/model",
            timeout_seconds=300,
            digest=digest,
            state=state,
            stop_reason="timeout",
        )

        self.assertIn("## Summary", result)
        self.assertIn("## Key Findings", result)
        self.assertIn("## Exploration Statistics", result)
        self.assertIn("## Files Explored", result)
        self.assertIn("## Recommendations", result)
        self.assertIn("## Questions / Unknowns", result)
        self.assertIn("test/model", result)
        self.assertIn("3", result)
        self.assertIn("src/main.py", result)
        self.assertIn("project_overview.md", result)
        self.assertIn("timeout", result)
        self.assertIn("side-call timed out after 37s", result)

    def test_update_records_read_file_window_path(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(
            digest,
            "read_file_window",
            '{"path": "src/utils.py"}',
            '{"path": "src/utils.py", "content": "..."}',
            False,
        )
        self.assertEqual(digest.files_read, ["src/utils.py"])
        self.assertEqual(digest.paths_visited, {"src/utils.py"})

    def test_update_records_find_symbol_path(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(
            digest,
            "find_symbol",
            '{"path": "src/main.py"}',
            '{"symbols": ["MyClass", "my_function"]}',
            False,
        )
        self.assertIn("src/main.py", digest.paths_visited)
        self.assertIn("MyClass", digest.symbols_found)
        self.assertIn("my_function", digest.symbols_found)

    def test_update_handles_unknown_tool_gracefully(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(
            digest,
            "unknown_tool_xyz",
            '{"path": "src/main.py"}',
            "not json at all",
            False,
        )
        self.assertEqual(digest.total_tool_calls, 1)
        self.assertIn("unknown_tool_xyz", digest.tools_invoked)
        self.assertEqual(digest.files_read, [])  # unknown tool, no path extracted

    def test_update_handles_malformed_json_gracefully(self) -> None:
        digest = ExplorationDigest()
        _update_exploration_digest(
            digest,
            "read_file",
            "not valid json{{{",
            "also not json",
            False,
        )
        self.assertEqual(digest.total_tool_calls, 1)
        self.assertEqual(digest.files_read, [])  # no crash, just empty

    def test_update_deduplicates_file_reads(self) -> None:
        digest = ExplorationDigest()
        for _ in range(3):
            _update_exploration_digest(
                digest,
                "read_file",
                '{"path": "src/main.py"}',
                '{"path": "src/main.py", "content": "..."}',
                False,
            )
        self.assertEqual(digest.total_tool_calls, 3)
        self.assertEqual(digest.files_read, ["src/main.py"])  # deduplicated

    def test_render_digest_with_provider_error_stop_reason(self) -> None:
        state = IntermittentReviewState()
        state.last_status = "provider_error"
        state.last_error = "API returned 429"
        digest = state.digest
        digest.total_tool_calls = 1

        result = _render_digest_snapshot(
            model="test/model",
            timeout_seconds=300,
            digest=digest,
            state=state,
            stop_reason="provider_error",
        )
        self.assertIn("stopped (provider_error)", result)
        self.assertIn("provider_error", result)
        self.assertIn("API returned 429", result)


# ---------------------------------------------------------------------------
# Phase 2: Side-call lifecycle tracking
# ---------------------------------------------------------------------------


class TestIntermittentStateDefaults(unittest.TestCase):
    def test_state_defaults(self) -> None:
        state = IntermittentReviewState()
        self.assertEqual(state.last_status, "never_dispatched")
        self.assertIsNone(state.last_error)
        self.assertIsNone(state.last_started_at)
        self.assertIsNone(state.last_finished_at)


class TestSideCallLifecycle(unittest.TestCase):
    """Test that _run_intermittent_review_call records lifecycle status in state."""

    def _make_service(self, client, repo: Path) -> ReviewService:
        return _make_service(intermittent_n=1, client=client, repo=repo)

    async def _invoke_side(self, service: ReviewService, state: IntermittentReviewState) -> None:
        await service._run_intermittent_review_call(
            model="test/model",
            messages_snapshot=[{"role": "user", "content": "hi"}],
            use_zai_direct=False,
            direct_model_name=None,
            use_kimi_direct=False,
            direct_kimi_model_name=None,
            extra_body=None,
            max_output_tokens=100,
            snapshot_tool_call_index=1,
            state=state,
        )

    def test_successful_side_call_sets_completed(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                client = _ScriptedSideClient([
                    OpenRouterCallResult(content="## Summary\nPartial review content.", tool_calls=[], raw={}),
                ])
                service = self._make_service(client, Path(td))
                state = IntermittentReviewState()
                await self._invoke_side(service, state)
                self.assertEqual(state.last_status, "completed")
                self.assertIsNone(state.last_error)
                self.assertIsNotNone(state.last_started_at)
                self.assertIsNotNone(state.last_finished_at)
                self.assertGreaterEqual(state.last_finished_at, state.last_started_at)

        asyncio.run(scenario())

    def test_empty_side_call_sets_empty(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                client = _ScriptedSideClient([
                    OpenRouterCallResult(content="   \n  ", tool_calls=[], raw={}),
                ])
                service = self._make_service(client, Path(td))
                state = IntermittentReviewState()
                await self._invoke_side(service, state)
                self.assertEqual(state.last_status, "empty")
                self.assertIsNone(state.last_error)

        asyncio.run(scenario())

    def test_provider_error_sets_provider_error(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                client = _ScriptedSideClient([RuntimeError("API returned 500")])
                service = self._make_service(client, Path(td))
                state = IntermittentReviewState()
                await self._invoke_side(service, state)
                self.assertEqual(state.last_status, "provider_error")
                self.assertIn("API returned 500", state.last_error or "")

        asyncio.run(scenario())

    def test_timeout_sets_timeout_status(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:

                class _SlowClient:
                    async def chat_completion(self, **kwargs):
                        await asyncio.sleep(3600)

                service = self._make_service(_SlowClient(), Path(td))
                state = IntermittentReviewState()
                with mock.patch.object(service, "_intermittent_timeout", 0.05):
                    await self._invoke_side(service, state)
                self.assertEqual(state.last_status, "timeout")
                self.assertIsNotNone(state.last_error)

        asyncio.run(scenario())

    def test_dispatch_sets_running(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:

                class _SlowClient:
                    async def chat_completion(self, **kwargs):
                        await asyncio.sleep(3600)

                service = self._make_service(_SlowClient(), Path(td))
                state = IntermittentReviewState()
                service._dispatch_intermittent_review(
                    state=state,
                    model="test/model",
                    messages=[{"role": "user", "content": "review"}],
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
                await asyncio.sleep(0.01)
                self.assertEqual(state.last_status, "running")
                self.assertIsNotNone(state.last_started_at)
                if state.in_flight_task and not state.in_flight_task.done():
                    state.in_flight_task.cancel()
                    try:
                        await state.in_flight_task
                    except asyncio.CancelledError:
                        pass

        asyncio.run(scenario())


class TestIntermittentStatusNote(unittest.TestCase):
    def test_never_dispatched_note(self) -> None:
        from lad_mcp_server.review_service import _format_intermittent_status_note
        state = IntermittentReviewState()
        note = _format_intermittent_status_note(state)
        self.assertIn("never_dispatched", note)

    def test_timeout_note_includes_error(self) -> None:
        from lad_mcp_server.review_service import _format_intermittent_status_note
        state = IntermittentReviewState()
        state.last_status = "timeout"
        state.last_error = "side-call timed out after 37s"
        note = _format_intermittent_status_note(state)
        self.assertIn("timeout", note)
        self.assertIn("side-call timed out after 37s", note)

    def test_completed_note(self) -> None:
        from lad_mcp_server.review_service import _format_intermittent_status_note
        state = IntermittentReviewState()
        state.last_status = "completed"
        note = _format_intermittent_status_note(state)
        self.assertIn("completed", note)
        self.assertNotIn("error", note.lower())


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_settings(intermittent_n: int) -> Settings:
    return Settings(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model="test/model",
        openrouter_secondary_reviewer_model="0",
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
        lad_serena_max_tool_calls=20,
        lad_serena_tool_timeout_seconds=5,
        lad_serena_max_tool_result_chars=12000,
        lad_serena_max_total_chars=50000,
        lad_serena_max_dir_entries=100,
        lad_serena_max_search_results=20,
        intermittent_review_calls=intermittent_n,
    )


class TestSelectBestInterimMarkdown(unittest.TestCase):
    """Test _select_best_interim_markdown priority: snapshot > digest > trace > None."""

    def _serena_ctx(self, *, has_evidence: bool = False) -> mock.Mock:
        ctx = mock.Mock()
        if has_evidence:
            ctx.used_tools = {"read_file"}
            ctx.used_memories = set()
            ctx.used_paths = {"src/main.py"}
        else:
            ctx.used_tools = set()
            ctx.used_memories = set()
            ctx.used_paths = set()
        return ctx

    def test_substantive_snapshot_wins_over_digest(self) -> None:
        state = IntermittentReviewState()
        state.latest_markdown = "## Summary\nReal review.\n## Key Findings\n- Bug found"
        state.digest.total_tool_calls = 5
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=self._serena_ctx(has_evidence=True),
            stop_reason="timeout",
        )
        self.assertIn("Real review", md)
        self.assertIn("intermittent snapshot", note)

    def test_digest_wins_over_trace_when_no_snapshot(self) -> None:
        state = IntermittentReviewState()
        state.digest.total_tool_calls = 3
        state.digest.files_read.append("src/main.py")
        state.digest.tools_invoked.add("read_file")
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=self._serena_ctx(has_evidence=True),
            stop_reason="timeout",
        )
        self.assertIsNotNone(md)
        self.assertIn("src/main.py", md)
        self.assertIn("deterministic exploration digest", note)

    def test_trace_wins_when_no_snapshot_and_empty_digest(self) -> None:
        state = IntermittentReviewState()
        # digest.total_tool_calls == 0 → no digest
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=self._serena_ctx(has_evidence=True),
            stop_reason="timeout",
        )
        self.assertIsNotNone(md)
        self.assertIn("tool-exploration trace", note)

    def test_none_when_no_evidence(self) -> None:
        state = IntermittentReviewState()
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=None,
            stop_reason="timeout",
        )
        self.assertIsNone(md)
        self.assertIn("no interim", note.lower())

    def test_provider_error_stop_reason_uses_digest(self) -> None:
        state = IntermittentReviewState()
        state.digest.total_tool_calls = 2
        state.digest.files_read.append("src/app.py")
        state.last_status = "provider_error"
        state.last_error = "API returned 500"
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=self._serena_ctx(has_evidence=True),
            stop_reason="provider_error",
        )
        self.assertIsNotNone(md)
        self.assertIn("src/app.py", md)
        self.assertIn("provider_error", note)


class _SerenaCtxStub:
    """Pre-activated Serena context with a single tool — no preflight forcing."""

    def __init__(self) -> None:
        self.activated_project: str | None = "."
        self.used_tools: set[str] = set()
        self.used_memories: set[str] = set()
        self.used_paths: set[str] = set()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}},
            }
        ]

    def call_tool(self, name: str, arguments_json: str) -> str:
        return "{}"


def _tool_call_response(call_id: str, name: str = "list_dir") -> Any:
    return type(
        "R",
        (),
        {
            "content": None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
            ],
            "raw": {},
        },
    )()


def _final_response(content: str = "## Summary\nFinal main review.") -> Any:
    return type("R", (), {"content": content, "tool_calls": [], "raw": {}})()


class _ModelsStub:
    def __init__(self, model_id: str):
        self._meta = ModelMetadata(
            model_id=model_id,
            context_length=50000,
            supported_parameters=("tools", "tool_choice", "max_tokens"),
            provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
        )

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._meta


class _MainOnlyClient:
    """Drives only the main tool loop. Refuses to accept side calls (would assert)."""

    def __init__(self, *, main_responses: list[Any]) -> None:
        self.main_responses = list(main_responses)
        self._idx = 0
        self.main_call_count = 0

    async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens,
                              tools=None, tool_choice=None, extra_body=None):
        await asyncio.sleep(0)  # mimic real async I/O yield
        assert tools is not None, "Side calls should be intercepted by mock _dispatch_intermittent_review"
        self.main_call_count += 1
        idx = min(self._idx, len(self.main_responses) - 1)
        self._idx += 1
        return self.main_responses[idx]


def _make_service(
    *,
    intermittent_n: int,
    client,
    repo: Path,
    primary_model: str = "test/model",
) -> ReviewService:
    settings = _build_settings(intermittent_n)
    object.__setattr__(settings, "openrouter_primary_reviewer_model", primary_model)
    return ReviewService(
        repo_root=repo,
        settings=settings,
        openrouter_client=client,
        models_client=_ModelsStub(primary_model),
    )


async def _run_tool_loop_with_mock_dispatch(
    service: ReviewService,
    *,
    intermittent_state: IntermittentReviewState | None,
    main_call_budget: int = 20,
):
    """
    Invoke _tool_loop with _dispatch_intermittent_review mocked to record calls.
    Returns (result_str, dispatch_calls_list).
    """
    dispatch_calls: list[dict[str, Any]] = []

    def fake_dispatch(**kwargs):
        # Mimic the real method: bump tool_calls_so_far is done by caller; here we just record.
        dispatch_calls.append(
            {
                "tool_calls_so_far": kwargs["state"].tool_calls_so_far,
                "model": kwargs["model"],
                "messages_len": len(kwargs["messages"]),
            }
        )

    with mock.patch.object(service, "_dispatch_intermittent_review", side_effect=fake_dispatch):
        result = await service._tool_loop(
            model="test/model",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}},
                }
            ],
            tool_choice_supported=False,
            serena_ctx=_SerenaCtxStub(),
            extra_body=None,
            reviewer_timeout_seconds=10,
            max_output_tokens=100,
            max_tool_calls=main_call_budget,
            tool_timeout_seconds=2,
            intermittent_state=intermittent_state,
        )
    return result, dispatch_calls


# ---------------------------------------------------------------------------
# Dispatch trigger logic (uses _dispatch_intermittent_review mocked out)
# ---------------------------------------------------------------------------


class TestDispatchTrigger(unittest.TestCase):
    def test_intermittent_disabled_when_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            main_responses = [_tool_call_response(f"tc{i}") for i in range(6)] + [_final_response()]
            client = _MainOnlyClient(main_responses=main_responses)
            service = _make_service(intermittent_n=0, client=client, repo=repo)
            state = IntermittentReviewState()
            _, dispatches = asyncio.run(
                _run_tool_loop_with_mock_dispatch(service, intermittent_state=state)
            )
            self.assertEqual(dispatches, [])

    def test_intermittent_disabled_when_state_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            main_responses = [_tool_call_response(f"tc{i}") for i in range(6)] + [_final_response()]
            client = _MainOnlyClient(main_responses=main_responses)
            service = _make_service(intermittent_n=5, client=client, repo=repo)
            _, dispatches = asyncio.run(
                _run_tool_loop_with_mock_dispatch(service, intermittent_state=None)
            )
            self.assertEqual(dispatches, [])

    def test_intermittent_dispatched_every_n_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            main_responses = [_tool_call_response(f"tc{i}") for i in range(5)] + [_final_response()]
            client = _MainOnlyClient(main_responses=main_responses)
            service = _make_service(intermittent_n=2, client=client, repo=repo)
            state = IntermittentReviewState()
            _, dispatches = asyncio.run(
                _run_tool_loop_with_mock_dispatch(service, intermittent_state=state)
            )
            # With N=2 and 5 tool calls, dispatch fires at counters 2 and 4.
            self.assertEqual([d["tool_calls_so_far"] for d in dispatches], [2, 4])

    def test_intermittent_dispatched_every_call_when_n_equals_1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            main_responses = [_tool_call_response(f"tc{i}") for i in range(3)] + [_final_response()]
            client = _MainOnlyClient(main_responses=main_responses)
            service = _make_service(intermittent_n=1, client=client, repo=repo)
            state = IntermittentReviewState()
            _, dispatches = asyncio.run(
                _run_tool_loop_with_mock_dispatch(service, intermittent_state=state)
            )
            self.assertEqual([d["tool_calls_so_far"] for d in dispatches], [1, 2, 3])

    def test_preflight_tools_do_not_trigger_dispatch(self) -> None:
        """Preflight tool calls should not count toward the intermittent dispatch counter."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            # 3 preflight + 4 exploration + final
            main_responses = [
                _tool_call_response("pf1", name="activate_project"),
                _tool_call_response("pf2", name="read_project_overview"),
                _tool_call_response("pf3", name="read_baseline_memories"),
                _tool_call_response("ex1", name="read_file"),
                _tool_call_response("ex2", name="read_file"),
                _tool_call_response("ex3", name="search_for_pattern"),
                _tool_call_response("ex4", name="read_file"),
                _final_response(),
            ]
            client = _MainOnlyClient(main_responses=main_responses)
            service = _make_service(intermittent_n=2, client=client, repo=repo)
            state = IntermittentReviewState()
            _, dispatches = asyncio.run(
                _run_tool_loop_with_mock_dispatch(service, intermittent_state=state)
            )
            # 4 exploration calls with N=2: dispatch at counters 2 and 4.
            self.assertEqual([d["tool_calls_so_far"] for d in dispatches], [2, 4])

    def test_preflight_tools_with_n_equals_1_dispatch_after_first_exploration(self) -> None:
        """With N=1, first dispatch should fire after the first non-preflight tool call."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            main_responses = [
                _tool_call_response("pf1", name="activate_project"),
                _tool_call_response("pf2", name="read_project_overview"),
                _tool_call_response("pf3", name="read_baseline_memories"),
                _tool_call_response("ex1", name="read_file"),
                _tool_call_response("ex2", name="read_file"),
                _final_response(),
            ]
            client = _MainOnlyClient(main_responses=main_responses)
            service = _make_service(intermittent_n=1, client=client, repo=repo)
            state = IntermittentReviewState()
            _, dispatches = asyncio.run(
                _run_tool_loop_with_mock_dispatch(service, intermittent_state=state)
            )
            # 2 exploration calls with N=1: dispatch at counters 1 and 2.
            self.assertEqual([d["tool_calls_so_far"] for d in dispatches], [1, 2])


# ---------------------------------------------------------------------------
# Side-call logic (direct unit tests of _run_intermittent_review_call)
# ---------------------------------------------------------------------------


class _ScriptedSideClient:
    """Client that handles only side calls; the responses are scripted in order."""

    def __init__(self, scripted_responses: list[Any]) -> None:
        self.scripted = list(scripted_responses)
        self.idx = 0
        self.calls = 0

    async def chat_completion(self, *, model, messages, timeout_seconds, max_output_tokens,
                              tools=None, tool_choice=None, extra_body=None):
        self.calls += 1
        idx = min(self.idx, len(self.scripted) - 1)
        self.idx += 1
        resp = self.scripted[idx]
        if isinstance(resp, BaseException):
            raise resp
        await asyncio.sleep(0)
        return resp


class TestSideCallLogic(unittest.TestCase):
    def _make(self, repo: Path, client) -> ReviewService:
        return _make_service(intermittent_n=1, client=client, repo=repo)

    async def _invoke_side(self, service: ReviewService, state: IntermittentReviewState, snapshot_idx: int) -> None:
        await service._run_intermittent_review_call(
            model="test/model",
            messages_snapshot=[{"role": "user", "content": "hi"}],
            use_zai_direct=False,
            direct_model_name=None,
            use_kimi_direct=False,
            direct_kimi_model_name=None,
            extra_body=None,
            max_output_tokens=100,
            snapshot_tool_call_index=snapshot_idx,
            state=state,
        )

    def test_cache_holds_latest_successful(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _ScriptedSideClient([
                _final_response("## Summary\nREV1"),
                _final_response("## Summary\nREV2"),
            ])
            service = self._make(repo, client)
            state = IntermittentReviewState()

            async def run():
                await self._invoke_side(service, state, 2)
                self.assertIn("REV1", state.latest_markdown or "")
                self.assertEqual(state.snapshot_tool_call_index, 2)
                await self._invoke_side(service, state, 4)
                self.assertIn("REV2", state.latest_markdown or "")
                self.assertEqual(state.snapshot_tool_call_index, 4)

            asyncio.run(run())

    def test_whitespace_response_does_not_overwrite_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _ScriptedSideClient([
                _final_response("## Summary\nREV1"),
                _final_response("   \n  "),
            ])
            service = self._make(repo, client)
            state = IntermittentReviewState()

            async def run():
                await self._invoke_side(service, state, 2)
                await self._invoke_side(service, state, 4)
                self.assertIn("REV1", state.latest_markdown or "")
                self.assertEqual(state.snapshot_tool_call_index, 2)  # unchanged

            asyncio.run(run())

    def test_side_call_openrouter_error_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _ScriptedSideClient([OpenRouterClientError("boom")])
            service = self._make(repo, client)
            state = IntermittentReviewState()

            async def run():
                # Must NOT raise.
                await self._invoke_side(service, state, 1)
                self.assertIsNone(state.latest_markdown)

            asyncio.run(run())

    def test_side_call_cancelled_error_propagates(self) -> None:
        """CancelledError must NOT be swallowed (it derives from BaseException, not Exception)."""

        class _CancellingClient:
            async def chat_completion(self, **kwargs):
                raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            service = self._make(repo, _CancellingClient())
            state = IntermittentReviewState()

            async def run():
                with self.assertRaises(asyncio.CancelledError):
                    await self._invoke_side(service, state, 1)

            asyncio.run(run())


# ---------------------------------------------------------------------------
# Cancellation on normal completion + done-before-cancel guard
# ---------------------------------------------------------------------------


class TestCancellationLogic(unittest.TestCase):
    def test_cancel_in_flight_intermittent_cancels_pending_task(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _MainOnlyClient(main_responses=[])
            service = _make_service(intermittent_n=1, client=client, repo=repo)
            state = IntermittentReviewState()

            async def run():
                async def never():
                    await asyncio.Event().wait()  # never completes

                task = asyncio.create_task(never())
                state.in_flight_task = task
                service._cancel_in_flight_intermittent(state)
                await asyncio.sleep(0)
                self.assertTrue(task.cancelled() or task.done())

            asyncio.run(run())

    def test_cancel_in_flight_intermittent_leaves_done_task_alone(self) -> None:
        """Done-before-cancel guard: a completed task is not re-cancelled."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _MainOnlyClient(main_responses=[])
            service = _make_service(intermittent_n=1, client=client, repo=repo)
            state = IntermittentReviewState()

            async def run():
                async def quick():
                    return "done"

                task = asyncio.create_task(quick())
                await task  # let it finish
                self.assertTrue(task.done())
                self.assertFalse(task.cancelled())
                state.in_flight_task = task
                service._cancel_in_flight_intermittent(state)
                # The done task must remain done, not cancelled.
                self.assertTrue(task.done())
                self.assertFalse(task.cancelled())

            asyncio.run(run())

    def test_cancel_in_flight_intermittent_on_none_state(self) -> None:
        """Safe to call with None state (used in back-compat path)."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            client = _MainOnlyClient(main_responses=[])
            service = _make_service(intermittent_n=0, client=client, repo=repo)
            # Should not raise.
            service._cancel_in_flight_intermittent(None)


# ---------------------------------------------------------------------------
# Non-blocking dispatch (structural)
# ---------------------------------------------------------------------------


class TestNonBlocking(unittest.TestCase):
    def test_dispatch_does_not_block_main_loop(self) -> None:
        """A side task that blocks forever must not prevent the main loop from finalizing."""

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                main_responses = [
                    _tool_call_response("tc0"),
                    _tool_call_response("tc1"),
                    _tool_call_response("tc2"),
                    _final_response("## Summary\nDone"),
                ]

                blocker = asyncio.Event()  # never set

                async def blocking_side():
                    await blocker.wait()

                class _BlockingDispatchService(ReviewService):
                    dispatched: int = 0

                    def _dispatch_intermittent_review(self, *, state, **kwargs):  # type: ignore[override]
                        state.tool_calls_so_far  # no-op read
                        prev = state.in_flight_task
                        if prev is not None and not prev.done():
                            prev.cancel()
                        state.in_flight_task = asyncio.create_task(blocking_side())
                        type(self).dispatched += 1

                client = _MainOnlyClient(main_responses=main_responses)
                settings = _build_settings(intermittent_n=2)
                service = _BlockingDispatchService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                state = IntermittentReviewState()
                result = await service._tool_loop(
                    model="test/model",
                    messages=[
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "user"},
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}},
                        }
                    ],
                    tool_choice_supported=False,
                    serena_ctx=_SerenaCtxStub(),
                    extra_body=None,
                    reviewer_timeout_seconds=10,
                    max_output_tokens=100,
                    max_tool_calls=20,
                    tool_timeout_seconds=2,
                    intermittent_state=state,
                )
                # Main loop completes despite the side task being stuck.
                self.assertIn("Done", result)
                self.assertGreaterEqual(_BlockingDispatchService.dispatched, 1)
                # Pending task got cancelled on normal completion.
                self.assertIsNotNone(state.in_flight_task)
                await asyncio.sleep(0)
                self.assertTrue(state.in_flight_task.cancelled() or state.in_flight_task.done())

        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Timeout fallback path
# ---------------------------------------------------------------------------


class _DummyFCB:
    def build(self, *, paths, max_chars):
        return type(
            "FC",
            (),
            {"formatted": "", "embedded_files": (), "skipped_files": ()},
        )()


class TestTimeoutReturnsSnapshot(unittest.TestCase):
    def test_timeout_returns_cached_intermittent_snapshot(self) -> None:
        """When the reviewer wall-clock fires and a snapshot exists, return it as ok=True intermittent."""

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                # Create .serena/ so the reviewer config attaches tools to main calls
                # (this distinguishes side calls from main calls inside the fake client).
                (repo / ".serena" / "memories").mkdir(parents=True)
                hang = asyncio.Event()

                class _HangingMainClient:
                    main_count = 0
                    side_count = 0

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
                        await asyncio.sleep(0)
                        if tools is None:
                            type(self).side_count += 1
                            return _final_response("## Summary\nFound three critical bugs today.")
                        type(self).main_count += 1
                        # Hang forever to force the reviewer timeout.
                        await hang.wait()
                        return _final_response("never")

                client = _HangingMainClient()
                settings = _build_settings(intermittent_n=1)
                object.__setattr__(settings, "openrouter_reviewer_timeout_seconds", 1)
                object.__setattr__(settings, "openrouter_tool_call_timeout_seconds", 3)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )

                # Pre-seed a snapshot directly via the side call helper (simulating a
                # snapshot captured before the timeout fires). This decouples the test
                # from event-loop scheduling races.
                state = IntermittentReviewState()
                state.latest_markdown = "## Summary\nFound three critical bugs today."
                state.snapshot_tool_call_index = 3
                state.tool_calls_so_far = 3

                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                outcome = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                    intermittent_state_override=state,
                )

                self.assertTrue(outcome.ok)
                self.assertTrue(outcome.is_intermittent)
                self.assertIn("critical bugs", outcome.markdown)
                self.assertIn("intermittent", (outcome.provider_note or "").lower())
                hang.set()

        asyncio.run(scenario())

    def test_timeout_without_snapshot_falls_back_to_tool_trace_when_serena_was_used(self) -> None:
        """When intermittent dispatch is disabled but Serena was active, timeout
        returns the tool-exploration trace (not the generic error stub)."""
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)
                hang = asyncio.Event()

                class _NoSideClient:
                    async def chat_completion(self, **kwargs):
                        await asyncio.sleep(0)
                        await hang.wait()
                        return _final_response("never")

                client = _NoSideClient()
                settings = _build_settings(intermittent_n=0)  # feature disabled
                object.__setattr__(settings, "openrouter_reviewer_timeout_seconds", 1)
                object.__setattr__(settings, "openrouter_tool_call_timeout_seconds", 3)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                outcome = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                # Tool-trace fallback when Serena was active (even with intermittent disabled)
                self.assertTrue(outcome.ok)
                self.assertTrue(outcome.is_intermittent)
                self.assertIn("tool-exploration trace", outcome.markdown)
                self.assertNotIn("Reviewer Error", outcome.markdown)
                hang.set()

        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Disclosure + synthesis
# ---------------------------------------------------------------------------


def _intermittent_outcome(model: str = "test/model") -> ReviewerOutcome:
    return ReviewerOutcome(
        ok=True,
        model=model,
        used_serena=True,
        serena_disabled_reason=None,
        serena_activated_project=".",
        serena_used_tools=(),
        serena_used_memories=(),
        serena_used_paths=(),
        markdown="## Summary\nPartial",
        error=None,
        provider="openrouter",
        provider_note="Reviewer timed out; intermittent snapshot.",
        is_intermittent=True,
    )


def _ok_outcome(model: str = "other/model") -> ReviewerOutcome:
    return ReviewerOutcome(
        ok=True,
        model=model,
        used_serena=False,
        serena_disabled_reason=None,
        serena_activated_project=None,
        serena_used_tools=(),
        serena_used_memories=(),
        serena_used_paths=(),
        markdown="## Summary\nDone",
        error=None,
        provider="openrouter",
    )


class TestTimeoutReturnsDigest(unittest.TestCase):
    """When no model snapshot exists but digest has evidence, timeout returns digest with ok=True."""

    def _make_service(self, repo: Path) -> ReviewService:
        openrouter = mock.Mock()

        async def _hang(**kwargs):
            await asyncio.sleep(3600)

        openrouter.chat_completion = _hang
        settings = _build_settings(intermittent_n=1)
        object.__setattr__(settings, "openrouter_reviewer_timeout_seconds", 1)
        object.__setattr__(settings, "openrouter_tool_call_timeout_seconds", 2)
        return ReviewService(
            repo_root=repo,
            settings=settings,
            openrouter_client=openrouter,
            models_client=_ModelsStub("test/model"),
        )

    def test_timeout_with_digest_returns_ok_true(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                service = self._make_service(repo)
                cfg = mock.Mock(
                    model="test/model",
                    budget=TokenBudget(effective_context_length=50000, effective_output_budget=2000, overhead_tokens=2000),
                    supported_parameters=("tools",),
                    tool_calling_supported=True,
                    tool_choice_supported=True,
                    serena_ctx=None,
                    serena_disabled_reason=None,
                )
                # Pre-populate digest to simulate tool exploration before timeout
                state = IntermittentReviewState()
                state.digest.total_tool_calls = 3
                state.digest.files_read.append("src/main.py")
                state.digest.tools_invoked.add("read_file")
                state.last_status = "timeout"
                state.last_error = "side-call timed out"

                result = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda **kw: "sys",
                    build_user_prompt=lambda tool_calling_enabled=False, redacted_inputs=None: "user",
                    redacted_inputs={"code": "x"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                    intermittent_state_override=state,
                )
                self.assertTrue(result.ok)
                self.assertTrue(result.is_intermittent)
                self.assertIn("src/main.py", result.markdown)
                self.assertIn("deterministic exploration digest", result.provider_note)

        asyncio.run(scenario())

    def test_timeout_with_no_evidence_returns_ok_false(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                service = self._make_service(repo)
                cfg = mock.Mock(
                    model="test/model",
                    budget=TokenBudget(effective_context_length=50000, effective_output_budget=2000, overhead_tokens=2000),
                    supported_parameters=("tools",),
                    tool_calling_supported=True,
                    tool_choice_supported=True,
                    serena_ctx=None,
                    serena_disabled_reason=None,
                )
                result = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda **kw: "sys",
                    build_user_prompt=lambda tool_calling_enabled=False, redacted_inputs=None: "user",
                    redacted_inputs={"code": "x"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                    intermittent_state_override=None,
                )
                self.assertFalse(result.ok)

        asyncio.run(scenario())


class TestExceptionReturnsDigest(unittest.TestCase):
    """Provider error after tool exploration returns digest, not error stub."""

    def test_provider_error_with_digest_returns_ok_true(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                openrouter = mock.Mock()
                openrouter.chat_completion = mock.AsyncMock(side_effect=RuntimeError("API 500"))
                service = _make_service(intermittent_n=1, client=openrouter, repo=repo)
                cfg = mock.Mock(
                    model="test/model",
                    budget=TokenBudget(effective_context_length=50000, effective_output_budget=2000, overhead_tokens=2000),
                    supported_parameters=("tools",),
                    tool_calling_supported=True,
                    tool_choice_supported=True,
                    serena_ctx=None,
                    serena_disabled_reason=None,
                )
                state = IntermittentReviewState()
                state.digest.total_tool_calls = 2
                state.digest.files_read.append("src/app.py")
                state.last_status = "provider_error"
                state.last_error = "API 500"

                result = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda **kw: "sys",
                    build_user_prompt=lambda tool_calling_enabled=False, redacted_inputs=None: "user",
                    redacted_inputs={"code": "x"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                    intermittent_state_override=state,
                )
                self.assertTrue(result.ok)
                self.assertTrue(result.is_intermittent)
                self.assertIn("src/app.py", result.markdown)
                self.assertIn("API 500", result.provider_note)

        asyncio.run(scenario())

    def test_provider_error_with_no_digest_returns_ok_false(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                openrouter = mock.Mock()
                openrouter.chat_completion = mock.AsyncMock(side_effect=RuntimeError("API 500"))
                service = _make_service(intermittent_n=1, client=openrouter, repo=repo)
                cfg = mock.Mock(
                    model="test/model",
                    budget=TokenBudget(effective_context_length=50000, effective_output_budget=2000, overhead_tokens=2000),
                    supported_parameters=("tools",),
                    tool_calling_supported=True,
                    tool_choice_supported=True,
                    serena_ctx=None,
                    serena_disabled_reason=None,
                )
                result = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda **kw: "sys",
                    build_user_prompt=lambda tool_calling_enabled=False, redacted_inputs=None: "user",
                    redacted_inputs={"code": "x"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                    intermittent_state_override=None,
                )
                self.assertFalse(result.ok)
                self.assertIn("Reviewer Error", result.markdown)

        asyncio.run(scenario())


class TestDisclosureAndSynthesis(unittest.TestCase):
    def test_disclosure_banner_for_intermittent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            service = _make_service(
                intermittent_n=0, client=_MainOnlyClient(main_responses=[]), repo=repo,
            )
            out = service._append_disclosure(_intermittent_outcome())
            self.assertIn("Intermittent review", out)
            banner_idx = out.find("Intermittent review")
            model_idx = out.find("*Model:")
            self.assertLess(banner_idx, model_idx)

    def test_disclosure_no_banner_for_normal_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            service = _make_service(
                intermittent_n=0, client=_MainOnlyClient(main_responses=[]), repo=repo,
            )
            out = service._append_disclosure(_ok_outcome())
            self.assertNotIn("Intermittent review", out)

    def test_synthesize_mentions_intermittent_for_primary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            service = _make_service(
                intermittent_n=0, client=_MainOnlyClient(main_responses=[]), repo=repo,
            )
            synth = service._synthesize(_intermittent_outcome(), _ok_outcome())
            self.assertIn("intermittent", synth.lower())

    def test_synthesize_mentions_intermittent_for_secondary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            service = _make_service(
                intermittent_n=0, client=_MainOnlyClient(main_responses=[]), repo=repo,
            )
            primary = _ok_outcome("primary/model")
            secondary = _intermittent_outcome("secondary/model")
            synth = service._synthesize(primary, secondary)
            self.assertIn("intermittent", synth.lower())


# ---------------------------------------------------------------------------
# Phase 1: Review-shaped tool trace fallback
# ---------------------------------------------------------------------------


class TestToolTraceRequiredSections(unittest.TestCase):
    """Phase 1.1: _build_tool_trace_summary must produce all required sections."""

    def test_trace_contains_all_required_sections(self) -> None:
        md = _build_tool_trace_summary(
            model="test/model",
            timeout_seconds=300,
            tool_calls_made=3,
            tools_invoked={"read_file", "find_symbol"},
            memories_used={"overview.md"},
            paths_used={"src/main.py"},
        )
        for section in ("## Summary", "## Key Findings", "## Recommendations", "## Questions / Unknowns"):
            self.assertIn(section, md, f"Missing required section: {section}")

    def test_normalize_does_not_append_placeholders_to_trace(self) -> None:
        from lad_mcp_server.markdown import normalize_reviewer_markdown
        md = _build_tool_trace_summary(
            model="test/model",
            timeout_seconds=300,
            tool_calls_made=1,
            tools_invoked={"read_file"},
            memories_used=set(),
            paths_used={"src/app.py"},
        )
        normalized = normalize_reviewer_markdown(md)
        self.assertNotIn("*(No Key Findings provided by reviewer)*", normalized)
        self.assertNotIn("*(No Questions / Unknowns provided by reviewer)*", normalized)

    def test_trace_with_empty_evidence_still_has_all_sections(self) -> None:
        md = _build_tool_trace_summary(
            model="test/model",
            timeout_seconds=60,
            tool_calls_made=0,
            tools_invoked=set(),
            memories_used=set(),
            paths_used=set(),
        )
        for section in ("## Summary", "## Key Findings", "## Recommendations", "## Questions / Unknowns"):
            self.assertIn(section, md, f"Missing required section with empty evidence: {section}")


class TestPlaceholderSnapshotFallsBackToTrace(unittest.TestCase):
    """Phase 1.2: Placeholder-only snapshot should fall back to trace/digest."""

    def test_select_best_prefers_digest_over_placeholder_snapshot(self) -> None:
        state = IntermittentReviewState()
        state.latest_markdown = "## Summary\n*(No Summary provided by reviewer)*"
        state.digest.total_tool_calls = 2
        state.digest.files_read.append("src/main.py")
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=None,
            stop_reason="timeout",
        )
        # Digest should win over placeholder snapshot
        self.assertIn("src/main.py", md)
        self.assertIn("deterministic exploration digest", note)

    def test_select_best_prefers_trace_over_placeholder_snapshot(self) -> None:
        state = IntermittentReviewState()
        state.latest_markdown = "## Summary\n*(No Summary provided by reviewer)*"
        # No digest evidence but serena has tools
        serena_ctx = mock.Mock()
        serena_ctx.used_tools = {"read_file"}
        serena_ctx.used_memories = set()
        serena_ctx.used_paths = {"src/main.py"}
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=serena_ctx,
            stop_reason="timeout",
        )
        self.assertIn("tool-exploration trace", note)
        self.assertIn("src/main.py", md)

    def test_substantive_snapshot_still_wins_over_digest(self) -> None:
        state = IntermittentReviewState()
        state.latest_markdown = "## Summary\nFound three critical bugs today.\n## Key Findings\n- Bug in auth"
        state.digest.total_tool_calls = 5
        state.digest.files_read.append("src/main.py")
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=None,
            stop_reason="timeout",
        )
        self.assertIn("critical bugs", md)
        self.assertIn("intermittent snapshot", note)


# ---------------------------------------------------------------------------
# Phase 6: Grace period semantics
# ---------------------------------------------------------------------------


class TestGracePeriodShield(unittest.TestCase):
    """Phase 6.1: Grace wait uses asyncio.shield so side-call survives grace expiry."""

    def test_grace_expiry_does_not_cancel_side_call(self) -> None:
        """When grace period expires, the in-flight side-call task is NOT cancelled."""
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)

                side_completed = asyncio.Event()

                class _Client:
                    call_count = 0

                    async def chat_completion(self, **kwargs):
                        type(self).call_count += 1
                        if kwargs.get("tools") is None:
                            # Side call — slow but completes after grace would expire
                            await asyncio.sleep(0.3)
                            side_completed.set()
                            return OpenRouterCallResult(
                                content="## Summary\nFound three critical bugs in the parser module.",
                                tool_calls=[],
                                raw={},
                            )
                        else:
                            # Main call — return one tool call then hang
                            return OpenRouterCallResult(
                                content="",
                                tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}],
                                raw={},
                            )

                client = _Client()
                settings = _build_settings(intermittent_n=1)
                object.__setattr__(settings, "openrouter_reviewer_timeout_seconds", 2)
                object.__setattr__(settings, "openrouter_tool_call_timeout_seconds", 5)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                _ = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                # Side-call should complete despite grace expiry
                self.assertTrue(side_completed.is_set(), "Side-call should have completed")

        asyncio.run(scenario())

    def test_side_call_completing_during_grace_updates_snapshot(self) -> None:
        """Side-call that completes during grace should update latest_markdown."""
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)

                class _Client:
                    async def chat_completion(self, **kwargs):
                        if kwargs.get("tools") is None:
                            return OpenRouterCallResult(
                                content="## Summary\nFound two bugs in authentication flow.",
                                tool_calls=[],
                                raw={},
                            )
                        else:
                            return OpenRouterCallResult(
                                content="",
                                tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}],
                                raw={},
                            )

                client = _Client()
                settings = _build_settings(intermittent_n=1)
                object.__setattr__(settings, "openrouter_reviewer_timeout_seconds", 2)
                object.__setattr__(settings, "openrouter_tool_call_timeout_seconds", 5)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                outcome = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                self.assertTrue(outcome.ok)
                self.assertIn("authentication flow", outcome.markdown)

        asyncio.run(scenario())


class TestGracePeriodBound(unittest.TestCase):
    """Phase 6.2: Grace period is bounded by outer-envelope remaining time."""

    def test_tight_timeout_skips_grace_entirely(self) -> None:
        """When remaining budget is exhausted, grace wait is skipped."""
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)

                side_started = asyncio.Event()

                class _Client:
                    async def chat_completion(self, **kwargs):
                        if kwargs.get("tools") is None:
                            side_started.set()
                            await asyncio.sleep(0.5)
                            return OpenRouterCallResult(
                                content="## Summary\nFound a bug.",
                                tool_calls=[],
                                raw={},
                            )
                        await asyncio.Event().wait()

                client = _Client()
                settings = _build_settings(intermittent_n=1)
                # Very tight timeout: 1s reviewer, 1s tool_call
                object.__setattr__(settings, "openrouter_reviewer_timeout_seconds", 1)
                object.__setattr__(settings, "openrouter_tool_call_timeout_seconds", 1)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                start = time.monotonic()
                _ = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                elapsed = time.monotonic() - start
                # With 1s reviewer + 1s tool_call timeout, total should not exceed ~3s
                # (grace should be 0 or very small, not 30s)
                self.assertLess(elapsed, 5.0, f"Total elapsed {elapsed:.1f}s — grace not bounded")

        asyncio.run(scenario())

    def test_grace_seconds_formula_clamps_to_remaining_budget(self) -> None:
        """Unit test: grace formula returns 0 when remaining budget is exhausted."""
        from lad_mcp_server.review_service import _compute_grace_seconds
        # remaining = 3s, which is < 5 (min grace) → grace = 0
        grace = _compute_grace_seconds(
            tool_call_timeout_seconds=10,
            reviewer_start=time.monotonic() - 8,  # 8s elapsed → 2s remaining - 5s reserve = negative
        )
        self.assertEqual(grace, 0)

    def test_grace_seconds_formula_normal_case(self) -> None:
        """Unit test: normal case with ample remaining budget."""
        from lad_mcp_server.review_service import _compute_grace_seconds
        # remaining = 60s - 5s reserve = 55s → base = max(5, min(15, 55//3)) = max(5,15) = 15
        grace = _compute_grace_seconds(
            tool_call_timeout_seconds=120,
            reviewer_start=time.monotonic() - 60,
        )
        self.assertGreater(grace, 0)
        self.assertLessEqual(grace, 15)


# ---------------------------------------------------------------------------
# Phase 7: Regression matrix — intermittent failure modes
# ---------------------------------------------------------------------------


class TestRegressionSideCallEmptyContent(unittest.TestCase):
    """Side-call that returns empty content should fall back to digest/trace."""

    def test_empty_side_call_falls_back_to_digest(self) -> None:
        state = IntermittentReviewState()
        state.latest_markdown = ""
        state.digest.total_tool_calls = 2
        state.digest.files_read.append("src/main.py")
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=None,
            stop_reason="timeout",
        )
        # Empty snapshot is not substantive → digest wins
        self.assertIn("src/main.py", md)
        self.assertIn("deterministic exploration digest", note)


class TestRegressionSideCallPlaceholderOnly(unittest.TestCase):
    """Side-call that returns placeholder-only markdown should fall back to digest/trace."""

    def test_placeholder_only_falls_back_to_digest(self) -> None:
        state = IntermittentReviewState()
        state.latest_markdown = (
            "## Summary\n*(No Summary provided by reviewer)*\n"
            "## Key Findings\n*(No Key Findings provided by reviewer)*"
        )
        state.digest.total_tool_calls = 3
        state.digest.files_read.append("src/app.py")
        md, note = _select_best_interim_markdown(
            model="test/model",
            timeout_seconds=300,
            state=state,
            serena_ctx=None,
            stop_reason="timeout",
        )
        self.assertIn("src/app.py", md)
        self.assertIn("deterministic exploration digest", note)


class TestRegressionQueuedSnapshotReplaces(unittest.TestCase):
    """Queued snapshot replaces older queued snapshot — state-level test."""

    def test_newer_queued_snapshot_replaces_older(self) -> None:
        state = IntermittentReviewState()
        # When in_flight_task is running, new dispatches overwrite queued_snapshot
        old_request = {"snapshot": "## Summary\nOld content.", "snapshot_tool_call_index": 3}
        state.queued_snapshot = old_request
        new_request = {"snapshot": "## Summary\nNew content.", "snapshot_tool_call_index": 5}
        state.queued_snapshot = new_request
        self.assertEqual(state.queued_snapshot["snapshot"], "## Summary\nNew content.")
        self.assertEqual(state.queued_snapshot["snapshot_tool_call_index"], 5)


class TestRegressionProviderNoteContainsError(unittest.TestCase):
    """provider_note contains original exception message on generic exception + interim recovery."""

    def test_provider_note_contains_exception_message(self) -> None:
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                openrouter = mock.Mock()
                openrouter.chat_completion = mock.AsyncMock(side_effect=RuntimeError("Connection refused: API gateway unreachable"))
                service = _make_service(intermittent_n=1, client=openrouter, repo=repo)
                cfg = mock.Mock(
                    model="test/model",
                    budget=TokenBudget(effective_context_length=50000, effective_output_budget=2000, overhead_tokens=2000),
                    supported_parameters=("tools",),
                    tool_calling_supported=True,
                    tool_choice_supported=True,
                    serena_ctx=None,
                    serena_disabled_reason=None,
                )
                state = IntermittentReviewState()
                state.digest.total_tool_calls = 2
                state.digest.files_read.append("src/main.py")
                state.last_status = "provider_error"
                state.last_error = "Connection refused"

                result = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda **kw: "sys",
                    build_user_prompt=lambda tool_calling_enabled=False, redacted_inputs=None: "user",
                    redacted_inputs={"code": "x"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                    intermittent_state_override=state,
                )
                self.assertTrue(result.ok)
                self.assertIn("Connection refused", result.provider_note)

        asyncio.run(scenario())


class TestRegressionStateResetBetweenReviews(unittest.TestCase):
    """IntermittentReviewState is discarded/reset between reviews."""

    def test_fresh_state_each_review(self) -> None:
        state1 = IntermittentReviewState()
        state1.latest_markdown = "## Summary\nOld review content."
        state1.digest.total_tool_calls = 5
        state1.snapshot_tool_call_index = 3

        state2 = IntermittentReviewState()
        self.assertIsNone(state2.latest_markdown)
        self.assertEqual(state2.digest.total_tool_calls, 0)
        self.assertEqual(state2.snapshot_tool_call_index, 0)
        self.assertEqual(state2.tool_calls_so_far, 0)


class TestRegressionDigestNoFullFileContents(unittest.TestCase):
    """ExplorationDigest does not persist full file contents."""

    def test_large_file_content_not_stored(self) -> None:
        digest = ExplorationDigest()
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        _update_exploration_digest(
            digest,
            "read_file",
            '{"path": "src/big.py"}',
            f'{{"path": "src/big.py", "content": "{large_content}"}}',
            False,
        )
        self.assertEqual(digest.files_read, ["src/big.py"])
        # Verify no massive string is stored in the digest
        for attr in vars(digest):
            val = getattr(digest, attr)
            if isinstance(val, str):
                self.assertLess(len(val), 1024 * 1024, f"Digest attr '{attr}' exceeds 1MB")
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        self.assertLess(len(item), 1024 * 1024, f"Digest list item in '{attr}' exceeds 1MB")

    def test_search_match_snippets_are_bounded(self) -> None:
        digest = ExplorationDigest()
        huge_match = "src/main.py:10:" + "x" * 5000
        _update_exploration_digest(
            digest,
            "search_for_pattern",
            '{"pattern": "x", "path": "src"}',
            f'{{"matches": ["{huge_match}"]}}',
            False,
        )
        self.assertEqual(len(digest.search_matches), 1)
        self.assertLessEqual(len(digest.search_matches[0]), EXPLORATION_DIGEST_MAX_SNIPPET_CHARS + 1)


# ---------------------------------------------------------------------------
# Success-path empty-content fallback
# ---------------------------------------------------------------------------


class TestSuccessPathEmptyContentFallback(unittest.TestCase):
    """When _tool_loop completes with empty/placeholder content, the success
    path should fall back to digest/trace instead of returning placeholder stubs."""

    def test_empty_final_content_returns_digest(self) -> None:
        """Reviewer completes with empty content after tool exploration → digest fallback."""
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)

                # Model does tool calls then returns empty string
                client = _MainOnlyClient(main_responses=[
                    # Turn 1: tool call
                    OpenRouterCallResult(
                        content="",
                        tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}],
                        raw={},
                    ),
                    # Turn 2: empty final content (model exhausted output tokens on reasoning)
                    OpenRouterCallResult(
                        content="",
                        tool_calls=[],
                        raw={},
                    ),
                    # Turn 3: retry also empty
                    OpenRouterCallResult(
                        content="   ",
                        tool_calls=[],
                        raw={},
                    ),
                ])
                settings = _build_settings(intermittent_n=1)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                outcome = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                self.assertTrue(outcome.ok, "Should return ok=True with digest fallback")
                self.assertNotIn("*(No Summary provided by reviewer)*", outcome.markdown)
                self.assertNotIn("*(No Key Findings provided by reviewer)*", outcome.markdown)
                # Should contain evidence from the tool call
                self.assertIn("src/main.py", outcome.markdown)

        asyncio.run(scenario())

    def test_placeholder_only_content_returns_digest(self) -> None:
        """Reviewer completes with placeholder-only content → digest fallback."""
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)

                client = _MainOnlyClient(main_responses=[
                    OpenRouterCallResult(
                        content="",
                        tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/app.py\"}"}}],
                        raw={},
                    ),
                    OpenRouterCallResult(
                        content="## Summary\n*(No Summary provided by reviewer)*",
                        tool_calls=[],
                        raw={},
                    ),
                ])
                settings = _build_settings(intermittent_n=1)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                outcome = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                self.assertTrue(outcome.ok)
                self.assertNotIn("*(No Summary provided by reviewer)*", outcome.markdown)
                self.assertIn("src/app.py", outcome.markdown)

        asyncio.run(scenario())

    def test_substantive_content_is_returned_as_is(self) -> None:
        """Reviewer completes with real content → no fallback, return as-is."""
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                (repo / ".serena" / "memories").mkdir(parents=True)

                client = _MainOnlyClient(main_responses=[
                    OpenRouterCallResult(
                        content="",
                        tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}],
                        raw={},
                    ),
                    OpenRouterCallResult(
                        content="## Summary\nFound 3 bugs.\n## Key Findings\n- Bug A\n## Recommendations\n- Fix A\n## Questions / Unknowns\n- Is this safe?",
                        tool_calls=[],
                        raw={},
                    ),
                ])
                settings = _build_settings(intermittent_n=1)
                service = ReviewService(
                    repo_root=repo,
                    settings=settings,
                    openrouter_client=client,
                    models_client=_ModelsStub("test/model"),
                )
                cfg = service._prepare_reviewer_config("test/model", repo_root=repo)
                outcome = await service._run_single_reviewer(
                    cfg=cfg,
                    tool_name="code_review",
                    build_system_prompt=lambda tool_calling_enabled: "system",
                    build_user_prompt=lambda *a, **kw: "user prompt",
                    redacted_inputs={"code": "hello"},
                    requested_paths=None,
                    file_context_builder=_DummyFCB(),
                )
                self.assertTrue(outcome.ok)
                self.assertIn("Found 3 bugs", outcome.markdown)
                self.assertNotIn("src/main.py", outcome.markdown)

        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants(unittest.TestCase):
    def test_intermittent_review_timeout_seconds_is_45(self) -> None:
        self.assertEqual(INTERMITTENT_REVIEW_TIMEOUT_SECONDS, 45)


if __name__ == "__main__":
    unittest.main()
