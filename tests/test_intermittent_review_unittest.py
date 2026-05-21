from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.openrouter_client import OpenRouterClientError
from lad_mcp_server.review_service import (
    INTERMITTENT_REVIEW_TIMEOUT_SECONDS,
    IntermittentReviewState,
    ReviewerOutcome,
    ReviewService,
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

    def test_settings_intermittent_review_calls_default_is_5(self) -> None:
        with mock.patch.dict(os.environ, self._required_env(), clear=True):
            s = Settings.from_env()
        self.assertEqual(s.intermittent_review_calls, 5)

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
                            return _final_response("## Summary\nIntermittent snapshot")
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
                state.latest_markdown = "## Summary\nIntermittent snapshot"
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
                self.assertIn("Intermittent snapshot", outcome.markdown)
                self.assertIn("intermittent", (outcome.provider_note or "").lower())
                hang.set()

        asyncio.run(scenario())

    def test_timeout_without_snapshot_falls_back_to_error_stub(self) -> None:
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
                self.assertFalse(outcome.ok)
                self.assertFalse(outcome.is_intermittent)
                self.assertIn("Reviewer Error", outcome.markdown)
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
# Constants
# ---------------------------------------------------------------------------


class TestConstants(unittest.TestCase):
    def test_intermittent_review_timeout_seconds_is_60(self) -> None:
        self.assertEqual(INTERMITTENT_REVIEW_TIMEOUT_SECONDS, 60)


if __name__ == "__main__":
    unittest.main()
