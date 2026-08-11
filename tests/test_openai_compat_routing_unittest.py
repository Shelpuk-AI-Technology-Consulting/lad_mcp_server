"""Routing, credential requirements and fallback for the OpenAI-compatible provider.

Issue #2's user cannot reach `openrouter.ai` at all, so the properties that matter
are observed at the `ReviewService` boundary rather than read from the code: does the
call actually go to the gateway, is OpenRouter genuinely never touched, and does a
failure surface as itself rather than as an authentication error from a fallback that
could never work.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadataError
from lad_mcp_server.review_service import ReviewService

_GATEWAY = "https://gateway.internal.example.com/v1"


def _settings(**overrides: Any) -> Settings:
    """Build settings for a gateway-only deployment unless overridden.

    Args:
        **overrides: Fields to replace.

    Returns:
        A populated :class:`Settings`.
    """
    base = dict(
        openrouter_api_key=None,
        openrouter_primary_reviewer_model="litellm/gateway-model",
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
        lad_serena_max_tool_calls=0,
        lad_serena_tool_timeout_seconds=1,
        lad_serena_max_tool_result_chars=12000,
        lad_serena_max_total_chars=50000,
        lad_serena_max_dir_entries=100,
        lad_serena_max_search_results=20,
        openai_compat_base_url=_GATEWAY,
        openai_compat_api_key="gateway-key",
        intermittent_review_calls=0,
    )
    base.update(overrides)
    return Settings(**base)


class _RecordingClient:
    """Records calls and returns a minimal successful review."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        """Configure the stub.

        Args:
            fail_with: When set, every call raises this instead of answering.
        """
        self.calls: list[dict[str, Any]] = []
        self._fail_with = fail_with

    async def chat_completion(self, **kwargs: Any) -> Any:
        """Record the call and answer or raise.

        Args:
            **kwargs: The call arguments.

        Returns:
            An object exposing ``content``, ``tool_calls`` and ``raw``.

        Raises:
            Exception: The configured failure, when one was given.
        """
        self.calls.append(kwargs)
        if self._fail_with is not None:
            raise self._fail_with
        return type("R", (), {"content": "## Summary\nOK", "tool_calls": [], "raw": {}})()


class _ExplodingModelsClient:
    """Fails if the OpenRouter models API is consulted at all."""

    def get_model(self, model_id: str) -> Any:
        """Fail loudly.

        Args:
            model_id: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError(f"OpenRouter models API must not be queried for {model_id!r}")


class TestRoutingToTheGateway(unittest.TestCase):
    """A prefixed model reaches the gateway and nothing else."""

    def test_prefixed_model_goes_to_the_gateway_not_openrouter(self) -> None:
        """The gateway client is called; OpenRouter and its metadata API are not."""
        gateway = _RecordingClient()
        openrouter = _RecordingClient()
        service = ReviewService(
            repo_root=None,
            settings=_settings(),
            openrouter_client=openrouter,
            models_client=_ExplodingModelsClient(),
            openai_compat_client=gateway,
        )

        out = asyncio.run(service.code_review(code="print('hello world')", paths=None))

        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0]["model"], "gateway-model", "prefix should be stripped")
        self.assertEqual(openrouter.calls, [], "OpenRouter must not be called")
        self.assertIn("*Provider: `openai_compat`*", out, "the disclosure must name who served it")

    def test_openai_compat_prefix_routes_identically(self) -> None:
        """The provider-neutral prefix behaves the same as `litellm/`."""
        gateway = _RecordingClient()
        service = ReviewService(
            repo_root=None,
            settings=_settings(openrouter_primary_reviewer_model="openai_compat/gateway-model"),
            openrouter_client=_RecordingClient(),
            models_client=_ExplodingModelsClient(),
            openai_compat_client=gateway,
        )

        asyncio.run(service.code_review(code="print('hello world')", paths=None))

        self.assertEqual(gateway.calls[0]["model"], "gateway-model")


class TestFallbackRequiresSomethingToFallBackTo(unittest.TestCase):
    """The fallback is skipped when it has no credential — or nothing to route to."""

    def test_without_an_openrouter_key_the_real_error_surfaces(self) -> None:
        """The gateway's own failure is reported, and OpenRouter is never attempted."""
        gateway = _RecordingClient(fail_with=RuntimeError("gateway refused the request"))
        openrouter = _RecordingClient()
        service = ReviewService(
            repo_root=None,
            settings=_settings(),
            openrouter_client=openrouter,
            models_client=_ExplodingModelsClient(),
            openai_compat_client=gateway,
        )

        out = asyncio.run(service.code_review(code="print('hello world')", paths=None))

        self.assertIn("gateway refused the request", out)
        self.assertEqual(openrouter.calls, [], "nothing to fall back to, so no attempt")
        self.assertNotIn("Fell back to OpenRouter", out)
        self.assertNotIn(
            "*Provider: `openrouter`*", out, "must not credit a provider that was never called"
        )

    def test_the_gateway_is_never_retried_on_openrouter_even_with_a_key(self) -> None:
        """A gateway model name has no OpenRouter equivalent, so the retry cannot work.

        The other providers' prefixes double as real OpenRouter vendor routes, so
        `deepseek/deepseek-v4` resolves on both. `litellm/` does not exist on
        OpenRouter, and the name after it means whatever the operator's gateway says —
        so retrying there replaces the gateway's real error with "model not found".
        """
        gateway = _RecordingClient(fail_with=RuntimeError("gateway refused the request"))
        openrouter = _RecordingClient()
        service = ReviewService(
            repo_root=None,
            settings=_settings(openrouter_api_key="sk-present"),
            openrouter_client=openrouter,
            models_client=_ExplodingModelsClient(),
            openai_compat_client=gateway,
        )

        out = asyncio.run(service.code_review(code="print('hello world')", paths=None))

        self.assertEqual(openrouter.calls, [], "a gateway-local name must not be sent to OpenRouter")
        self.assertIn("gateway refused the request", out)
        self.assertIn("Not retried on OpenRouter", out, "the disclosure must say why")
        self.assertNotIn("Fell back to OpenRouter", out)


class TestOpenRouterBoundModelWithoutAKey(unittest.TestCase):
    """A model that needs OpenRouter fails closed, before any network call.

    The raised message is what the user reads: `server.py` turns it into the tool's
    fatal-error output verbatim.
    """

    def test_it_names_the_missing_variable_and_queries_nothing(self) -> None:
        """The message must be actionable; the models API must not be consulted."""
        service = ReviewService(
            repo_root=None,
            settings=_settings(openrouter_primary_reviewer_model="google/gemma-4-31b-it"),
            openrouter_client=_RecordingClient(),
            models_client=_ExplodingModelsClient(),
            openai_compat_client=_RecordingClient(),
        )

        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(service.code_review(code="print('hello world')", paths=None))

        message = str(caught.exception)
        self.assertIn("OPENROUTER_API_KEY", message)
        self.assertIn("google/gemma-4-31b-it", message)
        self.assertIn("OPENAI_COMPAT_BASE_URL", message, "should say how to route it instead")

    def test_a_forgotten_secondary_model_says_which_reviewer_is_at_fault(self) -> None:
        """The most likely misconfiguration: primary routed, secondary left at default.

        The default secondary is an OpenRouter model, so a gateway-only user who sets
        only the primary hits this. The message has to name the model that is at
        fault, or they will look at the one they already configured.
        """
        service = ReviewService(
            repo_root=None,
            settings=_settings(openrouter_secondary_reviewer_model="minimax/minimax-m2.7"),
            openrouter_client=_RecordingClient(),
            models_client=_ExplodingModelsClient(),
            openai_compat_client=_RecordingClient(),
        )

        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(service.code_review(code="print('hello world')", paths=None))

        message = str(caught.exception)
        self.assertIn("minimax/minimax-m2.7", message)
        self.assertIn("OPENROUTER_API_KEY", message)
        # Naming the role, not just the model: both reviewers can be configured to the
        # same model string, and "which one do I fix" is the whole question here.
        self.assertIn("secondary reviewer", message)


class TestMisconfigurationDiagnostics(unittest.TestCase):
    """A prefix with no endpoint configured must not blame OpenRouter."""

    def test_prefix_without_a_base_url_warns_about_the_missing_variable(self) -> None:
        """Otherwise the only clue is "not found in OpenRouter models list"."""

        class _NotFound:
            """Reproduces what OpenRouter says about an unknown model id."""

            def get_model(self, model_id: str) -> Any:
                """Reject the lookup the way the real metadata client would.

                Args:
                    model_id: The model being looked up.

                Raises:
                    ModelMetadataError: Always.
                """
                raise ModelMetadataError(f"Model '{model_id}' not found in OpenRouter models list")

        service = ReviewService(
            repo_root=None,
            settings=_settings(openai_compat_base_url=None, openrouter_api_key="sk-present"),
            openrouter_client=_RecordingClient(),
            models_client=_NotFound(),
        )

        with self.assertLogs("lad_mcp_server.review_service", level="WARNING") as logs:
            with self.assertRaises(RuntimeError):
                asyncio.run(service.code_review(code="print('hello world')", paths=None))

        self.assertTrue(
            any("OPENAI_COMPAT_BASE_URL" in line for line in logs.output),
            f"expected a diagnostic naming the missing variable, got: {logs.output}",
        )


if __name__ == "__main__":
    unittest.main()
