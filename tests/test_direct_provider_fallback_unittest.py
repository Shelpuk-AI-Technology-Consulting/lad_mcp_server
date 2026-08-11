"""The OpenRouter fallback is attempted only when there is a credential for it.

Making `OPENROUTER_API_KEY` optional changed the runtime contract of the four direct
providers that predate this feature, not just the new one: a `DEEPSEEK_API_KEY`-only
deployment now reaches the fallback with nothing to fall back to. Attempting anyway
replaces the provider's real error with an authentication failure against a service
the user may not even be able to route to.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from lad_mcp_server.config import Settings
from lad_mcp_server.review_service import ReviewService


def _settings(**overrides: Any) -> Settings:
    """Build settings with no OpenRouter credential unless overridden.

    Args:
        **overrides: Fields to replace.

    Returns:
        A populated :class:`Settings`.
    """
    base = dict(
        openrouter_api_key=None,
        openrouter_primary_reviewer_model="deepseek/deepseek-v4",
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
        intermittent_review_calls=0,
    )
    base.update(overrides)
    return Settings(**base)


class _FailingClient:
    """A direct provider that is configured and reachable, but refuses the call."""

    def __init__(self, message: str) -> None:
        """Store the failure message.

        Args:
            message: Text the provider's own error carries.
        """
        self._message = message

    async def chat_completion(self, **kwargs: Any) -> Any:
        """Fail the way a misconfigured or overloaded endpoint would.

        Args:
            **kwargs: Ignored call arguments.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(self._message)


class _CountingOpenRouter:
    """Records every OpenRouter call so "none were made" is checkable."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> Any:
        """Record the call and answer.

        Args:
            **kwargs: The call arguments.

        Returns:
            A minimal successful result.
        """
        self.calls.append(kwargs)
        return type("R", (), {"content": "## Summary\nFrom OpenRouter", "tool_calls": [], "raw": {}})()


# One entry per direct provider: the credential that enables it, a model that routes
# to it, and the constructor argument that injects the stub.
_PROVIDERS = (
    ("deepseek", "deepseek_api_key", "deepseek/deepseek-v4", "deepseek_client"),
    ("ollama", "ollama_api_key", "ollama/gpt-oss:120b", "ollama_client"),
    ("zai", "zai_coding_plan_key", "z-ai/glm-5", "zai_client"),
    ("kimi", "kimi_code_api_key", "moonshotai/kimi-k2", "kimi_client"),
    ("openai_compat", "openai_compat_base_url", "litellm/gateway-model", "openai_compat_client"),
)


class TestNoFallbackWithoutACredential(unittest.TestCase):
    """FR8 holds for every direct provider, not only the newest one."""

    def test_the_providers_own_error_surfaces_and_openrouter_is_untouched(self) -> None:
        """Each provider fails on its own terms when there is no fallback available."""
        for name, credential, model, client_kwarg in _PROVIDERS:
            with self.subTest(provider=name):
                openrouter = _CountingOpenRouter()
                service = ReviewService(
                    repo_root=None,
                    settings=_settings(
                        **{credential: "configured", "openrouter_primary_reviewer_model": model}
                    ),
                    openrouter_client=openrouter,
                    models_client=None,
                    **{client_kwarg: _FailingClient(f"{name} endpoint refused the request")},
                )

                out = asyncio.run(service.code_review(code="print('hello world')", paths=None))

                self.assertIn(f"{name} endpoint refused the request", out)
                self.assertEqual(openrouter.calls, [], f"{name}: no credential, so no attempt")
                self.assertNotIn("Fell back to OpenRouter", out)
                # The provider's own error must be the *reported* failure, not a
                # footnote under a generic one. Dropping a provider's guard still
                # reaches the terminal fallthrough, which also refuses to call
                # OpenRouter — so without this the degradation is invisible.
                self.assertNotIn(
                    "No direct provider handled",
                    out,
                    f"{name}: the real diagnosis was replaced by the generic fallthrough",
                )


class TestTheFallbackStillWorksWhereItCan(unittest.TestCase):
    """FR9: a configured OpenRouter deployment keeps the behaviour it had.

    Kept on DeepSeek rather than the new provider, because the property depends on the
    prefix being a real OpenRouter vendor route: `deepseek/deepseek-v4` resolves on
    both services, so retrying there is a genuine second chance.
    """

    def test_a_deepseek_failure_is_retried_on_openrouter_under_the_same_name(self) -> None:
        openrouter = _CountingOpenRouter()

        class _Meta:
            """Minimal metadata so the OpenRouter path can build a budget."""

            supported_parameters = ("max_tokens",)

            def effective_context_length(self) -> int:
                """Return a context length large enough to validate.

                Returns:
                    A usable context length.
                """
                return 50000

            def effective_output_budget(self, fixed: int) -> int:
                """Return the output budget unchanged.

                Args:
                    fixed: The configured fixed output tokens.

                Returns:
                    The same value.
                """
                return fixed

            def supports_tools(self) -> bool:
                """Report tool support.

                Returns:
                    ``False``.
                """
                return False

        service = ReviewService(
            repo_root=None,
            settings=_settings(deepseek_api_key="configured", openrouter_api_key="sk-present"),
            openrouter_client=openrouter,
            models_client=type("M", (), {"get_model": lambda self, m: _Meta()})(),
            deepseek_client=_FailingClient("deepseek endpoint refused the request"),
        )

        out = asyncio.run(service.code_review(code="print('hello world')", paths=None))

        self.assertEqual(openrouter.calls[0]["model"], "deepseek/deepseek-v4")
        self.assertIn("Fell back to OpenRouter", out)


class TestKimiStickyFallbackHonoursTheGuard(unittest.TestCase):
    """A provider can decline to run without raising, landing on the fallthrough.

    After one failure Kimi records a 10-minute sticky fallback and skips its whole
    block on the next review. That path never enters an `except`, so guarding only the
    per-provider handlers is not enough. The flag also means the wrong thing here: it
    says "use OpenRouter instead", and there is no instead.
    """

    def _service(self, openrouter: _CountingOpenRouter) -> ReviewService:
        """Build a Kimi-only service whose Kimi endpoint always fails.

        Args:
            openrouter: The stub that records any fallback attempt.

        Returns:
            The configured service.
        """
        return ReviewService(
            repo_root=None,
            settings=_settings(
                kimi_code_api_key="configured",
                openrouter_primary_reviewer_model="moonshotai/kimi-k2",
            ),
            openrouter_client=openrouter,
            models_client=None,
            kimi_client=_FailingClient("kimi endpoint refused the request"),
        )

    def test_the_second_review_reports_the_same_thing_as_the_first(self) -> None:
        """Otherwise a Kimi-only user sees a different, worse error on every retry."""
        openrouter = _CountingOpenRouter()
        service = self._service(openrouter)

        first = asyncio.run(service.code_review(code="print('hello world')", paths=None))
        second = asyncio.run(service.code_review(code="print('hello again')", paths=None))

        self.assertIn("kimi endpoint refused the request", first)
        self.assertIn("kimi endpoint refused the request", second)
        self.assertEqual(openrouter.calls, [], "the sticky-fallback path must be guarded too")

    def test_a_call_no_provider_served_does_not_credit_openrouter(self) -> None:
        """The disclosure's provider line defaults to "openrouter"; it must not stand.

        Exercised directly against the dispatcher, because every route that reaches
        its terminal fallthrough is now closed further upstream — which is the point,
        but leaves this the only way to reach the guard without faking internal state.
        """
        openrouter = _CountingOpenRouter()
        service = ReviewService(
            repo_root=None,
            settings=_settings(),
            openrouter_client=openrouter,
            models_client=None,
        )
        provider_used = ["openrouter"]

        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(
                service._call_model_with_provider_fallback(
                    model="some/model",
                    direct_model_name=None,
                    use_zai_direct=False,
                    direct_kimi_model_name=None,
                    use_kimi_direct=False,
                    direct_deepseek_model_name=None,
                    use_deepseek_direct=False,
                    direct_ollama_model_name=None,
                    use_ollama_direct=False,
                    messages=[{"role": "user", "content": "hi"}],
                    timeout_seconds=5,
                    max_output_tokens=10,
                    tools=None,
                    preferred_tool_choice=None,
                    extra_body=None,
                    provider_used=provider_used,
                    provider_notes=[],
                )
            )

        self.assertIn("OPENROUTER_API_KEY", str(caught.exception))
        self.assertEqual(openrouter.calls, [])
        self.assertNotEqual(provider_used, ["openrouter"], "nothing served this call")


if __name__ == "__main__":
    unittest.main()
