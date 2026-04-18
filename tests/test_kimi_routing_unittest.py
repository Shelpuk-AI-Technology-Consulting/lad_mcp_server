import asyncio
import contextlib
import unittest

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.review_service import ReviewService

if not hasattr(asyncio, "timeout"):
    @contextlib.asynccontextmanager
    async def _compat_timeout(_seconds):
        yield

    asyncio.timeout = _compat_timeout  # type: ignore[attr-defined]


class _ModelsStub:
    def __init__(self, models: dict[str, ModelMetadata]):
        self._models = models

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._models[model_id]


class _OpenRouterStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

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
        self.calls.append(
            {
                "model": model,
                "tool_choice": tool_choice,
                "tools": tools,
            }
        )
        return type("R", (), {"content": "## Summary\nOpenRouter OK", "tool_calls": [], "raw": {}})()


class _KimiStub:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

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
        self.calls.append(
            {
                "model": model,
                "tool_choice": tool_choice,
                "tools": tools,
            }
        )
        if self.fail:
            raise RuntimeError("kimi endpoint unavailable")
        return type("R", (), {"content": "## Summary\nKimi Code OK", "tool_calls": [], "raw": {}})()


class TestKimiRouting(unittest.TestCase):
    def _settings(self, *, model: str, kimi_key: str | None) -> Settings:
        return Settings(
            openrouter_api_key="test-openrouter",
            openrouter_primary_reviewer_model=model,
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
            lad_serena_max_tool_calls=4,
            lad_serena_tool_timeout_seconds=1,
            lad_serena_max_tool_result_chars=12000,
            lad_serena_max_total_chars=50000,
            lad_serena_max_dir_entries=100,
            lad_serena_max_search_results=20,
            kimi_code_api_key=kimi_key,
        )

    def _models(self) -> _ModelsStub:
        return _ModelsStub(
            {
                "moonshotai/kimi-k2.5": ModelMetadata(
                    model_id="moonshotai/kimi-k2.5",
                    context_length=50000,
                    supported_parameters=("max_tokens",),
                    provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                ),
                "other/model": ModelMetadata(
                    model_id="other/model",
                    context_length=50000,
                    supported_parameters=("max_tokens",),
                    provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
                ),
            }
        )

    def test_uses_direct_kimi_for_kimi_model_when_key_present(self) -> None:
        openrouter = _OpenRouterStub()
        kimi = _KimiStub(fail=False)
        service = ReviewService(
            settings=self._settings(model="moonshotai/kimi-k2.5", kimi_key="kimi-key"),
            openrouter_client=openrouter,
            models_client=self._models(),
            kimi_client=kimi,
        )

        out = asyncio.run(service.code_review(code="print('x')", context=None, paths=None))

        self.assertIn("Kimi Code OK", out)
        self.assertEqual(len(kimi.calls), 1)
        self.assertEqual(kimi.calls[0]["model"], "kimi-for-coding")
        self.assertEqual(len(openrouter.calls), 0)

    def test_falls_back_to_openrouter_when_direct_kimi_fails_and_adds_note(self) -> None:
        openrouter = _OpenRouterStub()
        kimi = _KimiStub(fail=True)
        service = ReviewService(
            settings=self._settings(model="moonshotai/kimi-k2.5", kimi_key="kimi-key"),
            openrouter_client=openrouter,
            models_client=self._models(),
            kimi_client=kimi,
        )

        out = asyncio.run(service.code_review(code="print('x')", context=None, paths=None))

        self.assertIn("OpenRouter OK", out)
        self.assertEqual(len(kimi.calls), 1)
        self.assertEqual(len(openrouter.calls), 1)
        self.assertIn("Kimi Code endpoint failed", out)

    def test_uses_openrouter_for_kimi_model_when_key_missing(self) -> None:
        openrouter = _OpenRouterStub()
        kimi = _KimiStub(fail=False)
        service = ReviewService(
            settings=self._settings(model="moonshotai/kimi-k2.5", kimi_key=None),
            openrouter_client=openrouter,
            models_client=self._models(),
            kimi_client=kimi,
        )

        out = asyncio.run(service.code_review(code="print('x')", context=None, paths=None))

        self.assertIn("OpenRouter OK", out)
        self.assertEqual(len(openrouter.calls), 1)
        self.assertEqual(len(kimi.calls), 0)

    def test_non_kimi_model_ignores_kimi_key(self) -> None:
        openrouter = _OpenRouterStub()
        kimi = _KimiStub(fail=False)
        service = ReviewService(
            settings=self._settings(model="other/model", kimi_key="kimi-key"),
            openrouter_client=openrouter,
            models_client=self._models(),
            kimi_client=kimi,
        )

        out = asyncio.run(service.code_review(code="print('x')", context=None, paths=None))

        self.assertIn("OpenRouter OK", out)
        self.assertEqual(len(openrouter.calls), 1)
        self.assertEqual(len(kimi.calls), 0)


if __name__ == "__main__":
    unittest.main()
