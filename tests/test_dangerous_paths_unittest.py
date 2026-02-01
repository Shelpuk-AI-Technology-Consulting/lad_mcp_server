import asyncio
import os
import unittest
from pathlib import Path

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.review_service import ReviewService
from lad_mcp_server.schemas import ValidationError


class _ModelsStub:
    def __init__(self, models: dict[str, ModelMetadata]):
        self._models = models

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._models[model_id]


class _OpenRouterNeverCalled:
    async def chat_completion(self, **kwargs):  # pragma: no cover
        raise AssertionError("OpenRouter client should not be called for invalid path inputs")


class TestDangerousPaths(unittest.TestCase):
    def test_rejects_dangerous_system_paths(self) -> None:
        primary = "moonshotai/kimi-k2-thinking"
        meta = ModelMetadata(
            model_id=primary,
            context_length=50000,
            supported_parameters=("max_tokens",),
            provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
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
            lad_serena_max_tool_calls=0,
            lad_serena_tool_timeout_seconds=1,
            lad_serena_max_tool_result_chars=12000,
            lad_serena_max_total_chars=50000,
            lad_serena_max_dir_entries=100,
            lad_serena_max_search_results=20,
        )

        service = ReviewService(
            repo_root=None,
            settings=settings,
            openrouter_client=_OpenRouterNeverCalled(),
            models_client=_ModelsStub({primary: meta}),
        )

        if os.name == "nt":
            windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
            dangerous = windir / "System32" / "drivers" / "etc" / "hosts"
        else:
            dangerous = Path("/etc/passwd")

        with self.assertRaises(ValidationError):
            asyncio.run(service.code_review(code=None, paths=[str(dangerous)]))


if __name__ == "__main__":
    unittest.main()
