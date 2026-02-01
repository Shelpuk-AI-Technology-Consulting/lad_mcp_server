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


class _OpenRouterCaptureStub:
    def __init__(self) -> None:
        self.user_messages: list[str] = []

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
        # Capture the user prompt so the test can assert that file content was embedded.
        for msg in messages:
            if msg.get("role") == "user":
                self.user_messages.append(msg.get("content", ""))
        return type("R", (), {"content": "## Summary\nOK", "tool_calls": [], "raw": {}})()


class TestRepoRootEnv(unittest.TestCase):
    def test_lad_repo_root_is_used_when_repo_root_not_provided(self) -> None:
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
            lad_serena_max_tool_calls=0,
            lad_serena_tool_timeout_seconds=1,
            lad_serena_max_tool_result_chars=12000,
            lad_serena_max_total_chars=50000,
            lad_serena_max_dir_entries=100,
            lad_serena_max_search_results=20,
        )

        with tempfile.TemporaryDirectory() as repo_td, tempfile.TemporaryDirectory() as other_cwd_td:
            repo = Path(repo_td)
            (repo / "hello.js").write_text("console.log('hello');\n", encoding="utf-8")

            capture = _OpenRouterCaptureStub()
            models = _ModelsStub({primary: meta})

            prev_cwd = Path.cwd()
            prev_env = os.environ.get("LAD_REPO_ROOT")
            os.environ["LAD_REPO_ROOT"] = str(repo)
            try:
                os.chdir(other_cwd_td)
                service = ReviewService(
                    repo_root=None,
                    settings=settings,
                    openrouter_client=capture,
                    models_client=models,
                )
                out = asyncio.run(
                    service.code_review(
                        code=None,
                        paths=["hello.js"],
                        language=None,
                        focus=None,
                        model=None,
                    )
                )
            finally:
                os.chdir(prev_cwd)
                if prev_env is None:
                    os.environ.pop("LAD_REPO_ROOT", None)
                else:
                    os.environ["LAD_REPO_ROOT"] = prev_env

        self.assertIn("## Primary Reviewer", out)
        self.assertTrue(capture.user_messages)
        joined = "\n".join(capture.user_messages)
        self.assertIn("--- BEGIN FILE: hello.js", joined)


if __name__ == "__main__":
    unittest.main()

