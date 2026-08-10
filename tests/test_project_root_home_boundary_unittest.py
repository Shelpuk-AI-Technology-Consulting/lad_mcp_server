"""Tests for the home-directory boundary in project root inference.

Serena creates a global ``~/.serena`` config directory on every machine where it
runs. It is not a project marker, so ``_walk_up_for_project_root`` must not treat
it as one and promote the project root to the user's home directory — a location
``is_dangerous_repo_root`` rejects, which previously failed the whole review.

Every test patches :meth:`pathlib.Path.home` to a temporary directory so results
do not depend on whether the developer's real home directory happens to contain
``.serena``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadata, ProviderLimits
from lad_mcp_server.path_utils import is_dangerous_repo_root
from lad_mcp_server.review_service import ReviewService

PRIMARY_MODEL = "moonshotai/kimi-k2-thinking"


class _ModelsStub:
    """Return canned :class:`ModelMetadata` without calling the OpenRouter API."""

    def __init__(self, models: dict[str, ModelMetadata]):
        """Store the model metadata this stub will serve.

        Args:
            models: Mapping of model id to its metadata.
        """
        self._models = models

    def get_model(self, model_id: str) -> ModelMetadata:
        """Return metadata for ``model_id``.

        Args:
            model_id: The OpenRouter model identifier.

        Returns:
            The stored :class:`ModelMetadata` for that model.
        """
        return self._models[model_id]


class _OpenRouterCaptureStub:
    """Capture the user prompts a reviewer would have sent to OpenRouter."""

    def __init__(self) -> None:
        """Initialise an empty capture buffer."""
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
        """Record user-role messages and return a minimal valid review.

        Args:
            model: Model identifier (ignored).
            messages: Chat messages; user-role entries are captured.
            timeout_seconds: Request timeout (ignored).
            max_output_tokens: Output cap (ignored).
            tools: Tool schemas (ignored).
            tool_choice: Tool choice directive (ignored).
            extra_body: Extra request body fields (ignored).

        Returns:
            An object exposing ``content``, ``tool_calls`` and ``raw``.
        """
        for msg in messages:
            if msg.get("role") == "user":
                self.user_messages.append(msg.get("content", ""))
        return type("R", (), {"content": "## Summary\nOK", "tool_calls": [], "raw": {}})()


def _settings() -> Settings:
    """Build a single-reviewer Settings instance that never touches the network.

    Returns:
        A fully populated :class:`Settings` for tests.
    """
    return Settings(
        openrouter_api_key="test",
        openrouter_primary_reviewer_model=PRIMARY_MODEL,
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


class TestProjectRootHomeBoundary(unittest.TestCase):
    """Verify the walk-up never promotes the project root past the home directory."""

    def setUp(self) -> None:
        """Create a temp directory, verify it is safe, and patch it in as home."""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        # Resolve so the patched home compares equal to the resolved paths that
        # `is_dangerous_repo_root` builds internally (matters on macOS, where
        # /var/folders/... resolves to /private/var/folders/...).
        self.home = Path(td.name).resolve()

        # Guard against a TMPDIR under a blocked prefix (e.g. TMPDIR=/var/tmp in
        # containers), which would make these tests fail for an unrelated reason.
        # Checked before patching, while the temp dir is not yet "home".
        if is_dangerous_repo_root(self.home):
            self.skipTest(f"temp directory {self.home} is under a blocked prefix")

        # `_resolve_project_root` short-circuits on CODEX_WORKSPACE_ROOT before any
        # path inference, which would bypass the code under test entirely.
        env_patcher = mock.patch.dict(os.environ)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        os.environ.pop("CODEX_WORKSPACE_ROOT", None)

        home_patcher = mock.patch.object(Path, "home", return_value=self.home)
        home_patcher.start()
        self.addCleanup(home_patcher.stop)

    def test_walk_up_stops_before_home_directory(self) -> None:
        """A global ``~/.serena`` must not make the home directory the project root."""
        (self.home / ".serena").mkdir()
        start = self.home / "sub" / "dir"
        start.mkdir(parents=True)

        result = ReviewService._walk_up_for_project_root(start)

        self.assertEqual(result, start)
        self.assertNotEqual(result, self.home)

    def test_walk_up_still_finds_markers_below_home(self) -> None:
        """Both marker kinds are still honoured below the home boundary."""
        for marker in (".git", ".serena"):
            with self.subTest(marker=marker):
                repo = self.home / f"repo_{marker.lstrip('.')}"
                (repo / marker).mkdir(parents=True)
                start = repo / "src" / "pkg"
                start.mkdir(parents=True)

                self.assertEqual(ReviewService._walk_up_for_project_root(start), repo)

    def test_walk_up_returns_nearest_marker_ancestor(self) -> None:
        """The nearest marked ancestor wins over a more distant one."""
        outer = self.home / "a"
        (outer / ".git").mkdir(parents=True)
        inner = outer / "b"
        (inner / ".serena").mkdir(parents=True)
        start = inner / "c"
        start.mkdir(parents=True)

        self.assertEqual(ReviewService._walk_up_for_project_root(start), inner)

    def test_walk_up_returns_start_when_start_itself_is_dangerous(self) -> None:
        """A dangerous ``start`` is returned unchanged so the caller's guard fires.

        Deliberately plants no marker: the guard must break on the first
        iteration, rather than the result depending on marker discovery.
        """
        result = ReviewService._walk_up_for_project_root(self.home)

        self.assertEqual(result, self.home)
        self.assertTrue(is_dangerous_repo_root(result))

    def test_walk_up_stops_at_any_dangerous_boundary(self) -> None:
        """The guard is generic, not a home-directory special case.

        Pins the claim that a marker in any blocked location — a stray ``.git``
        at a drive root, for instance — is skipped for the same reason.
        """
        boundary = self.home / "boundary"
        (boundary / ".git").mkdir(parents=True)
        start = boundary / "proj" / "src"
        start.mkdir(parents=True)

        with mock.patch(
            "lad_mcp_server.review_service.is_dangerous_repo_root",
            side_effect=lambda p: p == boundary,
        ):
            self.assertEqual(ReviewService._walk_up_for_project_root(start), start)

    def test_directory_under_home_is_not_itself_dangerous(self) -> None:
        """The home check must stay exact-match, not prefix-match.

        Making it a prefix match would silently disable project root inference
        for every path under a user's home directory.
        """
        sub = self.home / "sub"
        sub.mkdir()

        self.assertTrue(is_dangerous_repo_root(self.home))
        self.assertFalse(is_dangerous_repo_root(sub))

    def test_code_review_succeeds_for_unmarked_dir_under_home(self) -> None:
        """Reviewing files in an unmarked directory under home must not fail."""
        (self.home / ".serena").mkdir()
        scratch = self.home / "scratch"
        scratch.mkdir()
        target = scratch / "a.txt"
        target.write_text("hello\n", encoding="utf-8")

        meta = ModelMetadata(
            model_id=PRIMARY_MODEL,
            context_length=50000,
            supported_parameters=("max_tokens",),
            provider_limits=ProviderLimits(context_length=50000, max_completion_tokens=2000),
        )
        capture = _OpenRouterCaptureStub()
        service = ReviewService(
            repo_root=None,
            settings=_settings(),
            openrouter_client=capture,
            models_client=_ModelsStub({PRIMARY_MODEL: meta}),
        )

        asyncio.run(service.code_review(code=None, paths=[str(target)]))

        joined = "\n".join(capture.user_messages)
        self.assertIn("--- BEGIN FILE: a.txt", joined)


if __name__ == "__main__":
    unittest.main()
