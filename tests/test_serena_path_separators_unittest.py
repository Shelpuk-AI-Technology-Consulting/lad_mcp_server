"""Repo-relative paths from :class:`SerenaContext` are POSIX identifiers.

Every path the Serena bridge hands back reaches a reviewer LLM — in tool results,
and in the disclosure block via ``used_paths``. ``FileContextBuilder`` already
normalises with ``.as_posix()`` (``file_context.py:137``), so without the same
treatment here a model sees ``src\\a.txt`` and ``src/a.txt`` in one prompt.

The invariant these tests pin: **no absolute paths, and no backslashes in a
repo-relative path, on any platform.**

Assertions check for the *absence* of a backslash rather than only the presence of
the POSIX form, so they stay meaningful on POSIX — where they cannot fail today, but
still guard the contract against a future regression.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lad_mcp_server.review_service import ExplorationDigest, _extract_path_from_arguments, _update_exploration_digest
from lad_mcp_server.serena_bridge import SerenaContext, SerenaLimits

_LIMITS = SerenaLimits(
    max_dir_entries=50,
    max_search_results=20,
    max_tool_result_chars=100_000,
    max_total_chars=500_000,
    tool_timeout_seconds=5,
)


def _tool_result(raw_output: str) -> dict:
    """Unwrap the inner tool payload from a ``call_tool`` envelope.

    Args:
        raw_output: The JSON envelope returned by :meth:`SerenaContext.call_tool`.

    Returns:
        The decoded ``tool_result_json`` object.
    """
    envelope = json.loads(raw_output)
    return json.loads(envelope["tool_result_json"])


class TestSerenaPathSeparators(unittest.TestCase):
    """Verify every path-bearing Serena tool emits POSIX repo-relative paths."""

    def setUp(self) -> None:
        """Build a temp repo containing a nested file and activate the bridge."""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.repo = Path(td.name).resolve()
        (self.repo / ".serena").mkdir()
        (self.repo / "src" / "pkg").mkdir(parents=True)
        self.target = self.repo / "src" / "pkg" / "a.txt"
        self.target.write_text("alpha needle beta\nsecond line\n", encoding="utf-8")

        ctx = SerenaContext.detect(self.repo, _LIMITS)
        assert ctx is not None
        ctx.call_tool("activate_project", json.dumps({"project": "."}))
        self.ctx = ctx

    def assert_posix_relative(self, value: str) -> None:
        """Assert an emitted path string is backslash-free and not absolute.

        Args:
            value: A path or a ``path:line:text`` match string.
        """
        self.assertNotIn("\\", value)
        head = value.split(":")[0]
        self.assertFalse(Path(head).is_absolute())
        # `Path("D").is_absolute()` is False, so the drive-letter form needs its own
        # check — that shape is exactly what the ripgrep parsing bug produced.
        self.assertIsNone(re.match(r"^[A-Za-z]:", value))

    def test_list_dir_path_is_posix(self) -> None:
        """`list_dir` reports a nested directory with forward slashes."""
        result = _tool_result(self.ctx.call_tool("list_dir", json.dumps({"path": "src/pkg"})))

        self.assertEqual(result["path"], "src/pkg")
        self.assert_posix_relative(result["path"])

    def test_read_file_path_is_posix(self) -> None:
        """`read_file` reports a nested file with forward slashes."""
        result = _tool_result(self.ctx.call_tool("read_file", json.dumps({"path": "src/pkg/a.txt"})))

        self.assertEqual(result["path"], "src/pkg/a.txt")

    def test_read_file_window_path_is_posix(self) -> None:
        """`read_file_window` reports a nested file with forward slashes."""
        args = json.dumps({"path": "src/pkg/a.txt", "start_line": 1, "num_lines": 1})
        result = _tool_result(self.ctx.call_tool("read_file_window", args))

        self.assertEqual(result["path"], "src/pkg/a.txt")

    def test_search_substring_in_file_path_is_posix(self) -> None:
        """`search_substring_in_file` reports a nested file with forward slashes."""
        args = json.dumps({"path": "src/pkg/a.txt", "substring": "needle"})
        result = _tool_result(self.ctx.call_tool("search_substring_in_file", args))

        self.assertEqual(result["path"], "src/pkg/a.txt")

    def test_search_for_pattern_python_fallback_paths_are_posix(self) -> None:
        """The pure-Python search branch emits POSIX match prefixes."""
        with mock.patch("lad_mcp_server.serena_bridge.subprocess.run", side_effect=FileNotFoundError()):
            result = _tool_result(self.ctx.call_tool("search_for_pattern", json.dumps({"pattern": "needle"})))

        self.assertTrue(result["matches"], "expected the fallback search to find the needle")
        for match in result["matches"]:
            self.assert_posix_relative(match)
        self.assertTrue(any(m.startswith("src/pkg/a.txt:") for m in result["matches"]))

    def test_search_for_pattern_ripgrep_paths_are_posix(self) -> None:
        """The ripgrep branch normalises the absolute paths rg prints.

        Guards the drive-letter defect: `line.split(":", 1)[0]` on a Windows
        absolute path yields ``"D"``, which used to be recorded as a repo path.
        """
        rg_stdout = f"{self.target}:1:alpha needle beta\n"
        fake = mock.Mock(stdout=rg_stdout, returncode=0)

        with mock.patch("lad_mcp_server.serena_bridge.subprocess.run", return_value=fake):
            result = _tool_result(self.ctx.call_tool("search_for_pattern", json.dumps({"pattern": "needle"})))

        self.assertEqual(result["matches"], ["src/pkg/a.txt:1:alpha needle beta"])
        self.assertEqual(self.ctx.used_paths, {"src/pkg/a.txt"})

    def test_search_for_pattern_handles_colons_in_matched_text(self) -> None:
        """A `:<digits>:` inside the matched source line does not confuse the parser.

        Pins the non-greedy quantifier in ``_MATCH_LINE_RE``. Source lines contain
        such sequences routinely (timestamps, slice syntax), and a greedy pattern
        swallows them into the captured path.
        """
        rg_stdout = f'{self.target}:1:cfg = {{"t":12:}}\n'
        fake = mock.Mock(stdout=rg_stdout, returncode=0)

        with mock.patch("lad_mcp_server.serena_bridge.subprocess.run", return_value=fake):
            result = _tool_result(self.ctx.call_tool("search_for_pattern", json.dumps({"pattern": "cfg"})))

        self.assertEqual(result["matches"], ['src/pkg/a.txt:1:cfg = {"t":12:}'])
        self.assertEqual(self.ctx.used_paths, {"src/pkg/a.txt"})

    def test_ripgrep_is_invoked_from_the_repo_root_with_a_relative_target(self) -> None:
        """rg runs with cwd=repo root and a repo-relative target, so it prints relative paths.

        The parser tolerates absolute output, so this is defence in depth — but
        nothing else would notice a regression back to an absolute search target.
        """
        fake = mock.Mock(stdout="", returncode=1)

        with mock.patch("lad_mcp_server.serena_bridge.subprocess.run", return_value=fake) as run:
            self.ctx.call_tool("search_for_pattern", json.dumps({"pattern": "needle", "path": "src/pkg"}))

        argv, kwargs = run.call_args[0][0], run.call_args[1]
        self.assertEqual(kwargs["cwd"], str(self.repo))
        self.assertEqual(argv[-1], "src/pkg")
        self.assertFalse(Path(argv[-1]).is_absolute())

    def test_search_for_pattern_drops_matches_outside_the_repo(self) -> None:
        """A match rg reports outside the repo root is dropped, not emitted raw."""
        with tempfile.TemporaryDirectory() as outside_td:
            stray = Path(outside_td).resolve() / "elsewhere.txt"
            fake = mock.Mock(stdout=f"{stray}:1:alpha needle beta\n", returncode=0)

            with mock.patch("lad_mcp_server.serena_bridge.subprocess.run", return_value=fake):
                result = _tool_result(self.ctx.call_tool("search_for_pattern", json.dumps({"pattern": "needle"})))

        self.assertEqual(result["matches"], [])
        self.assertEqual(self.ctx.used_paths, set())

    def test_used_paths_are_posix_and_relative(self) -> None:
        """Every accumulated `used_paths` entry is repo-relative and POSIX."""
        self.ctx.call_tool("list_dir", json.dumps({"path": "src/pkg"}))
        self.ctx.call_tool("read_file", json.dumps({"path": "src/pkg/a.txt"}))
        self.ctx.call_tool(
            "read_file_window",
            json.dumps({"path": "src/pkg/a.txt", "start_line": 1, "num_lines": 1}),
        )
        self.ctx.call_tool(
            "search_substring_in_file",
            json.dumps({"path": "src/pkg/a.txt", "substring": "needle"}),
        )

        self.assertTrue(self.ctx.used_paths)
        for entry in self.ctx.used_paths:
            self.assert_posix_relative(entry)


class TestExplorationDigestPathNormalization(unittest.TestCase):
    """The digest treats tool *arguments* as POSIX identifiers too.

    Tool results are normalised by the bridge, but `_extract_path_from_arguments`
    reads whatever the model wrote. Without matching normalisation the same file
    lands in ``paths_visited`` twice and inflates the reported count.
    """

    def test_windows_style_argument_is_normalized(self) -> None:
        """A backslash path in tool arguments is read as its POSIX equivalent."""
        self.assertEqual(
            _extract_path_from_arguments(json.dumps({"path": r"src\pkg\a.txt"})),
            "src/pkg/a.txt",
        )

    def test_same_file_via_result_and_argument_counts_once(self) -> None:
        """A result path and a backslash argument path collapse to one entry."""
        digest = ExplorationDigest()

        _update_exploration_digest(
            digest,
            "read_file",
            json.dumps({"path": "src/pkg/a.txt"}),
            json.dumps({"tool_result_json": json.dumps({"path": "src/pkg/a.txt"})}),
            False,
        )
        _update_exploration_digest(
            digest,
            "find_symbol",
            json.dumps({"name": "thing", "path": r"src\pkg\a.txt"}),
            json.dumps({"tool_result_json": json.dumps({"symbols": []})}),
            False,
        )

        self.assertEqual(digest.paths_visited, {"src/pkg/a.txt"})


if __name__ == "__main__":
    unittest.main()
