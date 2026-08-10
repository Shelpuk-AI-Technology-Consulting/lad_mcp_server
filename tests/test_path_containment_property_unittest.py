"""No path a model supplies may read outside the repository.

`safe_resolve_under_repo` confines every file access Lad performs — the embedded
file context and every path-taking Serena tool. Before this file it had **no direct
test coverage at all**; `tests/test_dangerous_paths_unittest.py` exercises
`is_dangerous_repo_root`, a different function, and never reaches the containment
logic.

The paths reaching it are attacker-influenced in the relevant sense: they come from
a reviewer LLM's tool arguments, which are in turn shaped by whatever is in the
repository under review.

The property is a disjunction, because both outcomes are correct: either the call
raises `ValueError`, or it returns a path inside the repo root. What must never
happen is a returned path that escapes.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from lad_mcp_server.path_utils import safe_resolve_under_repo

# A merge gate: the corpus is fixed so CI cannot go red on a fresh draw unrelated to
# a code change. See the note in test_redaction_property_unittest.py.
#
# `deadline=None` here, unlike the redaction properties, and the distinction matters.
# The cost is `Path.resolve()` making OS syscalls on pathological names — a Windows
# UNC path such as `\\server\share` triggers a network lookup that measured 2.3s —
# not a super-linear algorithm of ours. There is nothing in-process to fix, and in
# production `LAD_SERENA_TOOL_TIMEOUT_SECONDS` bounds it. The redaction properties
# keep their deadline precisely because a slow run *there* did indicate a real bug.
_SECURITY = settings(
    max_examples=300,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

# Segments deliberately include traversal and separator forms rather than arbitrary
# text: `..` is the whole point, and a generator of random unicode would spend its
# budget on strings that never approach the boundary.
_SEGMENTS = st.sampled_from(
    [
        "..", ".", "src", "a", "nested", "..%2f", "....//", "~", "$HOME",
        "sub dir", "uni\u00e7ode", ".git", "..\\", "../", "%2e%2e",
    ]
)

_RELATIVE_PATHS = st.lists(_SEGMENTS, min_size=1, max_size=6).map("/".join)

_ABSOLUTE_PATHS = st.sampled_from(
    [
        "/etc/passwd",
        "/",
        "//server/share/file",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "C:/Windows",
        "\\\\server\\share",
        "D:relative",
    ]
)


class TestPathsNeverEscapeTheRepo(unittest.TestCase):
    """`safe_resolve_under_repo` either contains the path or refuses it."""

    def setUp(self) -> None:
        """Create a repo root with some real structure to resolve against."""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.repo = Path(td.name).resolve()
        (self.repo / "src" / "nested").mkdir(parents=True)
        (self.repo / "src" / "a").write_text("x", encoding="utf-8")

    def assert_contained_or_rejected(self, path_str: str) -> None:
        """Assert the single security property for one input.

        Args:
            path_str: The candidate path.
        """
        try:
            resolved = safe_resolve_under_repo(repo_root=self.repo, path_str=path_str)
        except ValueError:
            return  # Refusing is always an acceptable outcome.

        # The only other acceptable outcome: the result is inside the repo root.
        try:
            resolved.relative_to(self.repo)
        except ValueError:  # pragma: no cover - this is the failure being guarded
            self.fail(f"{path_str!r} resolved to {resolved!r}, outside {self.repo!r}")

    @given(path_str=_RELATIVE_PATHS)
    @settings(_SECURITY)
    @example(path_str="../../../../../../etc/passwd")
    @example(path_str="src/../../..")
    @example(path_str="..")
    def test_relative_paths_never_escape(self, path_str: str) -> None:
        """No sequence of traversal segments escapes the repo root."""
        self.assert_contained_or_rejected(path_str)

    @given(path_str=_ABSOLUTE_PATHS)
    @settings(_SECURITY)
    def test_absolute_paths_never_escape(self, path_str: str) -> None:
        """An absolute path outside the repo is refused, never returned."""
        self.assert_contained_or_rejected(path_str)

    @given(path_str=st.text(max_size=60))
    @settings(_SECURITY)
    def test_arbitrary_text_never_escapes(self, path_str: str) -> None:
        """Arbitrary text is contained or refused — never an escape, never a crash.

        `ValueError` is the documented refusal. Anything else escaping the function
        would surface as an unhandled error inside a review.
        """
        try:
            self.assert_contained_or_rejected(path_str)
        except (ValueError, AssertionError):
            raise
        except OSError:
            # A malformed name (NUL bytes, over-long components) is rejected by the
            # OS during resolution. Acceptable: nothing is read.
            pass

    def test_repo_root_itself_is_contained(self) -> None:
        """The repo root resolves to itself rather than being refused."""
        resolved = safe_resolve_under_repo(repo_root=self.repo, path_str=".")

        self.assertEqual(resolved, self.repo)

    @given(path_str=st.sampled_from(["src", "src/nested", "src/a", "."]))
    @settings(_SECURITY)
    def test_resolution_is_idempotent_for_contained_paths(self, path_str: str) -> None:
        """Re-resolving the function's own output is a no-op.

        Pins the *shape* of the output — already absolute, already normalised — rather
        than adding containment strength.
        """
        once = safe_resolve_under_repo(repo_root=self.repo, path_str=path_str)

        twice = safe_resolve_under_repo(repo_root=self.repo, path_str=str(once))

        self.assertEqual(twice, once)

    @unittest.skipIf(os.name == "nt", "POSIX-only: Windows resolves drive-letter paths natively")
    def test_windows_absolute_paths_are_refused_on_posix(self) -> None:
        """A Windows-style absolute path is refused rather than treated as relative.

        On POSIX `C:\\Windows` is a legal *filename*, so without the explicit guard it
        would resolve to `<repo>/C:\\Windows` and be reported as contained.
        """
        with self.assertRaises(ValueError):
            safe_resolve_under_repo(repo_root=self.repo, path_str="C:\\Windows\\System32")


if __name__ == "__main__":
    unittest.main()
