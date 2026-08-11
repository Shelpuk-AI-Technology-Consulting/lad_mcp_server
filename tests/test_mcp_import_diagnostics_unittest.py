"""The mcp import failure must diagnose itself.

Issue #3: a user's server stopped booting when `mcp` 2.0 removed
`mcp.server.fastmcp`, and Lad told them "mcp dependency is not installed" — while
mcp *was* installed. The prescribed fix ("install dependencies") does nothing for
that cause, so the reporter had to find the real problem and the `mcp<2` workaround
themselves, then open an issue to tell us.

The root cause is pinned in `pyproject.toml`. These tests cover the other half:
that the next dependency break explains itself instead of sending someone hunting.
"""

from __future__ import annotations

import importlib.metadata
import re
import unittest
from pathlib import Path
from unittest import mock

from lad_mcp_server import server

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_REQUIREMENTS = _REPO_ROOT / "requirements.txt"


def _normalise_spec(spec: str) -> frozenset[str]:
    """Compare version specifiers by content, not by spelling.

    `>=1.2.0,<2`, `<2,>=1.2.0` and `>=1.2.0, <2` all mean the same thing; a drift
    test that fails on reordering or whitespace is friction, not signal.

    Args:
        spec: A PEP 508 version specifier such as ``>=1.2.0,<2``.

    Returns:
        The set of individual clauses, whitespace stripped.
    """
    return frozenset(clause.strip() for clause in spec.split(",") if clause.strip())


_UNDERLYING = ModuleNotFoundError("No module named 'mcp.server.fastmcp'")


class TestMcpImportFailureMessage(unittest.TestCase):
    """Each cause gets a message naming it, and a fix that works for it."""

    def test_absent_mcp_says_it_is_not_installed(self) -> None:
        """With no mcp present, the message prescribes installing dependencies."""
        with mock.patch.object(server, "_installed_mcp_version", return_value=None):
            message = server._mcp_import_failure_message(_UNDERLYING)

        self.assertIn("not installed", message)
        self.assertRegex(message, r"pip install|uv sync")
        # The mirror of the FR2 assertion below: a message that claims both "not
        # installed" and a version number is self-contradictory, and mutation testing
        # showed the suite accepted exactly that.
        self.assertNotRegex(message, r"\d+\.\d+", "must not claim a version it did not find")

    def test_incompatible_mcp_names_the_version_and_the_constraint(self) -> None:
        """With mcp 2.0 present, the message says what is wrong and how to recover.

        Pins the exact failure from issue #3: the reporter needed the installed
        version, the constraint, and the fact that a cached uvx build must be
        refreshed. None of that was in the old message.
        """
        with mock.patch.object(server, "_installed_mcp_version", return_value="2.0.0"):
            message = server._mcp_import_failure_message(_UNDERLYING)

        self.assertIn("2.0.0", message)
        self.assertIn("mcp>=1.2.0,<2", message)
        self.assertIn("--refresh", message)

    def test_incompatible_mcp_does_not_claim_it_is_missing(self) -> None:
        """The specific falsehood that misled the reporter must not reappear."""
        with mock.patch.object(server, "_installed_mcp_version", return_value="2.0.0"):
            message = server._mcp_import_failure_message(_UNDERLYING)

        self.assertNotIn("is not installed", message)

    def test_both_messages_carry_the_underlying_error(self) -> None:
        """A third cause — a real breakage inside mcp — must not be misreported."""
        for installed in (None, "2.0.0"):
            with self.subTest(installed_mcp=installed):
                with mock.patch.object(server, "_installed_mcp_version", return_value=installed):
                    message = server._mcp_import_failure_message(_UNDERLYING)

                self.assertIn("ModuleNotFoundError", message)
                self.assertIn("No module named 'mcp.server.fastmcp'", message)

    def test_quoted_constraint_matches_both_dependency_files(self) -> None:
        """The literal in the message must track the declared dependency.

        Read from the dependency files rather than from installed metadata, which is
        stale on an editable install: `importlib.metadata.requires` on this checkout
        still reports the pre-pin `mcp[cli]>=1.2.0`.

        `requirements.txt` is checked too — it is documented as mirroring
        `pyproject.toml`, and nothing else in the suite holds it to that.
        """
        with mock.patch.object(server, "_installed_mcp_version", return_value="2.0.0"):
            message = server._mcp_import_failure_message(_UNDERLYING)

        quoted = re.search(r"`mcp(?P<spec>[<>=~!][^`]*)`", message)
        self.assertIsNotNone(quoted, "message no longer quotes an mcp constraint")
        expected = _normalise_spec(quoted.group("spec"))

        for path, pattern in (
            (_PYPROJECT, r'"mcp(?:\[cli\])?\s*(?P<spec>[<>=~!][^"]*)"'),
            (_REQUIREMENTS, r'^mcp(?:\[cli\])?\s*(?P<spec>[<>=~!].*)$'),
        ):
            with self.subTest(file=path.name):
                declared = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
                self.assertIsNotNone(declared, f"{path.name} no longer declares an mcp requirement")
                self.assertEqual(
                    _normalise_spec(declared.group("spec")),
                    expected,
                    f"the constraint in the error message has drifted from {path.name}",
                )

    def test_an_mcp_below_the_minimum_is_told_to_upgrade(self) -> None:
        """A version predating FastMCP gets the upgrade path, not "you satisfy this".

        Verified against real installs: `mcp.server.fastmcp` is absent in 1.1.0 and
        present in 1.2.0, so this branch is reachable — and an earlier version of
        this file told such a user their version "satisfies Lad's requirement",
        which is both false and useless. The same class of false claim the whole
        change exists to remove.
        """
        with mock.patch.object(server, "_installed_mcp_version", return_value="1.1.0"):
            message = server._mcp_import_failure_message(_UNDERLYING)

        self.assertIn("1.1.0", message)
        self.assertIn("added in mcp 1.2.0", message)
        self.assertNotIn("satisfies", message)
        self.assertNotIn("mcp 2.0 removed", message)

    def test_version_verdict_classifies_each_side_of_both_bounds(self) -> None:
        """Both bounds are enforced, including pre-release spellings."""
        cases = {
            "0.9.0": "too_old",
            "1.1.0": "too_old",
            "1.2.0": "supported",
            "1.2rc1": "supported",
            "1.29.0": "supported",
            "2.0.0": "too_new",
            "2.0.0rc1": "too_new",
            # Unparseable defaults to the failure that actually happens.
            "not-a-version": "too_new",
        }
        for installed, expected in cases.items():
            with self.subTest(installed_mcp=installed):
                self.assertEqual(server._mcp_version_verdict(installed), expected)

    def test_a_supported_mcp_is_not_blamed_for_the_failure(self) -> None:
        """An mcp that satisfies the constraint must not be told to downgrade.

        A user on mcp 1.9 with a broken transitive dependency hits this path. Telling
        them "mcp 2.0 removed that module" would be the same wrong turn issue #3 was
        about, one scenario over.
        """
        broken_dep = ModuleNotFoundError("No module named 'anyio'")

        with mock.patch.object(server, "_installed_mcp_version", return_value="1.9.0"):
            message = server._mcp_import_failure_message(broken_dep)

        self.assertNotIn("mcp 2.0 removed", message)
        self.assertIn("satisfies", message)
        self.assertIn("anyio", message)


class TestInstalledVersionLookup(unittest.TestCase):
    """The lookup that chooses which message you get needs its own coverage.

    Every message test mocks it, so a regression here — querying the wrong
    distribution name, say — would send every user down the "not installed" branch
    with the suite still green. That is the precise falsehood this change removes.
    """

    def test_reports_the_real_installed_mcp_version(self) -> None:
        """The unmocked lookup agrees with the package metadata."""
        self.assertEqual(server._installed_mcp_version(), importlib.metadata.version("mcp"))

    def test_returns_none_when_the_package_is_absent(self) -> None:
        """An absent package yields ``None`` rather than raising."""
        with mock.patch(
            "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError
        ):
            self.assertIsNone(server._installed_mcp_version())

    def test_returns_none_when_metadata_cannot_be_read(self) -> None:
        """A filesystem error while walking sys.path must not replace the diagnostic."""
        with mock.patch("importlib.metadata.version", side_effect=OSError("unreadable")):
            self.assertIsNone(server._installed_mcp_version())


class TestImportGuardScope(unittest.TestCase):
    """The guard must not relabel unrelated failures as a dependency problem."""

    def test_a_non_import_error_propagates_unchanged(self) -> None:
        """Only `ImportError` means "the dependency is wrong".

        The original guard caught bare `Exception`, so a `SyntaxError` or a failure
        in Lad's own imports was reported as a missing mcp — the same class of
        misdirection as issue #3, one level up.
        """
        with mock.patch.object(
            server, "_import_fastmcp", side_effect=ValueError("something else entirely")
        ):
            with self.assertRaises(ValueError) as caught:
                server.create_app()

        self.assertIn("something else entirely", str(caught.exception))

    def test_an_import_error_becomes_the_diagnostic_runtime_error(self) -> None:
        """An `ImportError` is converted to the actionable message."""
        with mock.patch.object(server, "_import_fastmcp", side_effect=_UNDERLYING):
            with mock.patch.object(server, "_installed_mcp_version", return_value="2.0.0"):
                with self.assertRaises(RuntimeError) as caught:
                    server.create_app()

        self.assertIn("mcp>=1.2.0,<2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
