"""Documentation is a checked contract against ``config.py``.

``README.md`` and ``.env.example`` describe the environment variables Lad reads and
the defaults it applies. Nothing used to verify those claims, so they drifted: the
primary model default, ``LAD_SERENA_MAX_TOTAL_CHARS``, and an entire provider
integration (Ollama Cloud) were wrong or missing while the code moved on.

Two contracts are enforced here:

* **Values** — every default stated in either document matches what
  :meth:`Settings.from_env` actually produces, and every setting that *has* a fixed
  default is stated in the README.
* **Coverage** — every environment variable ``config.py`` reads appears in both
  documents. ``LAD_ENV_FILE`` is exempt from ``.env.example``: it names an env file,
  so listing it inside one is circular.

Note the deliberate consequence for ``.env.example``: because every ``NAME=VALUE``
in it is compared against the real default, it is a *defaults mirror* and cannot
ship a recommended non-default. Recommended overrides belong in README prose.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import re
import unittest
from pathlib import Path
from typing import Iterator
from unittest import mock

from lad_mcp_server.config import Settings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PY = _REPO_ROOT / "lad_mcp_server" / "config.py"
_README = _REPO_ROOT / "README.md"
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"

# `NAME` (default: `VALUE`) — trailing prose after the value is tolerated.
_README_DEFAULT_RE = re.compile(r"`(?P<name>[A-Z][A-Z0-9_]*)`\s*\(default:\s*`(?P<value>[^`]*)`")

_ENV_READERS = {"_get_int", "_get_bool", "_get_str", "getenv"}

# Required, so it has no default to document.
_NO_DEFAULT = {"OPENROUTER_API_KEY"}
# Consumed before a Settings object exists, so it has no attribute to compare.
_NO_SETTINGS_ATTR = {"LAD_ENV_FILE"}
# Listing the env file's own name inside the env file would be circular.
_ENV_EXAMPLE_EXEMPT = {"LAD_ENV_FILE"}


@contextlib.contextmanager
def _pristine_environment() -> Iterator[None]:
    """Run with an environment holding only ``OPENROUTER_API_KEY``.

    ``Settings.from_env`` reads the live environment, an optional ``LAD_ENV_FILE``,
    and ``.env`` via python-dotenv. Any of those would make this test depend on the
    developer's machine, so the environment is cleared and dotenv neutralised.

    ``dotenv.load_dotenv`` is the correct patch target rather than
    ``lad_mcp_server.config.load_dotenv``: ``config.py`` imports it *inside* the
    function, so patching the module attribute would silently no-op and leave the
    real ``.env`` in play.

    Yields:
        None, for the duration of the isolated environment.
    """
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}, clear=True):
        # config.py treats python-dotenv as optional even though pyproject declares it.
        if importlib.util.find_spec("dotenv") is None:
            yield
        else:
            with mock.patch("dotenv.load_dotenv", lambda *args, **kwargs: None):
                yield


def _is_os_environ(node: ast.AST) -> bool:
    """Report whether an AST node refers to ``os.environ``.

    Args:
        node: Any AST node.

    Returns:
        ``True`` when the node is the ``os.environ`` attribute access.
    """
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _string_constant(node: ast.AST) -> str | None:
    """Return a node's value when it is a string literal.

    Args:
        node: Any AST node.

    Returns:
        The string value, or ``None`` when the node is not a string constant.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _env_names_read_by_config() -> set[str]:
    """Collect the environment variable names ``config.py`` reads.

    Uses ``ast`` rather than a regex because ``config.py`` wraps some name literals
    onto the line after the opening parenthesis — which is exactly where the two
    model settings that drifted live, so a naive pattern would miss them.

    Recognises all four access shapes, not just the helpers: a variable read via
    ``os.environ.get`` or ``os.environ[...]`` would otherwise be invisible to this
    contract, and could be added undocumented with the suite still green.

    Returns:
        The set of environment variable names read by the config module.
    """
    tree = ast.parse(_CONFIG_PY.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        # `os.environ["NAME"]`
        if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            name = _string_constant(node.slice)
            if name is not None:
                names.add(name)
            continue

        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        # `os.environ.get("NAME")` — checked before the bare-name helpers so an
        # unrelated `.get` on some other object cannot slip through.
        if isinstance(func, ast.Attribute) and func.attr == "get" and _is_os_environ(func.value):
            name = _string_constant(node.args[0])
            if name is not None:
                names.add(name)
            continue

        # `_get_int("NAME", ...)` / `os.getenv("NAME")`
        func_name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if func_name not in _ENV_READERS:
            continue
        name = _string_constant(node.args[0])
        if name is not None:
            names.add(name)
    return names


def _readme_documented_defaults() -> dict[str, str]:
    """Scrape ``NAME`` -> documented default from the README.

    Returns:
        Mapping of environment variable name to the default it claims.
    """
    text = _README.read_text(encoding="utf-8")
    return {m.group("name"): m.group("value") for m in _README_DEFAULT_RE.finditer(text)}


def _env_example_assignments() -> dict[str, str]:
    """Scrape ``NAME=VALUE`` from ``.env.example``, skipping empty values.

    An empty value means "unset", which *is* the default, so those entries carry no
    claim to check.

    Returns:
        Mapping of environment variable name to the value it assigns.
    """
    out: dict[str, str] = {}
    for raw in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if name and value:
            out[name] = value
    return out


_MISSING = object()


def _actual_default(settings: Settings, name: str) -> object:
    """Read the ``Settings`` attribute corresponding to an env var name.

    Args:
        settings: A ``Settings`` built from a pristine environment.
        name: The environment variable name.

    Returns:
        The default value the code applies, or a sentinel when no attribute of
        that name exists — so the caller can report which setting is unmapped
        rather than raising an opaque ``AttributeError``.
    """
    return getattr(settings, name.lower(), _MISSING)


class TestDocsMatchConfig(unittest.TestCase):
    """Enforce the documentation-to-code contract."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the defaults once and scrape both documents."""
        with _pristine_environment():
            cls.settings = Settings.from_env()
        cls.env_names = _env_names_read_by_config()
        cls.readme_defaults = _readme_documented_defaults()
        cls.env_example = _env_example_assignments()

    def test_scraper_finds_the_known_settings(self) -> None:
        """Guard the scrapers themselves — a silent zero-match would pass everything."""
        self.assertIn("OPENROUTER_PRIMARY_REVIEWER_MODEL", self.env_names)
        self.assertIn("OLLAMA_API_KEY", self.env_names)
        self.assertGreater(len(self.readme_defaults), 10)
        self.assertGreater(len(self.env_example), 10)

    def test_readme_defaults_match_code(self) -> None:
        """Every default the README states is the one the code applies."""
        for name, documented in sorted(self.readme_defaults.items()):
            if name in _NO_DEFAULT or name in _NO_SETTINGS_ATTR or name not in self.env_names:
                continue
            with self.subTest(setting=name):
                actual = _actual_default(self.settings, name)
                self.assertIsNot(actual, _MISSING, f"{name} has no matching Settings attribute")
                self.assertEqual(
                    str(actual).lower(),
                    documented.lower(),
                    f"README says {name}={documented!r}, code applies {actual!r}",
                )

    def test_env_example_values_match_code(self) -> None:
        """Every value .env.example assigns is the real default."""
        for name, documented in sorted(self.env_example.items()):
            if name in _NO_DEFAULT or name in _NO_SETTINGS_ATTR or name not in self.env_names:
                continue
            with self.subTest(setting=name):
                actual = _actual_default(self.settings, name)
                self.assertIsNot(actual, _MISSING, f"{name} has no matching Settings attribute")
                self.assertEqual(
                    str(actual).lower(),
                    documented.lower(),
                    f".env.example says {name}={documented!r}, code applies {actual!r}",
                )

    def test_every_setting_with_a_default_is_documented_in_readme(self) -> None:
        """A setting with a fixed default must state it in a recognised shape.

        Catches both an undocumented setting and one written in a shape the scraper
        cannot see — which would otherwise go silently unchecked.
        """
        for name in sorted(self.env_names):
            if name in _NO_DEFAULT or name in _NO_SETTINGS_ATTR:
                continue
            if _actual_default(self.settings, name) is None:
                continue  # optional; nothing to state
            with self.subTest(setting=name):
                self.assertIn(
                    name,
                    self.readme_defaults,
                    f"{name} has a default but the README does not state it as `{name}` (default: `...`)",
                )

    def test_every_env_var_appears_in_readme(self) -> None:
        """Every environment variable config.py reads is mentioned in the README."""
        readme_text = _README.read_text(encoding="utf-8")
        for name in sorted(self.env_names):
            with self.subTest(setting=name):
                self.assertIn(name, readme_text, f"{name} is read by config.py but absent from README.md")

    def test_every_env_var_appears_in_env_example(self) -> None:
        """Every environment variable config.py reads is present in .env.example."""
        env_text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        for name in sorted(self.env_names):
            if name in _ENV_EXAMPLE_EXEMPT:
                continue
            with self.subTest(setting=name):
                self.assertIn(name, env_text, f"{name} is read by config.py but absent from .env.example")


if __name__ == "__main__":
    unittest.main()
