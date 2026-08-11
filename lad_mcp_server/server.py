from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from lad_mcp_server.errors import format_fatal_error, format_validation_error
from lad_mcp_server.review_service import ReviewService
from lad_mcp_server.schemas import ValidationError


# Kept as a literal rather than read from installed metadata. This text is produced
# on an error path — it runs when the environment is already broken, so it must hold
# the least logic that could itself fail. `importlib.metadata.requires` is also
# unreliable here: on an editable install it reports whatever was declared when the
# package was installed, which on a stale checkout is the pre-pin constraint.
# `tests/test_mcp_import_diagnostics_unittest.py` asserts this tracks pyproject.toml.
_REQUIRED_MCP_SPEC = ">=1.2.0,<2"


def _installed_mcp_version() -> str | None:
    """Return the installed ``mcp`` version, or ``None`` if it cannot be determined.

    ``None`` means "unknown", **not** "absent" — corrupt or unreadable metadata
    lands here too. Use :func:`_mcp_is_importable` to tell those apart; conflating
    them is how the message ends up claiming a package is missing when it is not.

    Returns:
        The version string from the package metadata, or ``None``.
    """
    try:
        from importlib.metadata import version

        return version("mcp")
    except Exception:
        return None


def _mcp_is_importable() -> bool:
    """Report whether the ``mcp`` package exists, independent of its metadata.

    Deliberately does not consult ``importlib.metadata``: the point is to answer
    "is it there?" when the metadata is exactly what cannot be trusted. ``find_spec``
    locates the module without executing it, so a package that raises on import is
    still correctly reported as present.

    Returns:
        ``True`` when the package can be located on the import path.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("mcp") is not None
    except Exception:
        return False


def _import_fastmcp() -> Any:
    """Import :class:`FastMCP`, kept separate so the failure path can be tested.

    Returns:
        The ``FastMCP`` class.

    Raises:
        ImportError: When ``mcp.server.fastmcp`` is unavailable.
    """
    from mcp.server.fastmcp import FastMCP

    return FastMCP


def _leading_int(part: str) -> int | None:
    """Read the leading integer of a version component.

    Tolerates pre-release suffixes, so ``"0rc1"`` reads as ``0``. Without this a
    version like ``1.2rc1`` would fail to parse and be misreported.

    Args:
        part: One dot-separated component of a version string.

    Returns:
        The leading integer, or ``None`` when the component does not start with one.
    """
    digits = ""
    for char in part:
        if not char.isdigit():
            break
        digits += char
    return int(digits) if digits else None


def _mcp_version_verdict(installed: str) -> str:
    """Classify an installed ``mcp`` against Lad's requirement.

    Both bounds matter, and both were verified against real installs:
    ``mcp.server.fastmcp`` is absent in 1.1.0, present in 1.2.0, and removed again
    in 2.0.0. So ``>=1.2.0,<2`` is exactly the window in which the module exists —
    the lower bound is load-bearing, not caution.

    Args:
        installed: The version string from the package metadata.

    Returns:
        ``"too_new"``, ``"too_old"`` or ``"supported"``. An unparseable version is
        reported as ``"too_new"`` because that is the failure that actually happens;
        the underlying error is printed in every case regardless.
    """
    parts = installed.split(".")
    major = _leading_int(parts[0]) if parts else None
    if major is None:
        return "too_new"
    minor = (_leading_int(parts[1]) if len(parts) > 1 else 0) or 0

    if major >= 2:
        return "too_new"
    if (major, minor) < (1, 2):
        return "too_old"
    return "supported"


def _mcp_import_failure_message(exc: BaseException) -> str:
    """Explain why ``mcp.server.fastmcp`` could not be imported.

    Distinguishes "the dependency is missing" from "the dependency is installed but
    incompatible", because they need different fixes. Reporting the second as the
    first is what made issue #3 a support ticket: mcp 2.0 removed this module, Lad
    said "mcp dependency is not installed", and the prescribed fix did nothing.

    Args:
        exc: The import error that was caught.

    Returns:
        A message naming the cause, the fix, and the underlying error.
    """
    underlying = f"Underlying import error: {type(exc).__name__}: {exc}"
    installed = _installed_mcp_version()

    # Empty as well as None: corrupt metadata can yield a blank version, and claiming
    # a nameless version is worse than saying nothing.
    if not installed:
        if _mcp_is_importable():
            # Present, but its version is unknowable. Saying "not installed" here
            # would be the same false claim this message exists to remove, reached
            # by a different route.
            return (
                "The `mcp` package is present but its version could not be read, so "
                f"its compatibility cannot be checked. Lad requires `mcp{_REQUIRED_MCP_SPEC}`; "
                "reinstall it if the install looks incomplete.\n"
                f"{underlying}"
            )
        return (
            "The `mcp` package is not installed. Install Lad's dependencies with "
            "`uv sync` or `pip install -e .`.\n"
            f"{underlying}"
        )

    verdict = _mcp_version_verdict(installed)
    if verdict == "too_new":
        # `--with mcp<2` is unquoted deliberately: the README's client configs pass
        # args as JSON arrays, where quotes would be taken literally and uv rejects
        # the result. They are only needed in a POSIX shell, where bare `<2` redirects.
        cause = (
            f"Lad requires `mcp{_REQUIRED_MCP_SPEC}`: mcp 2.0 removed that module.\n"
            "If you launch Lad with uvx, add `--refresh` so the pinned dependency is "
            "re-resolved (a cached build keeps the old one). Otherwise reinstall Lad, "
            "or pin it at the call site with `--with mcp<2` (quote that argument in a "
            "shell)."
        )
    elif verdict == "too_old":
        cause = (
            f"Lad requires `mcp{_REQUIRED_MCP_SPEC}`: `mcp.server.fastmcp` was only "
            "added in mcp 1.2.0. Upgrade mcp, or reinstall Lad so its pinned "
            "dependency is applied."
        )
    else:
        # Do not blame a version that is fine. Telling this user to change it would be
        # the same wrong turn issue #3 was about, one scenario over.
        cause = (
            f"That satisfies Lad's requirement (`mcp{_REQUIRED_MCP_SPEC}`), so the "
            "import failed for some other reason - see the underlying error below."
        )

    # Deliberately ASCII-only. This is printed while the environment is already
    # broken, and a console using an older code page renders a non-ASCII dash as a
    # literal escape - noise in the one message that has to be readable.
    return (
        f"`mcp` {installed} is installed, but it does not provide "
        f"`mcp.server.fastmcp`. {cause}\n"
        f"{underlying}"
    )


def create_app() -> Any:
    """
    Create the FastMCP application.

    Imports `mcp` lazily so unit tests that don't have dependencies installed can still run.
    """
    try:
        FastMCP = _import_fastmcp()
    except ImportError as exc:
        # Narrowed from bare `Exception`: a SyntaxError or a failure in Lad's own
        # imports is not a dependency problem, and relabelling it as one is the same
        # misdirection this message exists to remove.
        raise RuntimeError(_mcp_import_failure_message(exc)) from exc

    logging.basicConfig(level=logging.INFO)  # logs go to stderr by default

    mcp = FastMCP("lad-mcp-server")
    service = ReviewService()

    @mcp.tool()
    async def system_design_review(
        proposal: Annotated[
            str | None,
            Field(
                description=(
                    "The system design document to review. "
                    "Must be at least 10 characters (and is ignored when `paths` is also provided). "
                    "Required unless `paths` is provided (both can also be provided together)."
                ),
            ),
        ] = None,
        paths: Annotated[
            list[str] | str | None,
            Field(
                description=(
                    "File paths to existing implementation files that provide context for the review. "
                    "Pass either: a list of paths (e.g. ['src/main.py', 'src/utils.py']), "
                    "a newline-separated string of paths, or a JSON array string. "
                    "Paths are relative to the current working directory; absolute paths also work. "
                    "Files are read automatically from disk. "
                    "Required unless `proposal` is provided (both can also be provided together)."
                ),
            ),
        ] = None,
        constraints: Annotated[
            str | None,
            Field(
                description=(
                    "Explicit constraints the design must satisfy (e.g., non-functional requirements, "
                    "tech stack restrictions, performance targets). "
                    "Max 10,000 characters."
                ),
            ),
        ] = None,
        context: Annotated[
            str | None,
            Field(
                description=(
                    "Background information to help the reviewer understand the design context "
                    "(e.g., business goals, prior architectural decisions, known issues, scope). "
                    "Max 10,000 characters."
                ),
            ),
        ] = None,
    ) -> str:
        """
        Review a system design proposal and constraints using two LLM reviewers in parallel.

        At least one of `proposal` or `paths` must be provided.
        """
        try:
            return await service.system_design_review(
                proposal=proposal,
                paths=paths,
                constraints=constraints,
                context=context,
            )
        except ValidationError as exc:
            return format_validation_error(str(exc))
        except Exception as exc:  # pragma: no cover
            msg = str(exc).strip()
            if not msg:
                msg = exc.__class__.__name__
            return format_fatal_error(msg)

    @mcp.tool()
    async def code_review(
        code: Annotated[
            str | None,
            Field(
                description=(
                    "The code snippet or diff to review. "
                    "Pass a meaningful amount of code for a useful review. "
                    "Required unless `paths` is provided (both can also be provided together)."
                ),
            ),
        ] = None,
        paths: Annotated[
            list[str] | str | None,
            Field(
                description=(
                    "File paths to source files to review. "
                    "Pass either: a list of paths (e.g. ['src/main.py', 'src/utils.py']), "
                    "a newline-separated string of paths, or a JSON array string. "
                    "Paths are relative to the current working directory; absolute paths also work. "
                    "Files are read automatically from disk. "
                    "Required unless `code` is provided (both can also be provided together)."
                ),
            ),
        ] = None,
        context: Annotated[
            str | None,
            Field(
                description=(
                    "Additional guidance for the reviewer (e.g., review goals, "
                    "known issues, areas to focus on, style conventions, constraints). "
                    "Max 10,000 characters."
                ),
            ),
        ] = None,
    ) -> str:
        """
        Review a code snippet or diff using two LLM reviewers in parallel.

        At least one of `code` or `paths` must be provided.
        """
        try:
            return await service.code_review(
                code=code,
                paths=paths,
                context=context,
            )
        except ValidationError as exc:
            return format_validation_error(str(exc))
        except Exception as exc:  # pragma: no cover
            msg = str(exc).strip()
            if not msg:
                msg = exc.__class__.__name__
            return format_fatal_error(msg)

    return mcp
