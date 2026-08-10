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
    """Return the installed ``mcp`` version, or ``None`` if it is absent.

    Returns:
        The version string reported by the package metadata, or ``None`` when
        ``mcp`` is not installed or its metadata cannot be read.
    """
    try:
        from importlib.metadata import version

        return version("mcp")
    except Exception:
        return None


def _import_fastmcp() -> Any:
    """Import :class:`FastMCP`, kept separate so the failure path can be tested.

    Returns:
        The ``FastMCP`` class.

    Raises:
        ImportError: When ``mcp.server.fastmcp`` is unavailable.
    """
    from mcp.server.fastmcp import FastMCP

    return FastMCP


def _is_incompatible_mcp_major(installed: str) -> bool:
    """Report whether an installed ``mcp`` version is the one that dropped FastMCP.

    Args:
        installed: The version string from the package metadata.

    Returns:
        ``True`` when the major version is 2 or newer, or cannot be parsed — an
        unparseable version is treated as incompatible because that is by far the
        likelier case, and the message still prints the underlying error either way.
    """
    try:
        return int(installed.split(".", 1)[0]) >= 2
    except (ValueError, IndexError):
        return True


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
        return (
            "The `mcp` package is not installed. Install Lad's dependencies with "
            "`uv sync` or `pip install -e .`.\n"
            f"{underlying}"
        )

    if _is_incompatible_mcp_major(installed):
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
    else:
        # Do not blame the version when the version is fine. Telling this user to
        # downgrade would be the same wrong turn issue #3 was about.
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
