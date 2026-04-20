from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from lad_mcp_server.errors import format_fatal_error, format_validation_error
from lad_mcp_server.review_service import ReviewService
from lad_mcp_server.schemas import ValidationError


def create_app() -> Any:
    """
    Create the FastMCP application.

    Imports `mcp` lazily so unit tests that don't have dependencies installed can still run.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mcp dependency is not installed. Install dependencies from pyproject.toml."
        ) from exc

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
