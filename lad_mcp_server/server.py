from __future__ import annotations

import logging
from typing import Any

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
        proposal: str | None = None,
        paths: list[str] | None = None,
        project_root: str | None = None,
        constraints: str | None = None,
        context: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Review a system design proposal and constraints using two LLM reviewers in parallel.
        """
        try:
            return await service.system_design_review(
                proposal=proposal,
                paths=paths,
                project_root=project_root,
                constraints=constraints,
                context=context,
                model=model,
            )
        except ValidationError as exc:
            return format_validation_error(str(exc))
        except Exception as exc:  # pragma: no cover
            return format_fatal_error(str(exc))

    @mcp.tool()
    async def code_review(
        code: str | None = None,
        paths: list[str] | None = None,
        project_root: str | None = None,
        language: str | None = None,
        focus: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Review a code snippet or diff using two LLM reviewers in parallel.
        """
        try:
            return await service.code_review(
                code=code,
                paths=paths,
                project_root=project_root,
                language=language,
                focus=focus,
                model=model,
            )
        except ValidationError as exc:
            return format_validation_error(str(exc))
        except Exception as exc:  # pragma: no cover
            return format_fatal_error(str(exc))

    return mcp
