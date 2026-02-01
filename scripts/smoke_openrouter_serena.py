#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from lad_mcp_server.review_service import ReviewService


def load_env_file(path: Path) -> None:
    """
    Minimal .env loader to avoid extra dependencies.
    """
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "test.env"
    if env_path.is_file() and "LAD_ENV_FILE" not in os.environ:
        # Keep backwards compatibility for local smoke runs.
        load_env_file(env_path)

    # Force a prompt that strongly encourages tool usage to prove Serena-backed tools are actually invoked.
    proposal = (
        "We need a review. IMPORTANT: If tools are available, you MUST call these tools before answering:\n"
        "1) list_memories\n"
        "2) read_memory for 'project_overview'\n"
        "Then include the first line of that memory in your Summary.\n"
        "If tools are not available, explicitly say so.\n\n"
        "Design: Build a simple MCP server with two tools and dual reviewers."
    )

    service = ReviewService(repo_root=repo_root)
    out = await service.system_design_review(proposal=proposal, constraints=None, context=None, model=None)
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
