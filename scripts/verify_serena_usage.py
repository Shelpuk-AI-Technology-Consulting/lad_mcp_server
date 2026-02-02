#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from lad_mcp_server.review_service import ReviewService


def _load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


async def _run() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    service = ReviewService(repo_root=repo_root)
    proposal = (
        "TEST: Serena integration check.\n"
        "If tools are available, you MUST do all of the following IN ORDER:\n"
        "0) call activate_project with project '.'\n"
        "1) call list_memories\n"
        "2) call read_memory with name 'project_overview'\n"
        "Then include the FIRST LINE of that memory verbatim in your Summary.\n"
        "If tools are not available, explicitly say 'TOOLS_UNAVAILABLE'.\n\n"
        "Design proposal: minimal placeholder.\n"
    )
    return await service.system_design_review(proposal=proposal, constraints=None, context=None, model=None)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    env_file = os.getenv("LAD_ENV_FILE")
    if env_file:
        _load_env_file(Path(env_file))
    else:
        test_env = repo_root / "test.env"
        if test_env.is_file():
            _load_env_file(test_env)

    # Sanity: ensure `.serena/memories/project_overview.md` exists
    mem = repo_root / ".serena" / "memories" / "project_overview.md"
    if not mem.is_file():
        print(f"Missing Serena memory file: {mem}")
        return 2
    sentinel = mem.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    if not sentinel:
        print("First line of project_overview.md is empty; set a sentinel first line and retry.")
        return 3

    out = asyncio.run(_run())
    secondary_expected = os.getenv("OPENROUTER_SECONDARY_REVIEWER_MODEL", "z-ai/glm-4.7").strip() != "0"

    def section(start_marker: str, end_markers: list[str]) -> str:
        # Use substring slicing (not strict ^...$ regex) to be resilient to
        # model-inserted invisible characters / newline variants.
        start = out.find(start_marker)
        if start == -1:
            return ""
        end = len(out)
        for m in end_markers:
            idx = out.find(m, start + len(start_marker))
            if idx != -1:
                end = min(end, idx)
        return out[start:end]

    primary_sec = section("## Primary Reviewer", ["## Secondary Reviewer", "## Synthesized Summary"])
    secondary_sec = section("## Secondary Reviewer", ["## Synthesized Summary"])

    primary_used = "*Serena tools used: yes*" in primary_sec
    secondary_used = "*Serena tools used: yes*" in secondary_sec
    primary_activated = "Serena project activated:" in primary_sec and "`.`" in primary_sec
    secondary_activated = "Serena project activated:" in secondary_sec and "`.`" in secondary_sec
    primary_activate_invoked = "Serena tools invoked:" in primary_sec and "`activate_project`" in primary_sec
    secondary_activate_invoked = "Serena tools invoked:" in secondary_sec and "`activate_project`" in secondary_sec
    primary_memory_used = "Serena memories used:" in primary_sec and "project_overview.md" in primary_sec
    secondary_memory_used = "Serena memories used:" in secondary_sec and "project_overview.md" in secondary_sec
    primary_sentinel = sentinel in primary_sec
    secondary_sentinel = sentinel in secondary_sec

    if secondary_expected:
        if (
            primary_used
            and secondary_used
            and primary_activated
            and secondary_activated
            and primary_activate_invoked
            and secondary_activate_invoked
            and primary_memory_used
            and secondary_memory_used
            and primary_sentinel
            and secondary_sentinel
        ):
            print("OK: Both reviewers used Serena and read project_overview.md")
            return 0
    else:
        if (
            primary_used
            and primary_activated
            and primary_activate_invoked
            and primary_memory_used
            and primary_sentinel
            and "## Secondary Reviewer" not in out
        ):
            print("OK: Primary reviewer used Serena and read project_overview.md (Secondary disabled)")
            return 0

    print("FAILED: Serena usage verification did not pass.")
    print(f"- primary_used={primary_used}")
    print(f"- secondary_used={secondary_used}")
    print(f"- primary_activated={primary_activated}")
    print(f"- secondary_activated={secondary_activated}")
    print(f"- primary_activate_invoked={primary_activate_invoked}")
    print(f"- secondary_activate_invoked={secondary_activate_invoked}")
    print(f"- primary_memory_used={primary_memory_used}")
    print(f"- secondary_memory_used={secondary_memory_used}")
    print(f"- primary_sentinel_included={primary_sentinel}")
    print(f"- secondary_sentinel_included={secondary_sentinel}")
    print(f"- secondary_expected={secondary_expected}")
    print("\n--- Output (truncated) ---\n")
    print(out[:4000])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
