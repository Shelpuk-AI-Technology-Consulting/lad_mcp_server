from __future__ import annotations


def system_prompt_system_design_review(*, tool_calling_enabled: bool) -> str:
    tool_note = (
        "You MAY call tools to inspect repo context and Serena memories when needed."
        if tool_calling_enabled
        else "You do NOT have access to any tools or repository context beyond the user-provided text."
    )
    serena_preflight = (
        "SERENA WORKFLOW (mandatory):\n"
        "1) Immediately call `activate_project` with `project=\".\"` before any other tool.\n"
        "2) Call `read_project_overview` to load baseline project context.\n"
        "3) Call `list_memories`, then read the most relevant memories via `read_memory`:\n"
        "   - Always try: `project_overview`, `research_summary`\n"
        "   - If present: requirements/design constraints memories (e.g., `requirements`, `constraints`)\n"
        "4) If requirements/constraints are not in Serena memories, read `REQUIREMENTS.md` / `README.md` via `read_file`.\n"
        "5) Use Serena to explore beyond the provided snippets when needed (e.g., `list_dir`, `search_for_pattern`, `find_symbol`, `read_file`).\n"
        if tool_calling_enabled
        else ""
    )
    return (
        "You are an expert software architect and reviewer.\n"
        "Provide a thorough but concise critique.\n"
        f"{tool_note}\n"
        f"{serena_preflight}\n"
        "Return Markdown with sections:\n"
        "## Summary\n"
        "## Key Findings\n"
        "## Recommendations\n"
        "## Questions / Unknowns\n"
    )


def user_prompt_system_design_review(*, proposal: str, constraints: str | None, context: str | None) -> str:
    parts: list[str] = ["# System Design Review Request", "\n## Proposal\n", proposal]
    if constraints:
        parts += ["\n\n## Constraints\n", constraints]
    if context:
        parts += ["\n\n## Context\n", context]
    return "".join(parts)


def system_prompt_code_review(*, tool_calling_enabled: bool) -> str:
    tool_note = (
        "You MAY call tools to inspect repo context and Serena memories when needed."
        if tool_calling_enabled
        else "You do NOT have access to any tools or repository context beyond the user-provided text."
    )
    serena_preflight = (
        "SERENA WORKFLOW (mandatory):\n"
        "1) Immediately call `activate_project` with `project=\".\"` before any other tool.\n"
        "2) Call `read_project_overview` to load baseline project context.\n"
        "3) Call `list_memories`, then read the most relevant memories via `read_memory`:\n"
        "   - Always try: `project_overview`, `research_summary`\n"
        "   - If present: requirements/design constraints memories (e.g., `requirements`, `constraints`)\n"
        "4) If requirements/constraints are not in Serena memories, read `REQUIREMENTS.md` / `README.md` via `read_file`.\n"
        "5) Use Serena to explore beyond the provided snippets when needed (e.g., `list_dir`, `search_for_pattern`, `find_symbol`, `read_file`).\n"
        if tool_calling_enabled
        else ""
    )
    return (
        "You are an expert code reviewer focused on correctness, security, and maintainability.\n"
        f"{tool_note}\n"
        f"{serena_preflight}\n"
        "Return Markdown with sections:\n"
        "## Summary\n"
        "## Key Findings\n"
        "## Recommendations\n"
        "## Questions / Unknowns\n"
    )


def user_prompt_code_review(*, code: str) -> str:
    return (
        "# Code Review Request\n"
        "\n## Language\n"
        "Infer language(s) and frameworks from the code and embedded file paths/extensions.\n"
        "\n## Review Goal\n"
        "Find bugs, untracked failure modes, gaps or contradictions in business logic, and provide concrete improvement suggestions.\n"
        "\n## Code\n"
        f"```\n{code}\n```\n"
    )


def force_finalize_system_message() -> str:
    return "You have reached the maximum tool call budget. Provide your final review now without further tool calls."
