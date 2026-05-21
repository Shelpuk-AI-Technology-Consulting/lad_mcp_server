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
        "5) Use Serena to explore beyond the provided snippets when needed (e.g., `list_dir`, `search_for_pattern`, `find_symbol`, `read_file`, `read_file_window`).\n"
        "6) `read_file` semantics example: `head=50` means first 50 lines; `tail=30` means last 30 lines.\n"
        "7) Recommended workflow for large files: `search_for_pattern` -> `read_file_window(path, start_line, num_lines)`.\n"
        if tool_calling_enabled
        else ""
    )
    return (
        "You are an expert software architect and reviewer.\n"
        "Provide a thorough but concise critique of the proposed solution. Spot any issues, bugs, inconsistencies, failure modes, and corner cases.\n"
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
        "5) Use Serena to explore beyond the provided snippets when needed (e.g., `list_dir`, `search_for_pattern`, `find_symbol`, `read_file`, `read_file_window`).\n"
        "6) `read_file` semantics example: `head=50` means first 50 lines; `tail=30` means last 30 lines.\n"
        "7) Recommended workflow for large files: `search_for_pattern` -> `read_file_window(path, start_line, num_lines)`.\n"
        if tool_calling_enabled
        else ""
    )
    return (
        "You are an expert code reviewer focused on correctness, security, and maintainability. You spot any issues, bugs, inconsistencies, failure modes, and corner cases.\n"
        f"{tool_note}\n"
        f"{serena_preflight}\n"
        "Return Markdown with sections:\n"
        "## Summary\n"
        "## Key Findings\n"
        "## Recommendations\n"
        "## Questions / Unknowns\n"
    )


def user_prompt_code_review(*, code: str, context: str | None) -> str:
    context_block = ""
    if context:
        context_block = "\n## Context / Goals\n" + context + "\n"
    return (
        "# Code Review Request\n"
        "\n## Language\n"
        "Infer language(s) and frameworks from the code and embedded file paths/extensions.\n"
        "\n## Review Goal\n"
        "Find bugs, untracked failure modes, gaps or contradictions in business logic, and provide concrete improvement suggestions.\n"
        f"{context_block}"
        "\n## Code\n"
        f"```\n{code}\n```\n"
    )


def force_finalize_system_message() -> str:
    return "You have reached the maximum tool call budget. Provide your final review now without further tool calls."


def intermittent_review_finalize_user_message() -> str:
    """
    User-role nudge for an intermittent (partial) review snapshot.

    Sent as a user message (not system) to avoid provider quirks with mid-chain system messages.
    The side call is dispatched with `tools=None`, so the model has no tools available to call —
    there is no need to instruct it to refrain from tool use.
    """
    return (
        "PARTIAL REVIEW REQUESTED. Based on everything you have explored and gathered so far in this "
        "conversation, produce your best partial review NOW. Use these markdown sections exactly: "
        "`## Summary`, `## Key Findings`, `## Recommendations`, `## Questions / Unknowns`. "
        "Where you would otherwise have explored further, mark the gap explicitly under "
        "`## Questions / Unknowns` so the caller knows what is incomplete."
    )
