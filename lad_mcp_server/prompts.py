from __future__ import annotations


def system_prompt_system_design_review(*, tool_calling_enabled: bool) -> str:
    tool_note = (
        "You MAY call tools to inspect repo context and Serena memories when needed."
        if tool_calling_enabled
        else "You do NOT have access to any tools or repository context beyond the user-provided text."
    )
    serena_preflight = (
        "PRE-FLIGHT (mandatory): Immediately call `activate_project` with `project=\".\"` before any other tool. "
        "Then call `read_project_overview` to load baseline project context.\n"
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
        "PRE-FLIGHT (mandatory): Immediately call `activate_project` with `project=\".\"` before any other tool. "
        "Then call `read_project_overview` to load baseline project context.\n"
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


def user_prompt_code_review(*, code: str, language: str, focus: str | None) -> str:
    focus_text = focus or "general"
    return (
        "# Code Review Request\n"
        f"\n## Language\n{language}\n"
        f"\n## Focus\n{focus_text}\n"
        "\n## Code\n"
        f"```{language}\n{code}\n```\n"
    )


def force_finalize_system_message() -> str:
    return "You have reached the maximum tool call budget. Provide your final review now without further tool calls."
