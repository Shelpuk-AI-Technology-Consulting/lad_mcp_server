from __future__ import annotations


def format_validation_error(message: str) -> str:
    return (
        "## Summary\n"
        f"Validation error: {message}\n\n"
        "## Key Findings\n"
        "- **High**: Input validation failed.\n\n"
        "## Recommendations\n"
        "- Fix the input fields and retry.\n\n"
        "## Questions / Unknowns\n"
        "- None.\n"
    )


def format_fatal_error(message: str) -> str:
    return (
        "## Summary\n"
        "An error occurred while processing the request.\n\n"
        "## Key Findings\n"
        f"- **High**: {message}\n\n"
        "## Recommendations\n"
        "- Check configuration (OPENROUTER_API_KEY, model names).\n"
        "- Ensure OpenRouter Models API is reachable.\n\n"
        "## Questions / Unknowns\n"
        "- None.\n"
    )

