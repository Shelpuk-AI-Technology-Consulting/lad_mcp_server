from __future__ import annotations

import re

from lad_mcp_server.redaction import redact_text


REQUIRED_SECTIONS = (
    "Summary",
    "Key Findings",
    "Recommendations",
    "Questions / Unknowns",
)


def normalize_reviewer_markdown(markdown: str) -> str:
    """
    Best-effort normalization to ensure required headings exist.
    This is intentionally non-LLM and deterministic.
    """
    normalized = markdown.strip() if markdown is not None else ""
    if normalized == "":
        normalized = "## Summary\n*(No content provided by reviewer)*\n"

    # Ensure headings exist (match "## Heading" or "### Heading")
    for section in REQUIRED_SECTIONS:
        pattern = re.compile(rf"^#{2,3}\s+{re.escape(section)}\s*$", re.MULTILINE)
        if pattern.search(normalized) is None:
            normalized += f"\n\n## {section}\n*(No {section} provided by reviewer)*\n"
    return normalized.strip()


def format_aggregated_output(
    *,
    primary_markdown: str,
    secondary_markdown: str | None,
    synthesized_summary: str,
) -> str:
    primary_norm = normalize_reviewer_markdown(primary_markdown)
    summary_norm = synthesized_summary.strip() or "Primary and Secondary reviews are provided below."

    if secondary_markdown is None:
        out = (
            "## Primary Reviewer\n\n"
            f"{primary_norm}\n\n"
            "## Synthesized Summary\n\n"
            f"{summary_norm}\n"
        )
        return out.strip()

    secondary_norm = normalize_reviewer_markdown(secondary_markdown)
    out = (
        "## Primary Reviewer\n\n"
        f"{primary_norm}\n\n"
        "## Secondary Reviewer\n\n"
        f"{secondary_norm}\n\n"
        "## Synthesized Summary\n\n"
        f"{summary_norm}\n"
    )
    return out.strip()


def final_egress_redaction(markdown: str) -> str:
    return redact_text(markdown)
