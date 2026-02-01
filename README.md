# Lad MCP Server

An MCP (Model Context Protocol) server that exposes two tools:

- `system_design_review`
- `code_review`

Each tool runs **two OpenRouter-backed reviewers in parallel** (Primary + Secondary) and returns both outputs plus a synthesized summary. If the target repository contains `.serena/` and the selected models support tool calling, Lad exposes a **read-only Serena tool bridge** to both reviewers (with mandatory `activate_project(".")` preflight).

## Install & run (uv / uvx)

Lad ships a stdio CLI entrypoint: `lad-mcp-server`.

- Run from a local checkout (recommended for development): `uv run lad-mcp-server`
- One-off run from git (works well for MCP hosts launching tools via `uvx`):
  - `uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server lad-mcp-server`

Tip: `uvx` runs tools in an isolated environment; Lad infers the project root from the MCP host (or absolute paths, or CWD).

## Codex setup (CLI-only)

Install Codex CLI (one-liner):

```bash
npm i -g @openai/codex
```

Add Lad as a local stdio MCP server (macOS / Linux):

```bash
codex mcp add lad \
  --env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -- uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server lad-mcp-server
```

Windows (PowerShell):

```powershell
codex mcp add lad `
  --env OPENROUTER_API_KEY="$env:OPENROUTER_API_KEY" `
  -- uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server lad-mcp-server
```

Per-project usage: use `paths` from the project you want reviewed. Lad infers the project root from the MCP host (or from absolute `paths`, or finally from CWD).

## Configuration (environment variables)

Required:
- `OPENROUTER_API_KEY`

Common:
- `OPENROUTER_PRIMARY_REVIEWER_MODEL` (default: `moonshotai/kimi-k2-thinking`)
- `OPENROUTER_SECONDARY_REVIEWER_MODEL` (default: `z-ai/glm-4.7`)
- `OPENROUTER_REVIEWER_TIMEOUT_SECONDS` (default: `180`, per reviewer)
- `OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS` (default: `240`, per tool call)
- `LAD_ENV_FILE` (optional `KEY=VALUE` file, loaded only for missing vars)

## Path-based review requests

Both tools accept either direct text (`proposal` / `code`) or `paths` (files/dirs under inferred project root). When `paths` are provided, Lad reads and embeds **text-like** files from disk (language-agnostic) and skips common binary files and oversized files.
During directory expansion, hidden files and directories (dotfiles) are skipped; pass an explicit path (e.g., `.gitignore`) if you want it included.

Example tool payloads:
- `system_design_review`: `{\"paths\":[\"research/AI Code Review MCP Server Design.md\"],\"constraints\":\"...\"}`
- `code_review`: `{\"paths\":[\"lad_mcp_server\",\"tests\"]}`

## Serena integration

If the repo being reviewed contains `.serena/` and the model supports tool calling, both reviewers get a read-only Serena toolset (directory listing, file reads, memory reads, pattern search) and are instructed/forced to call `activate_project(\".\")` first.

## Dev helpers

- `python scripts/verify_openrouter_key.py` (fetches model list)
- `python scripts/verify_serena_usage.py` (ensures both reviewers used Serena and read `project_overview.md`)
