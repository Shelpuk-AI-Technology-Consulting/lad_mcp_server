# Lad MCP Server

An MCP (Model Context Protocol) server that exposes two tools:

- `system_design_review`
- `code_review`

Each tool runs **two OpenRouter-backed reviewers in parallel** (Primary + Secondary) and returns both outputs plus a synthesized summary. If the target repository contains `.serena/` and the selected models support tool calling, Lad exposes a **read-only Serena tool bridge** to both reviewers (with mandatory `activate_project(".")` preflight).

## Install & run (uv / uvx)

Lad ships a stdio CLI entrypoint: `lad-mcp-server`.

- Run from a local checkout (recommended for development): `uv run lad-mcp-server`
- One-off run from git (works well for MCP hosts launching tools via `uvx`):
  - `uvx --from git+https://github.com/<you>/<repo>.git lad-mcp-server`

Tip: `uvx` runs tools in an isolated environment; set `LAD_REPO_ROOT` so Lad knows which repo to read.

## Install Codex CLI (optional, one-liner)

`npm i -g @openai/codex`

## Configuration (environment variables)

Required:
- `OPENROUTER_API_KEY`

Common:
- `OPENROUTER_PRIMARY_REVIEWER_MODEL` (default: `moonshotai/kimi-k2-thinking`)
- `OPENROUTER_SECONDARY_REVIEWER_MODEL` (default: `z-ai/glm-4.7`)
- `OPENROUTER_REVIEWER_TIMEOUT_SECONDS` (default: `180`, per reviewer)
- `OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS` (default: `240`, per tool call)
- `LAD_REPO_ROOT` (repo to review; default: current working directory)
- `LAD_ENV_FILE` (optional `KEY=VALUE` file, loaded only for missing vars)

PowerShell example:
- `$env:LAD_ENV_FILE=\"test.env\"; $env:LAD_REPO_ROOT=\"D:\\PycharmProjects\\Shelpuk\\lad_mcp_server\"; uv run lad-mcp-server`

Timeout note: if Serena tool calling is enabled, a single review may require multiple OpenRouter requests; set `OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS` accordingly.

## Path-based review requests

Both tools accept either direct text (`proposal` / `code`) or `paths` (files/dirs under `LAD_REPO_ROOT`). When `paths` are provided, Lad reads and embeds **text-like** files from disk (language-agnostic) and skips common binary files and oversized files.
During directory expansion, hidden files and directories (dotfiles) are skipped; pass an explicit path (e.g., `.gitignore`) if you want it included.

Example tool payloads:
- `system_design_review`: `{\"paths\":[\"research/AI Code Review MCP Server Design.md\"],\"constraints\":\"...\"}`
- `code_review`: `{\"paths\":[\"lad_mcp_server\",\"tests\"],\"focus\":\"security\",\"language\":\"mixed\"}`

## Serena integration

If the repo being reviewed contains `.serena/` and the model supports tool calling, both reviewers get a read-only Serena toolset (directory listing, file reads, memory reads, pattern search) and are instructed/forced to call `activate_project(\".\")` first.

## Dev helpers

- `python scripts/verify_openrouter_key.py` (fetches model list)
- `python scripts/verify_serena_usage.py` (ensures both reviewers used Serena and read `project_overview.md`)
