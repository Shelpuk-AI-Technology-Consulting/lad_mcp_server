# Lad MCP Server

An MCP (Model Context Protocol) server that exposes two tools:

- `system_design_review`
- `code_review`

Each tool runs **two OpenRouter-backed reviewers in parallel** (Primary + Secondary) and returns both outputs plus a synthesized summary.

If the target repository contains `.serena/` and the selected models support tool calling, Lad exposes a **read-only Serena tool bridge** to both reviewers (with mandatory `activate_project(".")` preflight).

## Quickstart

### 1) Requirements

- Python 3.11+
- `uv` (recommended) or `pip`
- `OPENROUTER_API_KEY` (required)

### 2) Run command used by MCP clients (stdio)

```bash
uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server \
  lad-mcp-server
```

First-run note: the first `uvx` invocation may take 30–60 seconds while it builds the tool environment. If your MCP client times out on first start, run the command once in a terminal to “prewarm” it, then retry in your client.

## Install & run (local development)

Lad ships a stdio CLI entrypoint: `lad-mcp-server`.

- From a local checkout (recommended for development): `uv run lad-mcp-server`
- From a published package (when installed): `lad-mcp-server`

## Client setup

All client examples below run Lad over **stdio** (the MCP client spawns the server process).

Make sure `OPENROUTER_API_KEY` is available to the MCP client:
- either set it in your OS environment, or
- paste it directly into the client config (not recommended for repos you commit).

### Codex (Codex CLI)

Install Codex CLI:

```bash
npm i -g @openai/codex
```

CLI (no file editing) — add a local stdio MCP server (macOS / Linux):

```bash
codex mcp add lad \
  --env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -- uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server \
  lad-mcp-server
```

Windows (PowerShell):

```powershell
codex mcp add lad `
  --env OPENROUTER_API_KEY="$env:OPENROUTER_API_KEY" `
  -- uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server `
  lad-mcp-server
```

Alternative (file-based):
Edit `~/.codex/config.toml` (or project-scoped `.codex/config.toml` in trusted projects):

```toml
[mcp_servers.lad]
command = "uvx"
args = [
  "--from",
  "git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server",
  "lad-mcp-server",
]
# Forward variables from your shell/OS environment:
env_vars = ["OPENROUTER_API_KEY"]
startup_timeout_sec = 120.0
```

### Claude Code

CLI (no file editing) — add a local stdio MCP server (macOS / Linux):

```bash
claude mcp add --transport stdio \
  --env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  lad \
  -- uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server \
  lad-mcp-server
```

Windows (PowerShell):

```powershell
claude mcp add --transport stdio `
  --env OPENROUTER_API_KEY="$env:OPENROUTER_API_KEY" `
  lad `
  -- uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server `
  lad-mcp-server
```

If Claude Code times out while starting the server, increase the startup timeout (milliseconds):

macOS / Linux:
```bash
export MCP_TIMEOUT=120000
```

Windows (PowerShell):
```powershell
$env:MCP_TIMEOUT="120000"
```

Alternative (file-based):
Create/edit `.mcp.json` (project scope; recommended for teams):

```json
{
  "mcpServers": {
    "lad": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server",
        "lad-mcp-server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```

### Cursor

Create `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "lad": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server",
        "lad-mcp-server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}"
      }
    }
  }
}
```

Startup timeout: Cursor does not always expose a per-server startup timeout. If the first run is slow, run the `uvx` command once in a terminal to prebuild the tool environment, then restart Cursor.

### Gemini CLI

You can configure MCP servers via CLI or by editing `settings.json`.

CLI (adds to `.gemini/settings.json` by default; use `-s user` for `~/.gemini/settings.json`):

```bash
gemini mcp add -s user -t stdio \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  lad uvx --from git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server lad-mcp-server
```

Alternative (file-based):
Edit `~/.gemini/settings.json` (or `.gemini/settings.json` in a project):

```json
{
  "mcpServers": {
    "lad": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server",
        "lad-mcp-server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "$OPENROUTER_API_KEY"
      },
      "timeout": 120000,
      "trust": false
    }
  }
}
```

### Windsurf

Windsurf stores MCP servers in `~/.codeium/windsurf/mcp_config.json`.

In Windsurf:
1. Open **Cascade**
2. Click **MCPs** (top-right)
3. Click **Manage MCP Servers**
4. Click **View raw config** (opens `mcp_config.json`)
5. Add the server under `mcpServers`, save, then click **Refresh**

Paste this into your `mcpServers` object (don’t overwrite other servers):

```json
{
  "lad": {
    "command": "uvx",
    "args": [
      "--from",
      "git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server",
      "lad-mcp-server"
    ],
    "env": {
      "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}"
    }
  }
}
```

If Windsurf can’t find `uvx`, replace `"uvx"` with an absolute path (run `which uvx` in a terminal).

### Antigravity

In Antigravity, open the MCP store, then:
1. Click **Manage MCP Servers**
2. Click **View raw config** (opens `mcp_config.json`)
3. Add the server config under `mcpServers`, save, then go back and click **Refresh**

Paste this into your `mcpServers` object (don’t overwrite other servers):

```json
{
  "lad": {
    "command": "uvx",
    "args": [
      "--from",
      "git+https://github.com/Shelpuk-AI-Technology-Consulting/lad_mcp_server",
      "lad-mcp-server"
    ],
    "env": {
      "OPENROUTER_API_KEY": "PASTE_OPENROUTER_KEY_HERE"
    }
  }
}
```

Replace `PASTE_OPENROUTER_KEY_HERE` with your actual key.

If Antigravity can’t find `uvx`, replace `"uvx"` with an absolute path (run `which uvx` in a terminal).

## Configuration (environment variables)

### Required

- `OPENROUTER_API_KEY`

### OpenRouter models

- `OPENROUTER_PRIMARY_REVIEWER_MODEL` (default: `moonshotai/kimi-k2-thinking`)
- `OPENROUTER_SECONDARY_REVIEWER_MODEL` (default: `z-ai/glm-4.7`)
  - Set to `0` to disable the Secondary reviewer (Primary-only mode).

### OpenRouter request behavior

- `OPENROUTER_MAX_CONCURRENT_REQUESTS` (default: `4`)
- `OPENROUTER_REVIEWER_TIMEOUT_SECONDS` (default: `180`, per reviewer)
- `OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS` (default: `240`, per tool call)
- `OPENROUTER_HTTP_REFERER` (optional; forwarded to OpenRouter)
- `OPENROUTER_X_TITLE` (optional; forwarded to OpenRouter)

### Budgeting / limits

- `OPENROUTER_FIXED_OUTPUT_TOKENS` (default: `16384`)
- `OPENROUTER_CONTEXT_OVERHEAD_TOKENS` (default: `2000`)
- `OPENROUTER_MODEL_METADATA_TTL_SECONDS` (default: `3600`)
- `OPENROUTER_MAX_INPUT_CHARS` (default: `100000`)
- `OPENROUTER_INCLUDE_REASONING` (default: `false`)

### Serena bridge (only when `.serena/` exists and tool calling is supported)

- `LAD_SERENA_MAX_TOOL_CALLS` (default: `8`)
- `LAD_SERENA_TOOL_TIMEOUT_SECONDS` (default: `30`)
- `LAD_SERENA_MAX_TOOL_RESULT_CHARS` (default: `12000`)
- `LAD_SERENA_MAX_TOTAL_CHARS` (default: `50000`)
- `LAD_SERENA_MAX_DIR_ENTRIES` (default: `100`)
- `LAD_SERENA_MAX_SEARCH_RESULTS` (default: `20`)

### Env files (optional)

- `LAD_ENV_FILE` (optional `KEY=VALUE` file, loaded only for missing vars)
- `.env` (optional; loaded if `python-dotenv` is installed)

Precedence note: Lad never overrides variables that are already set in the process environment. It only fills missing variables from `LAD_ENV_FILE` first, then (optionally) from `.env`.

## Path-based review requests

Both tools accept either direct text (`proposal` / `code`) or `paths` (files/dirs under inferred project root). When `paths` are provided, Lad reads and embeds **text-like** files from disk (language-agnostic) and skips common binary files and oversized files.

During directory expansion, hidden files and directories (dotfiles) are skipped; pass an explicit path (e.g., `.gitignore`) if you want it included.

Example tool payloads:
- `system_design_review`: `{"paths":["research/AI Code Review MCP Server Design.md"],"constraints":"..."}`
- `code_review`: `{"paths":["lad_mcp_server","tests"],"context":"Please prioritize correctness and failure modes in auth/session logic."}`

Notes:
- For multi-project usage (one Lad config for many repos), prefer **absolute** `paths`. Relative `paths` work when the MCP host starts Lad with CWD set to the project root (or when the host provides a workspace root).
- For safety, Lad rejects path-based reviews that resolve to obvious system roots (e.g. `/etc`, `/proc`, `C:\Windows`).

## Serena integration

If the repo being reviewed contains `.serena/` and the model supports tool calling, both reviewers get a read-only Serena toolset (directory listing, file reads, memory reads, pattern search) and are instructed/forced to call `activate_project(".")` first.

## Docker deployment (local stdio)

This is the most compatible Docker setup: your MCP client still runs Lad as a stdio server, but the process it spawns is `docker run ...` instead of `uvx ...`.

Build the image:

```bash
docker build -t lad-mcp-server .
```

Don’t bake API keys into the image. Pass them via environment variables at runtime.

Run Lad over stdio (for manual testing):

```bash
docker run --rm -i \
  -e OPENROUTER_API_KEY="..." \
  lad-mcp-server
```

Use Docker from an MCP client (example config fragment):

```json
{
  "lad": {
    "command": "docker",
    "args": [
      "run",
      "-i",
      "--rm",
      "-e",
      "OPENROUTER_API_KEY",
      "lad-mcp-server"
    ],
    "env": {
      "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}"
    }
  }
}
```

Note: stdio servers are **local-process** servers. If your client config uses `command`/`args`, it must be able to execute that command (locally), even if the command is `docker`.

## Troubleshooting

- First run is slow / client times out: run the `uvx` command from Quickstart once in a terminal to prewarm, then restart the client.
- “OPENROUTER_API_KEY is required”: ensure the MCP client process has access to the variable (or paste it into the client config).
- Secondary reviewer is too expensive/slow: set `OPENROUTER_SECONDARY_REVIEWER_MODEL=0`.

## Security notes

- Don’t commit API keys.
- Treat code review as data egress: anything you pass via `code`/`proposal` or `paths` may be sent to OpenRouter (after redaction).
- Prefer environment-variable forwarding (`env_vars` / `${env:...}`) over hardcoding secrets in config files.

## Dev helpers

- `python3 scripts/verify_openrouter_key.py` (fetches model list)
- `python3 scripts/verify_serena_usage.py` (ensures both reviewers used Serena and read `project_overview.md`)
