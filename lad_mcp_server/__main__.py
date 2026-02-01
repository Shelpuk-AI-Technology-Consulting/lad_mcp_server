import sys


def main() -> None:
    """
    Entrypoint for running the Lad MCP server.

    Notes:
    - This server is intended to run via STDIO transport under an MCP host (Claude Desktop, Codex, etc.).
    - Logging must never write to stdout for STDIO transport.
    """
    try:
        from lad_mcp_server.server import create_app
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import Lad MCP server. Install dependencies first (see pyproject.toml).\n"
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    app = create_app()

    # FastMCP default transport is stdio when run() is called without args.
    app.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

