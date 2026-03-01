from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    raw_norm = raw.strip().lower()
    if raw_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if raw_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false)")


def _get_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw


def _load_env_file(path: Path) -> None:
    """
    Minimal env file loader (KEY=VALUE), intended for test/dev.
    Values are only loaded if the variable is not already set.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str

    openrouter_primary_reviewer_model: str
    openrouter_secondary_reviewer_model: str

    openrouter_http_referer: str | None
    openrouter_x_title: str | None

    openrouter_reviewer_timeout_seconds: int
    openrouter_tool_call_timeout_seconds: int
    openrouter_max_concurrent_requests: int

    openrouter_fixed_output_tokens: int
    openrouter_context_overhead_tokens: int
    openrouter_model_metadata_ttl_seconds: int

    openrouter_max_input_chars: int
    openrouter_include_reasoning: bool

    lad_serena_max_tool_calls: int
    lad_serena_tool_timeout_seconds: int
    lad_serena_max_tool_result_chars: int
    lad_serena_max_total_chars: int
    lad_serena_max_dir_entries: int
    lad_serena_max_search_results: int

    @staticmethod
    def from_env() -> "Settings":
        # Optional: load an explicit env file (useful for `test.env`).
        env_file = os.getenv("LAD_ENV_FILE")
        if env_file:
            p = Path(env_file)
            if p.is_file():
                _load_env_file(p)

        # Optional: load .env if python-dotenv is installed.
        try:  # pragma: no cover
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        api_key = _get_str("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        max_concurrency = _get_int("OPENROUTER_MAX_CONCURRENT_REQUESTS", 4)
        if max_concurrency <= 0:
            raise ValueError("OPENROUTER_MAX_CONCURRENT_REQUESTS must be > 0")

        reviewer_timeout = _get_int("OPENROUTER_REVIEWER_TIMEOUT_SECONDS", 300)
        if reviewer_timeout <= 0:
            raise ValueError("OPENROUTER_REVIEWER_TIMEOUT_SECONDS must be > 0")

        # Tool call timeout must not undercut per-reviewer timeout; otherwise the entire request gets cancelled
        # before any reviewer can finish, producing a misleading generic error.
        tool_call_timeout_default = reviewer_timeout + 60
        tool_call_timeout = _get_int("OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS", tool_call_timeout_default)
        if tool_call_timeout <= 0:
            raise ValueError("OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS must be > 0")
        if tool_call_timeout < reviewer_timeout:
            raise ValueError(
                "OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS must be >= OPENROUTER_REVIEWER_TIMEOUT_SECONDS"
            )

        fixed_output_tokens = _get_int("OPENROUTER_FIXED_OUTPUT_TOKENS", 8192)
        if fixed_output_tokens <= 0:
            raise ValueError("OPENROUTER_FIXED_OUTPUT_TOKENS must be > 0")

        overhead_tokens = _get_int("OPENROUTER_CONTEXT_OVERHEAD_TOKENS", 2000)
        if overhead_tokens < 0:
            raise ValueError("OPENROUTER_CONTEXT_OVERHEAD_TOKENS must be >= 0")

        max_input_chars = _get_int("OPENROUTER_MAX_INPUT_CHARS", 100000)
        if max_input_chars <= 0:
            raise ValueError("OPENROUTER_MAX_INPUT_CHARS must be > 0")

        return Settings(
            openrouter_api_key=api_key,
            openrouter_primary_reviewer_model=_get_str(
                "OPENROUTER_PRIMARY_REVIEWER_MODEL", "moonshotai/kimi-k2.5"
            )
            or "moonshotai/kimi-k2.5",
            openrouter_secondary_reviewer_model=_get_str(
                "OPENROUTER_SECONDARY_REVIEWER_MODEL", "z-ai/glm-5"
            )
            or "z-ai/glm-5",
            openrouter_http_referer=_get_str("OPENROUTER_HTTP_REFERER"),
            openrouter_x_title=_get_str("OPENROUTER_X_TITLE"),
            openrouter_reviewer_timeout_seconds=reviewer_timeout,
            openrouter_tool_call_timeout_seconds=tool_call_timeout,
            openrouter_max_concurrent_requests=max_concurrency,
            openrouter_fixed_output_tokens=fixed_output_tokens,
            openrouter_context_overhead_tokens=overhead_tokens,
            openrouter_model_metadata_ttl_seconds=_get_int("OPENROUTER_MODEL_METADATA_TTL_SECONDS", 3600),
            openrouter_max_input_chars=max_input_chars,
            openrouter_include_reasoning=_get_bool("OPENROUTER_INCLUDE_REASONING", False),
            lad_serena_max_tool_calls=_get_int("LAD_SERENA_MAX_TOOL_CALLS", 32),
            lad_serena_tool_timeout_seconds=_get_int("LAD_SERENA_TOOL_TIMEOUT_SECONDS", 30),
            lad_serena_max_tool_result_chars=_get_int("LAD_SERENA_MAX_TOOL_RESULT_CHARS", 12000),
            lad_serena_max_total_chars=_get_int("LAD_SERENA_MAX_TOTAL_CHARS", 50000),
            lad_serena_max_dir_entries=_get_int("LAD_SERENA_MAX_DIR_ENTRIES", 100),
            lad_serena_max_search_results=_get_int("LAD_SERENA_MAX_SEARCH_RESULTS", 20),
        )
