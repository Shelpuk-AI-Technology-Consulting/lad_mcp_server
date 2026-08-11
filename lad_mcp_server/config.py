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
    # Optional since a deployment may route every reviewer to a direct provider and
    # never touch OpenRouter. Normalised to "" by ReviewService so the clients keep
    # their `str` contract; they then omit auth entirely rather than construct an
    # OpenAI SDK client, which would read the ambient OPENAI_API_KEY.
    openrouter_api_key: str | None

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
    zai_coding_plan_key: str | None = None
    kimi_code_api_key: str | None = None
    deepseek_api_key: str | None = None
    ollama_api_key: str | None = None
    openai_compat_base_url: str | None = None
    openai_compat_api_key: str | None = None
    intermittent_review_calls: int = 2
    intermittent_max_output_tokens: int = 1500

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

        # Direct-provider credentials are read before validation, because
        # OPENROUTER_API_KEY is only required when nothing else can serve a review.
        # A user behind a corporate gateway cannot reach openrouter.ai at all, so
        # demanding a key they cannot obtain would block the whole product.
        zai_key = _get_str("ZAI_CODING_PLAN_KEY")
        kimi_key = _get_str("KIMI_CODE_API_KEY")
        deepseek_key = _get_str("DEEPSEEK_API_KEY")
        ollama_key = _get_str("OLLAMA_API_KEY")
        # Stripped because this value both enables the provider and becomes a URL: a
        # whitespace-only setting would otherwise look configured and then fail at
        # request time with "unknown url type".
        openai_compat_base_url = (_get_str("OPENAI_COMPAT_BASE_URL") or "").strip() or None
        openai_compat_api_key = _get_str("OPENAI_COMPAT_API_KEY")

        # The base URL alone enables the OpenAI-compatible provider: a local gateway
        # (vLLM, a dev LiteLLM) legitimately needs no key.
        has_direct_provider = any(
            [zai_key, kimi_key, deepseek_key, ollama_key, openai_compat_base_url]
        )
        if not api_key and not has_direct_provider:
            raise ValueError(
                "No provider credentials configured. Set OPENROUTER_API_KEY, or configure a "
                "direct provider: OPENAI_COMPAT_BASE_URL (any OpenAI-compatible endpoint), "
                "ZAI_CODING_PLAN_KEY, KIMI_CODE_API_KEY, DEEPSEEK_API_KEY or OLLAMA_API_KEY."
            )

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
        if tool_call_timeout <= reviewer_timeout:
            raise ValueError(
                "OPENROUTER_TOOL_CALL_TIMEOUT_SECONDS must be > OPENROUTER_REVIEWER_TIMEOUT_SECONDS "
                "(strictly greater, not equal — equal values risk the reviewer and tool-call "
                "timeouts firing simultaneously, causing premature reviewer cancellation)."
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

        intermittent_review_calls = _get_int("INTERMITTENT_REVIEW_CALLS", 2)
        if intermittent_review_calls < 0:
            raise ValueError("INTERMITTENT_REVIEW_CALLS must be >= 0")

        intermittent_max_output_tokens = _get_int("INTERMITTENT_MAX_OUTPUT_TOKENS", 1500)
        if intermittent_max_output_tokens <= 0:
            raise ValueError("INTERMITTENT_MAX_OUTPUT_TOKENS must be > 0")

        return Settings(
            openrouter_api_key=api_key,
            openrouter_primary_reviewer_model=_get_str(
                "OPENROUTER_PRIMARY_REVIEWER_MODEL", "google/gemma-4-31b-it"
            )
            or "google/gemma-4-31b-it",
            openrouter_secondary_reviewer_model=_get_str(
                "OPENROUTER_SECONDARY_REVIEWER_MODEL", "minimax/minimax-m2.7"
            )
            or "minimax/minimax-m2.7",
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
            lad_serena_max_total_chars=_get_int("LAD_SERENA_MAX_TOTAL_CHARS", 100000),
            lad_serena_max_dir_entries=_get_int("LAD_SERENA_MAX_DIR_ENTRIES", 100),
            lad_serena_max_search_results=_get_int("LAD_SERENA_MAX_SEARCH_RESULTS", 20),
            zai_coding_plan_key=zai_key,
            kimi_code_api_key=kimi_key,
            deepseek_api_key=deepseek_key,
            ollama_api_key=ollama_key,
            openai_compat_base_url=openai_compat_base_url,
            openai_compat_api_key=openai_compat_api_key,
            intermittent_review_calls=intermittent_review_calls,
            intermittent_max_output_tokens=intermittent_max_output_tokens,
        )
