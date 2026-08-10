from __future__ import annotations

import re
import asyncio
import atexit
import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from lad_mcp_server.config import Settings
from lad_mcp_server.file_context import FileContextBuilder
from lad_mcp_server.markdown import final_egress_redaction, format_aggregated_output
from lad_mcp_server.model_metadata import ModelMetadataError, OpenRouterModelsClient
from lad_mcp_server.ollama_cloud_client import OllamaCloudClient, is_ollama_model, normalize_ollama_model_name
from lad_mcp_server.openrouter_client import OpenRouterCallResult, OpenRouterClient, OpenRouterClientError
from lad_mcp_server.path_utils import is_dangerous_repo_root
from lad_mcp_server.prompts import (
    force_finalize_system_message,
    intermittent_review_finalize_user_message,
    system_prompt_code_review,
    system_prompt_system_design_review,
    user_prompt_code_review,
    user_prompt_system_design_review,
)
from lad_mcp_server.redaction import redact_text
from lad_mcp_server.schemas import CodeReviewRequest, SystemDesignReviewRequest, ValidationError
from lad_mcp_server.serena_bridge import BASELINE_REQUIRED_MEMORIES, SerenaContext, SerenaLimits, SerenaToolError
from lad_mcp_server.token_budget import TokenBudget, TokenBudgetError
from lad_mcp_server.deepseek_client import DeepSeekClient, is_deepseek_model, normalize_deepseek_model_name
from lad_mcp_server.kimi_code_client import KimiCodeClient, is_kimi_model, normalize_kimi_model_name
from lad_mcp_server.zai_coding_client import ZaiCodingClient, is_zai_model, normalize_zai_model_name


log = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 3  # conservative for mixed tokenizers
OPENROUTER_CALL_TIMEOUT_SAFETY_MARGIN_SECONDS = 5  # avoid racing external tool-call deadlines
TOOL_CHOICE_FALLBACK_TTL_SECONDS = 600
TOOL_CHOICE_FALLBACK_CACHE_MAX_MODELS = 128
SYSTEM_ROLE_FALLBACK_TTL_SECONDS = 3600
SYSTEM_ROLE_FALLBACK_CACHE_MAX_MODELS = 128
KIMI_FALLBACK_TTL_SECONDS = 600
KIMI_FALLBACK_CACHE_MAX_MODELS = 128
# Bounds on the intermittent side-call timeout, which is derived from the reviewer
# timeout rather than fixed — hence "MAX"/"MIN" rather than a single "TIMEOUT".
INTERMITTENT_REVIEW_MAX_TIMEOUT_SECONDS = 45
INTERMITTENT_REVIEW_MIN_TIMEOUT_SECONDS = 20
INTERMITTENT_REVIEW_TIMEOUT_DIVISOR = 8
CONSECUTIVE_DEGRADED_TOOL_OUTPUTS_GUARD = 2
TOOL_DEGRADATION_SYSTEM_HINT = (
    "Tool budget/state failed for the latest tool output. "
    "Treat recent tool output as unreliable and continue conservatively."
)
DEEPER_EXPLORATION_TOOL_NAMES = frozenset(
    {"list_dir", "read_file", "read_file_window", "search_for_pattern", "find_symbol"}
)

PREFLIGHT_TOOL_NAMES = frozenset(
    {"activate_project", "read_project_overview", "read_baseline_memories"}
)

EXPLORATION_DIGEST_MAX_SNIPPET_CHARS = 200

# Keys that are OpenRouter-specific and must not be forwarded to direct providers.
_OPENROUTER_ONLY_KEYS = frozenset({"include_reasoning", "max_completion_tokens"})

_SUBSTANTIVE_PLACEHOLDER_RE = re.compile(
    r"^\*\([^)]+\)\*$|"  # *(No Summary provided by reviewer)*
    r"^#{1,4}\s+.+$",     # markdown section headers
    re.MULTILINE,
)

FINAL_RENDER_RESERVE_SECONDS = 5  # reserved for rendering + aggregation after grace


def _bounded_digest_snippet(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    snippet = value.strip().replace("\n", " ")
    if len(snippet) > EXPLORATION_DIGEST_MAX_SNIPPET_CHARS:
        return snippet[:EXPLORATION_DIGEST_MAX_SNIPPET_CHARS] + "…"
    return snippet


_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8)
atexit.register(_TOOL_EXECUTOR.shutdown, wait=False, cancel_futures=True)


def _truncate_to_chars(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _exc_message(exc: BaseException) -> str:
    msg = str(exc).strip()
    return msg if msg else exc.__class__.__name__


def _build_tool_message(tool_call_id: str, name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def _build_assistant_tool_calls_message(
    tool_calls: list[dict[str, Any]],
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
    if content:
        msg["content"] = content
    # `reasoning_content` is required by Z.AI Coding Plan's Preserved Thinking contract.
    # Non-Z.AI clients strip the key before sending (see openrouter_client.strip_reasoning_content).
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    return msg


def _build_system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def _build_user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _is_degraded_tool_output(content: str) -> tuple[bool, str]:
    text = (content or "").strip()
    if not text:
        return True, "empty output"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return True, "invalid JSON output"
    if not isinstance(payload, dict):
        return True, "non-object JSON output"
    if payload.get("tool_status") == "budget_exhausted":
        return True, "tool_status=budget_exhausted"
    return False, ""


def _append_tooling_degradation_summary(
    markdown: str,
    *,
    degraded_outputs_count: int,
    consecutive_guard_triggered: bool,
) -> str:
    if degraded_outputs_count <= 0:
        return markdown
    base = markdown.rstrip()
    summary = (
        "\n\n## Tooling Degradation Summary\n"
        f"- Degraded tool outputs: {degraded_outputs_count}\n"
        f"- Consecutive guard triggered: {'yes' if consecutive_guard_triggered else 'no'}\n"
        "- Notes: tool budget/state failed in the Serena loop; output may be partially degraded.\n"
    )
    return base + summary


def _normalize_memory_name(name: str) -> str:
    return name if name.endswith(".md") else f"{name}.md"


def _load_json_object(text: str | None) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_read_memory_name(arguments_json: str) -> str | None:
    args = _load_json_object(arguments_json)
    if args is None:
        return None
    raw_name = args.get("name")
    if not isinstance(raw_name, str) or raw_name.strip() == "":
        return None
    return _normalize_memory_name(raw_name.strip())


def _extract_tool_result_object(tool_output: str) -> dict[str, Any] | None:
    outer = _load_json_object(tool_output)
    if outer is None:
        return None
    inner_raw = outer.get("tool_result_json")
    inner = _load_json_object(inner_raw) if isinstance(inner_raw, str) else None
    return inner if inner is not None else outer


def _add_unique_list_item(items: list[str], value: str | None) -> None:
    if value and value not in items:
        items.append(value)


def _extract_path_from_arguments(arguments_json: str) -> str | None:
    args = _load_json_object(arguments_json)
    if args is None:
        return None
    raw_path = args.get("path") or args.get("relative_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    # Repo-relative paths are POSIX identifiers on both sides of the tool boundary
    # (see `SerenaContext._repo_relative_posix`). Without normalizing the model's raw
    # argument, a Windows-style `src\a.txt` counts as a second distinct path in the
    # exploration digest alongside the normalized `src/a.txt` from the tool result.
    return raw_path.strip().replace("\\", "/")


def _update_exploration_digest(
    digest: "ExplorationDigest",
    fn_name: str,
    arguments_json: str,
    tool_output: str,
    is_degraded: bool,
) -> None:
    digest.total_tool_calls += 1
    if fn_name:
        digest.tools_invoked.add(fn_name)
    if is_degraded:
        digest.degraded_outputs += 1

    args_path = _extract_path_from_arguments(arguments_json)
    result = _extract_tool_result_object(tool_output) or {}

    if fn_name in {"read_file", "read_file_window"}:
        path = result.get("path") if isinstance(result.get("path"), str) else args_path
        if isinstance(path, str) and path:
            _add_unique_list_item(digest.files_read, path)
            digest.paths_visited.add(path)
    elif fn_name == "read_memory":
        memory_name = result.get("name") if isinstance(result.get("name"), str) else None
        if memory_name is None:
            memory_name = _extract_read_memory_name(arguments_json)
        if memory_name:
            digest.memories_used.add(_normalize_memory_name(memory_name))
    elif fn_name == "read_baseline_memories":
        loaded = result.get("loaded")
        if isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    digest.memories_used.add(_normalize_memory_name(item["name"]))
        present = result.get("present")
        if isinstance(present, list):
            for item in present:
                if isinstance(item, str):
                    digest.memories_used.add(_normalize_memory_name(item))
    elif fn_name == "find_symbol":
        if args_path:
            digest.paths_visited.add(args_path)
        symbols = result.get("symbols") or result.get("result")
        if isinstance(symbols, list):
            for item in symbols[:20]:
                if isinstance(item, str):
                    _add_unique_list_item(digest.symbols_found, item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("name_path")
                    if isinstance(name, str):
                        _add_unique_list_item(digest.symbols_found, name)
    elif fn_name == "search_for_pattern":
        if args_path:
            digest.paths_visited.add(args_path)
        matches = result.get("matches")
        if isinstance(matches, list):
            for match in matches[:20]:
                snippet = _bounded_digest_snippet(match)
                _add_unique_list_item(digest.search_matches, snippet)


def _preflight_validation_message(missing_required: set[str]) -> str:
    if not missing_required:
        return "Preflight memory checklist validation: all required preflight memories are loaded."
    missing = ", ".join(sorted(missing_required))
    return f"Preflight memory checklist validation: missing required preflight memories: {missing}."


def _skipped_preflight_warning_message(missing_required: set[str]) -> str:
    missing = ", ".join(sorted(missing_required))
    return (
        "Skipped required preflight memories before deeper exploration: "
        f"{missing}. Call `read_baseline_memories` or `read_memory` to fill gaps."
    )


def _is_substantive_review_content(content: str) -> bool:
    """Check whether review content has at least one substantive line beyond placeholders."""
    if not content or not content.strip():
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SUBSTANTIVE_PLACEHOLDER_RE.match(stripped):
            continue
        # At least one line with 3+ words that isn't a placeholder
        words = stripped.split()
        if len(words) >= 3:
            return True
    return False


def _build_tool_trace_summary(
    *,
    model: str,
    timeout_seconds: int,
    tool_calls_made: int,
    tools_invoked: set[str],
    memories_used: set[str],
    paths_used: set[str],
) -> str:
    """Synthesize a structured markdown summary from the Serena tool call trace."""
    tools_list = ", ".join(sorted(tools_invoked)) if tools_invoked else "(none)"
    memories_list = ", ".join(sorted(memories_used)) if memories_used else "(none)"
    paths_list = ", ".join(f"`{p}`" for p in sorted(paths_used)) if paths_used else "(none)"

    return (
        "## Summary\n"
        f"Reviewer (`{model}`) timed out after {timeout_seconds}s during tool-assisted exploration.\n"
        "This is a **tool-exploration trace summary**, not a model-generated review.\n\n"
        "## Key Findings\n"
        "- **Info**: No model-generated findings — reviewer timed out before synthesis.\n"
        f"- **Info**: Explored {len(paths_used)} file(s) using {tool_calls_made} tool call(s).\n\n"
        "## Exploration Statistics\n"
        f"- **Tool calls made**: {tool_calls_made}\n"
        f"- **Tools invoked**: {tools_list}\n"
        f"- **Memories accessed**: {memories_list}\n"
        f"- **Files explored**: {len(paths_used)}\n\n"
        "## Files Explored\n"
        f"{paths_list}\n\n"
        "## Tools Used\n"
        f"{tools_list}\n\n"
        "## Recommendations\n"
        "- Re-run the review with a longer timeout to get a full model-generated review.\n"
        "- Consider reducing the scope (fewer files) to fit within the timeout budget.\n\n"
        "## Questions / Unknowns\n"
        "- What substantive findings would the reviewer have produced?\n"
    )


@dataclass(frozen=True)
class ReviewerOutcome:
    ok: bool
    model: str
    used_serena: bool
    serena_disabled_reason: str | None
    serena_activated_project: str | None
    serena_used_tools: tuple[str, ...]
    serena_used_memories: tuple[str, ...]
    serena_used_paths: tuple[str, ...]
    markdown: str
    error: str | None
    provider: str = "openrouter"
    provider_note: str | None = None
    is_intermittent: bool = False



@dataclass
class ExplorationDigest:
    files_read: list[str] = field(default_factory=list)
    symbols_found: list[str] = field(default_factory=list)
    search_matches: list[str] = field(default_factory=list)
    llm_findings: list[str] = field(default_factory=list)
    llm_recommendations: list[str] = field(default_factory=list)
    llm_open_questions: list[str] = field(default_factory=list)
    tools_invoked: set[str] = field(default_factory=set)
    memories_used: set[str] = field(default_factory=set)
    paths_visited: set[str] = field(default_factory=set)
    degraded_outputs: int = 0
    total_tool_calls: int = 0  # includes preflight calls, used for coverage reporting


class IntermittentReviewState:
    """
    Per-reviewer holder for the latest intermittent (partial) review snapshot.

    Concurrency invariant: this holder is mutated ONLY from inside the side coroutine
    `_run_intermittent_review_call`, which runs in the same event loop as the main
    `_tool_loop`. The timeout handler in `_run_single_reviewer` reads the holder from
    the same event loop. Single-threaded access — no lock required.
    """

    __slots__ = (
        "latest_markdown",
        "snapshot_tool_call_index",
        "tool_calls_so_far",
        "in_flight_task",
        "queued_snapshot",
        "_preflight_complete",
        "digest",
        "last_status",
        "last_error",
        "last_started_at",
        "last_finished_at",
    )

    def __init__(self) -> None:
        self.latest_markdown: str | None = None
        self.snapshot_tool_call_index: int = 0
        self.tool_calls_so_far: int = 0
        self.in_flight_task: asyncio.Task[None] | None = None
        self.queued_snapshot: dict[str, Any] | None = None
        self._preflight_complete: bool = False
        self.digest: ExplorationDigest = ExplorationDigest()
        self.last_status: str = "never_dispatched"
        self.last_error: str | None = None
        self.last_started_at: float | None = None
        self.last_finished_at: float | None = None


@dataclass(frozen=True)
class ReviewerConfig:
    model: str
    budget: TokenBudget
    supported_parameters: tuple[str, ...]
    tool_calling_supported: bool
    tool_choice_supported: bool
    serena_ctx: SerenaContext | None
    serena_disabled_reason: str | None
    use_zai_direct: bool = False
    direct_model_name: str | None = None
    use_kimi_direct: bool = False
    direct_kimi_model_name: str | None = None
    use_deepseek_direct: bool = False
    direct_deepseek_model_name: str | None = None
    use_ollama_direct: bool = False
    direct_ollama_model_name: str | None = None


class ReviewService:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        settings: Settings | None = None,
        openrouter_client: OpenRouterClient | None = None,
        models_client: OpenRouterModelsClient | None = None,
        zai_client: ZaiCodingClient | Any | None = None,
        kimi_client: KimiCodeClient | Any | None = None,
        deepseek_client: DeepSeekClient | Any | None = None,
        ollama_client: OllamaCloudClient | Any | None = None,
    ) -> None:
        self._settings = settings or Settings.from_env()
        self._openrouter = openrouter_client or OpenRouterClient(
            api_key=self._settings.openrouter_api_key,
            http_referer=self._settings.openrouter_http_referer,
            x_title=self._settings.openrouter_x_title,
            max_concurrent_requests=self._settings.openrouter_max_concurrent_requests,
        )
        self._models = models_client or OpenRouterModelsClient(
            api_key=self._settings.openrouter_api_key,
            ttl_seconds=self._settings.openrouter_model_metadata_ttl_seconds,
        )
        self._zai = zai_client
        if self._zai is None and self._settings.zai_coding_plan_key:
            self._zai = ZaiCodingClient(
                api_key=self._settings.zai_coding_plan_key,
                max_concurrent_requests=self._settings.openrouter_max_concurrent_requests,
            )
        self._kimi = kimi_client
        if self._kimi is None and self._settings.kimi_code_api_key:
            self._kimi = KimiCodeClient(
                api_key=self._settings.kimi_code_api_key,
                max_concurrent_requests=self._settings.openrouter_max_concurrent_requests,
            )
        self._deepseek = deepseek_client
        if self._deepseek is None and self._settings.deepseek_api_key:
            self._deepseek = DeepSeekClient(
                api_key=self._settings.deepseek_api_key,
                max_concurrent_requests=self._settings.openrouter_max_concurrent_requests,
            )
        self._ollama = ollama_client
        if self._ollama is None and self._settings.ollama_api_key:
            self._ollama = OllamaCloudClient(
                api_key=self._settings.ollama_api_key,
                max_concurrent_requests=self._settings.openrouter_max_concurrent_requests,
            )
        # NOTE: `repo_root` here is treated as a *default* only.
        # The reviewed project is inferred per tool invocation (prefer CODEX_WORKSPACE_ROOT; otherwise absolute-path
        # inference; otherwise CWD), so Lad can be used across many projects with one MCP configuration.
        self._default_repo_root = repo_root.resolve() if repo_root is not None else None
        self._tool_executor = _TOOL_EXECUTOR
        self._intermittent_timeout = min(
            INTERMITTENT_REVIEW_MAX_TIMEOUT_SECONDS,
            max(
                INTERMITTENT_REVIEW_MIN_TIMEOUT_SECONDS,
                self._settings.openrouter_reviewer_timeout_seconds // INTERMITTENT_REVIEW_TIMEOUT_DIVISOR,
            ),
        )
        self._tool_choice_fallback_until_by_model: dict[str, float] = {}
        self._tool_choice_fallback_lock = threading.Lock()
        self._system_role_fallback_until_by_model: dict[str, float] = {}
        self._system_role_fallback_lock = threading.Lock()
        self._kimi_fallback_until_by_model: dict[str, float] = {}
        self._kimi_fallback_lock = threading.Lock()

    @staticmethod
    def _tool_choice_model_key(model: str) -> str:
        return model.strip()

    def _is_tool_choice_fallback_active(self, model: str) -> bool:
        key = self._tool_choice_model_key(model)
        now = time.monotonic()
        with self._tool_choice_fallback_lock:
            expires_at = self._tool_choice_fallback_until_by_model.get(key)
            if expires_at is None:
                self._cleanup_tool_choice_fallback_cache_locked(now)
                return False
            if expires_at <= now:
                self._tool_choice_fallback_until_by_model.pop(key, None)
                self._cleanup_tool_choice_fallback_cache_locked(now)
                return False
            self._cleanup_tool_choice_fallback_cache_locked(now)
            return True

    def _remember_tool_choice_fallback(self, model: str) -> None:
        key = self._tool_choice_model_key(model)
        now = time.monotonic()
        with self._tool_choice_fallback_lock:
            already_active = (self._tool_choice_fallback_until_by_model.get(key) or 0.0) > now
            self._tool_choice_fallback_until_by_model[key] = now + float(TOOL_CHOICE_FALLBACK_TTL_SECONDS)
            self._cleanup_tool_choice_fallback_cache_locked(now)
        if not already_active:
            log.info("Tool-choice fallback cache activated for model '%s' (%ss)", key, TOOL_CHOICE_FALLBACK_TTL_SECONDS)

    def _cleanup_tool_choice_fallback_cache_locked(self, now: float) -> None:
        expired = [k for k, v in self._tool_choice_fallback_until_by_model.items() if v <= now]
        for k in expired:
            self._tool_choice_fallback_until_by_model.pop(k, None)
        while len(self._tool_choice_fallback_until_by_model) > TOOL_CHOICE_FALLBACK_CACHE_MAX_MODELS:
            oldest_key = min(self._tool_choice_fallback_until_by_model, key=self._tool_choice_fallback_until_by_model.get)
            self._tool_choice_fallback_until_by_model.pop(oldest_key, None)

    def _kimi_model_key(self, model: str) -> str:
        return model.strip()

    def _is_kimi_fallback_active(self, model: str) -> bool:
        key = self._kimi_model_key(model)
        now = time.monotonic()
        with self._kimi_fallback_lock:
            expires_at = self._kimi_fallback_until_by_model.get(key)
            if expires_at is None:
                self._cleanup_kimi_fallback_cache_locked(now)
                return False
            if expires_at <= now:
                self._kimi_fallback_until_by_model.pop(key, None)
                self._cleanup_kimi_fallback_cache_locked(now)
                return False
            self._cleanup_kimi_fallback_cache_locked(now)
            return True

    def _remember_kimi_fallback(self, model: str) -> None:
        key = self._kimi_model_key(model)
        now = time.monotonic()
        with self._kimi_fallback_lock:
            already_active = (self._kimi_fallback_until_by_model.get(key) or 0.0) > now
            self._kimi_fallback_until_by_model[key] = now + float(KIMI_FALLBACK_TTL_SECONDS)
            self._cleanup_kimi_fallback_cache_locked(now)
        if not already_active:
            log.info("Kimi Code fallback cache activated for model '%s' (%ss)", key, KIMI_FALLBACK_TTL_SECONDS)

    def _cleanup_kimi_fallback_cache_locked(self, now: float) -> None:
        expired = [k for k, v in self._kimi_fallback_until_by_model.items() if v <= now]
        for k in expired:
            self._kimi_fallback_until_by_model.pop(k, None)
        while len(self._kimi_fallback_until_by_model) > KIMI_FALLBACK_CACHE_MAX_MODELS:
            oldest_key = min(self._kimi_fallback_until_by_model, key=self._kimi_fallback_until_by_model.get)
            self._kimi_fallback_until_by_model.pop(oldest_key, None)

    @staticmethod
    def _is_retryable_tool_choice_compatibility_error(exc: OpenRouterClientError) -> bool:
        msg = _exc_message(exc).lower()
        if "tool_choice" not in msg and "tool choice" not in msg:
            return False
        return (
            "no endpoints found" in msg
            or "support the provided" in msg
            or "unsupported" in msg
            or "routing" in msg
            or "must be auto" in msg
        )

    @staticmethod
    def _system_role_model_key(model: str) -> str:
        return model.strip()

    def _is_system_role_fallback_active(self, model: str) -> bool:
        key = self._system_role_model_key(model)
        now = time.monotonic()
        with self._system_role_fallback_lock:
            expires_at = self._system_role_fallback_until_by_model.get(key)
            if expires_at is None:
                self._cleanup_system_role_fallback_cache_locked(now)
                return False
            if expires_at <= now:
                self._system_role_fallback_until_by_model.pop(key, None)
                self._cleanup_system_role_fallback_cache_locked(now)
                return False
            self._cleanup_system_role_fallback_cache_locked(now)
            return True

    def _remember_system_role_fallback(self, model: str) -> None:
        key = self._system_role_model_key(model)
        now = time.monotonic()
        with self._system_role_fallback_lock:
            already_active = (self._system_role_fallback_until_by_model.get(key) or 0.0) > now
            self._system_role_fallback_until_by_model[key] = now + float(SYSTEM_ROLE_FALLBACK_TTL_SECONDS)
            self._cleanup_system_role_fallback_cache_locked(now)
        if not already_active:
            log.info("System-role fallback cache activated for model '%s' (%ss)", key, SYSTEM_ROLE_FALLBACK_TTL_SECONDS)

    def _cleanup_system_role_fallback_cache_locked(self, now: float) -> None:
        expired = [k for k, v in self._system_role_fallback_until_by_model.items() if v <= now]
        for k in expired:
            self._system_role_fallback_until_by_model.pop(k, None)
        while len(self._system_role_fallback_until_by_model) > SYSTEM_ROLE_FALLBACK_CACHE_MAX_MODELS:
            oldest_key = min(
                self._system_role_fallback_until_by_model,
                key=self._system_role_fallback_until_by_model.get,
            )
            self._system_role_fallback_until_by_model.pop(oldest_key, None)

    @staticmethod
    def _is_retryable_system_role_error(exc: OpenRouterClientError) -> bool:
        msg = _exc_message(exc).lower()
        return (
            "invalid message role: system" in msg
            or "invalid params, chat content has invalid message role" in msg and "system" in msg
            or "role: system" in msg and "invalid" in msg
        )

    def _adapt_messages_for_model(self, model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._is_system_role_fallback_active(model):
            return messages
        adapted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                new_msg = dict(msg)
                new_msg["role"] = "user"
                adapted.append(new_msg)
            else:
                adapted.append(msg)
        return adapted

    @staticmethod
    def _build_tool_choice_attempts(
        *,
        tools: list[dict[str, Any]] | None,
        preferred: str | dict[str, Any] | None,
    ) -> list[str | dict[str, Any] | None]:
        attempts: list[str | dict[str, Any] | None] = [preferred]
        if not tools:
            return attempts

        if not any(a == "auto" for a in attempts):
            attempts.append("auto")
        if not any(a is None for a in attempts):
            attempts.append(None)
        return attempts

    async def _call_openrouter_with_tool_choice_fallback(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout_seconds: int,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None,
        preferred_tool_choice: str | dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
    ) -> OpenRouterCallResult:
        attempts = self._build_tool_choice_attempts(tools=tools, preferred=preferred_tool_choice)
        last_exc: OpenRouterClientError | None = None
        for idx, tool_choice in enumerate(attempts):
            try:
                return await self._openrouter.chat_completion(
                    model=model,
                    messages=self._adapt_messages_for_model(model, messages),
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    extra_body=extra_body,
                )
            except OpenRouterClientError as exc:
                last_exc = exc
                if self._is_retryable_system_role_error(exc):
                    self._remember_system_role_fallback(model)
                    try:
                        return await self._openrouter.chat_completion(
                            model=model,
                            messages=self._adapt_messages_for_model(model, messages),
                            timeout_seconds=timeout_seconds,
                            max_output_tokens=max_output_tokens,
                            tools=tools,
                            tool_choice=tool_choice,
                            extra_body=extra_body,
                        )
                    except OpenRouterClientError as retry_exc:
                        last_exc = retry_exc
                        exc = retry_exc
                if not tools or not self._is_retryable_tool_choice_compatibility_error(exc):
                    raise exc
                if tool_choice is not None:
                    self._remember_tool_choice_fallback(model)
                if idx == len(attempts) - 1:
                    raise exc
                log.info(
                    "Retrying OpenRouter call for model '%s' with fallback tool_choice=%r",
                    model,
                    attempts[idx + 1],
                )
                continue

        # Defensive; loop always returns or raises.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenRouter fallback call ended without result")

    async def _call_model_with_provider_fallback(
        self,
        *,
        model: str,
        direct_model_name: str | None,
        use_zai_direct: bool,
        direct_kimi_model_name: str | None,
        use_kimi_direct: bool,
        direct_deepseek_model_name: str | None,
        use_deepseek_direct: bool,
        direct_ollama_model_name: str | None = None,
        use_ollama_direct: bool = False,
        messages: list[dict[str, Any]],
        timeout_seconds: int,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None,
        preferred_tool_choice: str | dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
        provider_used: list[str],
        provider_notes: list[str],
    ) -> OpenRouterCallResult:
        direct_extra_body = (
            {k: v for k, v in extra_body.items() if k not in _OPENROUTER_ONLY_KEYS}
            if extra_body
            else None
        )

        if use_deepseek_direct and self._deepseek is not None and direct_deepseek_model_name:
            try:
                result = await self._deepseek.chat_completion(
                    model=direct_deepseek_model_name,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    tools=tools,
                    tool_choice=preferred_tool_choice,
                    extra_body=direct_extra_body,
                )
                provider_used[:] = ["deepseek"]
                return result
            except Exception as exc:
                note = f"DeepSeek endpoint failed: {_exc_message(exc)}. Fell back to OpenRouter."
                provider_notes.append(note)
                provider_used[:] = ["openrouter"]
                log.warning("Direct DeepSeek call failed for model '%s'; falling back to OpenRouter: %s", model, note)

        if use_ollama_direct and self._ollama is not None and direct_ollama_model_name:
            try:
                result = await self._ollama.chat_completion(
                    model=direct_ollama_model_name,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    tools=tools,
                    tool_choice=preferred_tool_choice,
                    extra_body=direct_extra_body,
                )
                provider_used[:] = ["ollama"]
                return result
            except Exception as exc:
                note = f"Ollama Cloud endpoint failed: {_exc_message(exc)}. Fell back to OpenRouter."
                provider_notes.append(note)
                provider_used[:] = ["openrouter"]
                log.warning("Direct Ollama call failed for model '%s'; falling back to OpenRouter: %s", model, note)

        if use_zai_direct and self._zai is not None and direct_model_name:
            try:
                result = await self._zai.chat_completion(
                    model=direct_model_name,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                    tools=tools,
                    tool_choice=preferred_tool_choice,
                    extra_body=direct_extra_body,
                )
                provider_used[:] = ["zai_coding_plan"]
                return result
            except Exception as exc:
                note = f"Z.AI Coding Plan endpoint failed: {_exc_message(exc)}. Fell back to OpenRouter."
                provider_notes.append(note)
                provider_used[:] = ["openrouter"]
                log.warning("Direct Z.AI call failed for model '%s'; falling back to OpenRouter: %s", model, note)

        if use_kimi_direct and self._kimi is not None and direct_kimi_model_name:
            if not self._is_kimi_fallback_active(model):
                try:
                    result = await self._kimi.chat_completion(
                        model=direct_kimi_model_name,
                        messages=messages,
                        timeout_seconds=timeout_seconds,
                        max_output_tokens=max_output_tokens,
                        tools=tools,
                        tool_choice=preferred_tool_choice,
                        extra_body=direct_extra_body,
                    )
                    provider_used[:] = ["kimi_code"]
                    return result
                except Exception as exc:
                    self._remember_kimi_fallback(model)
                    note = f"Kimi Code endpoint failed: {_exc_message(exc)}. Fell back to OpenRouter."
                    provider_notes.append(note)
                    provider_used[:] = ["openrouter"]
                    log.warning("Direct Kimi Code call failed for model '%s'; falling back to OpenRouter: %s", model, note)

        provider_used[:] = ["openrouter"]
        return await self._call_openrouter_with_tool_choice_fallback(
            model=model,
            messages=messages,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            tools=tools,
            preferred_tool_choice=preferred_tool_choice,
            extra_body=extra_body,
        )

    async def _run_intermittent_review_call(
        self,
        *,
        model: str,
        messages_snapshot: list[dict[str, Any]],
        use_zai_direct: bool,
        direct_model_name: str | None,
        use_kimi_direct: bool,
        direct_kimi_model_name: str | None,
        use_deepseek_direct: bool = False,
        direct_deepseek_model_name: str | None = None,
        use_ollama_direct: bool = False,
        direct_ollama_model_name: str | None = None,
        extra_body: dict[str, Any] | None,
        max_output_tokens: int,
        snapshot_tool_call_index: int,
        state: IntermittentReviewState,
    ) -> None:
        """
        Fire a side LLM call (no tools) to produce a partial review snapshot.

        Real errors (timeout, OpenRouter errors, whitespace-only responses) are swallowed.
        `asyncio.CancelledError` is re-raised so cancellation propagates correctly; we use
        `except Exception` for the catch-all to avoid swallowing `SystemExit`/`KeyboardInterrupt`.
        Concurrency invariant: this coroutine is the ONLY writer of `state.latest_markdown`
        and `state.snapshot_tool_call_index`. Reads happen from the same event loop.
        """
        # Fresh per-call provider trackers so the side call doesn't mutate the main reviewer's.
        side_provider_used: list[str] = ["openrouter"]
        side_provider_notes: list[str] = []
        state.last_status = "running"
        state.last_started_at = time.monotonic()
        state.last_error = None
        try:
            result = await asyncio.wait_for(
                self._call_model_with_provider_fallback(
                    model=model,
                    direct_model_name=direct_model_name,
                    use_zai_direct=use_zai_direct,
                    direct_kimi_model_name=direct_kimi_model_name,
                    use_kimi_direct=use_kimi_direct,
                    direct_deepseek_model_name=direct_deepseek_model_name,
                    use_deepseek_direct=use_deepseek_direct,
                    direct_ollama_model_name=direct_ollama_model_name,
                    use_ollama_direct=use_ollama_direct,
                    messages=messages_snapshot,
                    timeout_seconds=self._intermittent_timeout,
                    max_output_tokens=max_output_tokens,
                    tools=None,
                    preferred_tool_choice=None,
                    extra_body=extra_body,
                    provider_used=side_provider_used,
                    provider_notes=side_provider_notes,
                ),
                timeout=self._intermittent_timeout,
            )
        except asyncio.CancelledError:
            state.last_status = "cancelled"
            state.last_finished_at = time.monotonic()
            raise
        except asyncio.TimeoutError:
            state.last_status = "timeout"
            state.last_error = f"side-call timed out after {self._intermittent_timeout}s"
            state.last_finished_at = time.monotonic()
            log.info("Intermittent review side call timed out for model '%s'", model)
            return
        except Exception as exc:
            state.last_status = "provider_error"
            state.last_error = str(exc)
            state.last_finished_at = time.monotonic()
            log.info("Intermittent review side call failed for model '%s': %s", model, exc)
            return

        content = (result.content or "").strip()
        state.last_finished_at = time.monotonic()
        if not content:
            state.last_status = "empty"
            log.info("Intermittent review side call for model '%s' returned empty/whitespace content", model)
        else:
            state.last_status = "completed"
            # Single-threaded event loop = atomic update without lock.
            state.latest_markdown = result.content
            state.snapshot_tool_call_index = snapshot_tool_call_index

        # Drop the reference to the completed task so the messages snapshot can be GC'd,
        # then start the latest queued snapshot request, if any.
        if state.in_flight_task is not None and state.in_flight_task.done():
            state.in_flight_task = None
        queued = state.queued_snapshot
        state.queued_snapshot = None
        if queued is not None:
            state.in_flight_task = asyncio.create_task(self._run_intermittent_review_call(**queued))

    def _cancel_in_flight_intermittent(self, state: IntermittentReviewState | None) -> None:
        """Best-effort cancellation of any pending intermittent side task. Safe to call on None / done."""
        if state is None:
            return
        task = state.in_flight_task
        if task is None or task.done():
            return
        task.cancel()

    def _dispatch_intermittent_review(
        self,
        *,
        state: IntermittentReviewState,
        model: str,
        messages: list[dict[str, Any]],
        use_zai_direct: bool,
        direct_model_name: str | None,
        use_kimi_direct: bool,
        direct_kimi_model_name: str | None,
        use_deepseek_direct: bool,
        direct_deepseek_model_name: str | None,
        use_ollama_direct: bool,
        direct_ollama_model_name: str | None,
        extra_body: dict[str, Any] | None,
        max_output_tokens: int,
    ) -> None:
        """
        Schedule a non-blocking side LLM call to produce a partial review snapshot.

        If a prior task is still in flight, keep it running and remember the latest
        requested snapshot. The running task will dispatch the queued snapshot after it completes.
        """
        # Use deepcopy: _tool_loop mutates message dicts in-place between dispatches.
        # A shallow copy would let the in-flight side-call observe partially mutated messages.
        snapshot_messages: list[dict[str, Any]] = copy.deepcopy(messages)
        snapshot_messages.append(_build_user_message(intermittent_review_finalize_user_message()))

        request = {
            "model": model,
            "messages_snapshot": snapshot_messages,
            "use_zai_direct": use_zai_direct,
            "direct_model_name": direct_model_name,
            "use_kimi_direct": use_kimi_direct,
            "direct_kimi_model_name": direct_kimi_model_name,
            "use_deepseek_direct": use_deepseek_direct,
            "direct_deepseek_model_name": direct_deepseek_model_name,
            "use_ollama_direct": use_ollama_direct,
            "direct_ollama_model_name": direct_ollama_model_name,
            "extra_body": extra_body,
            "max_output_tokens": max_output_tokens,
            "snapshot_tool_call_index": state.tool_calls_so_far,
            "state": state,
        }

        prev = state.in_flight_task
        if prev is not None and not prev.done():
            state.queued_snapshot = request
            return

        state.queued_snapshot = None
        task = asyncio.create_task(
            self._run_intermittent_review_call(
                model=model,
                messages_snapshot=snapshot_messages,
                use_zai_direct=use_zai_direct,
                direct_model_name=direct_model_name,
                use_kimi_direct=use_kimi_direct,
                direct_kimi_model_name=direct_kimi_model_name,
                use_deepseek_direct=use_deepseek_direct,
                direct_deepseek_model_name=direct_deepseek_model_name,
                use_ollama_direct=use_ollama_direct,
                direct_ollama_model_name=direct_ollama_model_name,
                extra_body=extra_body,
                max_output_tokens=max_output_tokens,
                snapshot_tool_call_index=state.tool_calls_so_far,
                state=state,
            )
        )
        state.in_flight_task = task

    @staticmethod
    def _walk_up_for_project_root(start: Path, *, max_depth: int = 25) -> Path:
        """
        Best-effort project root inference.

        Priority:
        - `.serena/` (enables Serena integration)
        - `.git/` (common VCS marker)

        The climb never promotes the root into a directory that
        :func:`~lad_mcp_server.path_utils.is_dangerous_repo_root` rejects, so a
        marker in such a location cannot make the whole review fail. A `start`
        that is itself dangerous is returned unchanged, leaving the caller's own
        guard to reject it.

        Args:
            start: Absolute, already-resolved directory to begin the climb from.
            max_depth: Maximum number of ancestors to inspect.

        Returns:
            The nearest marked ancestor strictly below the first dangerous
            ancestor. Falls back to ``start`` when no such marker exists,
            when ``start`` is itself dangerous, or when ``max_depth`` is
            exhausted.
        """
        cur = start
        for _ in range(max_depth):
            # Serena's global `~/.serena` is a user config directory, not a project
            # marker. Stop before adopting any root the caller is bound to reject.
            if is_dangerous_repo_root(cur):
                break
            if (cur / ".serena").is_dir():
                return cur
            if (cur / ".git").is_dir():
                return cur
            # Termination guarantee that does not depend on the guard above.
            # `is_dangerous_repo_root` happens to reject filesystem roots today, so
            # this rarely fires — but the loop must still terminate if that changes.
            if cur.parent == cur:
                break
            cur = cur.parent
        return start

    def _resolve_project_root(self, *, paths: list[str] | None) -> Path:
        # 1) Codex provides a workspace root for the current session.
        codex_root = os.getenv("CODEX_WORKSPACE_ROOT")
        if codex_root and codex_root.strip():
            pr = Path(codex_root).expanduser().resolve()
            if pr.exists() and pr.is_dir():
                return pr

        # 2) Infer from absolute paths (so one Lad process can review multiple repos).
        if paths:
            abs_dirs: list[str] = []
            for p in paths:
                pp = Path(p)
                if not pp.is_absolute():
                    abs_dirs = []
                    break
                resolved = pp.expanduser().resolve()
                if resolved.is_file():
                    resolved = resolved.parent
                abs_dirs.append(str(resolved))
            if abs_dirs:
                base = Path(os.path.commonpath(abs_dirs)).resolve()
                if base.is_file():
                    base = base.parent
                if base.exists() and base.is_dir():
                    return self._walk_up_for_project_root(base)

        # 3) Service default (if any), otherwise current working directory at call time.
        return (self._default_repo_root or Path.cwd()).resolve()

    async def system_design_review(self, **kwargs: Any) -> str:
        req = SystemDesignReviewRequest.validate(
            proposal=kwargs.get("proposal"),
            paths=kwargs.get("paths"),
            constraints=kwargs.get("constraints"),
            context=kwargs.get("context"),
            max_input_chars=self._settings.openrouter_max_input_chars,
        )

        async def _run() -> str:
            return await self._run_dual_review(
                tool_name="system_design_review",
                build_system_prompt=system_prompt_system_design_review,
                build_user_prompt=lambda tool_calling_enabled, redacted: user_prompt_system_design_review(
                    proposal=redacted.get("proposal")
                    or "(No proposal text provided. Use the embedded files below as the system design context.)",
                    constraints=redacted.get("constraints"),
                    context=redacted.get("context"),
                ),
                redaction_inputs={
                    "proposal": req.proposal,
                    "constraints": req.constraints,
                    "context": req.context,
                },
                requested_paths=req.paths,
            )

        try:
            return await asyncio.wait_for(_run(), timeout=self._settings.openrouter_tool_call_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Tool call timed out after {self._settings.openrouter_tool_call_timeout_seconds}s"
            ) from exc

    async def code_review(self, **kwargs: Any) -> str:
        req = CodeReviewRequest.validate(
            code=kwargs.get("code"),
            paths=kwargs.get("paths"),
            context=kwargs.get("context"),
            max_input_chars=self._settings.openrouter_max_input_chars,
        )

        async def _run() -> str:
            return await self._run_dual_review(
                tool_name="code_review",
                build_system_prompt=system_prompt_code_review,
                build_user_prompt=lambda tool_calling_enabled, redacted: user_prompt_code_review(
                    code=redacted.get("code") or "(No code snippet provided. Use the embedded files below.)",
                    context=redacted.get("context"),
                ),
                redaction_inputs={"code": req.code, "context": req.context},
                requested_paths=req.paths,
            )

        try:
            return await asyncio.wait_for(_run(), timeout=self._settings.openrouter_tool_call_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Tool call timed out after {self._settings.openrouter_tool_call_timeout_seconds}s"
            ) from exc

    async def _run_dual_review(
        self,
        *,
        tool_name: str,
        build_system_prompt: Any,
        build_user_prompt: Any,
        redaction_inputs: dict[str, str | None],
        requested_paths: list[str] | None,
    ) -> str:
        # Redact initial inputs (fail closed if redaction makes required content empty)
        redacted_inputs: dict[str, str] = {}
        for k, v in redaction_inputs.items():
            if v is None:
                continue
            redacted_inputs[k] = redact_text(v)

        direct_required = ["proposal"] if tool_name == "system_design_review" else ["code"]
        for req_field in direct_required:
            # Only enforce non-empty if direct input was actually supplied.
            if req_field in redaction_inputs and redaction_inputs.get(req_field) is not None:
                if redacted_inputs.get(req_field, "").strip() == "":
                    raise ValidationError("Content is empty after sanitization")

        if tool_name == "system_design_review":
            if redaction_inputs.get("proposal") is None and not requested_paths:
                raise ValidationError("Either proposal or paths must be provided")
        else:
            if redaction_inputs.get("code") is None and not requested_paths:
                raise ValidationError("Either code or paths must be provided")

        primary_model = self._settings.openrouter_primary_reviewer_model
        secondary_model = self._settings.openrouter_secondary_reviewer_model
        secondary_enabled = secondary_model != "0"

        resolved_root = self._resolve_project_root(paths=requested_paths)
        if requested_paths and is_dangerous_repo_root(resolved_root):
            raise ValidationError(
                "paths resolve to an unsafe project root; provide paths under a real repository directory"
            )
        file_context_builder = FileContextBuilder(repo_root=resolved_root)
        reviewer_start = time.monotonic()

        # R8: If model metadata fetch fails, fail closed (no OpenRouter completion requests are sent).
        primary_cfg = self._prepare_reviewer_config(primary_model, repo_root=resolved_root)
        secondary_cfg = (
            self._prepare_reviewer_config(secondary_model, repo_root=resolved_root) if secondary_enabled else None
        )

        primary_task = asyncio.create_task(
            self._run_single_reviewer(
                cfg=primary_cfg,
                tool_name=tool_name,
                build_system_prompt=build_system_prompt,
                build_user_prompt=build_user_prompt,
                redacted_inputs=redacted_inputs,
                requested_paths=requested_paths,
                file_context_builder=file_context_builder,
                reviewer_start=reviewer_start,
            )
        )

        if not secondary_enabled or secondary_cfg is None:
            primary = await primary_task
            synthesized = self._synthesize(primary, None)
            aggregated = format_aggregated_output(
                primary_markdown=self._append_disclosure(primary),
                secondary_markdown=None,
                synthesized_summary=synthesized,
            )
            return final_egress_redaction(aggregated)

        secondary_task = asyncio.create_task(
            self._run_single_reviewer(
                cfg=secondary_cfg,
                tool_name=tool_name,
                build_system_prompt=build_system_prompt,
                build_user_prompt=build_user_prompt,
                redacted_inputs=redacted_inputs,
                requested_paths=requested_paths,
                reviewer_start=reviewer_start,
                file_context_builder=file_context_builder,
            )
        )

        primary, secondary = await asyncio.gather(primary_task, secondary_task)

        synthesized = self._synthesize(primary, secondary)
        aggregated = format_aggregated_output(
            primary_markdown=self._append_disclosure(primary),
            secondary_markdown=self._append_disclosure(secondary),
            synthesized_summary=synthesized,
        )
        return final_egress_redaction(aggregated)

    def _append_disclosure(self, outcome: ReviewerOutcome) -> str:
        # Disclose additional resources used, without leaking secrets.
        body = outcome.markdown.rstrip()
        if outcome.is_intermittent:
            banner = (
                "> ⚠ *Intermittent review — reviewer timed out before completing exploration. "
                "This is a partial snapshot.*\n\n"
            )
            body = banner + body
        lines = []
        lines.append("---")
        lines.append(f"*Model: `{outcome.model}`*")
        lines.append(f"*Provider: `{outcome.provider}`*")
        if outcome.provider_note:
            lines.append(f"*Provider note: {outcome.provider_note}*")
        if outcome.used_serena:
            lines.append("*Serena tools used: yes*")
            if outcome.serena_activated_project is not None:
                lines.append(f"*Serena project activated: `{outcome.serena_activated_project}`*")
            if outcome.serena_used_tools:
                tools = ", ".join(f"`{t}`" for t in outcome.serena_used_tools)
                lines.append(f"*Serena tools invoked: {tools}*")
            if outcome.serena_used_memories:
                mems = ", ".join(f"`{m}`" for m in outcome.serena_used_memories)
                lines.append(f"*Serena memories used: {mems}*")
            if outcome.serena_used_paths:
                paths = ", ".join(f"`{p}`" for p in outcome.serena_used_paths)
                lines.append(f"*Repo paths used: {paths}*")
        else:
            lines.append("*Serena tools used: no*")
        if outcome.serena_disabled_reason:
            lines.append(f"*Serena note: {outcome.serena_disabled_reason}*")
        return body + "\n\n" + "\n".join(lines) + "\n"

    def _synthesize(self, primary: ReviewerOutcome, secondary: ReviewerOutcome | None) -> str:
        if secondary is None:
            if primary.ok:
                return "Only Primary review is provided (secondary reviewer disabled)."
            return f"Primary reviewer failed: {primary.error}"

        if primary.ok and secondary.ok:
            notes = []
            if primary.used_serena:
                notes.append("Primary reviewer used Serena-backed context.")
            elif primary.serena_disabled_reason:
                notes.append(f"Primary reviewer Serena context disabled: {primary.serena_disabled_reason}.")
            if secondary.used_serena:
                notes.append("Secondary reviewer used Serena-backed context.")
            elif secondary.serena_disabled_reason:
                notes.append(f"Secondary reviewer Serena context disabled: {secondary.serena_disabled_reason}.")
            if primary.is_intermittent:
                notes.append("Primary review is intermittent (partial snapshot due to timeout) — weight findings accordingly.")
            if secondary.is_intermittent:
                notes.append("Secondary review is intermittent (partial snapshot due to timeout) — weight findings accordingly.")
            base = "Primary and Secondary reviews are provided. Where recommendations conflict, consider severity and evidence in each section."
            if notes:
                return base + "\n\n" + "\n".join(f"- {n}" for n in notes)
            return base
        if primary.ok and not secondary.ok:
            return f"Only Primary review is available. Secondary reviewer failed: {secondary.error}"
        if not primary.ok and secondary.ok:
            return f"Only Secondary review is available. Primary reviewer failed: {primary.error}"
        return f"Both reviewers failed.\n- Primary error: {primary.error}\n- Secondary error: {secondary.error}"

    def _prepare_reviewer_config(self, model: str, *, repo_root: Path) -> ReviewerConfig:
        use_zai_direct = (
            bool(self._settings.zai_coding_plan_key)
            and self._zai is not None
            and is_zai_model(model)
        )
        direct_model_name = normalize_zai_model_name(model) if use_zai_direct else None

        use_kimi_direct = (
            bool(self._settings.kimi_code_api_key)
            and self._kimi is not None
            and is_kimi_model(model)
        )
        direct_kimi_model_name = normalize_kimi_model_name(model) if use_kimi_direct else None

        use_deepseek_direct = (
            bool(self._settings.deepseek_api_key)
            and self._deepseek is not None
            and is_deepseek_model(model)
        )
        direct_deepseek_model_name = normalize_deepseek_model_name(model) if use_deepseek_direct else None
        if is_deepseek_model(model) and not use_deepseek_direct:
            log.warning(
                "DeepSeek model '%s' not routed direct: api_key=%s, client=%s",
                model,
                bool(self._settings.deepseek_api_key),
                self._deepseek is not None,
            )

        use_ollama_direct = (
            bool(self._settings.ollama_api_key)
            and self._ollama is not None
            and is_ollama_model(model)
        )
        direct_ollama_model_name = normalize_ollama_model_name(model) if use_ollama_direct else None

        if use_zai_direct or use_kimi_direct or use_deepseek_direct or use_ollama_direct:
            input_budget_tokens = max(self._settings.openrouter_max_input_chars // CHARS_PER_TOKEN_ESTIMATE, 1)
            budget = TokenBudget(
                effective_context_length=(
                    input_budget_tokens
                    + self._settings.openrouter_fixed_output_tokens
                    + self._settings.openrouter_context_overhead_tokens
                ),
                effective_output_budget=self._settings.openrouter_fixed_output_tokens,
                overhead_tokens=self._settings.openrouter_context_overhead_tokens,
            )
            try:
                budget.validate()
            except TokenBudgetError as exc:
                raise RuntimeError(f"Model budget error for {model}: {exc}") from exc
            supported_parameters: tuple[str, ...] = ("tools", "tool_choice", "max_tokens")
            tool_calling_supported = True
        else:
            try:
                meta = self._models.get_model(model)
                budget = TokenBudget(
                    effective_context_length=meta.effective_context_length(),
                    effective_output_budget=meta.effective_output_budget(self._settings.openrouter_fixed_output_tokens),
                    overhead_tokens=self._settings.openrouter_context_overhead_tokens,
                )
                budget.validate()
            except (ModelMetadataError, TokenBudgetError) as exc:
                # Fail closed: prevent any LLM calls if model metadata/budget cannot be established.
                raise RuntimeError(f"Model metadata/budget error for {model}: {exc}") from exc
            supported_parameters = meta.supported_parameters
            tool_calling_supported = meta.supports_tools()

        serena_ctx = None
        serena_disabled_reason = None

        if tool_calling_supported:
            try:
                serena_ctx = SerenaContext.detect(
                    repo_root,
                    SerenaLimits(
                        max_dir_entries=self._settings.lad_serena_max_dir_entries,
                        max_search_results=self._settings.lad_serena_max_search_results,
                        max_tool_result_chars=self._settings.lad_serena_max_tool_result_chars,
                        max_total_chars=self._settings.lad_serena_max_total_chars,
                        tool_timeout_seconds=self._settings.lad_serena_tool_timeout_seconds,
                    ),
                )
            except Exception as exc:
                # R9: if Serena integration is enabled (via `.serena/`) but fails, fail closed.
                raise RuntimeError(f"Serena integration initialization failed: {exc}") from exc

            if serena_ctx is None and (repo_root / ".serena").is_dir():
                # `.serena/` exists but context could not be enabled; treat as failure per R9.
                raise RuntimeError("Serena integration required but could not be enabled")
            if serena_ctx is None:
                serena_disabled_reason = "No .serena directory detected"
        else:
            serena_disabled_reason = "Model does not support tool calling"

        return ReviewerConfig(
            model=model,
            budget=budget,
            supported_parameters=supported_parameters,
            tool_calling_supported=tool_calling_supported,
            tool_choice_supported="tool_choice" in supported_parameters,
            serena_ctx=serena_ctx,
            serena_disabled_reason=serena_disabled_reason,
            use_zai_direct=use_zai_direct,
            direct_model_name=direct_model_name,
            use_kimi_direct=use_kimi_direct,
            direct_kimi_model_name=direct_kimi_model_name,
            use_deepseek_direct=use_deepseek_direct,
            direct_deepseek_model_name=direct_deepseek_model_name,
            use_ollama_direct=use_ollama_direct,
            direct_ollama_model_name=direct_ollama_model_name,
        )

    async def _run_single_reviewer(
        self,
        *,
        cfg: ReviewerConfig,
        tool_name: str,
        build_system_prompt: Any,
        build_user_prompt: Any,
        redacted_inputs: dict[str, str],
        requested_paths: list[str] | None,
        file_context_builder: FileContextBuilder,
        intermittent_state_override: IntermittentReviewState | None = None,
        reviewer_start: float | None = None,
    ) -> ReviewerOutcome:
        if reviewer_start is None:
            reviewer_start = time.monotonic()
        model = cfg.model
        budget = cfg.budget
        serena_ctx = cfg.serena_ctx
        serena_disabled_reason = cfg.serena_disabled_reason
        if intermittent_state_override is not None:
            intermittent_state = intermittent_state_override
        elif self._settings.intermittent_review_calls > 0:
            intermittent_state = IntermittentReviewState()
        else:
            intermittent_state = None

        system_prompt = build_system_prompt(tool_calling_enabled=serena_ctx is not None)
        user_prompt = build_user_prompt(serena_ctx is not None, redacted_inputs)

        max_user_chars = min(
            self._settings.openrouter_max_input_chars,
            max(budget.input_budget_tokens, 1) * CHARS_PER_TOKEN_ESTIMATE,
        )

        if requested_paths:
            # Embed repo-scoped file context into the user prompt (path-based review).
            # Budget conservatively by reserving space for the existing prompt and a small buffer.
            buffer = 600
            remaining_for_files = max(max_user_chars - len(user_prompt) - buffer, 0)
            if remaining_for_files > 0:
                file_ctx = file_context_builder.build(paths=requested_paths, max_chars=remaining_for_files)

                embedded_list = "\n".join(f"- `{p}`" for p in file_ctx.embedded_files) or "- (none)"
                skipped_list = "\n".join(
                    f"- `{s.get('path')}` — {s.get('reason')}" for s in file_ctx.skipped_files
                ) or "- (none)"
                file_section = (
                    "\n\n## Files (from disk)\n"
                    "### Embedded\n"
                    f"{embedded_list}\n\n"
                    "### Skipped\n"
                    f"{skipped_list}\n\n"
                    "### Embedded Content\n"
                    f"{file_ctx.formatted}\n"
                )
                user_prompt += redact_text(file_section)
        user_prompt, truncated = _truncate_to_chars(user_prompt, max_user_chars)

        if truncated:
            note = "\n\n[NOTE: Input truncated to fit model context window.]\n"
            if len(user_prompt) + len(note) > max_user_chars:
                user_prompt = user_prompt[: max(max_user_chars - len(note), 0)]
            user_prompt += note

        messages: list[dict[str, Any]] = [
            _build_system_message(system_prompt),
            _build_user_message(user_prompt),
        ]

        tools = serena_ctx.tool_schemas() if serena_ctx is not None else None

        extra_body: dict[str, Any] = {}

        # Best-effort: only request reasoning traces when the model claims to support it.
        if self._settings.openrouter_include_reasoning and "include_reasoning" in cfg.supported_parameters:
            extra_body["include_reasoning"] = True

        # Best-effort: if model supports max_completion_tokens, pass it via extra_body as well.
        if "max_completion_tokens" in cfg.supported_parameters:
            extra_body["max_completion_tokens"] = budget.effective_output_budget
        extra_body_to_send = extra_body or None
        provider_used = ["openrouter"]
        provider_notes: list[str] = []

        try:
            # Enforce a wall-clock cap for the whole reviewer run (including multiple model calls and tool calls).
            markdown = await asyncio.wait_for(
                self._tool_loop(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice_supported=cfg.tool_choice_supported,
                    serena_ctx=serena_ctx,
                    extra_body=extra_body_to_send,
                    reviewer_timeout_seconds=self._settings.openrouter_reviewer_timeout_seconds,
                    max_output_tokens=budget.effective_output_budget,
                    max_tool_calls=self._settings.lad_serena_max_tool_calls,
                    tool_timeout_seconds=self._settings.lad_serena_tool_timeout_seconds,
                    use_zai_direct=cfg.use_zai_direct,
                    direct_model_name=cfg.direct_model_name,
                    use_kimi_direct=cfg.use_kimi_direct,
                    direct_kimi_model_name=cfg.direct_kimi_model_name,
                    use_deepseek_direct=cfg.use_deepseek_direct,
                    direct_deepseek_model_name=cfg.direct_deepseek_model_name,
                    use_ollama_direct=cfg.use_ollama_direct,
                    direct_ollama_model_name=cfg.direct_ollama_model_name,
                    provider_used=provider_used,
                    provider_notes=provider_notes,
                    intermittent_state=intermittent_state,
                ),
                timeout=self._settings.openrouter_reviewer_timeout_seconds,
            )
            # Defensive: ensure no in-flight intermittent task outlives the reviewer.
            self._cancel_in_flight_intermittent(intermittent_state)
            used_serena = serena_ctx is not None and (
                serena_ctx.used_tools or serena_ctx.used_memories or serena_ctx.used_paths
            )

            # If the model completed but returned empty or placeholder-only content,
            # fall back to the best available interim evidence (digest/trace) instead
            # of returning useless placeholder stubs.
            is_intermittent = False
            if not _is_substantive_review_content(markdown):
                if intermittent_state is not None:
                    best_md, best_note = _select_best_interim_markdown(
                        model=model,
                        timeout_seconds=self._settings.openrouter_reviewer_timeout_seconds,
                        state=intermittent_state,
                        serena_ctx=serena_ctx,
                        stop_reason="empty_content",
                    )
                    if best_md is not None:
                        markdown = best_md
                        is_intermittent = True
                        provider_notes.append(
                            f"Reviewer completed with empty content. {best_note}"
                        )
                if not _is_substantive_review_content(markdown) and serena_ctx is not None:
                    trace = _build_tool_trace_summary(
                        model=model,
                        timeout_seconds=self._settings.openrouter_reviewer_timeout_seconds,
                        tool_calls_made=serena_ctx.total_tool_calls,
                        tools_invoked=serena_ctx.used_tools,
                        memories_used=serena_ctx.used_memories,
                        paths_used=serena_ctx.used_paths,
                    )
                    if trace.strip():
                        markdown = trace
                        is_intermittent = True
                        provider_notes.append(
                            "Reviewer completed with empty content. Returning tool-exploration trace summary as fallback."
                        )

            return ReviewerOutcome(
                ok=True,
                model=model,
                used_serena=used_serena,
                serena_disabled_reason=serena_disabled_reason,
                serena_activated_project=serena_ctx.activated_project if serena_ctx is not None else None,
                serena_used_tools=tuple(sorted(serena_ctx.used_tools)) if serena_ctx is not None else (),
                serena_used_memories=tuple(sorted(serena_ctx.used_memories)) if serena_ctx is not None else (),
                serena_used_paths=tuple(sorted(serena_ctx.used_paths)) if serena_ctx is not None else (),
                markdown=markdown,
                error=None,
                provider=provider_used[0],
                provider_note="; ".join(provider_notes) if provider_notes else None,
                is_intermittent=is_intermittent,
            )
        except (TimeoutError, asyncio.TimeoutError):
            timeout_seconds = self._settings.openrouter_reviewer_timeout_seconds
            # Give an in-flight intermittent side-call a bounded grace period to finish.
            # asyncio.shield prevents grace expiry from cancelling the side-call.
            grace_seconds = _compute_grace_seconds(
                tool_call_timeout_seconds=self._settings.openrouter_tool_call_timeout_seconds,
                reviewer_start=reviewer_start,
            )
            if grace_seconds > 0 and intermittent_state is not None and intermittent_state.in_flight_task is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(intermittent_state.in_flight_task),
                        timeout=grace_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except (asyncio.TimeoutError, Exception):
                    pass
            used_serena = serena_ctx is not None and (serena_ctx.used_tools or serena_ctx.used_memories or serena_ctx.used_paths)

            if intermittent_state is not None:
                best_md, best_note = _select_best_interim_markdown(
                    model=model,
                    timeout_seconds=timeout_seconds,
                    state=intermittent_state,
                    serena_ctx=serena_ctx,
                    stop_reason="timeout",
                )
            elif serena_ctx is not None:
                trace = _build_tool_trace_summary(
                    model=model,
                    timeout_seconds=timeout_seconds,
                    tool_calls_made=serena_ctx.total_tool_calls,
                    tools_invoked=serena_ctx.used_tools,
                    memories_used=serena_ctx.used_memories,
                    paths_used=serena_ctx.used_paths,
                )
                if trace.strip():
                    best_md = trace
                    best_note = "Returning tool-exploration trace summary as fallback."
                else:
                    best_md, best_note = None, "No interim markdown available."
            else:
                best_md, best_note = None, "No intermittent state."

            # Clean up in-flight task after reading state — shield ensured it was
            # not cancelled by grace expiry, but we cancel now before returning.
            self._cancel_in_flight_intermittent(intermittent_state)

            timeout_prefix = f"Reviewer timed out after {timeout_seconds}s. "
            if best_md is not None:
                return ReviewerOutcome(
                    ok=True,
                    model=model,
                    used_serena=used_serena,
                    serena_disabled_reason=serena_disabled_reason,
                    serena_activated_project=serena_ctx.activated_project if serena_ctx is not None else None,
                    serena_used_tools=tuple(sorted(serena_ctx.used_tools)) if serena_ctx is not None else (),
                    serena_used_memories=tuple(sorted(serena_ctx.used_memories)) if serena_ctx is not None else (),
                    serena_used_paths=tuple(sorted(serena_ctx.used_paths)) if serena_ctx is not None else (),
                    markdown=best_md,
                    error=None,
                    provider=provider_used[0],
                    provider_note=timeout_prefix + best_note,
                    is_intermittent=True,
                )

            serena_tools = tuple(sorted(serena_ctx.used_tools)) if serena_ctx is not None else ()
            serena_memories = tuple(sorted(serena_ctx.used_memories)) if serena_ctx is not None else ()
            serena_paths = tuple(sorted(serena_ctx.used_paths)) if serena_ctx is not None else ()
            msg = f"Reviewer timed out after {timeout_seconds}s"
            markdown = _format_reviewer_error(model, msg)
            return ReviewerOutcome(
                ok=False,
                model=model,
                used_serena=used_serena,
                serena_disabled_reason=serena_disabled_reason,
                serena_activated_project=serena_ctx.activated_project if serena_ctx is not None else None,
                serena_used_tools=serena_tools,
                serena_used_memories=serena_memories,
                serena_used_paths=serena_paths,
                markdown=markdown,
                error=msg,
                provider=provider_used[0] if provider_used else "openrouter",
                provider_note="; ".join(provider_notes) if provider_notes else None,
            )
        except Exception as exc:
            msg = _exc_message(exc)
            used_serena = serena_ctx is not None and (
                serena_ctx.used_tools or serena_ctx.used_memories or serena_ctx.used_paths
            )
            # IMPORTANT: Read interim state BEFORE cancelling the in-flight task.
            # No await between here and _cancel_in_flight_intermittent below —
            # the single-threaded event loop ensures state is consistent.
            if intermittent_state is not None:
                best_md, best_note = _select_best_interim_markdown(
                    model=model,
                    timeout_seconds=self._settings.openrouter_reviewer_timeout_seconds,
                    state=intermittent_state,
                    serena_ctx=serena_ctx,
                    stop_reason="provider_error",
                )
            elif serena_ctx is not None:
                trace = _build_tool_trace_summary(
                    model=model,
                    timeout_seconds=self._settings.openrouter_reviewer_timeout_seconds,
                    tool_calls_made=serena_ctx.total_tool_calls,
                    tools_invoked=serena_ctx.used_tools,
                    memories_used=serena_ctx.used_memories,
                    paths_used=serena_ctx.used_paths,
                )
                if trace.strip():
                    best_md = trace
                    best_note = "Returning tool-exploration trace summary as fallback."
                else:
                    best_md, best_note = None, "No interim markdown available."
            else:
                best_md, best_note = None, "No intermittent state."
            self._cancel_in_flight_intermittent(intermittent_state)

            serena_tools = tuple(sorted(serena_ctx.used_tools)) if serena_ctx is not None else ()
            serena_memories = tuple(sorted(serena_ctx.used_memories)) if serena_ctx is not None else ()
            serena_paths = tuple(sorted(serena_ctx.used_paths)) if serena_ctx is not None else ()
            markdown = _format_reviewer_error(model, msg)

            if best_md is not None:
                error_prefix = f"Provider error: {msg}. "
                return ReviewerOutcome(
                    ok=True,
                    model=model,
                    used_serena=used_serena,
                    serena_disabled_reason=serena_disabled_reason,
                    serena_activated_project=serena_ctx.activated_project if serena_ctx is not None else None,
                    serena_used_tools=tuple(sorted(serena_ctx.used_tools)) if serena_ctx is not None else (),
                    serena_used_memories=tuple(sorted(serena_ctx.used_memories)) if serena_ctx is not None else (),
                    serena_used_paths=tuple(sorted(serena_ctx.used_paths)) if serena_ctx is not None else (),
                    markdown=best_md,
                    error=None,
                    provider=provider_used[0] if provider_used else "openrouter",
                    provider_note=error_prefix + best_note,
                    is_intermittent=True,
                )


            return ReviewerOutcome(
                ok=False,
                model=model,
                used_serena=used_serena,
                serena_disabled_reason=serena_disabled_reason,
                serena_activated_project=serena_ctx.activated_project if serena_ctx is not None else None,
                serena_used_tools=serena_tools,
                serena_used_memories=serena_memories,
                serena_used_paths=serena_paths,
                markdown=markdown,
                error=msg,
                provider=provider_used[0] if provider_used else "openrouter",
                provider_note="; ".join(provider_notes) if provider_notes else None,
            )

    async def _tool_loop(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice_supported: bool,
        serena_ctx: SerenaContext | None,
        extra_body: dict[str, Any] | None,
        reviewer_timeout_seconds: int,
        max_output_tokens: int,
        max_tool_calls: int,
        tool_timeout_seconds: int,
        use_zai_direct: bool = False,
        direct_model_name: str | None = None,
        use_kimi_direct: bool = False,
        direct_kimi_model_name: str | None = None,
        use_deepseek_direct: bool = False,
        direct_deepseek_model_name: str | None = None,
        use_ollama_direct: bool = False,
        direct_ollama_model_name: str | None = None,
        provider_used: list[str] | None = None,
        provider_notes: list[str] | None = None,
        intermittent_state: IntermittentReviewState | None = None,
    ) -> str:
        if provider_used is None:
            provider_used = ["openrouter"]
        if provider_notes is None:
            provider_notes = []
        intermittent_n = self._settings.intermittent_review_calls if intermittent_state is not None else 0
        remaining_tool_calls = max_tool_calls
        did_force_project_overview = False
        did_force_baseline_memories = False
        required_memories = set(BASELINE_REQUIRED_MEMORIES)
        covered_required_memories: set[str] = set()
        preflight_validation_emitted = False
        skipped_preflight_warning_emitted = False
        degraded_outputs_count = 0
        consecutive_degraded_outputs = 0
        consecutive_guard_triggered = False
        had_tool_interaction = False
        empty_post_tool_retry_used = False
        tools_disabled_hallucination_retry_used = False

        while True:
            tool_choice: str | dict[str, Any] | None = "auto" if tools else None
            # Preflight (Serena parity):
            # 1) activate_project (mandatory) must run before any other Serena tool.
            # 2) read_project_overview (best-effort) provides baseline context and enables deterministic validation.
            if tools and serena_ctx is not None and remaining_tool_calls > 0:
                if serena_ctx.activated_project is None:
                    if tool_choice_supported:
                        tool_choice = {"type": "function", "function": {"name": "activate_project"}}
                    else:
                        tool_choice = "auto"
                elif not did_force_project_overview:
                    did_force_project_overview = True
                    if tool_choice_supported:
                        tool_choice = {"type": "function", "function": {"name": "read_project_overview"}}
                    else:
                        tool_choice = "auto"
                elif not did_force_baseline_memories:
                    did_force_baseline_memories = True
                    if tool_choice_supported:
                        tool_choice = {"type": "function", "function": {"name": "read_baseline_memories"}}
                    else:
                        tool_choice = "auto"

            if tools and isinstance(tool_choice, dict) and self._is_tool_choice_fallback_active(model):
                log.info("Using cached fallback tool_choice='auto' for model '%s'", model)
                tool_choice = "auto"

            call_timeout_seconds = max(
                int(reviewer_timeout_seconds) - int(OPENROUTER_CALL_TIMEOUT_SAFETY_MARGIN_SECONDS),
                1,
            )

            result = await self._call_model_with_provider_fallback(
                model=model,
                direct_model_name=direct_model_name,
                use_zai_direct=use_zai_direct,
                direct_kimi_model_name=direct_kimi_model_name,
                use_kimi_direct=use_kimi_direct,
                direct_deepseek_model_name=direct_deepseek_model_name,
                use_deepseek_direct=use_deepseek_direct,
                direct_ollama_model_name=direct_ollama_model_name,
                use_ollama_direct=use_ollama_direct,
                messages=messages,
                timeout_seconds=call_timeout_seconds,
                max_output_tokens=max_output_tokens,
                tools=tools,
                preferred_tool_choice=tool_choice,
                extra_body=extra_body,
                provider_used=provider_used,
                provider_notes=provider_notes,
            )

            if not result.tool_calls:
                content = result.content or ""
                if (
                    content.strip() == ""
                    and had_tool_interaction
                    and not empty_post_tool_retry_used
                ):
                    empty_post_tool_retry_used = True
                    tools = None
                    messages.append(
                        _build_system_message(
                            "Previous model turn returned an empty response after tool interaction. "
                            "Provide a non-empty final review now."
                        )
                    )
                    continue
                self._cancel_in_flight_intermittent(intermittent_state)
                return _append_tooling_degradation_summary(
                    content,
                    degraded_outputs_count=degraded_outputs_count,
                    consecutive_guard_triggered=consecutive_guard_triggered,
                )

            if serena_ctx is None or tools is None:
                # Model returned tool calls despite tools being disabled (hallucination).
                # Attempt one-shot recovery by injecting a hint; if it happens again, return what we have.
                if not tools_disabled_hallucination_retry_used:
                    tools_disabled_hallucination_retry_used = True
                    messages.append(_build_system_message(
                        "You attempted to use a tool, but tools are no longer available. "
                        "Provide your final review in plain markdown without any tool calls."
                    ))
                    continue
                self._cancel_in_flight_intermittent(intermittent_state)
                return _append_tooling_degradation_summary(
                    result.content or "",
                    degraded_outputs_count=degraded_outputs_count,
                    consecutive_guard_triggered=consecutive_guard_triggered,
                )

            executable_tool_calls = result.tool_calls[: max(remaining_tool_calls, 0)]
            if executable_tool_calls:
                # Use `getattr` so this code path tolerates duck-typed result objects from tests
                # that predate the addition of `reasoning_content` to `OpenRouterCallResult`.
                messages.append(
                    _build_assistant_tool_calls_message(
                        executable_tool_calls,
                        content=result.content,
                        reasoning_content=getattr(result, "reasoning_content", None),
                    )
                )

            if remaining_tool_calls <= 0:
                messages.append(_build_system_message(force_finalize_system_message()))
                # Disable tool usage by dropping tools list and forcing none.
                tools = None
                continue

            for tool_call in executable_tool_calls:
                remaining_tool_calls -= 1

                tc_id = tool_call.get("id") or ""
                fn = tool_call.get("function") or {}
                fn_name = fn.get("name") or ""
                fn_args = fn.get("arguments") or "{}"

                def _run_tool_sync() -> str:
                    try:
                        return serena_ctx.call_tool(fn_name, fn_args)
                    except SerenaToolError as exc:
                        return json.dumps({"error": str(exc)})

                # Preflight tools are intentionally lightweight and safe to run inline; keeping them out of the
                # threadpool avoids startup/scheduling delays that can cause false timeouts in short-review tests.
                if fn_name in {"activate_project", "read_project_overview", "read_baseline_memories"}:
                    tool_out = _run_tool_sync()
                else:
                    loop = asyncio.get_running_loop()
                    try:
                        tool_out = await asyncio.wait_for(
                            loop.run_in_executor(self._tool_executor, _run_tool_sync),
                            timeout=tool_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        tool_out = json.dumps({"error": f"tool call timed out after {tool_timeout_seconds}s"})

                messages.append(_build_tool_message(tc_id, fn_name, tool_out))
                had_tool_interaction = True

                if fn_name == "read_project_overview":
                    covered_required_memories.add("project_overview.md")
                elif fn_name == "read_memory":
                    memory_name = _extract_read_memory_name(fn_args)
                    if memory_name in required_memories:
                        covered_required_memories.add(memory_name)
                elif fn_name == "read_baseline_memories":
                    baseline = _extract_tool_result_object(tool_out) or {}
                    present = baseline.get("present")
                    if isinstance(present, list):
                        for item in present:
                            if isinstance(item, str):
                                name = _normalize_memory_name(item)
                                if name in required_memories:
                                    covered_required_memories.add(name)
                    loaded = baseline.get("loaded")
                    if isinstance(loaded, list):
                        for item in loaded:
                            if isinstance(item, dict):
                                raw_name = item.get("name")
                                if isinstance(raw_name, str):
                                    name = _normalize_memory_name(raw_name)
                                    if name in required_memories:
                                        covered_required_memories.add(name)
                    if not preflight_validation_emitted:
                        missing_required = required_memories - covered_required_memories
                        messages.append(_build_system_message(_preflight_validation_message(missing_required)))
                        preflight_validation_emitted = True

                missing_required = required_memories - covered_required_memories
                if fn_name in DEEPER_EXPLORATION_TOOL_NAMES and missing_required:
                    if not preflight_validation_emitted:
                        messages.append(_build_system_message(_preflight_validation_message(missing_required)))
                        preflight_validation_emitted = True
                    if not skipped_preflight_warning_emitted:
                        messages.append(_build_system_message(_skipped_preflight_warning_message(missing_required)))
                        skipped_preflight_warning_emitted = True

                is_degraded, _reason = _is_degraded_tool_output(tool_out)
                if intermittent_state is not None:
                    _update_exploration_digest(
                        intermittent_state.digest,
                        fn_name,
                        fn_args,
                        tool_out,
                        is_degraded,
                    )
                if is_degraded:
                    degraded_outputs_count += 1
                    consecutive_degraded_outputs += 1
                    messages.append(_build_system_message(TOOL_DEGRADATION_SYSTEM_HINT))
                else:
                    consecutive_degraded_outputs = 0

                if consecutive_degraded_outputs >= CONSECUTIVE_DEGRADED_TOOL_OUTPUTS_GUARD and tools is not None:
                    consecutive_guard_triggered = True
                    messages.append(_build_system_message(force_finalize_system_message()))
                    tools = None
                    break

                # Intermittent review dispatch (R2/R3): non-blocking; runs concurrent with main loop.
                if intermittent_state is not None and intermittent_n > 0:
                    if fn_name in PREFLIGHT_TOOL_NAMES and not intermittent_state._preflight_complete:
                        pass  # Skip dispatch counter for preflight tools
                    else:
                        intermittent_state._preflight_complete = True
                        intermittent_state.tool_calls_so_far += 1
                        if intermittent_state.tool_calls_so_far % intermittent_n == 0:
                            self._dispatch_intermittent_review(
                                state=intermittent_state,
                                model=model,
                                messages=messages,
                                use_zai_direct=use_zai_direct,
                                direct_model_name=direct_model_name,
                                use_kimi_direct=use_kimi_direct,
                                direct_kimi_model_name=direct_kimi_model_name,
                                use_deepseek_direct=use_deepseek_direct,
                                direct_deepseek_model_name=direct_deepseek_model_name,
                                use_ollama_direct=use_ollama_direct,
                                direct_ollama_model_name=direct_ollama_model_name,
                                extra_body=extra_body,
                                max_output_tokens=min(
                                    self._settings.intermittent_max_output_tokens,
                                    max_output_tokens,
                                ),
                            )


def _format_intermittent_status_note(state: IntermittentReviewState) -> str:
    status = state.last_status
    error = state.last_error
    parts = [f"intermittent side-call status: {status}"]
    if error:
        parts.append(f"({error})")
    if state.last_started_at is not None and state.last_finished_at is not None:
        elapsed = state.last_finished_at - state.last_started_at
        parts.append(f"[{elapsed:.1f}s elapsed]")
    return " ".join(parts)


def _compute_grace_seconds(
    *,
    tool_call_timeout_seconds: int,
    reviewer_start: float,
) -> float:
    remaining = tool_call_timeout_seconds - (time.monotonic() - reviewer_start) - FINAL_RENDER_RESERVE_SECONDS
    if remaining <= 0:
        return 0.0
    return max(1.0, min(remaining, 15.0))


def _select_best_interim_markdown(
    *,
    model: str,
    timeout_seconds: int,
    state: IntermittentReviewState,
    serena_ctx: "SerenaContext | None",
    stop_reason: str,
) -> tuple[str | None, str]:
    """
    Priority-based selection of the best available interim markdown.

    Returns (markdown_or_None, provider_note_text).
    """
    # 1. Substantive model snapshot
    if state.latest_markdown and _is_substantive_review_content(state.latest_markdown):
        note = (
            f"Returning intermittent snapshot captured at tool call "
            f"{state.snapshot_tool_call_index}."
        )
        return state.latest_markdown, note

    # 2. Deterministic exploration digest (if any tool calls were tracked)
    if state.digest.total_tool_calls > 0:
        md = _render_digest_snapshot(
            model=model,
            timeout_seconds=timeout_seconds,
            digest=state.digest,
            state=state,
            stop_reason=stop_reason,
        )
        status_note = _format_intermittent_status_note(state)
        note = f"Returning deterministic exploration digest ({status_note})."
        return md, note

    # 3. Tool-trace summary from Serena context
    if serena_ctx is not None and (serena_ctx.used_tools or serena_ctx.used_paths):
        trace = _build_tool_trace_summary(
            model=model,
            timeout_seconds=timeout_seconds,
            tool_calls_made=serena_ctx.total_tool_calls,
            tools_invoked=serena_ctx.used_tools,
            memories_used=serena_ctx.used_memories,
            paths_used=serena_ctx.used_paths,
        )
        note = "Returning tool-exploration trace summary as fallback."
        return trace, note

    # 4. No evidence at all
    note = "No interim markdown available."
    return None, note


def _render_digest_snapshot(
    *,
    model: str,
    timeout_seconds: int,
    digest: ExplorationDigest,
    state: IntermittentReviewState,
    stop_reason: str,
) -> str:
    files_list = ", ".join(f"`{p}`" for p in sorted(digest.files_read)) if digest.files_read else "*(none)*"
    tools_list = ", ".join(sorted(digest.tools_invoked)) if digest.tools_invoked else "*(none)*"
    memories_list = ", ".join(f"`{m}`" for m in sorted(digest.memories_used)) if digest.memories_used else "*(none)*"
    paths_count = len(digest.paths_visited)

    side_call_status = state.last_status
    side_call_error = state.last_error

    if stop_reason == "timeout":
        summary_reason = f"timed out after {timeout_seconds}s"
    else:
        summary_reason = f"stopped ({stop_reason})"

    lines: list[str] = []
    lines.append("## Summary")
    lines.append(
        f"Partial review: reviewer (`{model}`) {summary_reason} during tool-assisted exploration.\n"
        f"Explored **{paths_count} path(s)** across **{digest.total_tool_calls}** tool call(s)."
    )
    if side_call_status not in ("never_dispatched", "completed"):
        note = f"Side-call status: {side_call_status}"
        if side_call_error:
            note += f" ({side_call_error})"
        lines.append(note)

    lines.append("\n## Key Findings")
    if digest.llm_findings:
        for finding in digest.llm_findings[:10]:
            lines.append(f"- {finding}")
    else:
        lines.append(f"- **Medium**: Reviewer {summary_reason} before model-generated final analysis.")
        lines.append(f"- **Info**: Tool exploration inspected {paths_count} path(s) and invoked {len(digest.tools_invoked)} tool type(s).")
    if digest.degraded_outputs:
        lines.append(f"- **Low**: {digest.degraded_outputs} degraded tool output(s) encountered during exploration.")

    lines.append("\n## Exploration Statistics")
    lines.append(f"- **Tool calls made**: {digest.total_tool_calls}")
    lines.append(f"- **Tools invoked**: {tools_list}")
    lines.append(f"- **Memories accessed**: {memories_list}")
    lines.append(f"- **Paths visited**: {paths_count}")
    if digest.degraded_outputs:
        lines.append(f"- **Degraded outputs**: {digest.degraded_outputs}")

    lines.append("\n## Files Explored")
    lines.append(files_list)

    if digest.symbols_found:
        lines.append("\n## Symbols Found")
        for sym in digest.symbols_found[:15]:
            lines.append(f"- `{sym}`")

    if digest.search_matches:
        lines.append("\n## Search Matches")
        for match in digest.search_matches[:15]:
            lines.append(f"- `{match}`")

    lines.append("\n## Recommendations")
    if digest.llm_recommendations:
        for rec in digest.llm_recommendations[:10]:
            lines.append(f"- {rec}")
    else:
        lines.append("- Re-run the review with a longer timeout to get a full model-generated review.")
        lines.append("- Consider reducing the scope (fewer files) to fit within the timeout budget.")

    lines.append("\n## Questions / Unknowns")
    if digest.llm_open_questions:
        for q in digest.llm_open_questions[:10]:
            lines.append(f"- {q}")
    else:
        lines.append("- What substantive conclusions would the reviewer have produced after final synthesis?")
        if side_call_status not in ("never_dispatched", "completed"):
            lines.append(f"- Did the intermittent side-call fail, time out, or return empty content? (status: {side_call_status})")

    return "\n".join(lines) + "\n"


def _format_reviewer_error(model: str, error: str) -> str:
    return (
        "## Summary\n"
        f"**Reviewer Error** for model `{model}`.\n\n"
        "## Key Findings\n"
        f"- **High**: {error}\n\n"
        "## Recommendations\n"
        "- Ensure OPENROUTER_API_KEY is set and model names are valid.\n"
        "- Verify OpenRouter Models API is reachable.\n\n"
        "## Questions / Unknowns\n"
        "- Did the model support tool calling and/or was Serena available?\n"
    )
