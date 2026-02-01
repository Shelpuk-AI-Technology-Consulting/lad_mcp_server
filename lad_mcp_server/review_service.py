from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from lad_mcp_server.config import Settings
from lad_mcp_server.file_context import FileContextBuilder
from lad_mcp_server.markdown import final_egress_redaction, format_aggregated_output
from lad_mcp_server.model_metadata import ModelMetadataError, OpenRouterModelsClient
from lad_mcp_server.openrouter_client import OpenRouterClient, OpenRouterClientError
from lad_mcp_server.prompts import (
    force_finalize_system_message,
    system_prompt_code_review,
    system_prompt_system_design_review,
    user_prompt_code_review,
    user_prompt_system_design_review,
)
from lad_mcp_server.redaction import redact_text
from lad_mcp_server.schemas import CodeReviewRequest, SystemDesignReviewRequest, ValidationError
from lad_mcp_server.serena_bridge import SerenaContext, SerenaLimits, SerenaToolError
from lad_mcp_server.token_budget import TokenBudget, TokenBudgetError


log = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 3  # conservative for mixed tokenizers

_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8)
atexit.register(_TOOL_EXECUTOR.shutdown, wait=False, cancel_futures=True)


def _truncate_to_chars(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _build_tool_message(tool_call_id: str, name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def _build_system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def _build_user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


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


@dataclass(frozen=True)
class ReviewerConfig:
    model: str
    budget: TokenBudget
    supported_parameters: tuple[str, ...]
    tool_calling_supported: bool
    tool_choice_supported: bool
    serena_ctx: SerenaContext | None
    serena_disabled_reason: str | None


class ReviewService:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        settings: Settings | None = None,
        openrouter_client: OpenRouterClient | None = None,
        models_client: OpenRouterModelsClient | None = None,
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
        if repo_root is None:
            env_root = os.getenv("LAD_REPO_ROOT")
            if env_root and env_root.strip():
                repo_root = Path(env_root)
        self._repo_root = (repo_root or Path.cwd()).resolve()
        self._file_context_builder = FileContextBuilder(repo_root=self._repo_root)
        self._tool_executor = _TOOL_EXECUTOR

    async def system_design_review(self, **kwargs: Any) -> str:
        req = SystemDesignReviewRequest.validate(
            proposal=kwargs.get("proposal"),
            paths=kwargs.get("paths"),
            constraints=kwargs.get("constraints"),
            context=kwargs.get("context"),
            model=kwargs.get("model"),
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
                override_model=req.model,
            )

        return await asyncio.wait_for(_run(), timeout=self._settings.openrouter_tool_call_timeout_seconds)

    async def code_review(self, **kwargs: Any) -> str:
        req = CodeReviewRequest.validate(
            code=kwargs.get("code"),
            paths=kwargs.get("paths"),
            language=kwargs.get("language"),
            focus=kwargs.get("focus"),
            model=kwargs.get("model"),
            max_input_chars=self._settings.openrouter_max_input_chars,
        )

        async def _run() -> str:
            return await self._run_dual_review(
                tool_name="code_review",
                build_system_prompt=system_prompt_code_review,
                build_user_prompt=lambda tool_calling_enabled, redacted: user_prompt_code_review(
                    code=redacted.get("code") or "(No code snippet provided. Use the embedded files below.)",
                    language=req.language or "mixed",
                    focus=req.focus,
                ),
                redaction_inputs={"code": req.code},
                requested_paths=req.paths,
                override_model=req.model,
            )

        return await asyncio.wait_for(_run(), timeout=self._settings.openrouter_tool_call_timeout_seconds)

    async def _run_dual_review(
        self,
        *,
        tool_name: str,
        build_system_prompt: Any,
        build_user_prompt: Any,
        redaction_inputs: dict[str, str | None],
        requested_paths: list[str] | None,
        override_model: str | None,
    ) -> str:
        # Redact initial inputs (fail closed if redaction makes required content empty)
        redacted_inputs: dict[str, str] = {}
        for k, v in redaction_inputs.items():
            if v is None:
                continue
            redacted_inputs[k] = redact_text(v)

        direct_required = ["proposal"] if tool_name == "system_design_review" else ["code"]
        for field in direct_required:
            # Only enforce non-empty if direct input was actually supplied.
            if field in redaction_inputs and redaction_inputs.get(field) is not None:
                if redacted_inputs.get(field, "").strip() == "":
                    raise ValidationError("Content is empty after sanitization")

        if tool_name == "system_design_review":
            if redaction_inputs.get("proposal") is None and not requested_paths:
                raise ValidationError("Either proposal or paths must be provided")
        else:
            if redaction_inputs.get("code") is None and not requested_paths:
                raise ValidationError("Either code or paths must be provided")

        primary_model = override_model or self._settings.openrouter_primary_reviewer_model
        secondary_model = override_model or self._settings.openrouter_secondary_reviewer_model

        # R8: If model metadata fetch fails, fail closed (no OpenRouter completion requests are sent).
        primary_cfg = self._prepare_reviewer_config(primary_model)
        secondary_cfg = self._prepare_reviewer_config(secondary_model)

        primary_task = asyncio.create_task(
            self._run_single_reviewer(
                cfg=primary_cfg,
                tool_name=tool_name,
                build_system_prompt=build_system_prompt,
                build_user_prompt=build_user_prompt,
                redacted_inputs=redacted_inputs,
                requested_paths=requested_paths,
            )
        )
        secondary_task = asyncio.create_task(
            self._run_single_reviewer(
                cfg=secondary_cfg,
                tool_name=tool_name,
                build_system_prompt=build_system_prompt,
                build_user_prompt=build_user_prompt,
                redacted_inputs=redacted_inputs,
                requested_paths=requested_paths,
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
        lines = []
        lines.append("---")
        lines.append(f"*Model: `{outcome.model}`*")
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
        return outcome.markdown.rstrip() + "\n\n" + "\n".join(lines) + "\n"

    def _synthesize(self, primary: ReviewerOutcome, secondary: ReviewerOutcome) -> str:
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
            base = "Primary and Secondary reviews are provided. Where recommendations conflict, consider severity and evidence in each section."
            if notes:
                return base + "\n\n" + "\n".join(f"- {n}" for n in notes)
            return base
        if primary.ok and not secondary.ok:
            return f"Only Primary review is available. Secondary reviewer failed: {secondary.error}"
        if not primary.ok and secondary.ok:
            return f"Only Secondary review is available. Primary reviewer failed: {primary.error}"
        return f"Both reviewers failed.\n- Primary error: {primary.error}\n- Secondary error: {secondary.error}"

    def _prepare_reviewer_config(self, model: str) -> ReviewerConfig:
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

        tool_calling_supported = meta.supports_tools()
        serena_ctx = None
        serena_disabled_reason = None

        if tool_calling_supported:
            try:
                serena_ctx = SerenaContext.detect(
                    self._repo_root,
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

            if serena_ctx is None and (self._repo_root / ".serena").is_dir():
                # `.serena/` exists but context could not be enabled; treat as failure per R9.
                raise RuntimeError("Serena integration required but could not be enabled")
            if serena_ctx is None:
                serena_disabled_reason = "No .serena directory detected"
        else:
            serena_disabled_reason = "Model does not support tool calling"

        return ReviewerConfig(
            model=model,
            budget=budget,
            supported_parameters=meta.supported_parameters,
            tool_calling_supported=tool_calling_supported,
            tool_choice_supported="tool_choice" in meta.supported_parameters,
            serena_ctx=serena_ctx,
            serena_disabled_reason=serena_disabled_reason,
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
    ) -> ReviewerOutcome:
        model = cfg.model
        budget = cfg.budget
        serena_ctx = cfg.serena_ctx
        serena_disabled_reason = cfg.serena_disabled_reason

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
                file_ctx = self._file_context_builder.build(paths=requested_paths, max_chars=remaining_for_files)

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

        try:
            markdown = await self._tool_loop(
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
            )
            used_serena = serena_ctx is not None and (
                serena_ctx.used_tools or serena_ctx.used_memories or serena_ctx.used_paths
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
            )
        except Exception as exc:
            return ReviewerOutcome(
                ok=False,
                model=model,
                used_serena=False,
                serena_disabled_reason=serena_disabled_reason,
                serena_activated_project=None,
                serena_used_tools=(),
                serena_used_memories=(),
                serena_used_paths=(),
                markdown=_format_reviewer_error(model, str(exc)),
                error=str(exc),
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
    ) -> str:
        remaining_tool_calls = max_tool_calls
        did_force_project_overview = False

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

            result = await self._openrouter.chat_completion(
                model=model,
                messages=messages,
                timeout_seconds=reviewer_timeout_seconds,
                max_output_tokens=max_output_tokens,
                tools=tools,
                tool_choice=tool_choice,
                extra_body=extra_body,
            )

            if not result.tool_calls:
                return result.content or ""

            if serena_ctx is None or tools is None:
                # Should not happen: model returned tool calls but tools weren't provided.
                return (result.content or "") + "\n\n*(Tool calls were requested, but no tools were available.)\n"

            if remaining_tool_calls <= 0:
                messages.append(_build_system_message(force_finalize_system_message()))
                # Disable tool usage by dropping tools list and forcing none.
                tools = None
                continue

            for tool_call in result.tool_calls:
                if remaining_tool_calls <= 0:
                    break
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

                loop = asyncio.get_running_loop()
                try:
                    tool_out = await asyncio.wait_for(
                        loop.run_in_executor(self._tool_executor, _run_tool_sync),
                        timeout=tool_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    tool_out = json.dumps({"error": f"tool call timed out after {tool_timeout_seconds}s"})

                messages.append(_build_tool_message(tc_id, fn_name, tool_out))


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
