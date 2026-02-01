from __future__ import annotations

import json
from dataclasses import dataclass


class ValidationError(ValueError):
    pass


def _require_non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if value.strip() == "":
        raise ValidationError(f"{field_name} must not be blank")
    return value


def _max_len(value: str | None, field_name: str, max_chars: int) -> str | None:
    if value is None:
        return None
    if len(value) > max_chars:
        raise ValidationError(f"{field_name} must be <= {max_chars} characters")
    return value


def _normalize_paths(paths: list[str] | str | None) -> list[str] | None:
    if paths is None:
        return None

    if isinstance(paths, str):
        s = paths.strip()
        if s == "":
            raise ValidationError("paths must be a non-empty list of strings when provided")
        # If the caller passed a JSON array as a string, try to parse it.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except Exception as exc:
                raise ValidationError(f"paths JSON could not be parsed: {exc}") from exc
            paths = parsed
        else:
            # Common failure mode: newline-separated list (e.g., copied from a UI textarea).
            paths = [p.strip() for p in s.splitlines() if p.strip()]

    if not isinstance(paths, list) or len(paths) == 0:
        raise ValidationError("paths must be a non-empty list of strings when provided")

    cleaned: list[str] = []
    for p in paths:
        p = _require_non_blank(p, "paths[]")
        cleaned.append(p)
    return cleaned


@dataclass(frozen=True)
class SystemDesignReviewRequest:
    proposal: str | None
    paths: list[str] | None = None
    constraints: str | None = None
    context: str | None = None

    @staticmethod
    def validate(
        *,
        proposal: str | None,
        paths: list[str] | str | None,
        constraints: str | None,
        context: str | None,
        max_input_chars: int,
    ) -> "SystemDesignReviewRequest":
        if proposal is not None:
            proposal = _require_non_blank(proposal, "proposal")
            if len(proposal) < 10:
                raise ValidationError("proposal must be at least 10 characters")
            if len(proposal) > max_input_chars:
                raise ValidationError(f"proposal must be <= OPENROUTER_MAX_INPUT_CHARS ({max_input_chars})")

        paths = _normalize_paths(paths)

        if proposal is None and not paths:
            raise ValidationError("Either proposal or paths must be provided")
        constraints = _max_len(constraints, "constraints", 10_000)
        context = _max_len(context, "context", 10_000)
        return SystemDesignReviewRequest(
            proposal=proposal,
            paths=paths,
            constraints=constraints,
            context=context,
        )


@dataclass(frozen=True)
class CodeReviewRequest:
    code: str | None
    paths: list[str] | None

    @staticmethod
    def validate(
        *,
        code: str | None,
        paths: list[str] | str | None,
        max_input_chars: int,
    ) -> "CodeReviewRequest":
        if code is not None:
            code = _require_non_blank(code, "code")
            if len(code) > max_input_chars:
                raise ValidationError(f"code must be <= OPENROUTER_MAX_INPUT_CHARS ({max_input_chars})")

        normalized_paths = _normalize_paths(paths)

        if code is None and not normalized_paths:
            raise ValidationError("Either code or paths must be provided")

        return CodeReviewRequest(
            code=code,
            paths=normalized_paths,
        )
