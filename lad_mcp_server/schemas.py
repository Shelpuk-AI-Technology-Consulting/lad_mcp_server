from __future__ import annotations

from dataclasses import dataclass


class ValidationError(ValueError):
    pass


ALLOWED_CODE_REVIEW_FOCUS: frozenset[str] = frozenset(
    {"security", "performance", "logic", "architecture", "maintainability", "tests"}
)


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


@dataclass(frozen=True)
class SystemDesignReviewRequest:
    proposal: str | None
    paths: list[str] | None = None
    project_root: str | None = None
    constraints: str | None = None
    context: str | None = None
    model: str | None = None

    @staticmethod
    def validate(
        *,
        proposal: str | None,
        paths: list[str] | None,
        project_root: str | None,
        constraints: str | None,
        context: str | None,
        model: str | None,
        max_input_chars: int,
    ) -> "SystemDesignReviewRequest":
        if proposal is not None:
            proposal = _require_non_blank(proposal, "proposal")
            if len(proposal) < 10:
                raise ValidationError("proposal must be at least 10 characters")
            if len(proposal) > max_input_chars:
                raise ValidationError(f"proposal must be <= OPENROUTER_MAX_INPUT_CHARS ({max_input_chars})")

        if paths is not None:
            if not isinstance(paths, list) or len(paths) == 0:
                raise ValidationError("paths must be a non-empty list of strings when provided")
            cleaned: list[str] = []
            for p in paths:
                p = _require_non_blank(p, "paths[]")
                cleaned.append(p)
            paths = cleaned

        if proposal is None and not paths:
            raise ValidationError("Either proposal or paths must be provided")
        if project_root is not None:
            project_root = _require_non_blank(project_root, "project_root")
        constraints = _max_len(constraints, "constraints", 10_000)
        context = _max_len(context, "context", 10_000)
        if model is not None and model.strip() == "":
            model = None
        return SystemDesignReviewRequest(
            proposal=proposal,
            paths=paths,
            project_root=project_root,
            constraints=constraints,
            context=context,
            model=model,
        )


@dataclass(frozen=True)
class CodeReviewRequest:
    code: str | None
    paths: list[str] | None
    project_root: str | None
    language: str | None
    focus: str | None = None
    model: str | None = None

    @staticmethod
    def validate(
        *,
        code: str | None,
        paths: list[str] | None,
        project_root: str | None,
        language: str | None,
        focus: str | None,
        model: str | None,
        max_input_chars: int,
    ) -> "CodeReviewRequest":
        if code is not None:
            code = _require_non_blank(code, "code")
            if len(code) > max_input_chars:
                raise ValidationError(f"code must be <= OPENROUTER_MAX_INPUT_CHARS ({max_input_chars})")

        if paths is not None:
            if not isinstance(paths, list) or len(paths) == 0:
                raise ValidationError("paths must be a non-empty list of strings when provided")
            cleaned_paths: list[str] = []
            for p in paths:
                p = _require_non_blank(p, "paths[]")
                cleaned_paths.append(p)
            paths = cleaned_paths

        if code is None and not paths:
            raise ValidationError("Either code or paths must be provided")

        if project_root is not None:
            project_root = _require_non_blank(project_root, "project_root")

        if language is not None:
            language = _require_non_blank(language, "language")
            if len(language) > 40:
                raise ValidationError("language must be <= 40 characters")

        if focus is not None:
            focus = _require_non_blank(focus, "focus")
            if focus not in ALLOWED_CODE_REVIEW_FOCUS:
                allowed = ", ".join(sorted(ALLOWED_CODE_REVIEW_FOCUS))
                raise ValidationError(f"focus must be one of: {allowed}")

        if model is not None and model.strip() == "":
            model = None

        return CodeReviewRequest(
            code=code,
            paths=paths,
            project_root=project_root,
            language=language,
            focus=focus,
            model=model,
        )
