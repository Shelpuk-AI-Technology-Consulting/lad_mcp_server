from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RedactionRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str = "[REDACTED]"


DEFAULT_RULES: tuple[RedactionRule, ...] = (
    # OpenAI/OpenRouter-like secret keys
    RedactionRule(
        name="openai_like_api_key",
        pattern=re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    ),
    # OpenRouter keys are often prefixed sk-or-... but keep generic sk- match above as well.
    RedactionRule(
        name="openrouter_api_key",
        pattern=re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}\b"),
    ),
    # GitHub tokens
    RedactionRule(
        name="github_pat",
        pattern=re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    ),
    RedactionRule(
        name="github_fine_grained_pat",
        pattern=re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ),
    # AWS access keys (best-effort)
    RedactionRule(
        name="aws_access_key_id",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    # JWT (best-effort)
    RedactionRule(
        name="jwt",
        pattern=re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    # PEM blocks (best-effort)
    RedactionRule(
        name="pem_private_key",
        pattern=re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |)?PRIVATE KEY-----"),
    ),
)


def redact_text(text: str, *, rules: Iterable[RedactionRule] = DEFAULT_RULES) -> str:
    """
    Redact common secret/PII patterns from text.

    Notes:
    - This is best-effort and intentionally conservative; it may over-redact.
    - Callers should also ensure logs never contain raw unredacted payloads.
    """
    redacted = text
    for rule in rules:
        redacted = rule.pattern.sub(rule.replacement, redacted)
    return redacted


def redact_maybe(text: str | None, *, rules: Iterable[RedactionRule] = DEFAULT_RULES) -> str | None:
    if text is None:
        return None
    return redact_text(text, rules=rules)


def contains_unredacted_secrets(text: str, *, rules: Iterable[RedactionRule] = DEFAULT_RULES) -> bool:
    for rule in rules:
        if rule.pattern.search(text) is not None:
            return True
    return False

