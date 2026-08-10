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
    # PEM blocks (best-effort).
    #
    # The body is a *tempered* token — it may not span another `-----BEGIN `. With a
    # plain `[\s\S]*?`, every unterminated BEGIN marker made the lazy body scan all
    # the way to end-of-input, so a document full of them was quadratic: 6400 markers
    # took 124 seconds. Stopping at the next marker makes a failed scan local, which
    # is linear — the same input now takes 69ms. Redaction runs 2-3 times per review
    # on the async path, so this was a CPU amplifier reachable from untrusted input.
    #
    # `[\s\S]` rather than a `-`-excluding class: encrypted PEMs carry `Proc-Type:`
    # and `DEK-Info: AES-128-CBC` headers, which contain hyphens.
    # The label is matched generically per RFC 7468 rather than as a fixed list. The
    # previous `(?:RSA |EC |OPENSSH |)?` alternation silently missed
    # `ENCRYPTED PRIVATE KEY` — a password-protected PKCS#8 key, the standard way to
    # store one with a passphrase — as well as `DSA PRIVATE KEY` and
    # `PGP PRIVATE KEY BLOCK`. All three leaked their whole body while
    # `contains_unredacted_secrets` reported clean.
    RedactionRule(
        name="pem_private_key",
        pattern=re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"
            r"(?:(?!-----BEGIN )[\s\S])*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"
        ),
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

