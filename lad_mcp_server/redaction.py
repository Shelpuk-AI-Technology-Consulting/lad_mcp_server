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


# Delimiters of a PEM block, matched individually. After `redact_text` has removed
# every complete BEGIN..END pair, a *surviving* delimiter can only belong to a block
# whose other end was cut away.
_PEM_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----")
_PEM_END_RE = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----")

_TRUNCATED_KEY_NOTE = "[REDACTED: truncated private key block]"


def suppress_incomplete_private_key_blocks(text: str) -> str:
    """Drop key material from a PEM block whose other delimiter was truncated away.

    `redact_text` matches a whole ``BEGIN``..``END`` block, so it is defeated by any
    caller that can only see a *slice* of a file: a key crossing the cut loses one
    delimiter and the rule can no longer match it. That is not hypothetical — it is
    how 15 lines of key material reached a prompt.

    Call this on each slice **after** ``redact_text``. Anything from a surviving
    ``BEGIN`` to the end of the slice, and from the start of the slice to a
    surviving ``END``, is treated as key material and removed.

    Deliberately conservative: it discards surrounding text rather than risk keeping
    key bytes. It cannot help with a slice that contains no delimiter at all — a
    window from the middle of a key is pure base64, and nothing in the text marks it
    as secret.

    Args:
        text: A slice of file content, already passed through :func:`redact_text`.

    Returns:
        The slice with any incomplete key block removed.
    """
    # A surviving END has no BEGIN before it, so everything up to it is key material.
    end = _PEM_END_RE.search(text)
    if end is not None:
        text = _TRUNCATED_KEY_NOTE + text[end.end():]

    # A surviving BEGIN has no END after it, so everything from it on is key material.
    begin = _PEM_BEGIN_RE.search(text)
    if begin is not None:
        text = text[: begin.start()] + _TRUNCATED_KEY_NOTE

    return text


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

