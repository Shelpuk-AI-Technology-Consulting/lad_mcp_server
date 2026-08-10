"""Secrets must not survive the ingest pipeline, and redaction must stay linear.

Redaction is only a security boundary if it sees the whole text. Two of the seven
rules are multi-line — a PEM block is matched from ``BEGIN`` to ``END`` — so a
truncation that runs *before* redaction decapitates the pattern and the key
material sails through unredacted, into a prompt bound for a third-party LLM.

Both ingest paths truncate: ``SerenaContext._read_file`` for files over
``READ_FILE_TRUNCATE_THRESHOLD``, and ``FileContextBuilder.build`` when the char
budget runs out. These tests pin the ordering invariant at both, and pin the
performance property that makes running redaction early affordable.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from lad_mcp_server.file_context import FileContextBuilder
from lad_mcp_server.redaction import redact_text
from lad_mcp_server.serena_bridge import (
    READ_FILE_TRUNCATE_THRESHOLD,
    SerenaContext,
    SerenaLimits,
)

_KEY_BODY_LINE = "MIIEowIBAAKCAQEAvVJ8ZlKq9m3wR2xN7pQ4tYbG6hJkL0sDfE1cX8aZnP5oT3uW"
_LIMITS = SerenaLimits(
    max_dir_entries=50,
    max_search_results=20,
    max_tool_result_chars=200_000,
    max_total_chars=1_000_000,
    tool_timeout_seconds=5,
)


def _file_with_key_near_the_top(total_chars: int) -> str:
    """Build source text with a PEM key at the top and padding after it.

    The key sits near the start so that head/tail truncation keeps its ``BEGIN``
    marker and its first lines while discarding the ``END`` marker — the shape
    that defeats the multi-line rule.

    Args:
        total_chars: Approximate total size of the returned text.

    Returns:
        File content containing one PEM private key.
    """
    key = "\n".join(
        ["-----BEGIN RSA PRIVATE KEY-----"] + [_KEY_BODY_LINE] * 40 + ["-----END RSA PRIVATE KEY-----"]
    )
    padding = "\n".join(f"# padding line {i} of ordinary source" for i in range(2000))
    return (key + "\n" + padding)[:total_chars]


class TestSecretsSurviveTruncation(unittest.TestCase):
    """Neither ingest path may emit key material from a truncated file."""

    def test_serena_read_file_does_not_leak_with_line_slicing(self) -> None:
        """`read_file` with an explicit `head` emits no key material.

        A different cut from the sibling test: line-based slicing rather than the
        char threshold. `head=20` keeps the `BEGIN` marker and part of the body while
        discarding the `END` marker.
        """
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td).resolve()
            (repo / ".serena").mkdir()
            content = _file_with_key_near_the_top(READ_FILE_TRUNCATE_THRESHOLD + 5000)
            (repo / "secrets.py").write_text(content, encoding="utf-8")

            ctx = SerenaContext(repo_root=repo, limits=_LIMITS)
            ctx.activated_project = "."
            raw = ctx.call_tool("read_file", json.dumps({"path": "secrets.py", "head": 20}))

        self.assertNotIn("BEGIN RSA PRIVATE KEY", raw)
        self.assertNotIn(_KEY_BODY_LINE, raw)

    def test_encrypted_and_legacy_pem_labels_are_redacted(self) -> None:
        """Every private-key label OpenSSL/OpenSSH/GnuPG emit is covered.

        `ENCRYPTED PRIVATE KEY` is a passphrase-protected PKCS#8 key — the normal way
        to store one with a password — and it, `DSA` and `PGP` all used to leak their
        whole body while `contains_unredacted_secrets` reported clean.
        """
        for label in (
            "PRIVATE KEY",
            "ENCRYPTED PRIVATE KEY",
            "RSA PRIVATE KEY",
            "DSA PRIVATE KEY",
            "EC PRIVATE KEY",
            "OPENSSH PRIVATE KEY",
            "PGP PRIVATE KEY BLOCK",
        ):
            with self.subTest(label=label):
                pem = f"-----BEGIN {label}-----\n{_KEY_BODY_LINE}\n-----END {label}-----"
                self.assertEqual(redact_text(pem), "[REDACTED]")

    def test_serena_read_file_below_threshold_is_not_truncated(self) -> None:
        """Control: a file under the threshold is redacted whole."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td).resolve()
            (repo / ".serena").mkdir()
            content = _file_with_key_near_the_top(READ_FILE_TRUNCATE_THRESHOLD - 1000)
            (repo / "secrets.py").write_text(content, encoding="utf-8")

            ctx = SerenaContext(repo_root=repo, limits=_LIMITS)
            ctx.activated_project = "."
            raw = ctx.call_tool("read_file", json.dumps({"path": "secrets.py"}))

        self.assertNotIn("BEGIN RSA PRIVATE KEY", raw)
        self.assertNotIn(_KEY_BODY_LINE, raw)

    def test_serena_read_file_does_not_leak_across_the_truncation_boundary(self) -> None:
        """The same holds for a file large enough to trigger head/tail truncation."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td).resolve()
            (repo / ".serena").mkdir()
            content = _file_with_key_near_the_top(READ_FILE_TRUNCATE_THRESHOLD + 5000)
            (repo / "secrets.py").write_text(content, encoding="utf-8")

            ctx = SerenaContext(repo_root=repo, limits=_LIMITS)
            ctx.activated_project = "."
            raw = ctx.call_tool("read_file", json.dumps({"path": "secrets.py"}))

        self.assertNotIn("BEGIN RSA PRIVATE KEY", raw)
        self.assertNotIn(_KEY_BODY_LINE, raw)

    def test_file_context_does_not_leak_a_budget_truncated_key(self) -> None:
        """A file cut short by the char budget emits no key material either."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td).resolve()
            content = _file_with_key_near_the_top(9000)
            (repo / "secrets.py").write_text(content, encoding="utf-8")

            builder = FileContextBuilder(repo_root=repo)
            # A budget that lands mid-key, discarding the END marker.
            built = builder.build(paths=["secrets.py"], max_chars=1500)

        self.assertNotIn("BEGIN RSA PRIVATE KEY", built.formatted)
        self.assertNotIn(_KEY_BODY_LINE, built.formatted)


class TestRedactionStaysLinear(unittest.TestCase):
    """Redaction runs on every prompt; it must not be a CPU amplifier."""

    def test_many_unterminated_begin_markers_redact_quickly(self) -> None:
        """Repeated `BEGIN` markers must not make the PEM rule quadratic.

        Each unterminated marker used to make the lazy body scan to end-of-input,
        costing 124s for this input. Redaction runs 2-3 times per review on the
        async path, so that blocks every concurrent review.
        """
        payload = ("-----BEGIN PRIVATE KEY-----\n" + "A" * 64 + "\n") * 6400

        started = time.perf_counter()
        redact_text(payload)
        elapsed = time.perf_counter() - started

        print(f"\nredacted {len(payload):,} chars of BEGIN markers in {elapsed * 1000:.0f}ms")
        self.assertLess(elapsed, 1.0, "PEM rule is super-linear in the number of BEGIN markers")

    def test_real_pem_shapes_still_redact(self) -> None:
        """The performance fix must not narrow what the rule matches.

        Encrypted PEMs carry `Proc-Type:` and `DEK-Info:` headers containing `-`,
        so a fix that simply excluded `-` from the body would silently stop
        redacting them.
        """
        plain = "-----BEGIN RSA PRIVATE KEY-----\n" + _KEY_BODY_LINE + "\n-----END RSA PRIVATE KEY-----"
        encrypted = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "Proc-Type: 4,ENCRYPTED\n"
            "DEK-Info: AES-128-CBC,0123456789ABCDEF\n\n"
            + _KEY_BODY_LINE
            + "\n-----END RSA PRIVATE KEY-----"
        )
        openssh = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + _KEY_BODY_LINE + "\n-----END OPENSSH PRIVATE KEY-----"

        for label, pem in (("plain", plain), ("encrypted", encrypted), ("openssh", openssh)):
            with self.subTest(pem=label):
                self.assertEqual(redact_text(pem), "[REDACTED]")

    def test_two_keys_in_one_document_are_both_redacted(self) -> None:
        """A tempered body must not swallow the gap between two keys."""
        key = "-----BEGIN RSA PRIVATE KEY-----\n" + _KEY_BODY_LINE + "\n-----END RSA PRIVATE KEY-----"
        document = key + "\nordinary source between the keys\n" + key

        redacted = redact_text(document)

        self.assertNotIn(_KEY_BODY_LINE, redacted)
        self.assertEqual(len(re.findall(r"\[REDACTED\]", redacted)), 2)
        self.assertIn("ordinary source between the keys", redacted)


if __name__ == "__main__":
    unittest.main()
