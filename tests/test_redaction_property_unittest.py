"""No credential of a recognised class may survive `redact_text`.

Redaction is the only thing between a user's secrets and a third-party LLM, and it
was covered by three example tests for seven rules — deleting the GitHub, AWS or
JWT rules left the entire suite green.

**The generators here are derived from vendor documentation, never from
``DEFAULT_RULES``.** That is the whole point: a strategy built from the same regex
the code matches with would prove only that a regex matches itself, and would pass
just as happily if both were wrong. Each strategy cites its source.

Two assertions per property, because either alone is too weak:

* the generated secret must be **literally absent** from the output, and
* ``contains_unredacted_secrets`` must be ``False``.

The predicate alone is the weaker check — it re-runs the same patterns, so once an
anchor is consumed or cut it cannot see a surviving remnant. That is exactly how a
head-truncated PEM used to report "clean" while leaking 15 lines of key material.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from lad_mcp_server.redaction import contains_unredacted_secrets, redact_text

_REPO_ROOT = Path(__file__).resolve().parents[1]

# A merge gate, not a fuzzing campaign: `derandomize` fixes the corpus so CI cannot
# go red on a fresh random draw unrelated to a code change. The cost, stated plainly:
# this is a fixed regression corpus, not a search. New counterexamples appear only
# when a strategy or the Hypothesis version changes.
# `deadline=None`, and the reasoning matters because an earlier version of this file
# argued the opposite. These properties generate PEM blocks of at most eight body
# lines, so Hypothesis's 200ms default could never have caught the quadratic PEM bug
# — `test_many_unterminated_begin_markers_redact_quickly` is what catches that, with
# an explicit 1s budget on a 595,000-char input. So the deadline here buys no
# protection while producing real flakes: both properties failed on a loaded run
# (214s wall clock) and passed on the next (71s). A security gate that goes red for
# reasons unrelated to the code trains people to ignore it.
_SECURITY = settings(
    max_examples=200,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

_ALNUM = st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1)


def _fixed_alnum(length: int) -> st.SearchStrategy[str]:
    """Generate a fixed-length run of ASCII alphanumerics.

    Args:
        length: Exact number of characters.

    Returns:
        A strategy producing strings of exactly ``length`` alphanumerics.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return st.text(alphabet=alphabet, min_size=length, max_size=length)


# --- Strategies, each derived from the vendor's documented format ---------------

def github_classic_pats() -> st.SearchStrategy[str]:
    """Generate GitHub classic personal access tokens.

    Format ``ghp_`` + 36 alphanumerics (40 total), per GitHub's authentication
    token format changelog.

    Returns:
        A strategy producing classic PATs.
    """
    return _fixed_alnum(36).map(lambda body: f"ghp_{body}")


def github_fine_grained_pats() -> st.SearchStrategy[str]:
    """Generate GitHub fine-grained personal access tokens.

    Format ``github_pat_`` + 22 alphanumerics + ``_`` + 59 alphanumerics (93 total).

    Returns:
        A strategy producing fine-grained PATs.
    """
    return st.tuples(_fixed_alnum(22), _fixed_alnum(59)).map(
        lambda parts: f"github_pat_{parts[0]}_{parts[1]}"
    )


def aws_access_key_ids() -> st.SearchStrategy[str]:
    """Generate AWS access key ids.

    Format ``AKIA`` + 16 uppercase alphanumerics (20 total). Note the rule matches
    the ``AKIA`` prefix only — ``ASIA`` (STS) credentials are out of scope.

    Returns:
        A strategy producing AWS access key ids.
    """
    return st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=16, max_size=16).map(
        lambda body: f"AKIA{body}"
    )


def jwts() -> st.SearchStrategy[str]:
    """Generate JSON Web Tokens.

    Three ``.``-separated base64url segments (RFC 7519), the first being a base64url
    header that begins ``eyJ``. Segments are kept at 10+ characters because the rule
    does not match an empty signature (``alg:none``).

    Returns:
        A strategy producing JWT-shaped strings.
    """
    b64url = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=10,
        max_size=40,
    )
    return st.tuples(b64url, b64url, b64url).map(
        lambda parts: f"eyJ{parts[0]}.{parts[1]}.{parts[2]}"
    )


def openai_style_keys() -> st.SearchStrategy[str]:
    """Generate OpenAI-style API keys (``sk-`` + 16 or more alphanumerics).

    Returns:
        A strategy producing OpenAI-style keys.
    """
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=16,
        max_size=48,
    ).map(lambda body: f"sk-{body}")


def openrouter_keys() -> st.SearchStrategy[str]:
    """Generate OpenRouter API keys (``sk-or-v1-`` + 16 or more alphanumerics).

    These are *not* covered by the generic ``sk-`` rule: the hyphens in ``or-v1-``
    break its ``{16,}`` run, so both rules are required.

    Returns:
        A strategy producing OpenRouter keys.
    """
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=16,
        max_size=64,
    ).map(lambda body: f"sk-or-v1-{body}")


def pem_private_keys() -> st.SearchStrategy[str]:
    """Generate PEM private key blocks (RFC 7468).

    The labels come from the formats OpenSSL, OpenSSH and GnuPG actually emit —
    **not** from the redaction rule's own alternation. An earlier version of this
    strategy copied that alternation, which is precisely the circularity this file
    exists to avoid: it made the property vacuously green for
    `ENCRYPTED PRIVATE KEY`, `DSA PRIVATE KEY` and `PGP PRIVATE KEY BLOCK`, all
    three of which leaked their entire body.

    Returns:
        A strategy producing complete PEM blocks.
    """
    kinds = st.sampled_from(
        [
            "",                 # PKCS#8, unencrypted
            "ENCRYPTED ",       # PKCS#8, passphrase-protected
            "RSA ",             # PKCS#1 / traditional OpenSSL
            "DSA ",
            "EC ",
            "OPENSSH ",
        ]
    )
    body = st.lists(_fixed_alnum(64), min_size=1, max_size=8).map("\n".join)
    return st.tuples(kinds, body).map(
        lambda parts: (
            f"-----BEGIN {parts[0]}PRIVATE KEY-----\n{parts[1]}\n-----END {parts[0]}PRIVATE KEY-----"
        )
    )


_SECRET_CLASSES = {
    "github_classic_pat": github_classic_pats(),
    "github_fine_grained_pat": github_fine_grained_pats(),
    "aws_access_key_id": aws_access_key_ids(),
    "jwt": jwts(),
    "openai_style_key": openai_style_keys(),
    "openrouter_key": openrouter_keys(),
    "pem_private_key": pem_private_keys(),
}

_ANY_SECRET = st.one_of(*_SECRET_CLASSES.values())

# `redact_text`'s rules are `\b`-anchored, so a secret abutting `[A-Za-z0-9_]` is not
# redacted — `AKIA…` is missed entirely, since its `{16}` length is exact. Surrounding
# text is therefore drawn from a non-word alphabet. This is a real limitation of the
# anchors, recorded rather than worked around; widening them risks over-redaction
# across the whole product and deserves its own change.
_NON_WORD = st.text(alphabet=" \t\n\"'=:,;<>(){}[]#/\\|!?*&^%$@+", max_size=40)


class TestGeneratorsMatchVendorFormats(unittest.TestCase):
    """Guard the generators themselves — a broken strategy would silently pass."""

    @given(data=st.data())
    @settings(max_examples=25, derandomize=True)
    def test_each_strategy_matches_its_documented_format(self, data: st.DataObject) -> None:
        """Every drawn value matches the vendor's published pattern.

        Args:
            data: Hypothesis draw handle, so each class is sampled many times rather
                than once via `.example()`.
        """
        expected = {
            "github_classic_pat": r"^ghp_[A-Za-z0-9]{36}$",
            "github_fine_grained_pat": r"^github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}$",
            "aws_access_key_id": r"^AKIA[0-9A-Z]{16}$",
            "jwt": r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$",
            "openai_style_key": r"^sk-[A-Za-z0-9]{16,}$",
            "openrouter_key": r"^sk-or-v1-[A-Za-z0-9]{16,}$",
            # RFC 7468 label shape, not the redaction rule's alternation.
            "pem_private_key": r"^-----BEGIN [A-Z ]*PRIVATE KEY-----\n[\s\S]+\n-----END [A-Z ]*PRIVATE KEY-----$",
        }
        # No `subTest` here: Hypothesis warns that per-case reporting interacts badly
        # with hundreds of generated cases. The class name goes in the message instead.
        for name, strategy in _SECRET_CLASSES.items():
            sample = data.draw(strategy)
            self.assertRegex(sample, expected[name], f"{name} generator drifted from its vendor format")


class TestNoSecretSurvivesRedaction(unittest.TestCase):
    """Every recognised secret class is removed from any text it appears in."""

    @given(secret=github_classic_pats(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    @example(secret="ghp_" + "a" * 36, before="token=", after="\n")
    def test_github_classic_pat_never_survives(self, secret: str, before: str, after: str) -> None:
        """A classic GitHub PAT is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    @given(secret=github_fine_grained_pats(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    @example(secret="github_pat_" + "a" * 22 + "_" + "b" * 59, before="TOKEN='", after="'\n")
    def test_github_fine_grained_pat_never_survives(self, secret: str, before: str, after: str) -> None:
        """A fine-grained GitHub PAT is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    @given(secret=aws_access_key_ids(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    @example(secret="AKIAIOSFODNN7EXAMPLE", before="aws_access_key_id = ", after="\n")
    def test_aws_access_key_never_survives(self, secret: str, before: str, after: str) -> None:
        """An AWS access key id is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    @given(secret=jwts(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    @example(secret="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27u", before="Bearer ", after="\n")
    def test_jwt_never_survives(self, secret: str, before: str, after: str) -> None:
        """A JWT is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    @given(secret=openai_style_keys(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    @example(secret="sk-" + "a" * 48, before="OPENAI_API_KEY=", after="\n")
    def test_openai_style_key_never_survives(self, secret: str, before: str, after: str) -> None:
        """An OpenAI-style key is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    @given(secret=openrouter_keys(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    @example(secret="sk-or-v1-" + "a" * 64, before="OPENROUTER_API_KEY=", after="\n")
    def test_openrouter_key_never_survives(self, secret: str, before: str, after: str) -> None:
        """An OpenRouter key is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    @given(secret=pem_private_keys(), before=_NON_WORD, after=_NON_WORD)
    @settings(_SECURITY)
    def test_pem_private_key_never_survives(self, secret: str, before: str, after: str) -> None:
        """A PEM private key block is redacted wherever it appears."""
        self._assert_redacted(secret, before, after)

    def _assert_redacted(self, secret: str, before: str, after: str) -> None:
        """Assert a secret is gone from redacted text, by both available measures.

        Args:
            secret: The generated credential.
            before: Non-word text preceding it.
            after: Non-word text following it.
        """
        redacted = redact_text(f"{before}{secret}{after}")

        self.assertNotIn(secret, redacted)
        self.assertFalse(
            contains_unredacted_secrets(redacted),
            f"a rule still matches the output for {secret[:12]}…",
        )


class TestRedactionProperties(unittest.TestCase):
    """Structural properties that must hold for redaction to be safe to re-run."""

    @given(text=st.text(max_size=500), secret=_ANY_SECRET)
    @settings(_SECURITY)
    def test_redaction_is_idempotent(self, text: str, secret: str) -> None:
        """A second pass changes nothing.

        Lad redacts on ingest *and* on egress, and the fixes for the truncation leak
        add a third pass. Idempotence is what makes that safe. It holds structurally
        because `[REDACTED]` is bracketed by non-word characters, so a substitution
        cannot splice its neighbours into a new `\\b`-anchored match — a replacement
        of bare alphanumerics would break this.
        """
        once = redact_text(f"{text} {secret}")

        self.assertEqual(redact_text(once), once)

    def test_ordinary_source_is_returned_unchanged(self) -> None:
        """Real source files pass through untouched.

        Over-redaction is the other failure mode: the module's own docstring warns it
        "may over-redact". This runs the repo's own code through it, which is a
        realistic corpus rather than generated noise.
        """
        checked = 0
        for path in sorted((_REPO_ROOT / "lad_mcp_server").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                self.assertEqual(redact_text(source), source)
            checked += 1

        self.assertGreater(checked, 5, "expected to check several modules")


if __name__ == "__main__":
    unittest.main()
