import unittest

from lad_mcp_server.redaction import contains_unredacted_secrets, redact_text


class TestRedaction(unittest.TestCase):
    def test_redacts_sk_keys(self) -> None:
        raw = "here is a key sk-1234567890abcdef1234567890abcdef and more"
        redacted = redact_text(raw)
        self.assertNotIn("sk-1234567890abcdef", redacted)
        self.assertIn("[REDACTED]", redacted)
        self.assertFalse(contains_unredacted_secrets(redacted))

    def test_redacts_openrouter_sk_or(self) -> None:
        raw = "OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuvwxyz0123456789"
        redacted = redact_text(raw)
        self.assertIn("[REDACTED]", redacted)
        self.assertFalse(contains_unredacted_secrets(redacted))

    def test_redacts_pem_block(self) -> None:
        raw = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
        redacted = redact_text(raw)
        self.assertEqual(redacted, "[REDACTED]")


if __name__ == "__main__":
    unittest.main()

