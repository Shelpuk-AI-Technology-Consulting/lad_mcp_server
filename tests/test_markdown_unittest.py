import unittest

from lad_mcp_server.markdown import format_aggregated_output, normalize_reviewer_markdown


class TestMarkdown(unittest.TestCase):
    def test_normalization_inserts_sections(self) -> None:
        raw = "hello"
        normalized = normalize_reviewer_markdown(raw)
        self.assertIn("## Summary", normalized)
        self.assertIn("## Key Findings", normalized)
        self.assertIn("## Recommendations", normalized)
        self.assertIn("## Questions / Unknowns", normalized)

    def test_aggregated_structure(self) -> None:
        out = format_aggregated_output(
            primary_markdown="## Summary\nA",
            secondary_markdown="## Summary\nB",
            synthesized_summary="S",
        )
        self.assertIn("## Primary Reviewer", out)
        self.assertIn("## Secondary Reviewer", out)
        self.assertIn("## Synthesized Summary", out)

    def test_aggregated_structure_primary_only(self) -> None:
        out = format_aggregated_output(
            primary_markdown="## Summary\nA",
            secondary_markdown=None,
            synthesized_summary="S",
        )
        self.assertIn("## Primary Reviewer", out)
        self.assertNotIn("## Secondary Reviewer", out)
        self.assertIn("## Synthesized Summary", out)


if __name__ == "__main__":
    unittest.main()
