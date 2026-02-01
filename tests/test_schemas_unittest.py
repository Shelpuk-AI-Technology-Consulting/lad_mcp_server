import unittest

from lad_mcp_server.schemas import CodeReviewRequest, SystemDesignReviewRequest, ValidationError


class TestSchemas(unittest.TestCase):
    def test_system_design_rejects_short(self) -> None:
        with self.assertRaises(ValidationError):
            SystemDesignReviewRequest.validate(
                proposal="too short",
                paths=None,
                constraints=None,
                context=None,
                max_input_chars=1000,
            )

    def test_system_design_accepts_paths_only(self) -> None:
        req = SystemDesignReviewRequest.validate(
            proposal=None,
            paths=["README.md"],
            constraints=None,
            context=None,
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["README.md"])

    def test_system_design_accepts_newline_separated_paths_string(self) -> None:
        req = SystemDesignReviewRequest.validate(
            proposal=None,
            paths="docs/adr.md\nREADME.md\n",
            constraints=None,
            context=None,
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["docs/adr.md", "README.md"])

    # NOTE: `focus` was removed from `code_review` to keep the interface simple and consistent.

    def test_code_review_accepts_paths_only(self) -> None:
        req = CodeReviewRequest.validate(
            code=None,
            paths=["lad_mcp_server/server.py"],
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["lad_mcp_server/server.py"])

    def test_code_review_accepts_context(self) -> None:
        req = CodeReviewRequest.validate(
            code="print('hi')\n",
            paths=None,
            context="Please focus on edge cases and error handling.",
            max_input_chars=10_000,
        )
        self.assertEqual(req.context, "Please focus on edge cases and error handling.")

    def test_code_review_accepts_newline_separated_paths_string(self) -> None:
        req = CodeReviewRequest.validate(
            code=None,
            paths="a.py\nb.py\n",
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["a.py", "b.py"])


if __name__ == "__main__":
    unittest.main()
