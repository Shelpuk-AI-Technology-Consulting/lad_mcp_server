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
                model=None,
                max_input_chars=1000,
            )

    def test_system_design_accepts_paths_only(self) -> None:
        req = SystemDesignReviewRequest.validate(
            proposal=None,
            paths=["README.md"],
            constraints=None,
            context=None,
            model=None,
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["README.md"])

    def test_code_review_rejects_unknown_focus(self) -> None:
        with self.assertRaises(ValidationError):
            CodeReviewRequest.validate(
                code="print('hi')",
                paths=None,
                language="python",
                focus="unknown",
                model=None,
                max_input_chars=1000,
            )

    def test_code_review_accepts_known_focus(self) -> None:
        req = CodeReviewRequest.validate(
            code="print('hi')",
            paths=None,
            language="python",
            focus="security",
            model=None,
            max_input_chars=1000,
        )
        self.assertEqual(req.focus, "security")

    def test_code_review_accepts_paths_only(self) -> None:
        req = CodeReviewRequest.validate(
            code=None,
            paths=["lad_mcp_server/server.py"],
            language=None,
            focus=None,
            model=None,
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["lad_mcp_server/server.py"])


if __name__ == "__main__":
    unittest.main()
