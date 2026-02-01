import unittest

from lad_mcp_server.schemas import CodeReviewRequest, SystemDesignReviewRequest, ValidationError


class TestSchemas(unittest.TestCase):
    def test_system_design_rejects_short(self) -> None:
        with self.assertRaises(ValidationError):
            SystemDesignReviewRequest.validate(
                proposal="too short",
                paths=None,
                project_root=None,
                constraints=None,
                context=None,
                model=None,
                max_input_chars=1000,
            )

    def test_system_design_accepts_paths_only(self) -> None:
        req = SystemDesignReviewRequest.validate(
            proposal=None,
            paths=["README.md"],
            project_root=None,
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
                project_root=None,
                language="python",
                focus="unknown",
                model=None,
                max_input_chars=1000,
            )

    def test_code_review_accepts_known_focus(self) -> None:
        req = CodeReviewRequest.validate(
            code="print('hi')",
            paths=None,
            project_root=None,
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
            project_root=None,
            language=None,
            focus=None,
            model=None,
            max_input_chars=1000,
        )
        self.assertEqual(req.paths, ["lad_mcp_server/server.py"])

    def test_project_root_blank_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SystemDesignReviewRequest.validate(
                proposal="0123456789",
                paths=None,
                project_root="",
                constraints=None,
                context=None,
                model=None,
                max_input_chars=1000,
            )


if __name__ == "__main__":
    unittest.main()
