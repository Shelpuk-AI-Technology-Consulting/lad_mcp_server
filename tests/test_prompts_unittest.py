import unittest

from lad_mcp_server.prompts import (
    intermittent_review_finalize_user_message,
    system_prompt_code_review,
    system_prompt_system_design_review,
)


class TestPrompts(unittest.TestCase):
    def test_code_review_prompt_includes_serena_workflow_when_tools_enabled(self) -> None:
        p = system_prompt_code_review(tool_calling_enabled=True)
        self.assertIn("activate_project", p)
        self.assertIn("list_memories", p)
        self.assertIn("read_memory", p)
        self.assertIn("read_file", p)
        self.assertIn("read_file_window", p)
        self.assertIn("search_for_pattern", p)
        self.assertIn("head", p)
        self.assertIn("tail", p)
        self.assertIn("search_for_pattern", p)
        self.assertIn("->", p)
        self.assertIn("read_file_window", p)

    def test_system_design_prompt_includes_serena_workflow_when_tools_enabled(self) -> None:
        p = system_prompt_system_design_review(tool_calling_enabled=True)
        self.assertIn("activate_project", p)
        self.assertIn("list_memories", p)
        self.assertIn("read_memory", p)
        self.assertIn("read_file", p)
        self.assertIn("read_file_window", p)
        self.assertIn("search_for_pattern", p)
        self.assertIn("head", p)
        self.assertIn("tail", p)
        self.assertIn("->", p)

    def test_prompts_do_not_mention_serena_when_tools_disabled(self) -> None:
        p1 = system_prompt_code_review(tool_calling_enabled=False)
        p2 = system_prompt_system_design_review(tool_calling_enabled=False)
        self.assertNotIn("activate_project", p1)
        self.assertNotIn("activate_project", p2)

    def test_intermittent_review_prompt_requests_bounded_partial_snapshot(self) -> None:
        p = intermittent_review_finalize_user_message()
        self.assertIn("time-bounded partial review", p)
        self.assertIn("concise snapshot", p)
        self.assertIn("top 5-10", p)
        self.assertIn("not yet explored", p)
        self.assertIn("## Summary", p)
        self.assertIn("## Key Findings", p)
        self.assertIn("## Recommendations", p)
        self.assertIn("## Questions / Unknowns", p)


if __name__ == "__main__":
    unittest.main()
