import unittest

from lad_mcp_server.prompts import force_finalize_system_message, system_prompt_code_review, system_prompt_system_design_review


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

    def test_force_finalize_prompt_demands_actionable_partial_review(self) -> None:
        p = force_finalize_system_message()
        self.assertIn("current message chain", p)
        self.assertIn("tool results", p)
        self.assertIn("gaps", p)
        self.assertIn("contradictions", p)
        self.assertIn("inconsistencies", p)
        self.assertIn("improvement opportunities", p)
        self.assertIn("Do not summarize which tools", p)
        self.assertIn("## Key Findings", p)


if __name__ == "__main__":
    unittest.main()
