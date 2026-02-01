import tempfile
import unittest
from pathlib import Path

from lad_mcp_server.file_context import FileContextBuilder


class TestFileContext(unittest.TestCase):
    def test_expands_directories_and_excludes_common_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".git").mkdir()
            (repo / ".git" / "config").write_text("ignored", encoding="utf-8")
            (repo / ".venv").mkdir()
            (repo / ".venv" / "x.py").write_text("ignored", encoding="utf-8")

            (repo / "src").mkdir()
            (repo / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
            (repo / "src" / "b.md").write_text("# b\n", encoding="utf-8")

            builder = FileContextBuilder(repo_root=repo)
            ctx = builder.build(paths=["src"], max_chars=10_000)

            self.assertIn("src/a.py", ctx.embedded_files)
            self.assertIn("src/b.md", ctx.embedded_files)
            self.assertNotIn(".git/config", ctx.embedded_files)
            self.assertNotIn(".venv/x.py", ctx.embedded_files)
            self.assertIn("--- BEGIN FILE:", ctx.formatted)

    def test_serena_dir_excluded_on_directory_expansion_but_explicit_file_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena" / "memories").mkdir(parents=True)
            (repo / ".serena" / "memories" / "project_overview.md").write_text("hello\n", encoding="utf-8")
            (repo / "docs").mkdir()
            (repo / "docs" / "note.md").write_text("note\n", encoding="utf-8")

            builder = FileContextBuilder(repo_root=repo)
            ctx1 = builder.build(paths=["."], max_chars=10_000)
            self.assertNotIn(".serena/memories/project_overview.md", ctx1.embedded_files)

            ctx2 = builder.build(paths=[".serena/memories/project_overview.md"], max_chars=10_000)
            self.assertIn(".serena/memories/project_overview.md", ctx2.embedded_files)

    def test_rejects_paths_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            builder = FileContextBuilder(repo_root=repo)
            outside = repo.parent / "outside.txt"
            outside.write_text("nope\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                builder.build(paths=[str(outside)], max_chars=1000)

    def test_stops_when_budget_exhausted_and_records_skips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "src").mkdir()
            (repo / "src" / "a.py").write_text("a" * 2000, encoding="utf-8")
            (repo / "src" / "b.py").write_text("b" * 2000, encoding="utf-8")

            builder = FileContextBuilder(repo_root=repo)
            ctx = builder.build(paths=["src"], max_chars=500)  # too small for both

            self.assertGreaterEqual(len(ctx.embedded_files), 1)
            self.assertTrue(any(s.get("reason") == "budget_exhausted" for s in ctx.skipped_files))

    def test_embeds_non_python_languages_and_skips_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "src").mkdir()
            (repo / "src" / "app.js").write_text("console.log('hi')\n", encoding="utf-8")
            (repo / "src" / "main.go").write_text("package main\nfunc main(){}\n", encoding="utf-8")
            (repo / "src" / "Dockerfile").write_text("FROM alpine\n", encoding="utf-8")
            # binary-ish file
            (repo / "src" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00")

            builder = FileContextBuilder(repo_root=repo)
            ctx = builder.build(paths=["src"], max_chars=10_000)

            self.assertIn("src/app.js", ctx.embedded_files)
            self.assertIn("src/main.go", ctx.embedded_files)
            self.assertIn("src/Dockerfile", ctx.embedded_files)
            self.assertNotIn("src/image.png", ctx.embedded_files)


if __name__ == "__main__":
    unittest.main()
