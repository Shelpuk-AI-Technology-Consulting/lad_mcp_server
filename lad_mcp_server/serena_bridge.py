from __future__ import annotations

import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from lad_mcp_server.redaction import redact_text
from lad_mcp_server.path_utils import safe_resolve_under_repo

LARGE_FILE_READ_MAX_BYTES = 100_000_000
LARGE_FILE_HEAD_WARNING_LINES = 400
READ_FILE_TRUNCATE_THRESHOLD = 10_000
READ_FILE_MAX_HEAD_TAIL_CHARS = 1_000
BASELINE_REQUIRED_MEMORIES = ("project_overview.md", "research_summary.md")


class SerenaToolError(RuntimeError):
    pass


def _commonpath_is_within(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(child), str(parent)]) == str(parent)
    except Exception:
        return False


@dataclass(frozen=True)
class SerenaLimits:
    max_dir_entries: int
    max_search_results: int
    max_tool_result_chars: int
    max_total_chars: int
    tool_timeout_seconds: int


class SerenaContext:
    """
    A minimal, repo-scoped, read-only bridge providing Serena-like context capabilities.

    This does NOT start or depend on the Serena MCP server process. It uses the local `.serena/` folder
    (memories) and safe filesystem/search operations to provide additional context to reviewer LLMs.
    """

    def __init__(self, *, repo_root: Path, limits: SerenaLimits) -> None:
        self.repo_root = repo_root.resolve()
        self._limits = limits

        self.serena_dir = self.repo_root / ".serena"
        self.memories_dir = self.serena_dir / "memories"

        self._total_chars_emitted = 0
        self.used_tools: set[str] = set()
        self.total_tool_calls: int = 0
        self.used_memories: set[str] = set()
        self.used_paths: set[str] = set()
        self.activated_project: str | None = None

    def _require_activated(self) -> None:
        if self.activated_project is None:
            raise SerenaToolError("activate_project must be called first")

    @staticmethod
    def detect(repo_root: Path, limits: SerenaLimits) -> "SerenaContext | None":
        repo_root = repo_root.resolve()
        if (repo_root / ".serena").is_dir():
            return SerenaContext(repo_root=repo_root, limits=limits)
        return None

    def _budget_snapshot(self, *, emitted_chars_this_call: int) -> dict[str, int]:
        remaining = max(self._limits.max_total_chars - self._total_chars_emitted, 0)
        return {
            "max_total_chars": int(self._limits.max_total_chars),
            "used_chars": int(self._total_chars_emitted),
            "remaining_chars": int(remaining),
            "emitted_chars_this_call": int(emitted_chars_this_call),
        }

    def _safe_resolve_under_repo(self, relative_path: str) -> Path:
        try:
            return safe_resolve_under_repo(repo_root=self.repo_root, path_str=relative_path)
        except ValueError as exc:
            raise SerenaToolError(str(exc)) from exc

    def _repo_relative_posix(self, path: Path) -> str | None:
        """Convert a path to a repo-relative POSIX string.

        Repo-relative paths are identifiers here, not filesystem arguments: they are
        embedded in reviewer prompts, compared against :class:`FileContextBuilder`
        output, and printed in the disclosure block. They must never be absolute and
        never contain a backslash, on any platform.

        Args:
            path: A filesystem path, absolute or relative to the repo root.

        Returns:
            The repo-relative path using forward slashes, or ``None`` when it
            resolves outside the repo root.
        """
        candidate = path if path.is_absolute() else (self.repo_root / path)
        try:
            return candidate.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return None

    def _normalize_match_line(self, line: str) -> tuple[str, str] | None:
        """Rewrite one ``rg -n --null`` output line to carry a repo-relative POSIX path.

        ``--null`` makes ``rg`` terminate the path with a NUL byte instead of the
        usual colon, which is the only way to split it unambiguously: the path may
        be absolute (a Windows drive letter contains a colon), and on POSIX a
        filename may itself contain ``:<digits>:``. Guessing with a ``path:line:``
        pattern fabricated entries in both cases — ``D`` from ``D:\\repo\\...``, and
        ``src`` from a file named ``src:2:notes.txt`` in a repo that has a ``src/``
        directory.

        A line without a NUL is therefore dropped rather than guessed at. The caller
        detects wholesale parse failure and re-runs the search in Python, so an
        ``rg`` build that ignored the flag degrades instead of reporting no matches.

        Args:
            line: One line of ``rg -n --null`` output, i.e. ``path\\0line:text``.

        Returns:
            A ``(repo_relative_path, rewritten_line)`` pair, or ``None`` when the
            line carries no NUL or names a path outside the repo root.
        """
        raw_path, sep, remainder = line.partition("\0")
        if not sep:
            return None
        rel = self._repo_relative_posix(Path(raw_path))
        if rel is None:
            return None
        # Restore the human-readable `path:line:text` shape for the model.
        return rel, f"{rel}:{remainder}"

    def tool_schemas(self) -> list[dict[str, Any]]:
        """
        OpenAI-compatible tool schema definitions to pass to models via OpenRouter.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "activate_project",
                    "description": "Activate the current project (required preflight). Call with project='.' to activate the repo root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project": {
                                "type": "string",
                                "description": "Must be '.' or the absolute path to the repo root.",
                            }
                        },
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_memories",
                    "description": "List available Serena memories for this project (from .serena/memories).",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_project_overview",
                    "description": "Read the Serena memory `.serena/memories/project_overview.md` (if present).",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_baseline_memories",
                    "description": (
                        "Fetch required baseline memories in one preflight call "
                        "(project_overview.md and research_summary.md)."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_memory",
                    "description": "Read a Serena memory file by name (no path traversal).",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "description": "Memory name (with or without .md)."}},
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List files/directories under a repo-relative path (read-only).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Repo-relative path. Use '.' for repo root."}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a text file under the repo root (read-only). "
                        "For files larger than 10,000 characters, returns only the first 1000 and last 1000 "
                        "characters (in 'content' and 'tail' fields) plus 'file_size'. "
                        "For smaller files, returns full content. "
                        "Use head for first N lines and tail for last N lines (small files only)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Repo-relative file path."},
                            "head": {"type": "integer", "description": "Optional: read only first N lines."},
                            "tail": {"type": "integer", "description": "Optional: read only last N lines."},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file_window",
                    "description": (
                        "Read a targeted line window from a repo file. "
                        "start_line is 1-based; num_lines is the number of lines to read. "
                        "Useful for targeted JSON sections and focused code inspection "
                        "(for example, locate a function and read only its nearby implementation window)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Repo-relative file path."},
                            "start_line": {
                                "type": "integer",
                                "description": "1-based line number to start from (>= 1).",
                                "minimum": 1,
                            },
                            "num_lines": {
                                "type": "integer",
                                "description": "Number of lines to read (>= 0, 0 returns empty content).",
                                "minimum": 0,
                            },
                        },
                        "required": ["path", "start_line", "num_lines"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_for_pattern",
                    "description": "Search for a plain substring or regex pattern in repo files (best-effort, read-only).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Substring/regex pattern to search for."},
                            "path": {"type": "string", "description": "Optional repo-relative path to restrict search."},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_substring_in_file",
                    "description": (
                        "Search a single repo file for a literal, case-sensitive substring. "
                        "Returns total non-overlapping match count, file_size in characters, and up to 30 occurrences. "
                        "First 5 occurrences include index, line, and context (up to 100 chars before + substring + up to 100 chars after). "
                        "Occurrences 6-30 include index and line only. "
                        "Files up to 100 MB are supported."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Repo-relative file path."},
                            "substring": {"type": "string", "description": "Literal substring to search for."},
                        },
                        "required": ["path", "substring"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_symbol",
                    "description": "Best-effort symbol lookup (Python def/class) across repo files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Symbol name to find (e.g., MyClass, my_func)."},
                            "path": {"type": "string", "description": "Optional repo-relative path to restrict search."},
                        },
                        "required": ["name"],
                    },
                },
            },
        ]

    def call_tool(self, name: str, arguments_json: str) -> str:
        """
        Execute a tool call and return tool output as a JSON string.

        Output is redacted and capped to configured budgets.
        """
        self.total_tool_calls += 1
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            raise SerenaToolError(f"Invalid tool arguments JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise SerenaToolError("Tool arguments must be a JSON object")

        # Serena parity: require project activation before any other tool call.
        if name != "activate_project":
            self._require_activated()

        if name == "list_memories":
            result = self._list_memories()
        elif name == "activate_project":
            result = self._activate_project(args.get("project"))
        elif name == "read_project_overview":
            result = self._read_memory("project_overview")
        elif name == "read_baseline_memories":
            result = self._read_baseline_memories()
        elif name == "read_memory":
            result = self._read_memory(args.get("name"))
        elif name == "list_dir":
            result = self._list_dir(args.get("path"))
        elif name == "read_file":
            result = self._read_file(args.get("path"), args.get("head"), args.get("tail"))
        elif name == "read_file_window":
            result = self._read_file_window(args.get("path"), args.get("start_line"), args.get("num_lines"))
        elif name == "search_for_pattern":
            result = self._search_for_pattern(args.get("pattern"), args.get("path"))
        elif name == "search_substring_in_file":
            result = self._search_substring_in_file(args.get("path"), args.get("substring"))
        elif name == "find_symbol":
            result = self._find_symbol(args.get("name"), args.get("path"))
        else:
            raise SerenaToolError(f"Unknown tool: {name}")

        self.used_tools.add(name)

        raw_result_json = json.dumps(result, ensure_ascii=False, indent=2)
        raw_result_json = redact_text(raw_result_json)
        remaining_before = max(self._limits.max_total_chars - self._total_chars_emitted, 0)

        if remaining_before <= 0:
            payload = {
                "tool_status": "budget_exhausted",
                "tool_name": name,
                "tool_params": args,
                "tool_budget": self._budget_snapshot(emitted_chars_this_call=0),
                "error": "serena output budget exhausted",
                "hint": "Reduce scope or increase LAD_SERENA_MAX_TOTAL_CHARS.",
                "tool_result_json": "",
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        max_result_chars = min(len(raw_result_json), int(self._limits.max_tool_result_chars), int(remaining_before))
        emitted_result_json = raw_result_json[:max_result_chars]
        self._total_chars_emitted += len(emitted_result_json)

        status = "ok" if len(emitted_result_json) == len(raw_result_json) else "truncated"
        payload: dict[str, Any] = {
            "tool_status": status,
            "tool_name": name,
            "tool_params": args,
            "tool_budget": self._budget_snapshot(emitted_chars_this_call=len(emitted_result_json)),
            "tool_result_json": emitted_result_json,
        }
        if status == "truncated":
            payload["note"] = "Tool output was truncated by per-call or total Serena budget limits."
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _list_memories(self) -> dict[str, Any]:
        if not self.memories_dir.is_dir():
            return {"memories": [], "note": "No .serena/memories directory found."}
        memories = sorted(p.name for p in self.memories_dir.glob("*.md") if p.is_file())
        return {"memories": memories}

    def _activate_project(self, project: Any) -> dict[str, Any]:
        # Some models may call `activate_project` without arguments even when tool_choice forces it.
        # Be permissive here to improve robustness: default to "." (repo root).
        if project is None:
            project = "."
        if not isinstance(project, str) or project.strip() == "":
            project = "."

        # In Serena, `activate_project` accepts a project name or a path.
        # In this Lad MCP server bridge, we only allow activating the current repo root.
        allowed = {
            ".",
            str(self.repo_root),
            str(self.repo_root.as_posix()),
            str(self.repo_root.as_posix()).rstrip("/"),
        }
        if project not in allowed:
            raise SerenaToolError("Only the current repo root can be activated by this server")

        self.activated_project = project
        return {
            "status": "activated",
            "project": project,
            "note": "Project activated. You may now use Serena tools like list_memories/read_memory.",
        }

    def _read_memory(self, name: Any) -> dict[str, Any]:
        if not isinstance(name, str) or name.strip() == "":
            raise SerenaToolError("name must be a non-empty string")

        filename = name if name.endswith(".md") else f"{name}.md"
        if ".." in Path(filename).parts:
            raise SerenaToolError("path traversal is not allowed")

        path = (self.memories_dir / filename).resolve()
        if not _commonpath_is_within(path, self.memories_dir.resolve()):
            raise SerenaToolError("invalid memory path")
        if not path.is_file():
            raise SerenaToolError("memory not found")

        self.used_memories.add(filename)
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"name": filename, "content": content}

    def _read_baseline_memories(self) -> dict[str, Any]:
        required = list(BASELINE_REQUIRED_MEMORIES)
        if not self.memories_dir.is_dir():
            return {"required": required, "present": [], "loaded": [], "missing": required}

        available = {p.name for p in self.memories_dir.glob("*.md") if p.is_file()}
        present = [name for name in required if name in available]
        missing = [name for name in required if name not in available]

        loaded: list[dict[str, str]] = []
        for name in present:
            try:
                mem = self._read_memory(name)
            except SerenaToolError:
                missing.append(name)
                continue
            loaded.append({"name": mem["name"], "content": mem["content"]})

        # Keep deterministic order.
        missing = sorted(set(missing), key=required.index if required else None)
        return {
            "required": required,
            "present": present,
            "loaded": loaded,
            "missing": missing,
        }

    def _list_dir(self, path: Any) -> dict[str, Any]:
        if not isinstance(path, str) or path.strip() == "":
            raise SerenaToolError("path must be a non-empty string")
        rel = "." if path == "." else path
        target = self.repo_root if rel == "." else self._safe_resolve_under_repo(rel)
        if not target.exists():
            raise SerenaToolError("path not found")
        if not target.is_dir():
            raise SerenaToolError("path is not a directory")

        entries = []
        for child in sorted(target.iterdir(), key=lambda p: p.name)[: self._limits.max_dir_entries]:
            entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
        rel = target.relative_to(self.repo_root).as_posix()
        self.used_paths.add(rel)
        return {"path": rel, "entries": entries}

    def _search_for_pattern(self, pattern: Any, path: Any) -> dict[str, Any]:
        if not isinstance(pattern, str) or pattern.strip() == "":
            raise SerenaToolError("pattern must be a non-empty string")

        restrict_dir = self.repo_root
        if isinstance(path, str) and path.strip():
            restrict_dir = self.repo_root if path == "." else self._safe_resolve_under_repo(path)
            if restrict_dir.is_file():
                restrict_dir = restrict_dir.parent

        # Prefer ripgrep if available (fast, with max-count). Run from the repo root
        # with a repo-relative target so paths come back relative, and with `--null`
        # so the path is NUL-terminated — the only unambiguous way to split it.
        cmd = [
            "rg",
            "-n",
            "--null",
            "--max-count",
            str(self._limits.max_search_results),
            pattern,
            restrict_dir.relative_to(self.repo_root).as_posix(),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._limits.tool_timeout_seconds,
                cwd=str(self.repo_root),
            )
        except FileNotFoundError:
            return self._search_for_pattern_fallback(pattern, restrict_dir)
        except subprocess.TimeoutExpired:
            raise SerenaToolError("search timed out")

        # rg exit codes: 0 = matched, 1 = no match, >= 2 = error. An error with no
        # output (an unknown flag, say) previously read as "no matches"; search in
        # Python instead of silently reporting nothing.
        stdout = proc.stdout.strip()
        if proc.returncode >= 2 and not stdout:
            return self._search_for_pattern_fallback(pattern, restrict_dir)
        matches = stdout.splitlines() if stdout else []

        # Normalize paths to repo-relative POSIX; drop anything we cannot place
        # inside the repo rather than leaking an absolute host path to the model.
        rel_matches: list[str] = []
        delimited = 0
        for line in matches[: self._limits.max_search_results]:
            # Counted separately from the result: a line we cannot split is a parser
            # failure, whereas one that splits but resolves outside the repo is a
            # legitimate drop. Conflating them would make one stray match re-run the
            # whole search.
            if "\0" not in line:
                continue
            delimited += 1
            normalized = self._normalize_match_line(line)
            if normalized is None:
                continue
            rel, rewritten = normalized
            self.used_paths.add(rel)
            rel_matches.append(rewritten)

        # rg produced output but not one line carried a NUL — most likely a build
        # that ignored `--null`. Reporting "no matches" would be a wrong answer
        # rather than a degraded one, so redo the search in Python.
        if matches and delimited == 0:
            return self._search_for_pattern_fallback(pattern, restrict_dir)
        return {"matches": rel_matches}

    def _search_for_pattern_fallback(self, pattern: str, restrict_dir: Path) -> dict[str, Any]:
        """
        Pure-Python fallback when `rg` is unavailable.

        Behavior is best-effort and intentionally conservative to avoid scanning huge repos:
        - skips common excluded directories
        - skips very large files (1MB)
        - stops after max_search_results matches
        """
        if len(pattern) > 500:
            return {"matches": [], "note": "Python fallback search rejected an excessively long pattern."}

        # SECURITY: treat pattern as a literal substring only.
        # Python's `re` module can be vulnerable to catastrophic backtracking (ReDoS) for attacker-controlled
        # patterns, and it does not support timeouts. Full regex search is available via `rg`.
        if re.search(r"[.^$*+?{}\[\]\\\\|()]", pattern):
            return {
                "matches": [],
                "note": "Python fallback search does not support regex patterns (ReDoS-safe). Install `rg` for regex search.",
            }

        excluded = {".git", ".venv", "__pycache__", "node_modules"}
        matches: list[str] = []
        start = time.monotonic()

        for root, dirnames, filenames in os.walk(restrict_dir):
            if time.monotonic() - start > float(self._limits.tool_timeout_seconds):
                return {"matches": matches, "note": "Python fallback search timed out."}
            dirnames[:] = [d for d in dirnames if d not in excluded and not d.startswith(".")]
            dirnames.sort()
            for fn in sorted(filenames):
                if len(matches) >= self._limits.max_search_results:
                    break
                if time.monotonic() - start > float(self._limits.tool_timeout_seconds):
                    return {"matches": matches, "note": "Python fallback search timed out."}
                if fn.startswith("."):
                    continue
                fp = Path(root) / fn
                try:
                    # size guard
                    if fp.stat().st_size > 1_000_000:
                        continue
                    with fp.open("rb") as fh:
                        sample = fh.read(8192)
                        if b"\x00" in sample:
                            continue
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                for i, line in enumerate(text.splitlines(), start=1):
                    if len(matches) >= self._limits.max_search_results:
                        break
                    if pattern in line:
                        rel = self._repo_relative_posix(fp)
                        if rel is None:
                            continue
                        self.used_paths.add(rel)
                        matches.append(f"{rel}:{i}:{line[:200]}")

            if len(matches) >= self._limits.max_search_results:
                break

        note = "Used Python fallback search (rg not available)."
        return {"matches": matches, "note": note}

    def _find_symbol(self, name: Any, path: Any) -> dict[str, Any]:
        if not isinstance(name, str) or name.strip() == "":
            raise SerenaToolError("name must be a non-empty string")

        # Best-effort: search for python def/class declarations.
        pattern = rf"^(?:def|class)\s+{name}\b"
        return self._search_for_pattern(pattern, path)

    def _search_substring_in_file(self, path: Any, substring: Any) -> dict[str, Any]:
        if not isinstance(path, str) or path.strip() == "":
            raise SerenaToolError("path must be a non-empty string")
        if not isinstance(substring, str) or substring == "":
            raise SerenaToolError("substring must be a non-empty string")

        target = self._safe_resolve_under_repo(path)
        if not target.is_file():
            raise SerenaToolError("path is not a file")
        try:
            size = target.stat().st_size
        except OSError:
            raise SerenaToolError("failed to stat file")
        if size > LARGE_FILE_READ_MAX_BYTES:
            raise SerenaToolError("file is too large to search")

        text = target.read_text(encoding="utf-8", errors="replace")
        rel = target.relative_to(self.repo_root).as_posix()
        self.used_paths.add(rel)

        sub_len = len(substring)
        count = 0
        match_positions: list[int] = []
        search_from = 0
        while True:
            pos = text.find(substring, search_from)
            if pos < 0:
                break
            count += 1
            if count <= 30:
                match_positions.append(pos)
            search_from = pos + sub_len

        result: dict[str, Any] = {"path": rel, "file_size": len(text), "count": count}
        if count > 0:
            occurrences: list[dict[str, Any]] = []
            for i, pos in enumerate(match_positions):
                line = text.count("\n", 0, pos) + 1
                if i < 5:
                    head = text[max(0, pos - 100) : pos]
                    tail_start = pos + sub_len
                    tail = text[tail_start : tail_start + 100]
                    occurrences.append(
                        {"index": pos, "line": line, "context": head + substring + tail}
                    )
                else:
                    occurrences.append({"index": pos, "line": line})
            result["occurrences"] = occurrences
        return result

    def _read_file(self, path: Any, head: Any, tail: Any) -> dict[str, Any]:
        if not isinstance(path, str) or path.strip() == "":
            raise SerenaToolError("path must be a non-empty string")
        target = self._safe_resolve_under_repo(path)
        if not target.is_file():
            raise SerenaToolError("path is not a file")

        head_n = int(head) if isinstance(head, int) else None
        tail_n = int(tail) if isinstance(tail, int) else None
        if head_n is not None and head_n < 0:
            raise SerenaToolError("head must be >= 0")
        if tail_n is not None and tail_n < 0:
            raise SerenaToolError("tail must be >= 0")

        try:
            size = target.stat().st_size
        except OSError:
            raise SerenaToolError("failed to stat file")

        # Reject files > LARGE_FILE_READ_MAX_BYTES without explicit head/tail.
        if size > LARGE_FILE_READ_MAX_BYTES and head_n is None and tail_n is None:
            raise SerenaToolError("file is too large to read without head/tail")

        rel = target.relative_to(self.repo_root).as_posix()
        self.used_paths.add(rel)

        # Streaming path for files > LARGE_FILE_READ_MAX_BYTES with head/tail.
        if size > LARGE_FILE_READ_MAX_BYTES and (head_n is not None or tail_n is not None):
            head_lines: list[str] = []
            tail_lines: deque[str] | None = deque(maxlen=tail_n) if tail_n else None
            with target.open("r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    if head_n is not None and i <= head_n:
                        head_lines.append(line)
                    if tail_lines is not None:
                        tail_lines.append(line)
            lines: list[str] = []
            if head_n is not None:
                lines.extend(head_lines)
            if tail_lines is not None:
                if head_n is not None:
                    lines.append("\n[NOTE: Middle of file omitted due to size.]\n")
                lines.extend(list(tail_lines))
            # Redact here too: this branch returns without reaching the whole-text
            # redaction below, and its head/tail line slicing cuts PEM terminators
            # exactly as the char-based truncation does.
            content = redact_text("".join(lines))
            result: dict[str, Any] = {"path": rel, "content": content}
            if (
                size > LARGE_FILE_READ_MAX_BYTES
                and head_n is not None
                and head_n >= LARGE_FILE_HEAD_WARNING_LINES
                and tail_n is None
            ):
                result["warning"] = (
                    "Large file with large head request. Prefer `search_for_pattern` first, then "
                    "`read_file_window(path, start_line, num_lines)` for a focused read."
                )
            return result

        # Normal path: load full text.
        #
        # Redact BEFORE truncating. `call_tool` redacts the assembled result too, but
        # by then the head/tail slice below has already cut the `-----END-----` marker
        # off any PEM block, and the multi-line rule cannot match what is no longer
        # there — so the key material would egress. A second pass only ever redacts more
        # (monotonic), so the later pass is harmless.
        text = redact_text(target.read_text(encoding="utf-8", errors="replace"))

        # Auto-truncate files above the character threshold.
        if len(text) > READ_FILE_TRUNCATE_THRESHOLD:
            return {
                "path": rel,
                "content": text[:READ_FILE_MAX_HEAD_TAIL_CHARS],
                "tail": text[-READ_FILE_MAX_HEAD_TAIL_CHARS:],
                "file_size": len(text),
            }

        # Small file: apply optional line-based head/tail slicing.
        lines = text.splitlines(keepends=True)
        if head_n is not None:
            lines = lines[:head_n]
        if tail_n is not None:
            lines = lines[-tail_n:] if tail_n != 0 else []
        content = "".join(lines)

        return {"path": rel, "content": content}

    def _read_file_window(self, path: Any, start_line: Any, num_lines: Any) -> dict[str, Any]:
        if not isinstance(path, str) or path.strip() == "":
            raise SerenaToolError("path must be a non-empty string")
        target = self._safe_resolve_under_repo(path)
        if not target.is_file():
            raise SerenaToolError("path is not a file")
        if not isinstance(start_line, int):
            raise SerenaToolError("start_line must be an integer >= 1")
        if start_line < 1:
            raise SerenaToolError("start_line must be >= 1")
        if not isinstance(num_lines, int):
            raise SerenaToolError("num_lines must be an integer >= 0")
        if num_lines < 0:
            raise SerenaToolError("num_lines must be >= 0")

        rel = target.relative_to(self.repo_root).as_posix()
        self.used_paths.add(rel)
        if num_lines == 0:
            return {"path": rel, "start_line": start_line, "num_lines": num_lines, "content": ""}

        stop_line = start_line + num_lines - 1
        lines: list[str] = []
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh, start=1):
                if idx < start_line:
                    continue
                if idx > stop_line:
                    break
                lines.append(line)

        return {"path": rel, "start_line": start_line, "num_lines": num_lines, "content": "".join(lines)}
