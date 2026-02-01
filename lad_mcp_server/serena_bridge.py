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

    def _cap_and_track(self, content: str) -> str:
        remaining = max(self._limits.max_total_chars - self._total_chars_emitted, 0)
        capped = content[: min(len(content), self._limits.max_tool_result_chars, remaining)]
        self._total_chars_emitted += len(capped)
        return capped

    def _safe_resolve_under_repo(self, relative_path: str) -> Path:
        try:
            return safe_resolve_under_repo(repo_root=self.repo_root, path_str=relative_path)
        except ValueError as exc:
            raise SerenaToolError(str(exc)) from exc

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
                    "description": "Read a text file under the repo root (read-only).",
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
        elif name == "read_memory":
            result = self._read_memory(args.get("name"))
        elif name == "list_dir":
            result = self._list_dir(args.get("path"))
        elif name == "read_file":
            result = self._read_file(args.get("path"), args.get("head"), args.get("tail"))
        elif name == "search_for_pattern":
            result = self._search_for_pattern(args.get("pattern"), args.get("path"))
        elif name == "find_symbol":
            result = self._find_symbol(args.get("name"), args.get("path"))
        else:
            raise SerenaToolError(f"Unknown tool: {name}")

        self.used_tools.add(name)
        out = json.dumps(result, ensure_ascii=False, indent=2)
        out = redact_text(out)
        out = self._cap_and_track(out)
        return out

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
        self.used_paths.add(str(target.relative_to(self.repo_root)))
        return {"path": str(target.relative_to(self.repo_root)), "entries": entries}

    def _search_for_pattern(self, pattern: Any, path: Any) -> dict[str, Any]:
        if not isinstance(pattern, str) or pattern.strip() == "":
            raise SerenaToolError("pattern must be a non-empty string")

        restrict_dir = self.repo_root
        if isinstance(path, str) and path.strip():
            restrict_dir = self.repo_root if path == "." else self._safe_resolve_under_repo(path)
            if restrict_dir.is_file():
                restrict_dir = restrict_dir.parent

        # Prefer ripgrep if available (fast, with max-count).
        cmd = [
            "rg",
            "-n",
            "--max-count",
            str(self._limits.max_search_results),
            pattern,
            str(restrict_dir),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self._limits.tool_timeout_seconds)
        except FileNotFoundError:
            return self._search_for_pattern_fallback(pattern, restrict_dir)
        except subprocess.TimeoutExpired:
            raise SerenaToolError("search timed out")

        # rg exit code 1 means no matches; treat as empty.
        stdout = proc.stdout.strip()
        matches = stdout.splitlines() if stdout else []

        # Normalize paths to repo-relative for reporting.
        rel_matches: list[str] = []
        for line in matches[: self._limits.max_search_results]:
            # rg format: path:line:match
            if ":" in line:
                p = line.split(":", 1)[0]
                try:
                    rel_p = str(Path(p).resolve().relative_to(self.repo_root))
                    self.used_paths.add(rel_p)
                    rel_matches.append(line.replace(p, rel_p, 1))
                except Exception:
                    rel_matches.append(line)
            else:
                rel_matches.append(line)
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
                        try:
                            rel = str(fp.resolve().relative_to(self.repo_root))
                            self.used_paths.add(rel)
                        except Exception:
                            rel = str(fp)
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

        max_bytes = 1_000_000
        try:
            size = target.stat().st_size
        except OSError:
            raise SerenaToolError("failed to stat file")

        # Avoid loading huge files into memory unless the caller explicitly requests head/tail slices.
        if size > max_bytes and head_n is None and tail_n is None:
            raise SerenaToolError("file is too large to read without head/tail")

        if size > max_bytes and (head_n is not None or tail_n is not None):
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
            content = "".join(lines)
        else:
            text = target.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            if head_n is not None:
                lines = lines[:head_n]
            if tail_n is not None:
                lines = lines[-tail_n:] if tail_n != 0 else []
            content = "".join(lines)

        rel = str(target.relative_to(self.repo_root))
        self.used_paths.add(rel)
        return {"path": rel, "content": content}
