from __future__ import annotations

import os
import re
from pathlib import Path


def _looks_like_windows_absolute(path_str: str) -> bool:
    # Drive-letter paths (C:\...) and UNC paths (\\server\share\...)
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_str)) or path_str.startswith("\\\\")


def _is_filesystem_root(path: Path) -> bool:
    resolved = path.resolve()
    return resolved.parent == resolved


def is_dangerous_repo_root(repo_root: Path) -> bool:
    """
    Best-effort guard against reviewing arbitrary system directories.

    This is intentionally conservative: Lad should be usable across many repos, but path-based
    reviews should not allow embedding from locations like /etc or C:\\Windows.
    """
    root = repo_root.resolve()

    if _is_filesystem_root(root):
        return True

    # Avoid allowing "home directory root" as a project root (too broad).
    try:
        home = Path.home().resolve()
        if root == home:
            return True
    except Exception:
        # If home cannot be resolved, skip this check.
        pass

    if os.name == "nt":
        # Drive root (e.g., C:\) is always too broad.
        if root.parent == root:
            return True

        windir = Path(os.environ.get("WINDIR", r"C:\Windows")).resolve()
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files")).resolve()
        program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")).resolve()
        program_data = Path(os.environ.get("ProgramData", r"C:\ProgramData")).resolve()

        blocked_prefixes = (windir, program_files, program_files_x86, program_data)
    else:
        blocked_prefixes = (
            Path("/etc"),
            Path("/proc"),
            Path("/sys"),
            Path("/dev"),
            Path("/run"),
            Path("/var"),
            Path("/bin"),
            Path("/sbin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/boot"),
        )

    for prefix in blocked_prefixes:
        try:
            root.relative_to(prefix)
            return True
        except ValueError:
            continue

    return False


def safe_resolve_under_repo(*, repo_root: Path, path_str: str) -> Path:
    """
    Resolve `path_str` (absolute or repo-relative) and ensure it stays under `repo_root`.

    Raises ValueError on invalid/unsafe paths.
    """
    if not isinstance(path_str, str) or path_str.strip() == "":
        raise ValueError("path must be a non-empty string")

    # Defensive: if running on POSIX but given a Windows absolute path, reject.
    if os.name != "nt" and _looks_like_windows_absolute(path_str):
        raise ValueError("windows absolute paths are not supported on this platform")

    p = Path(path_str)
    if ".." in p.parts:
        raise ValueError("path traversal is not allowed")

    resolved = p.resolve() if p.is_absolute() else (repo_root / p).resolve()

    # Compare using normcase on Windows (case-insensitive filesystem).
    resolved_s = str(resolved)
    root_s = str(repo_root.resolve())
    if os.name == "nt":
        resolved_s = os.path.normcase(resolved_s)
        root_s = os.path.normcase(root_s)

    try:
        common = os.path.commonpath([resolved_s, root_s])
    except Exception as exc:
        raise ValueError(f"invalid path: {exc}") from exc

    if common != root_s:
        raise ValueError("path is outside repo root")

    return resolved
