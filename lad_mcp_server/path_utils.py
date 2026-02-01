from __future__ import annotations

import os
import re
from pathlib import Path


def _looks_like_windows_absolute(path_str: str) -> bool:
    # Drive-letter paths (C:\...) and UNC paths (\\server\share\...)
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_str)) or path_str.startswith("\\\\")


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

