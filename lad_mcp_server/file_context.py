from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lad_mcp_server.path_utils import safe_resolve_under_repo

_EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".serena",
}

_DEFAULT_BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".jar",
    ".wasm",
    ".pyc",
    ".pyo",
    ".db",
    ".sqlite",
    ".parquet",
    ".feather",
    ".bin",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
}


@dataclass(frozen=True)
class FileContext:
    formatted: str
    embedded_files: tuple[str, ...]
    skipped_files: tuple[dict[str, Any], ...]


class FileContextBuilder:
    def __init__(
        self,
        *,
        repo_root: Path,
        allowed_extensions: set[str] | None = None,
        binary_extensions: set[str] | None = None,
        excluded_dir_names: set[str] | None = None,
        max_bytes_per_file: int = 1_000_000,
        max_files: int = 2000,
    ) -> None:
        self.repo_root = repo_root.resolve()
        # If not provided, allow any extension and rely on binary detection + size caps.
        self.allowed_extensions = allowed_extensions
        self.binary_extensions = binary_extensions or set(_DEFAULT_BINARY_EXTENSIONS)
        self.excluded_dir_names = excluded_dir_names or set(_EXCLUDED_DIR_NAMES)
        self.max_bytes_per_file = max_bytes_per_file
        self.max_files = max_files

    @staticmethod
    def _is_likely_binary(sample: bytes) -> bool:
        return b"\x00" in sample

    def _safe_resolve_under_repo(self, path_str: str) -> Path:
        return safe_resolve_under_repo(repo_root=self.repo_root, path_str=path_str)

    def _iter_files(self, resolved_paths: list[Path]) -> Iterable[Path]:
        for p in resolved_paths:
            if p.is_dir():
                # Deterministic walk (sorted) with directory pruning.
                for root, dirnames, filenames in os.walk(p):
                    dirnames[:] = sorted(
                        d
                        for d in dirnames
                        if d not in self.excluded_dir_names and not d.startswith(".")
                    )
                    for fn in sorted(filenames):
                        if fn.startswith("."):
                            continue
                        yield Path(root) / fn
            else:
                yield p

    def build(self, *, paths: list[str], max_chars: int) -> FileContext:
        if not isinstance(paths, list) or not paths:
            raise ValueError("paths must be a non-empty list of strings")
        if max_chars <= 0:
            return FileContext(formatted="", embedded_files=(), skipped_files=())

        resolved_inputs = [self._safe_resolve_under_repo(p) for p in paths]
        files: list[Path] = []
        scan_truncated = False
        for f in self._iter_files(resolved_inputs):
            if not f.exists():
                continue
            if f.is_dir():
                continue
            if self.max_files > 0 and len(files) >= self.max_files:
                scan_truncated = True
                break
            files.append(f)

        embedded: list[str] = []
        skipped: list[dict[str, Any]] = []
        chunks: list[str] = []
        remaining = max_chars

        if scan_truncated:
            skipped.append(
                {
                    "path": "(directory scan)",
                    "reason": "too_many_files",
                    "note": f"stopped after {self.max_files} files; additional files were not considered",
                }
            )

        for idx, f in enumerate(files):
            rel = f.relative_to(self.repo_root).as_posix()
            ext = f.suffix.lower()
            if ext in self.binary_extensions:
                skipped.append({"path": rel, "reason": "binary_extension"})
                continue
            if self.allowed_extensions is not None and ext and ext not in self.allowed_extensions:
                skipped.append({"path": rel, "reason": "unsupported_extension"})
                continue

            try:
                st = f.stat()
            except OSError:
                skipped.append({"path": rel, "reason": "stat_failed"})
                continue
            if st.st_size > self.max_bytes_per_file:
                skipped.append({"path": rel, "reason": "too_large"})
                continue

            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            header = f"--- BEGIN FILE: {rel} (Last modified: {mtime}) ---\n"
            footer = f"\n--- END FILE: {rel} ---\n"

            try:
                with f.open("rb") as fh:
                    data = fh.read(self.max_bytes_per_file)
                sample = data[: min(len(data), 65536)]
                if self._is_likely_binary(sample):
                    skipped.append({"path": rel, "reason": "binary"})
                    continue
                content = data.decode("utf-8", errors="replace")
            except OSError:
                skipped.append({"path": rel, "reason": "read_failed"})
                continue

            block = header + content + footer

            if len(block) <= remaining:
                chunks.append(block)
                embedded.append(rel)
                remaining -= len(block)
                continue

            # Not enough budget for full block: embed the first file partially, otherwise skip.
            min_overhead = len(header) + len(footer) + 50
            if not embedded and remaining > min_overhead:
                usable = max(remaining - len(header) - len(footer), 0)
                partial = content[:usable]
                chunks.append(header + partial + "\n[NOTE: File content truncated due to budget.]\n" + footer)
                embedded.append(rel)
                remaining = 0
                remaining_count = len(files) - idx - 1
                if remaining_count > 0:
                    skipped.append(
                        {
                            "path": rel,
                            "reason": "budget_exhausted",
                            "note": f"{remaining_count} additional files also skipped due to budget",
                        }
                    )
            else:
                remaining_count = len(files) - idx - 1
                skipped.append(
                    {
                        "path": rel,
                        "reason": "budget_exhausted",
                        "note": f"{remaining_count} additional files also skipped due to budget",
                    }
                )
            break

        formatted = "".join(chunks)
        return FileContext(
            formatted=formatted,
            embedded_files=tuple(embedded),
            skipped_files=tuple(skipped),
        )
