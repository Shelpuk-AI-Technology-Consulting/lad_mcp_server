#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import urllib.request


def load_env_file(path: Path) -> None:
    """
    Minimal env file loader to support `LAD_ENV_FILE=test.env`.
    Loads KEY=VALUE lines into the process environment if not already set.
    """
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def main() -> int:
    env_file = os.getenv("LAD_ENV_FILE")
    if env_file:
        p = Path(env_file)
        if p.is_file():
            load_env_file(p)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.")
        return 2

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"FAILED: could not call OpenRouter Models API: {exc}")
        return 3

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("FAILED: response was not valid JSON")
        return 4

    models = data.get("data")
    if not isinstance(models, list) or not models:
        print("FAILED: unexpected models payload (missing/empty data list)")
        return 5

    print(f"OK: fetched {len(models)} models from OpenRouter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
