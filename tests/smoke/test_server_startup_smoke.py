"""The server actually starts and serves MCP, for every way a client launches it.

Everything else in ``tests/`` exercises Python objects in-process, which is why a
dependency break that stopped the server starting for *every* client went unnoticed:
nothing under ``tests/`` imports the MCP SDK, so ``pyproject.toml`` resolving to an
SDK that removed ``mcp.server.fastmcp`` left CI green and the product dead.

Runtime invariants pinned here, none of which live anywhere else in the repo:

* stdio is the only transport Lad speaks — ``__main__.py`` calls ``run()`` with no
  arguments and ignores ``sys.argv``;
* stdout carries JSON-RPC and nothing else;
* the tool set is exactly ``system_design_review`` and ``code_review``;
* startup succeeds for a gateway-only deployment, and fails closed — loudly — when no
  provider credential is configured at all;
* ``mcp`` must stay below 2.0.

Distinct from ``scripts/smoke_openrouter_serena.py``, which is an in-process run
against a live key that costs money. Nothing here reaches OpenRouter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from mcp_stdio_client import build_requests, handshake

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAKE_KEY = "sk-smoke-test-not-a-real-key"

# Clients that forward only named variables give the server a very narrow
# environment. On Windows, omitting SystemRoot fails in Winsock (WinError 10106)
# before any Lad code runs, so the keep-list is platform-specific.
_MINIMAL_ENV_KEYS = ("PATH",) if os.name != "nt" else (
    "PATH", "SystemRoot", "SystemDrive", "TEMP", "PATHEXT", "COMSPEC",
)


def _client_env(**overrides: str) -> dict[str, str]:
    """Build a full environment with a dummy key, as an MCP client would.

    Args:
        **overrides: Extra variables to set, or `""` to drop one.

    Returns:
        The environment mapping for the child process.
    """
    # Strip the developer's own Lad configuration: an inherited OPENROUTER_* or LAD_*
    # override could fail a startup case for a reason unrelated to the code.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENROUTER_", "LAD_", "INTERMITTENT_"))
    }
    # Put this checkout ahead of any editable install. Without it, `python -m
    # lad_mcp_server` resolves through site-packages to whatever path the install
    # points at — in a git worktree that is the *main* checkout, so the smoke suite
    # would start a different copy of the server than the one under test.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])]
    )
    env["OPENROUTER_API_KEY"] = _FAKE_KEY
    for key, value in overrides.items():
        if value == "":
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _docker_available() -> bool:
    """Report whether a usable Docker daemon is present.

    Probes the daemon rather than the binary: the CLI can be installed while the
    daemon is stopped, in which case a build would fail slowly instead of skipping.

    Returns:
        ``True`` when ``docker info`` succeeds.
    """
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=60
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _uvx_available() -> bool:
    """Report whether ``uvx`` is on PATH.

    Returns:
        ``True`` when the binary resolves.
    """
    return shutil.which("uvx") is not None


# ---------------------------------------------------------------------------
# Invocation shapes. `neutral_cwd` matters: run from inside the checkout, Python
# resolves `lad_mcp_server` from the source tree, so an in-tree case would pass with
# the package uninstalled and prove nothing about packaging.
# ---------------------------------------------------------------------------

def _find_console_script() -> str | None:
    """Locate the ``lad-mcp-server`` console script.

    Tries PATH first, because that is how a client invokes it. Falls back to the
    directory beside ``sys.executable``: a venv puts the script there without
    necessarily being on PATH, which is the common local layout. Neither alone is
    sufficient — PATH misses the venv case, and the adjacent directory misses a
    system Python or ``pip install --user`` layout.

    Returns:
        The resolved path, or ``None`` when the script is not installed.
    """
    found = shutil.which("lad-mcp-server")
    if found:
        return found
    suffix = ".exe" if os.name == "nt" else ""
    candidate = Path(sys.executable).parent / f"lad-mcp-server{suffix}"
    return str(candidate) if candidate.exists() else None


_CONSOLE_SCRIPT = _find_console_script()

_IN_TREE_SHAPES = [
    pytest.param([sys.executable, "-m", "lad_mcp_server"], 60.0, id="python-m"),
    pytest.param([_CONSOLE_SCRIPT or "lad-mcp-server"], 60.0, id="console-script"),
]


def _skip_if_missing_console_script(argv: list[str]) -> None:
    """Skip only the console-script row when that script is not installed.

    Gated on `argv`, not on `_CONSOLE_SCRIPT` alone: the `python -m` row does not
    need the entry point, and skipping it too would silently drop module-entrypoint
    coverage on any checkout that has dependencies but no installed script.

    Args:
        argv: The invocation being tested.
    """
    if _CONSOLE_SCRIPT is None and argv[0] == "lad-mcp-server":
        pytest.skip("console script 'lad-mcp-server' is not installed")


@pytest.mark.parametrize(("argv", "timeout"), _IN_TREE_SHAPES)
def test_in_tree_invocation_completes_handshake(argv: list[str], timeout: float, tmp_path: Path) -> None:
    """Each in-tree invocation initializes and advertises both tools."""
    _skip_if_missing_console_script(argv)

    result = handshake(argv, env=_client_env(), cwd=str(tmp_path), timeout=timeout)

    init = result.result_for(1)
    assert init["serverInfo"]["name"] == "lad-mcp-server"
    # `serverInfo.version` is deliberately not asserted: FastMCP reports the SDK's
    # version, not Lad's, so pinning it would break on every SDK upgrade.
    assert set(_tool_names(result)) == {"system_design_review", "code_review"}


def _tool_names(result) -> list[str]:
    """Extract advertised tool names from a handshake result.

    Args:
        result: A completed :class:`HandshakeResult`.

    Returns:
        The tool names in advertisement order.
    """
    return [tool["name"] for tool in result.result_for(2)["tools"]]


def test_advertised_tools_have_usable_schemas(tmp_path: Path) -> None:
    """Both tools carry a description and an object-typed input schema."""
    result = handshake([sys.executable, "-m", "lad_mcp_server"], env=_client_env(), cwd=str(tmp_path))

    tools = {tool["name"]: tool for tool in result.result_for(2)["tools"]}
    assert set(tools) == {"system_design_review", "code_review"}
    for name, tool in tools.items():
        assert tool.get("description", "").strip(), f"{name} has no description"
        assert tool["inputSchema"]["type"] == "object", f"{name} has a non-object input schema"


def test_stdout_carries_only_json_rpc(tmp_path: Path) -> None:
    """A stray write to stdout would corrupt the stream for every stdio client."""
    result = handshake([sys.executable, "-m", "lad_mcp_server"], env=_client_env(), cwd=str(tmp_path))

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Every line must be JSON-RPC — note the server may legitimately emit
        # `notifications/message`, which is still a valid message.
        message = json.loads(line)
        assert message.get("jsonrpc") == "2.0", f"non-JSON-RPC line on stdout: {line!r}"


def test_package_never_writes_to_stdout() -> None:
    """Static guard: the handshake only covers the lines it happens to trigger."""
    offenders = []
    for path in (_REPO_ROOT / "lad_mcp_server").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "print(" in stripped or "sys.stdout" in stripped:
                offenders.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{number}: {stripped}")
    assert not offenders, "stdio transport requires a silent stdout:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "protocol_version", ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
)
def test_supported_protocol_version_is_echoed(protocol_version: str, tmp_path: Path) -> None:
    """Each version a documented client may offer is accepted verbatim."""
    result = handshake(
        [sys.executable, "-m", "lad_mcp_server"],
        env=_client_env(),
        cwd=str(tmp_path),
        requests=build_requests(protocol_version=protocol_version),
    )

    assert result.result_for(1)["protocolVersion"] == protocol_version


def test_unsupported_protocol_version_is_downgraded_not_rejected(tmp_path: Path) -> None:
    """An unknown version negotiates down rather than failing the client."""
    result = handshake(
        [sys.executable, "-m", "lad_mcp_server"],
        env=_client_env(),
        cwd=str(tmp_path),
        requests=build_requests(protocol_version="1999-01-01"),
    )

    negotiated = result.result_for(1)["protocolVersion"]
    assert negotiated and negotiated != "1999-01-01"


def test_starts_with_a_minimal_client_environment(tmp_path: Path) -> None:
    """Clients that forward only named variables must still get a working server."""
    minimal = {key: os.environ[key] for key in _MINIMAL_ENV_KEYS if key in os.environ}
    minimal["OPENROUTER_API_KEY"] = _FAKE_KEY
    # See `_client_env`: without this the child imports the installed package rather
    # than this checkout.
    minimal["PYTHONPATH"] = str(_REPO_ROOT)

    result = handshake([sys.executable, "-m", "lad_mcp_server"], env=minimal, cwd=str(tmp_path))

    assert set(_tool_names(result)) == {"system_design_review", "code_review"}


def test_starts_for_a_gateway_only_deployment(tmp_path: Path) -> None:
    """The headline capability of issue #2: no OpenRouter account anywhere.

    Every other test in this file supplies `OPENROUTER_API_KEY`, so nothing would
    notice if the server started only when that variable happened to be set — which is
    precisely the environment the reporter cannot create.
    """
    env = _client_env(
        OPENROUTER_API_KEY="",
        OPENAI_COMPAT_BASE_URL="https://gateway.invalid/v1",
        OPENROUTER_PRIMARY_REVIEWER_MODEL="litellm/gateway-model",
        OPENROUTER_SECONDARY_REVIEWER_MODEL="0",
    )

    result = handshake([sys.executable, "-m", "lad_mcp_server"], env=env, cwd=str(tmp_path))

    assert set(_tool_names(result)) == {"system_design_review", "code_review"}


def test_tool_call_returns_a_structured_error_without_network(tmp_path: Path) -> None:
    """A validation failure round-trips as an MCP result, reaching no provider."""
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "code_review", "arguments": {}},
    }
    result = handshake(
        [sys.executable, "-m", "lad_mcp_server"],
        env=_client_env(),
        cwd=str(tmp_path),
        requests=build_requests(follow_up=call),
    )

    text = result.result_for(2)["content"][0]["text"]
    assert "Either code or paths must be provided" in text


def test_start_is_fast_enough_for_client_timeouts(tmp_path: Path) -> None:
    """Startup latency is the client-visible contract, so measure it.

    Codex defaults `startup_timeout_sec` to 60, Claude Code documents
    `MCP_TIMEOUT=120000`. The bound is half the tightest of those rather than
    something tighter: it still catches a start that regressed from ~1s to tens
    of seconds, without flaking when the machine is loaded.
    """
    result = handshake([sys.executable, "-m", "lad_mcp_server"], env=_client_env(), cwd=str(tmp_path))

    print(f"\ninitialize responded in {result.initialize_seconds:.2f}s")
    assert result.initialize_seconds < 30.0


def test_missing_api_key_fails_closed(tmp_path: Path) -> None:
    """Without the key the server exits non-zero and says which variable is missing.

    Run against a *copy* of the package in a temp tree: python-dotenv's
    `find_dotenv` walks up from `config.py`'s own directory rather than the cwd, so
    a developer with a repo-root `.env` would otherwise see this pass spuriously.
    """
    package_copy = tmp_path / "pkg"
    package_copy.mkdir()
    shutil.copytree(_REPO_ROOT / "lad_mcp_server", package_copy / "lad_mcp_server")

    env = {key: os.environ[key] for key in _MINIMAL_ENV_KEYS if key in os.environ}
    env["PYTHONPATH"] = str(package_copy)

    proc = subprocess.run(
        [sys.executable, "-m", "lad_mcp_server"],
        cwd=str(package_copy),
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode != 0, "server should refuse to start with no credentials"
    assert proc.stdout == "", "nothing may reach stdout before the protocol starts"
    # `OPENROUTER_API_KEY` is no longer unconditionally required — a deployment may
    # route every reviewer to a direct provider. The behaviour worth pinning is that
    # *no* credentials at all still fails closed, and says which ones would do.
    assert "No provider credentials configured" in proc.stderr
    assert "OPENROUTER_API_KEY" in proc.stderr
    assert "OPENAI_COMPAT_BASE_URL" in proc.stderr


@pytest.mark.slow
@pytest.mark.skipif(not _uvx_available(), reason="uvx is not installed")
def test_uvx_built_distribution_completes_handshake(tmp_path: Path) -> None:
    """The install path every documented client uses actually starts.

    This is the case that catches packaging faults, because it builds and installs a
    distribution rather than importing from the source tree. It is also the case
    that would have caught the `mcp>=2` break.
    """
    argv = ["uvx", "--from", str(_REPO_ROOT), "lad-mcp-server"]

    # Cold environment builds were measured at ~112s; warm runs at ~4s.
    result = handshake(argv, env=_client_env(), cwd=str(tmp_path), timeout=300.0)

    assert set(_tool_names(result)) == {"system_design_review", "code_review"}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_uv_run_resolves_dependencies_and_starts(tmp_path: Path) -> None:
    """`uv run` re-resolves from pyproject.toml, so it sees dependency breaks.

    The documented local-development path (README:200). Materially stronger than
    the already-installed console script: this resolves the dependency set afresh,
    which is exactly how the `mcp>=2` break reached users.
    """
    argv = ["uv", "run", "--project", str(_REPO_ROOT), "lad-mcp-server"]

    result = handshake(argv, env=_client_env(), cwd=str(tmp_path), timeout=300.0)

    assert set(_tool_names(result)) == {"system_design_review", "code_review"}


@pytest.mark.slow
@pytest.mark.skipif(not _uvx_available(), reason="uvx is not installed")
@pytest.mark.skipif(
    not os.environ.get("LAD_SMOKE_GIT_REF"),
    reason="LAD_SMOKE_GIT_REF not set (CI sets it to the pushed commit)",
)
def test_uvx_from_git_completes_handshake(tmp_path: Path) -> None:
    """The literal install command every documented client uses.

    Builds from a *clone*, not the working tree, so this is the only case that can
    catch a file that was never committed. Requires the commit to be pushed, so it
    runs on CI where `LAD_SMOKE_GIT_REF` names the pushed SHA.
    """
    ref = os.environ["LAD_SMOKE_GIT_REF"]
    repo = os.environ.get("LAD_SMOKE_GIT_REPO", "Shelpuk-AI-Technology-Consulting/lad_mcp_server")
    argv = ["uvx", "--from", f"git+https://github.com/{repo}@{ref}", "lad-mcp-server"]

    result = handshake(argv, env=_client_env(), cwd=str(tmp_path), timeout=600.0)

    assert set(_tool_names(result)) == {"system_design_review", "code_review"}


@pytest.fixture(scope="module")
def docker_image() -> str:
    """Build the Docker image once, under a tag that cannot clobber a local one.

    README:537 tells developers to build `-t lad-mcp-server`; reusing that tag would
    silently overwrite their image.

    Yields:
        The unique image tag.
    """
    if not _docker_available():
        pytest.skip("docker daemon is not available")

    tag = f"lad-mcp-server:smoke-{uuid.uuid4().hex[:8]}"
    build = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        pytest.fail(f"docker build failed:\n{build.stderr[-3000:]}")
    try:
        yield tag
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)


@pytest.mark.slow
def test_docker_image_completes_handshake(docker_image: str, tmp_path: Path) -> None:
    """The documented Docker deployment starts and serves MCP.

    Named explicitly and force-removed afterwards: a container is not a child
    process, so killing the `docker run` client does not stop it. Process-tree kill
    cannot reach across the daemon boundary — only the name can.
    """
    container = f"lad-smoke-{uuid.uuid4().hex[:8]}"
    argv = [
        "docker", "run", "-i", "--rm", "--name", container,
        "-e", f"OPENROUTER_API_KEY={_FAKE_KEY}",
        docker_image,
    ]

    try:
        result = handshake(argv, env=_client_env(), cwd=str(tmp_path), timeout=180.0)
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=120)

    assert set(_tool_names(result)) == {"system_design_review", "code_review"}
