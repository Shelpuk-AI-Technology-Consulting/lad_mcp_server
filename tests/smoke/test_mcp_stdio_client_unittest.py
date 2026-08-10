"""The handshake helper fails usefully, and never leaves a process behind.

The helper is test infrastructure, but the startup suite's diagnosability rests on
it: if it masks the child's stderr or hangs instead of failing, a real breakage
becomes an opaque CI timeout. Its failure paths are therefore tested against
synthetic children rather than against Lad.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest
from mcp_stdio_client import McpHandshakeError, handshake


def _python(*source: str) -> list[str]:
    """Build an argv running an inline Python program.

    Args:
        *source: Lines of the program.

    Returns:
        An argv list.
    """
    return [sys.executable, "-c", "\n".join(source)]


def test_child_that_exits_immediately_reports_its_stderr() -> None:
    """A child that dies before reading stdin surfaces its own error, not a pipe error."""
    argv = _python(
        "import sys",
        "sys.stderr.write('CONFIG EXPLOSION\\n')",
        "sys.exit(3)",
    )

    with pytest.raises(McpHandshakeError) as excinfo:
        handshake(argv, env=dict(os.environ), timeout=30.0)

    assert "CONFIG EXPLOSION" in str(excinfo.value)


def test_child_that_never_answers_times_out_rather_than_hanging() -> None:
    """A silent child fails on the deadline instead of blocking forever.

    `readline()` blocks indefinitely, so an earlier version checked the timeout only
    *between* lines and would have hung here until the CI job was killed.
    """
    argv = _python(
        "import sys, time",
        "sys.stdin.read()",
        "time.sleep(120)",
    )

    with pytest.raises(McpHandshakeError) as excinfo:
        handshake(argv, env=dict(os.environ), timeout=3.0)

    assert "timed out after 3.0s" in str(excinfo.value)


def test_child_closing_stdout_reports_the_exit_code_not_a_timeout() -> None:
    """Closing stdout without answering gives the intended message, not TimeoutExpired."""
    argv = _python(
        "import sys",
        "sys.stdout.close()",
        "sys.stderr.write('BYE\\n')",
        "sys.exit(7)",
    )

    with pytest.raises(McpHandshakeError) as excinfo:
        handshake(argv, env=dict(os.environ), timeout=30.0)

    message = str(excinfo.value)
    assert "TimeoutExpired" not in message
    assert "BYE" in message or "rc=7" in message


def _process_alive(pid: int) -> bool:
    """Report whether a process id is still running.

    Args:
        pid: The process id to probe.

    Returns:
        ``True`` when the process still exists.
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    # A filtered query, because a full `tasklist /v` scan costs tens of seconds.
    listing = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    return str(pid) in listing


def test_failed_handshake_leaves_no_running_child_or_threads(tmp_path) -> None:
    """Teardown reaps the child and joins both reader threads even on failure.

    `Popen.__del__` does not kill, so an assertion failing mid-handshake used to
    leave the server, both reader threads and three pipes alive for the rest of the
    pytest process. The child here deliberately ignores stdin EOF, so this exercises
    the `kill()` fallback rather than the graceful path.
    """
    pid_file = tmp_path / "child.pid"
    argv = _python(
        "import os, sys, time",
        f"open(r'{pid_file}', 'w').write(str(os.getpid()))",
        "time.sleep(120)",
    )
    threads_before = threading.active_count()

    with pytest.raises(McpHandshakeError):
        handshake(argv, env=dict(os.environ), timeout=3.0)

    assert threading.active_count() == threads_before, "reader threads outlived the handshake"
    assert pid_file.exists(), "child never started; the test would be vacuous"
    assert not _process_alive(int(pid_file.read_text())), "child process outlived the handshake"


def test_failed_handshake_reaps_a_wrapper_grandchild(tmp_path) -> None:
    """Teardown reaps the real server behind a wrapper, not just the wrapper.

    `uvx`, `uv run` and `docker run` are all wrappers: the process we spawn is not
    the server. `proc.kill()` reaps only the wrapper, so a server that ignores stdin
    EOF survives holding its pipes open — the exact leak the direct-child test above
    cannot see. Verified failing before the process-tree kill existed.
    """
    pid_file = tmp_path / "grandchild.pid"
    grandchild = (
        "import os,sys,time;"
        f"open(r'{pid_file}','w').write(str(os.getpid()));"
        "sys.stdout.flush();"
        "time.sleep(120)"
    )
    wrapper = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{grandchild!r}])"

    with pytest.raises(McpHandshakeError):
        handshake(_python(wrapper), env=dict(os.environ), timeout=4.0)

    # The grandchild writes its pid at startup; give the spawn a moment to land.
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)

    assert pid_file.exists(), "grandchild never started; the test would be vacuous"
    assert not _process_alive(int(pid_file.read_text())), "grandchild outlived the handshake"
