"""A minimal MCP stdio client, for driving the server as a real subprocess.

Deliberately does **not** use the ``mcp`` client SDK. The point of these tests is to
exercise the wire format an actual client speaks; using the same library the server
is built on would let a framing bug pass unnoticed on both sides.

The framing is newline-delimited JSON-RPC, as implemented by ``mcp/server/stdio.py``
and unchanged across the SDK's 1.x/2.x major versions.

Three hazards this helper exists to avoid, all of them reachable:

* **Deadlock.** The SDK's stdout writer awaits every write on a zero-capacity stream,
  so a client that stops reading stalls the server. The measured ``initialize`` +
  ``tools/list`` responses total roughly 3.5 KB against a Windows pipe buffer of about
  4 KB — one more tool would cross it. So the whole request batch is written once, up
  front, and responses are read incrementally.
* **Orphaned processes.** ``Popen.__del__`` does not kill. An assertion failing
  mid-handshake would leave a server running, so teardown always kills and waits.
* **Silent death.** A child that exits before the write turns ``stdin.write`` into
  ``BrokenPipeError`` (POSIX) or ``OSError`` (Windows), hiding the real cause. Those
  are caught and re-raised with the child's stderr attached.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence


class McpHandshakeError(RuntimeError):
    """Raised when the server fails to complete an MCP handshake."""


def _spawn_kwargs() -> dict[str, Any]:
    """Platform options that make the child reapable as a whole tree.

    On POSIX the child gets its own session, so the process *group* can be signalled
    — otherwise killing a wrapper such as ``uvx`` leaves the real server running.
    Windows needs no spawn-time flag; ``taskkill /T`` walks the tree instead.

    Returns:
        Keyword arguments for :class:`subprocess.Popen`.
    """
    return {} if os.name == "nt" else {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process and every descendant it spawned.

    ``proc.kill()`` reaps only the process we spawned. For every wrapper invocation
    — ``uvx``, ``uv run``, ``docker run`` — that process is *not* the server, so a
    server which ignores stdin EOF survives with its pipes held open, and the reader
    threads never see EOF. Verified: a grandchild outlived teardown until this
    existed.

    Args:
        proc: The spawned process.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=30,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        # Fall back to the single-process kill rather than leaving it running.
        with contextlib.suppress(OSError):
            proc.kill()


@dataclass
class HandshakeResult:
    """Outcome of a stdio handshake.

    Attributes:
        messages: Every JSON-RPC message the server wrote to stdout, in order.
        stdout: Raw stdout text, for asserting nothing non-JSON was emitted.
        stderr: Raw stderr text, for diagnosis.
        initialize_seconds: Wall-clock time from spawn to the initialize response.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    initialize_seconds: float = 0.0

    def result_for(self, request_id: int) -> dict[str, Any]:
        """Return the ``result`` payload for a request id.

        Args:
            request_id: The JSON-RPC id to look up.

        Returns:
            The ``result`` object.

        Raises:
            McpHandshakeError: If no successful response carries that id.
        """
        for msg in self.messages:
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise McpHandshakeError(f"request {request_id} failed: {msg['error']}")
                return msg.get("result", {})
        raise McpHandshakeError(f"no response for request id {request_id}; stderr:\n{self.stderr}")


def _frame(payload: dict[str, Any]) -> str:
    """Encode one JSON-RPC message as a newline-delimited frame.

    Args:
        payload: The JSON-RPC message.

    Returns:
        A single line of JSON ending in a newline.
    """
    return json.dumps(payload) + "\n"


def build_requests(
    *,
    protocol_version: str = "2025-06-18",
    follow_up: dict[str, Any] | None = None,
) -> str:
    """Build the standard opening exchange a client sends.

    Args:
        protocol_version: The version to offer in ``initialize``.
        follow_up: An optional request sent after ``notifications/initialized``.
            Defaults to ``tools/list`` with id 2.

    Returns:
        The concatenated frames, ready to write to the server's stdin.
    """
    if follow_up is None:
        follow_up = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    return "".join([
        _frame({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "lad-smoke-test", "version": "0"},
            },
        }),
        _frame({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        _frame(follow_up),
    ])


def handshake(
    argv: Sequence[str],
    *,
    env: dict[str, str],
    cwd: str | None = None,
    timeout: float = 60.0,
    expected_ids: Sequence[int] = (1, 2),
    requests: str | None = None,
) -> HandshakeResult:
    """Start a server and run one MCP exchange over stdio.

    Args:
        argv: The command to spawn.
        env: The complete environment for the child. Passed as-is, so callers can
            test the narrow environments some MCP clients forward.
        cwd: Working directory for the child.
        timeout: Seconds to wait for all expected responses. Defaults to 60 —
            the tightest documented client budget (Codex `startup_timeout_sec`),
            so a server that misses it is broken for a real client, not merely
            slow on a loaded machine.
        expected_ids: JSON-RPC ids to wait for before returning.
        requests: Frames to send. Defaults to :func:`build_requests`.

    Returns:
        A :class:`HandshakeResult`.

    Raises:
        McpHandshakeError: On spawn failure, timeout, or premature exit.
    """
    payload = build_requests() if requests is None else requests
    started = time.monotonic()

    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_spawn_kwargs(),
        )
    except OSError as exc:
        raise McpHandshakeError(f"could not start {argv[0]!r}: {exc}") from exc

    # Drained continuously on its own thread: a cold `uvx` writes build progress here,
    # and a full stderr pipe would stall the child just as a full stdout one would.
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        """Accumulate the child's stderr until the stream closes."""
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # Stdout is read on its own thread and handed over through a queue, so the
    # deadline below is a real one. A plain `readline()` blocks indefinitely and would
    # only check the timeout *between* lines, so a server that writes nothing would
    # hang the test rather than fail it. Started before the write, so the `finally`
    # block can always join it — and so nothing is missed if the child answers fast.
    stdout_queue: queue.Queue[str | None] = queue.Queue()

    def _read_stdout() -> None:
        """Feed stdout lines to the queue, then signal EOF with ``None``."""
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_queue.put(line)
        stdout_queue.put(None)

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stdout_thread.start()

    result = HandshakeResult()
    stdout_lines: list[str] = []
    try:
        try:
            assert proc.stdin is not None
            proc.stdin.write(payload)
            proc.stdin.flush()
        except OSError as exc:  # BrokenPipeError is an OSError
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
            stderr_thread.join(timeout=5)
            raise McpHandshakeError(
                f"server exited before accepting input (rc={proc.returncode}): {exc}\n"
                f"stderr:\n{''.join(stderr_chunks)}"
            ) from exc

        seen: set[int] = set()
        while not seen.issuperset(expected_ids):
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise McpHandshakeError(
                    f"timed out after {timeout}s waiting for ids {sorted(set(expected_ids) - seen)}\n"
                    f"stdout so far:\n{''.join(stdout_lines)}\nstderr:\n{''.join(stderr_chunks)}"
                )
            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                stderr_thread.join(timeout=5)
                raise McpHandshakeError(
                    f"server closed stdout before responding (rc={proc.returncode})\n"
                    f"stdout:\n{''.join(stdout_lines)}\nstderr:\n{''.join(stderr_chunks)}"
                )
            stdout_lines.append(line)
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # Kept, not raised on: stdout purity is asserted by its own test, and
                # a clear assertion there beats an opaque parse error here.
                continue
            result.messages.append(message)
            if isinstance(message.get("id"), int):
                seen.add(message["id"])
                if message["id"] == 1:
                    result.initialize_seconds = time.monotonic() - started
    finally:
        # Close stdin first. An MCP stdio server shuts down on EOF, and for wrapper
        # invocations (`uvx`, `uv run`, `docker run`) that is the *only* thing that
        # reaches the real server: `proc.kill()` kills the wrapper, leaving the
        # server, both reader threads and all three pipes alive for the rest of the
        # pytest process. Verified: EOF brings the whole chain down in ~0.3s.
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        # 5s is ample: a healthy chain exits on EOF in well under a second (measured
        # ~0.3s through `uvx`). A longer grace would be paid in full by every child
        # that ignores EOF, for no benefit.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        # Only reached if it ignored EOF. Kill the whole tree, not just the process
        # we spawned — for a wrapper invocation that process is not the server.
        _kill_process_tree(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    # Drain anything written after the last expected response, so a stray write
    # during shutdown is still visible to the stdout-purity assertion.
    while True:
        try:
            trailing = stdout_queue.get_nowait()
        except queue.Empty:
            break
        if trailing is None:
            break
        stdout_lines.append(trailing)

    result.stdout = "".join(stdout_lines)
    result.stderr = "".join(stderr_chunks)
    return result
