#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2026 Russ Fellows <russ.fellows@gmail.com>
"""Regression test for issue #192: SIGSEGV at interpreter exit after async calls.

Root cause: pyo3-async-runtimes delivers an awaitable's result from a tokio
blocking-pool thread (``Python::attach`` -> ``loop.call_soon_threadsafe``).
``call_soon_threadsafe`` wakes the event loop's self-pipe and releases the GIL;
the main thread can then return from ``asyncio.run`` and start finalizing the
interpreter while the delivery thread is still inside its ``Python::attach``
scope.  When that thread tries to retake the GIL, CPython force-exits it
(``PyThread_exit_thread``) and the unwind frees Python objects without a thread
state -> SIGSEGV.  Timing dependent: roughly 0-25 percent of runs.

The fix registers an ``atexit`` hook that first closes a delivery gate (a late
delivery is dropped unrun, never entering Python) and then, with the GIL released, waits
(bounded) for in-flight result deliveries to finish before finalization.

Each scenario runs in a FRESH subprocess, K times, sequentially; every run must
exit with code 0.  Processes are only ever stopped with SIGTERM (never SIGKILL).

Run against an installed wheel:
    uv run --no-project --with <wheel> --with pytest pytest tests/test_async_exit_no_segfault.py
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytest.importorskip("s3dlio")

# Number of fresh-process repetitions per scenario.
RUNS_PER_SCENARIO = 30
# Wall-clock budget for one subprocess before it is asked (SIGTERM) to stop.
RUN_TIMEOUT_SECS = 60
# Grace period after SIGTERM before the test gives up waiting for the process.
TERMINATE_GRACE_SECS = 15

# Each scenario is a script that ends right after asyncio.run(): the process
# exits while a result delivery may still be finishing on a tokio thread.
SCENARIOS = {
    # Plain get_async, success, file:// store: no network.
    "get_async_file_ok": """
        import asyncio, sys, s3dlio
        async def main():
            data = await s3dlio.get_async(sys.argv[1])
            print("RESULT ok", len(bytes(data)), flush=True)
        asyncio.run(main())
        print("END-OF-SCRIPT", flush=True)
    """,
    # Plain get_async, connection refused (issue #192 "S9"): error delivery path.
    "get_async_connection_refused": """
        import asyncio, s3dlio
        async def main():
            try:
                await s3dlio.get_async("s3://127.0.0.1:1/s3dlio-fix192/obj")
            except BaseException as exc:
                print("RESULT exc", type(exc).__name__, flush=True)
        asyncio.run(main())
        print("END-OF-SCRIPT", flush=True)
    """,
    # MultiEndpointStore.get against an unreachable endpoint (issue #192 repro).
    "multi_endpoint_connection_refused": """
        import asyncio, s3dlio
        async def main():
            store = s3dlio.create_multi_endpoint_store(
                ["s3://127.0.0.1:1/s3dlio-fix192/"], "round_robin")
            try:
                await store.get("s3://127.0.0.1:1/s3dlio-fix192/obj")
            except BaseException as exc:
                print("RESULT exc", type(exc).__name__, flush=True)
        asyncio.run(main())
        print("END-OF-SCRIPT", flush=True)
    """,
}


def _run_once(script, argv, env):
    """Run one fresh interpreter; return (exit_code, stdout, stderr, seconds).

    The child is stopped with SIGTERM only if it exceeds the time budget.
    """
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", script, *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=RUN_TIMEOUT_SECS)
    except subprocess.TimeoutExpired:
        proc.terminate()
        out, err = proc.communicate(timeout=TERMINATE_GRACE_SECS)
        pytest.fail(f"subprocess exceeded {RUN_TIMEOUT_SECS}s and was terminated")
    return proc.returncode, out, err, time.monotonic() - started


@pytest.fixture(scope="module")
def child_env():
    env = dict(os.environ)
    # Offline, deterministic: dummy credentials, one attempt, short connect timeout.
    env.update(
        AWS_ACCESS_KEY_ID="test",
        AWS_SECRET_ACCESS_KEY="test",
        AWS_REGION="us-east-1",
        S3DLIO_MAX_RETRY_ATTEMPTS="1",
        S3DLIO_CONNECT_TIMEOUT_SECS="2",
    )
    return env


@pytest.fixture(scope="module")
def payload_uri(tmp_path_factory):
    path = tmp_path_factory.mktemp("fix192") / "obj.bin"
    path.write_bytes(b"s3dlio-issue-192" * 64)
    return f"file://{path}"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_fresh_process_exits_cleanly_after_async_call(name, child_env, payload_uri):
    script = textwrap.dedent(SCENARIOS[name])
    failures = []
    for i in range(RUNS_PER_SCENARIO):
        code, out, err, _ = _run_once(script, [payload_uri], child_env)
        if code != 0 or "END-OF-SCRIPT" not in out:
            failures.append((i, code, err.strip()[-200:]))
    assert not failures, (
        f"{name}: {len(failures)}/{RUNS_PER_SCENARIO} runs did not exit 0 "
        f"(negative = killed by that signal, -11 = SIGSEGV); first: {failures[:3]}"
    )


def test_exit_is_not_delayed_when_nothing_is_in_flight(child_env):
    """The atexit drain must be a no-op fast path for a script with no async work."""
    script = "import s3dlio\nprint('END-OF-SCRIPT', flush=True)\n"
    # Warm bytecode and page cache once, then take the best of several runs.
    _run_once(script, [], child_env)
    best = min(_run_once(script, [], child_env)[3] for _ in range(5))
    assert best < 5.0, f"idle import+exit took {best:.2f}s"


def test_pending_never_completing_task_does_not_hang_exit(child_env):
    """An awaitable that never completes must not be waited on at exit.

    A local TCP listener accepts connections and never answers.  The script
    starts get_async against it, lets the loop spin briefly and then exits
    without awaiting the result: exit must stay prompt (bounded), not hang.
    """
    script = textwrap.dedent(
        """
        import asyncio, socket, s3dlio
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port = srv.getsockname()[1]
        async def main():
            pending = s3dlio.get_async(f"s3://127.0.0.1:{port}/b/k")
            await asyncio.sleep(0.3)  # the request is in flight; never awaited
        loop = asyncio.new_event_loop()
        loop.run_until_complete(main())
        print("END-OF-SCRIPT", flush=True)
        """
    )
    code, out, err, secs = _run_once(script, [], child_env)
    assert "END-OF-SCRIPT" in out, err
    assert code == 0, f"exit code {code}: {err.strip()[-300:]}"
    # The drain timeout is 2 s; a never-completing task is not 'in flight', so
    # exit must come well inside that bound plus interpreter start-up.
    assert secs < 10.0, f"exit took {secs:.2f}s"


def test_late_delivery_after_exit_drain_never_enters_python(child_env):
    """A result that arrives after the exit drain must not be delivered.

    An ``atexit`` handler registered BEFORE ``import s3dlio`` runs AFTER
    s3dlio's drain (LIFO), so it sleeps with the GIL released once the delivery
    gate has closed.  The abandoned ``get_async`` gets its HTTP 404 during that
    sleep.  With the gate closed the Rust side must drop the delivery without
    calling ``loop.call_soon_threadsafe`` (i.e. without entering Python), and
    the process must still exit 0.
    """
    script = textwrap.dedent(
        """
        import atexit, socket, threading, time
        calls = [0]
        def late():
            time.sleep(1.2)
            print("LATE-DELIVERY-CALLS", calls[0], flush=True)
        atexit.register(late)

        import asyncio, s3dlio
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port = srv.getsockname()[1]
        def serve():
            conn, _ = srv.accept()
            conn.recv(65536)
            time.sleep(0.6)
            conn.sendall(b"HTTP/1.1 404 Not Found\\r\\nContent-Length: 0\\r\\n"
                         b"Connection: close\\r\\n\\r\\n")
            time.sleep(2)
        threading.Thread(target=serve, daemon=True).start()

        loop = asyncio.new_event_loop()
        orig = loop.call_soon_threadsafe
        def counting(*a, **k):
            calls[0] += 1
            return orig(*a, **k)
        loop.call_soon_threadsafe = counting

        async def main():
            pending = s3dlio.get_async(f"s3://127.0.0.1:{port}/b/k")
            await asyncio.sleep(0.1)  # request in flight; never awaited
        loop.run_until_complete(main())
        """
    )
    for _ in range(3):
        code, out, err, _ = _run_once(script, [], child_env)
        assert code == 0, f"exit code {code}: {err.strip()[-300:]}"
        assert "LATE-DELIVERY-CALLS 0" in out, (out, err.strip()[-300:])
