"""Completion barrier of put_bytes() / put_bytes_async(): a put does not complete (normally, with an
error, OR by cancellation) until every native reader of the caller's buffer is done and the buffer
export has really been released (Blue review P1, "a cancelled borrowed put can still read the
caller's memory").

Run under both supported interpreters, each with its own wheel:
    S3DLIO_TEST_DIR=<real fs> uv run --python 3.12 --no-project --isolated --with <cp312 wheel> \
        --with pytest --with numpy python -m pytest python/tests/test_put_release_barrier.py
    (same with --python 3.13 and the cp313 wheel)
"""

import asyncio
import os
import random
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import s3dlio

MIB = 1024 * 1024


@pytest.fixture
def work_dir():
    base = os.environ.get("S3DLIO_TEST_DIR") or None
    with tempfile.TemporaryDirectory(dir=base) as d:
        yield d


def resizable(ba):
    """True when no export is held on the bytearray (extend/shrink succeed)."""
    try:
        ba.append(0)
        del ba[-1:]
        return True
    except BufferError:
        return False


def read_fifo(path):
    with open(path, "rb") as f:
        return f.read()


def test_cancel_after_the_native_write_started_is_not_observable_until_it_finished(
    work_dir,
):
    fifo = os.path.join(work_dir, "pipe")
    os.mkfifo(fifo)
    original = bytes(range(251)) * 4000
    ba = bytearray(original)
    seen = {}

    async def go():
        task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        await asyncio.sleep(0.3)  # the blocking writer is stuck opening the FIFO
        task.cancel()
        await asyncio.sleep(0.5)
        # The caller cannot observe completion or cancellation while the write is still pending,
        # and the buffer is still pinned.
        seen["terminal_while_write_pending"] = task.done()
        seen["pinned_while_write_pending"] = not resizable(ba)
        reader = threading.Thread(
            target=lambda: seen.__setitem__("data", read_fifo(fifo))
        )
        reader.start()  # lets the pending write run to completion
        with pytest.raises(asyncio.CancelledError):
            await task
        reader.join(timeout=30)
        seen["released_at_cancel"] = resizable(ba)

    asyncio.run(go())
    assert seen["terminal_while_write_pending"] is False, (
        "cancel completed under a running write"
    )
    assert seen["pinned_while_write_pending"] is True
    assert seen["data"] == original, (
        "the write read memory that was not the caller's data"
    )
    assert seen["released_at_cancel"] is True, (
        "CancelledError raised before the export was released"
    )


def test_a_second_cancel_while_waiting_does_not_cut_the_barrier_short(work_dir):
    fifo = os.path.join(work_dir, "pipe")
    os.mkfifo(fifo)
    ba = bytearray(b"x" * 100_000)
    seen = {}

    async def go():
        task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.sleep(0.2)
        seen["done"] = task.done()
        t = threading.Thread(target=lambda: read_fifo(fifo))
        t.start()
        with pytest.raises(asyncio.CancelledError):
            await task
        t.join(timeout=30)
        seen["released"] = resizable(ba)

    asyncio.run(go())
    assert seen["done"] is False
    assert seen["released"] is True


def test_cancellation_at_any_moment_leaves_the_buffer_free_and_the_file_whole(work_dir):
    rng = random.Random(7)
    data = bytes(range(256)) * 4096  # 1 MiB
    for i in range(40):
        ba = bytearray(data)
        path = os.path.join(work_dir, f"o{i}.bin")

        async def go(path=path, ba=ba):
            task = s3dlio.put_bytes_async(f"file://{path}", ba, borrow=True)
            await asyncio.sleep(rng.random() * 0.004)
            task.cancel()
            try:
                await task
                return "completed"
            except asyncio.CancelledError:
                return "cancelled"

        outcome = asyncio.run(go())
        assert resizable(ba), f"export still held after {outcome} (iteration {i})"
        ba[:] = b"Z" * len(ba)  # reuse the buffer right away: must not reach the file
        time.sleep(0.01)
        if os.path.exists(path) and outcome == "completed":
            with open(path, "rb") as f:
                assert f.read() == data, f"iteration {i}"


def test_error_releases_the_export_before_raising(work_dir):
    blocker = os.path.join(work_dir, "afile")
    open(blocker, "w").close()
    uri = f"file://{blocker}/child.bin"  # parent is a regular file: ENOTDIR
    ba = bytearray(b"e" * 200_000)
    with pytest.raises(Exception):  # noqa: B017 - any error type is fine; the export must be free
        s3dlio.put_bytes(uri, ba)
    assert resizable(ba), "sync error returned with the export still held"

    async def go():
        with pytest.raises(Exception):  # noqa: B017 - any error type is fine; the export must be free
            await s3dlio.put_bytes_async(uri, ba, borrow=True)
        return resizable(ba)

    assert asyncio.run(go()) is True, "async error raised with the export still held"


def test_success_releases_the_export_before_resolving(work_dir):
    ba = bytearray(b"s" * 300_000)

    async def go():
        await s3dlio.put_bytes_async(f"file://{work_dir}/ok.bin", ba, borrow=True)
        return resizable(ba)

    assert asyncio.run(go()) is True
    s3dlio.put_bytes(f"file://{work_dir}/ok2.bin", ba)
    assert resizable(ba)


def test_no_quiesce_timeouts_on_the_normal_paths(work_dir):
    before = s3dlio._put_export_stats()["quiesce_timeouts"]
    ba = bytearray(b"q" * 100_000)
    s3dlio.put_bytes(f"file://{work_dir}/q.bin", ba)

    async def go():
        await s3dlio.put_bytes_async(f"file://{work_dir}/q2.bin", ba, borrow=True)

    asyncio.run(go())
    assert s3dlio._put_export_stats()["quiesce_timeouts"] == before


# ---------------------------------------------------------------------------------------------
# Interpreter teardown and subinterpreter hosts (fresh subprocesses, whichever interpreter runs
# pytest: run the file under 3.12 and 3.13).
# ---------------------------------------------------------------------------------------------

ABANDONED = r"""
import asyncio, os, sys, time
import s3dlio
d = sys.argv[1]
fifo = os.path.join(d, "stuck")
os.mkfifo(fifo)
ba = bytearray(b"a" * 1_000_000)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
async def start():
    return s3dlio.put_bytes_async("file://" + fifo, ba, borrow=True)
task = loop.run_until_complete(start())
loop.run_until_complete(asyncio.sleep(0.2))
# exit with the borrowed put still running natively (its writer is stuck on the FIFO)
"""

STRESS = r"""
import asyncio, sys, threading
import s3dlio
d = sys.argv[1]
def sync_puts(i):
    ba = bytearray(b"t" * 200_000)
    for j in range(60):
        s3dlio.put_bytes(f"file://{d}/s{i}-{j}.bin", ba)
        ba.append(0); del ba[-1:]
ts = [threading.Thread(target=sync_puts, args=(i,)) for i in range(4)]
[t.start() for t in ts]
async def many():
    ba = bytearray(b"u" * 200_000)
    for j in range(60):
        await s3dlio.put_bytes_async(f"file://{d}/a{j}.bin", ba, borrow=True)
        ba.append(0); del ba[-1:]
asyncio.run(many())
[t.join() for t in ts]
"""

SUBINTERPRETER = r"""
import asyncio, sys, threading
try:
    import _interpreters as _i
except ImportError:
    try:
        import _xxsubinterpreters as _i
    except ImportError:
        print("NO_SUBINTERPRETERS"); sys.exit(0)
# a host that created (and destroyed) a subinterpreter BEFORE using s3dlio: from now on
# PyGILState_Check() answers "held" on threads that do not hold the GIL
x = _i.create(); _i.destroy(x)
import s3dlio
d = sys.argv[1]
def sync_puts(i):
    ba = bytearray(b"t" * 300_000)
    for j in range(40):
        s3dlio.put_bytes(f"file://{d}/s{i}-{j}.bin", ba)
        ba.append(0); del ba[-1:]
ts = [threading.Thread(target=sync_puts, args=(i,)) for i in range(4)]
[t.start() for t in ts]
async def many():
    ba = bytearray(b"u" * 300_000)
    for j in range(40):
        await s3dlio.put_bytes_async(f"file://{d}/a{j}.bin", ba, borrow=True)
        ba.append(0); del ba[-1:]
asyncio.run(many())
[t.join() for t in ts]
print("OK")
"""


def run_script(script, directory, timeout=120):
    return subprocess.run(
        [sys.executable, "-c", script, directory],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_exit_with_an_abandoned_borrowed_put_is_clean_and_repeatable(work_dir):
    for i in range(8):
        d = os.path.join(work_dir, f"run{i}")
        os.mkdir(d)
        r = run_script(ABANDONED, d, timeout=60)
        assert r.returncode == 0, f"exit {i}: rc={r.returncode}\n{r.stderr[-2000:]}"


def test_exit_after_heavy_put_traffic_is_clean_and_repeatable(work_dir):
    for i in range(8):
        d = os.path.join(work_dir, f"run{i}")
        os.mkdir(d)
        r = run_script(STRESS, d)
        assert r.returncode == 0, f"exit {i}: rc={r.returncode}\n{r.stderr[-2000:]}"


def test_host_that_created_a_subinterpreter_before_using_s3dlio(work_dir):
    d = os.path.join(work_dir, "sub")
    os.mkdir(d)
    r = run_script(SUBINTERPRETER, d)
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stderr[-2000:]}"
    if "NO_SUBINTERPRETERS" in r.stdout:
        pytest.skip("this interpreter has no _interpreters/_xxsubinterpreters module")
    assert "OK" in r.stdout
