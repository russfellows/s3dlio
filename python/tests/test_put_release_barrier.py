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


def run_bounded(make_coro, fifo=None, timeout=60):
    """Run `make_coro()` on its own event loop in a daemon thread and FAIL (not hang) when it
    does not finish within `timeout` seconds. `fifo`: a stuck native writer is unblocked by
    attaching a reader so the process can end."""
    box = {}

    def target():
        try:
            box["value"] = asyncio.run(make_coro())
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        if fifo is not None:
            threading.Thread(target=read_fifo, args=(fifo,), daemon=True).start()
            t.join(10)
        pytest.fail(f"timed out after {timeout}s: the awaitable never completed (hang)")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class Reader:
    """Attaches a reader to a FIFO (so a stuck native writer can finish) and keeps what it read."""

    def __init__(self, path):
        self.data = None
        self.thread = threading.Thread(target=self._run, args=(path,), daemon=True)

    def _run(self, path):
        self.data = read_fifo(path)

    def start(self):
        self.thread.start()

    def join(self, timeout=30):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the FIFO reader never finished"
        return self.data


def nonblocking_drain(path, seconds):
    """Open the FIFO for reading WITHOUT blocking and return whatever arrives within `seconds`
    (a native writer stuck opening it completes the moment a reader exists)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    got = bytearray()
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 1 << 20)
            except BlockingIOError:
                chunk = None
            if chunk:
                got += chunk
            else:
                time.sleep(0.01)
    finally:
        os.close(fd)
    return bytes(got)


def stuck_put(work_dir, original):
    fifo = os.path.join(work_dir, "pipe")
    os.mkfifo(fifo)
    return fifo, bytearray(original)


def test_cancel_after_the_native_write_started_is_not_observable_until_it_finished(
    work_dir,
):
    original = bytes(range(251)) * 4000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        await asyncio.sleep(0.3)  # the blocking writer is stuck opening the FIFO
        task.cancel()
        await asyncio.sleep(0.5)
        # The caller cannot observe completion or cancellation while the write is pending,
        # and the buffer is still pinned.
        seen["terminal_while_write_pending"] = task.done()
        seen["pinned_while_write_pending"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()  # lets the pending write run to completion
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
        seen["data"] = reader.join()
        seen["released_at_cancel"] = resizable(ba)

    run_bounded(go, fifo)
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


def test_cancel_synchronously_right_after_the_call(work_dir):
    """The early window: cancel() before the awaitable ever ran a step. The invariant, whichever
    way the race with the native writer goes: a terminal awaitable means no native reader is left."""
    original = bytes(range(250)) * 4000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        task.cancel()  # synchronously: no loop iteration has happened
        seen["done_immediately"] = task.done()
        await asyncio.sleep(0.5)
        seen["terminal"] = task.done()
        if seen["terminal"]:
            # Nothing may still be reading the buffer: a reader must see NO bytes.
            seen["late_bytes"] = nonblocking_drain(fifo, 0.5)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 30)
        else:
            seen["pinned"] = not resizable(ba)
            reader = Reader(fifo)
            reader.start()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 30)
            seen["data"] = reader.join()
        seen["released"] = resizable(ba)

    run_bounded(go, fifo)
    assert seen["done_immediately"] is False, (
        "cancel() made the awaitable terminal at once"
    )
    if seen["terminal"]:
        assert seen.get("late_bytes") == b"", (
            "the awaitable was terminal while a native write still read the buffer"
        )
    else:
        assert seen["pinned"] is True
        assert seen["data"] == original
    assert seen["released"] is True


async def _expect_deferred_timeout(make_body, fifo, ba, original, seen):
    """A timeout cancels the put at the very start, so the native write may or may not have begun
    when it fires. The invariant, either way: the timeout is raised only once no native reader
    of the buffer remains."""
    t = asyncio.ensure_future(make_body())
    await asyncio.sleep(0.6)
    seen["terminal_early"] = t.done()
    if seen["terminal_early"]:
        # nothing may still be reading the buffer: a reader must see no bytes
        seen["late_bytes"] = nonblocking_drain(fifo, 0.5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(t, 30)
    else:
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(t, 30)
        seen["data"] = reader.join()
    seen["released"] = resizable(ba)


def _check_deferred(seen, original):
    if seen["terminal_early"]:
        assert seen.get("late_bytes") == b"", (
            "the timeout fired while a native write still read the buffer"
        )
    else:
        assert seen["pinned"] is True
        assert seen["data"] == original
    assert seen["released"] is True


def test_asyncio_timeout_zero_ends_only_when_no_native_reader_remains(work_dir):
    original = b"timeout-zero" * 20_000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        async def body():
            async with asyncio.timeout(0):
                await s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)

        await _expect_deferred_timeout(body, fifo, ba, original, seen)

    run_bounded(go, fifo)
    _check_deferred(seen, original)


def test_wait_for_zero_ends_only_when_no_native_reader_remains(work_dir):
    original = b"wait-for-zero" * 20_000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        async def body():
            return await asyncio.wait_for(
                s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True), 0
            )

        await _expect_deferred_timeout(body, fifo, ba, original, seen)

    run_bounded(go, fifo)
    _check_deferred(seen, original)


def test_timeout_after_the_native_write_started_waits_for_it(work_dir):
    """Deterministic variant: the write IS running (stuck opening the FIFO) when the timeout fires."""
    original = b"started-then-timeout" * 10_000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        async def body():
            async with asyncio.timeout(0.5):
                await s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)

        t = asyncio.ensure_future(body())
        await asyncio.sleep(
            1.5
        )  # the timeout (0.5 s) has fired; the write is still stuck
        seen["terminal_early"] = t.done()
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(t, 30)
        seen["data"] = reader.join()
        seen["released"] = resizable(ba)

    run_bounded(go, fifo)
    assert seen["terminal_early"] is False, "timeout ended under a running native write"
    assert seen["pinned"] is True
    assert seen["data"] == original
    assert seen["released"] is True


def test_a_second_cancel_while_waiting_does_not_cut_the_barrier_short(work_dir):
    fifo, ba = stuck_put(work_dir, b"x" * 100_000)
    seen = {}

    async def go():
        task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.sleep(0.2)
        seen["done"] = task.done()
        reader = Reader(fifo)
        reader.start()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
        reader.join()
        seen["released"] = resizable(ba)

    run_bounded(go, fifo)
    assert seen["done"] is False
    assert seen["released"] is True


def test_a_put_that_fails_to_start_leaves_no_spurious_buffer_error(work_dir):
    """put_bytes_async outside an event loop fails AFTER the buffer was exported: the error path
    must release it before raising (the caller may resize at once)."""
    ba = bytearray(b"e" * 100_000)
    with pytest.raises(RuntimeError):
        s3dlio.put_bytes_async(f"file://{work_dir}/x.bin", ba, borrow=True)
    assert resizable(ba), "BufferError right after a failed put_bytes_async"


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


QUIESCE_TIMEOUT = r"""
import asyncio, os, sys, threading
import s3dlio
d = sys.argv[1]
fifo = os.path.join(d, "pipe")
os.mkfifo(fifo)
ba = bytearray(b"q" * 100_000)
out = {}
async def go():
    task = s3dlio.put_bytes_async("file://" + fifo, ba, borrow=True)
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), 20)
    except BaseException as e:
        out["exc"] = type(e).__name__
        out["is_subclass"] = isinstance(e, s3dlio.ExportStillReferencedError)
    try:
        ba.append(0)
        out["pinned"] = False
    except BufferError:
        out["pinned"] = True
    out["timeouts"] = s3dlio._put_export_stats()["quiesce_timeouts"]
asyncio.run(go())
# let the stuck writer finish so the process can end
t = threading.Thread(target=lambda: open(fifo, "rb").read(), daemon=True)
t.start(); t.join(10)
print(out)
"""


def test_quiesce_timeout_raises_a_distinct_error_and_keeps_the_buffer_pinned(work_dir):
    env = dict(os.environ, S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS="1")
    r = subprocess.run(
        [sys.executable, "-c", QUIESCE_TIMEOUT, work_dir],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    out = eval(r.stdout.strip().splitlines()[-1])
    assert out["exc"] == "ExportStillReferencedError", out
    assert out["is_subclass"] is True
    assert out["pinned"] is True, "the buffer must stay pinned after a quiesce timeout"
    assert out["timeouts"] >= 1
