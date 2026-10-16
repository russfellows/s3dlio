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


async def finished_with(t, exc_type, seconds=30):
    """Wait for `t` with asyncio.wait - which cannot raise TimeoutError by itself, so a hang is
    never mistaken for the expected TimeoutError - then check HOW it ended."""
    done, _ = await asyncio.wait({t}, timeout=seconds)
    assert t in done, f"did not complete within {seconds}s (hang)"
    if exc_type is asyncio.CancelledError:
        assert t.cancelled(), f"expected cancellation, got {t!r}"
    else:
        assert not t.cancelled(), f"expected {exc_type.__name__}, got a cancellation"
        assert isinstance(t.exception(), exc_type), (
            f"expected {exc_type.__name__}, got {t.exception()!r}"
        )


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
        await finished_with(task, asyncio.CancelledError)
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


BRANCHES = {}


def record_branch(name, terminal):
    BRANCHES.setdefault(name, []).append(terminal)


def test_cancel_synchronously_right_after_the_call(work_dir):
    """The early window: cancel() before the awaitable ever ran a step. Whichever way the race
    with the native writer goes, a terminal awaitable means no native reader is left. The branch
    taken is RECORDED per iteration (run with -s to see the counts)."""
    iterations = 5
    for i in range(iterations):
        d = os.path.join(work_dir, f"it{i}")
        os.mkdir(d)
        original = bytes(range(250)) * 4000
        fifo, ba = stuck_put(d, original)
        seen = {}

        async def go(fifo=fifo, ba=ba, seen=seen):
            task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
            task.cancel()  # synchronously: no loop iteration has happened
            seen["done_immediately"] = task.done()
            await asyncio.sleep(0.5)
            seen["terminal"] = task.done()
            if seen["terminal"]:
                # Nothing may still be reading the buffer: a reader must see NO bytes.
                seen["late_bytes"] = nonblocking_drain(fifo, 0.5)
                await finished_with(task, asyncio.CancelledError)
            else:
                seen["pinned"] = not resizable(ba)
                reader = Reader(fifo)
                reader.start()
                await finished_with(task, asyncio.CancelledError)
                seen["data"] = reader.join()
            seen["released"] = resizable(ba)

        run_bounded(go, fifo)
        record_branch("sync-cancel", seen["terminal"])
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
    print(f"\nsync-cancel branches (terminal early?): {BRANCHES['sync-cancel']}")


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
        await finished_with(t, TimeoutError)
    else:
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()
        await finished_with(t, TimeoutError)
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


def _timeout_zero_iterations(name, work_dir, make_put_body):
    iterations = 4
    for i in range(iterations):
        d = os.path.join(work_dir, f"it{i}")
        os.mkdir(d)
        original = b"timeout-zero" * 20_000
        fifo, ba = stuck_put(d, original)
        seen = {}

        async def go(fifo=fifo, ba=ba, original=original, seen=seen):
            async def body():
                await make_put_body(fifo, ba)

            await _expect_deferred_timeout(body, fifo, ba, original, seen)

        run_bounded(go, fifo)
        record_branch(name, seen["terminal_early"])
        _check_deferred(seen, original)
    print(f"\n{name} branches (terminal early?): {BRANCHES[name]}")


def test_asyncio_timeout_zero_ends_only_when_no_native_reader_remains(work_dir):
    async def put(fifo, ba):
        async with asyncio.timeout(0):
            await s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)

    _timeout_zero_iterations("timeout(0)", work_dir, put)


def test_wait_for_zero_ends_only_when_no_native_reader_remains(work_dir):
    async def put(fifo, ba):
        await asyncio.wait_for(
            s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True), 0
        )

    _timeout_zero_iterations("wait_for(0)", work_dir, put)


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
        await finished_with(t, TimeoutError)
        seen["data"] = reader.join()
        seen["released"] = resizable(ba)

    run_bounded(go, fifo)
    assert seen["terminal_early"] is False, "timeout ended under a running native write"
    assert seen["pinned"] is True
    assert seen["data"] == original
    assert seen["released"] is True


def test_wait_for_after_the_native_write_started_waits_for_it(work_dir):
    """Deterministic: the write IS running (stuck opening the FIFO) when wait_for's timeout
    fires; the TimeoutError may only arrive after the write finished. Bounded by elapsed time."""
    original = b"started-then-wait-for" * 10_000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        async def body():
            return await asyncio.wait_for(
                s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True), 0.5
            )

        t = asyncio.ensure_future(body())
        await asyncio.sleep(1.5)  # wait_for (0.5 s) has fired; the write is still stuck
        seen["terminal_early"] = t.done()
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        started = time.monotonic()
        reader.start()
        await finished_with(t, TimeoutError)
        seen["elapsed"] = time.monotonic() - started
        seen["data"] = reader.join()
        seen["released"] = resizable(ba)

    run_bounded(go, fifo)
    assert seen["terminal_early"] is False, (
        "wait_for ended under a running native write"
    )
    assert seen["pinned"] is True
    assert seen["data"] == original
    assert seen["released"] is True
    assert seen["elapsed"] < 20, "completion took far longer than the write"


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
        await finished_with(task, asyncio.CancelledError)
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
    done, _ = await asyncio.wait({task}, timeout=20)
    out["done"] = task in done
    try:
        task.result()
    except BaseException as e:
        out["exc"] = type(e).__name__
        out["is_subclass"] = isinstance(e, s3dlio.ExportStillReferencedError)
        out["msg"] = str(e)
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
    assert out["done"] is True, out
    assert out["exc"] == "ExportStillReferencedError", out
    assert out["is_subclass"] is True
    assert "1s" in out["msg"], "the message must name the timeout value: " + out["msg"]
    assert "the put was cancelled" in out["msg"], "and what the put did: " + out["msg"]
    assert "do not modify" in out["msg"]
    assert out["pinned"] is True, "the buffer must stay pinned after a quiesce timeout"
    assert out["timeouts"] >= 1


# ---------------------------------------------------------------------------------------------
# GuardedFuture semantics with controllable stand-ins (the class is whatever a borrowed put returns)
# ---------------------------------------------------------------------------------------------


def guarded_future_class(work_dir):
    async def go():
        fut = s3dlio.put_bytes_async(
            f"file://{work_dir}/g.bin", bytearray(b"x" * 1000), borrow=True
        )
        await asyncio.wait_for(fut, 30)
        return type(fut)

    return asyncio.run(go())


def fake_error(kind, text, detail):
    return s3dlio.ExportStillReferencedError(f"{kind}|{text}|{detail}")


def test_cancel_before_the_put_ends_is_deferred_until_the_barrier(work_dir):
    cls = guarded_future_class(work_dir)

    async def go():
        loop = asyncio.get_running_loop()
        inner = loop.create_future()
        barrier = loop.create_future()
        fut = cls(inner, lambda: False, lambda: barrier, fake_error)
        assert fut.cancel() is True
        assert inner.cancelled(), "the put was not asked to cancel"
        await asyncio.sleep(0.05)
        assert not fut.done(), "terminal before the barrier"
        barrier.set_result(False)
        done, _ = await asyncio.wait({fut}, timeout=10)
        assert fut in done and fut.cancelled()

    run_bounded(go)


def test_cancel_after_the_put_ended_returns_false_so_a_task_still_cancels(work_dir):
    """inner already done, barrier still pending: cancel() must return False, so a Task awaiting
    the future records the cancellation itself and raises CancelledError once the future
    completes - instead of the outcome (a result) silently swallowing the cancellation."""
    cls = guarded_future_class(work_dir)
    seen = {}

    async def go():
        loop = asyncio.get_running_loop()
        inner = loop.create_future()
        barrier = loop.create_future()
        fut = cls(inner, lambda: False, lambda: barrier, fake_error)
        inner.set_result("uploaded")
        await asyncio.sleep(0.05)  # _inner_done ran; waiting for the barrier
        seen["cancel_returned"] = fut.cancel()

        async def waiter():
            return await fut

        task = asyncio.ensure_future(waiter())
        await asyncio.sleep(0.05)
        task.cancel()
        barrier.set_result(False)
        await finished_with(task, asyncio.CancelledError)
        seen["future_result"] = fut.result()

    run_bounded(go)
    assert seen["cancel_returned"] is False
    assert seen["future_result"] == "uploaded", "the put's own outcome must be kept"


def test_barrier_failures_become_distinct_errors_with_the_outcome(work_dir):
    cls = guarded_future_class(work_dir)

    async def run_case(put_result, barrier_result, ready=False):
        loop = asyncio.get_running_loop()
        inner = loop.create_future()
        barrier = loop.create_future()
        fut = cls(inner, lambda: ready, lambda: barrier, fake_error)
        if isinstance(put_result, BaseException):
            inner.set_exception(put_result)
        else:
            inner.set_result(put_result)
        await asyncio.sleep(0.02)
        if isinstance(barrier_result, BaseException):
            barrier.set_exception(barrier_result)
        else:
            barrier.set_result(barrier_result)
        done, _ = await asyncio.wait({fut}, timeout=10)
        assert fut in done
        return fut

    async def go():
        timeout_ok = await run_case("r", True)
        e = timeout_ok.exception()
        assert isinstance(e, s3dlio.ExportStillReferencedError)
        assert str(e) == "timeout|the put itself succeeded|"

        timeout_failed = await run_case(OSError("disk full"), True)
        e = timeout_failed.exception()
        assert isinstance(e, s3dlio.ExportStillReferencedError)
        assert "the put itself failed" in str(e) and "disk full" in str(e)
        assert isinstance(e.__cause__, OSError)

        internal = await run_case("r", RuntimeError("quiesce broke"))
        e = internal.exception()
        assert isinstance(e, s3dlio.ExportStillReferencedError)
        assert str(e).startswith("internal|the put itself succeeded|")
        assert "quiesce broke" in str(e)

        ok = await run_case("r", False)
        assert ok.result() == "r"
        fast = await run_case("r", False, ready=True)
        assert fast.result() == "r"

    run_bounded(go)


# ---------------------------------------------------------------------------------------------
# R2: the autonomous releaser's attach gate (Python 3.12 shutdown race). Deterministic: a test-only
# hook parks the releaser thread BETWEEN entering the attach gate and Python::try_attach, then the
# interpreter exits. The atexit callback must wait (GIL detached) for that attach scope to finish
# BEFORE it returns, i.e. before CPython finalization can start.
# ---------------------------------------------------------------------------------------------

PARKED_RELEASER_EXIT = r"""
import atexit, json, sys, time

def report():
    # Registered BEFORE `import s3dlio`, so (atexit is LIFO) it runs AFTER s3dlio's own exit
    # callback has returned: the stats are the state at the moment finalization could begin.
    import s3dlio
    print("STATS " + json.dumps(s3dlio._put_export_stats()), flush=True)

atexit.register(report)
import s3dlio

ba = bytearray(b"r" * 4096)
# an export nobody drains + the releaser parks 1.5 s between its gate entry and try_attach
s3dlio._put_test_queue_export(ba, 1500)
deadline = time.monotonic() + 30
while s3dlio._put_export_stats()["test_releaser_parked"] < 1:
    assert time.monotonic() < deadline, "the releaser never reached the park point"
    time.sleep(0.005)
print("PARKED", flush=True)
# fall off the end: the interpreter exits while the releaser is inside its attach scope
"""


def test_exit_waits_for_a_releaser_parked_before_try_attach(work_dir):
    import json

    started = time.monotonic()
    r = subprocess.run(
        [sys.executable, "-c", PARKED_RELEASER_EXIT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    elapsed = time.monotonic() - started
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}"
    assert "PARKED" in r.stdout, r.stdout
    stats_lines = [l for l in r.stdout.splitlines() if l.startswith("STATS ")]
    assert stats_lines, f"no stats line:\n{r.stdout}\n{r.stderr[-2000:]}"
    stats = json.loads(stats_lines[-1][len("STATS ") :])
    assert stats["shutdown_attach_waits"] == 1, (
        f"atexit did not wait for the releaser's attach scope: {stats}"
    )
    assert stats["releaser_attaches"] == 1, (
        f"atexit returned before the releaser finished attaching: {stats}"
    )
    assert stats["shutdown_attach_wait_timeouts"] == 0, stats
    assert elapsed >= 1.4, f"exit did not wait for the parked releaser ({elapsed:.2f}s)"
