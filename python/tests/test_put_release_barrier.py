"""Completion barrier of put_bytes() / put_bytes_async(): a put does not complete (normally, with an
error, OR by cancellation) until every native reader of the caller's buffer is done and the buffer
export has really been released (Blue review P1, "a cancelled borrowed put can still read the
caller's memory").

Run under every supported interpreter (3.11, 3.12, 3.13), each with its own wheel:
    S3DLIO_TEST_DIR=<real fs> uv run --python 3.12 --no-project --isolated --with <cp312 wheel> \
        --with pytest --with pytest-timeout --with numpy \
        python -m pytest --timeout=120 python/tests/test_put_release_barrier.py
    (same with --python 3.11 / 3.13 and the matching wheel)

Interpreter differences (#278, #288). Ownership and eventual completion are IDENTICAL on every
interpreter: nothing is terminal while a native reader remains or before the export is released,
and the guard future itself completes after the release. What differs is asyncio's or CPython's
behaviour AROUND the guard, not s3dlio's:
  * `asyncio.wait_for` after the put itself finished but while its barrier is pending: the guard
    refuses the cancellation (the outcome is fixed). 3.12+ (wait_for is built on asyncio.timeout)
    then raises TimeoutError once the barrier completes; 3.11 returns the put's result. Each is
    asserted exactly, per version.
  * #288, a 3.11 LIMITATION no library change can remove: 3.11's `wait_for` cancels the inner
    awaitable and then waits for it with `_cancel_and_wait`, which a further cancellation
    interrupts by design. A task inside `wait_for` that is cancelled a SECOND time, or cancelled
    once AFTER the deadline fired, therefore ENDS (CancelledError) while the guard future is
    still pending and the buffer still exported. 3.12+ keeps the task pending until the guard
    ends. `asyncio.timeout()` is safe on every version (3.11 included). The tests pin both
    behaviours, and the 3.11 tests then show the safe pattern: await the guard future itself
    (not the task) and touch the buffer only after it ended.
  * A Python-level `__buffer__` / `__release_buffer__` exporter (PEP 688) exists from 3.12 only;
    the finalizer-based exporter tests cover the same release schedule on every version.
"""

import asyncio
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import warnings

import pytest
import s3dlio

MIB = 1024 * 1024


@pytest.fixture
def work_dir():
    base = os.environ.get("S3DLIO_TEST_DIR") or None
    with tempfile.TemporaryDirectory(dir=base) as d:
        yield d


@pytest.fixture(autouse=True)
def no_leftover_retention():
    """A test that armed `_put_test_retain_next_clone()` and then failed before a put consumed
    the arming must not make the NEXT test's put retain a clone forever (#278): that put would
    never complete, because its barrier is never satisfied.

    Newer builds make `_put_test_release_retained()` disarm a pending arming as well; older
    builds (including the 0.9.128 and 0.10.0 wheels) only drop retained clones. The teardown
    therefore first CONSUMES a pending arming with a throw-away export (the arming is taken by
    the next payload that holds a buffer export) and only then releases whatever was retained,
    which is correct on both. Without a pending arming the throw-away export is simply released
    at once."""
    yield
    s3dlio._put_test_queue_export(bytearray(1))
    s3dlio._put_test_release_retained()


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
    never mistaken for the expected TimeoutError - then check HOW it ended. `exc_type=None`
    means it must have finished normally."""
    done, _ = await asyncio.wait({t}, timeout=seconds)
    assert t in done, f"did not complete within {seconds}s (hang)"
    if exc_type is None:
        assert not t.cancelled(), f"expected a normal finish, got a cancellation: {t!r}"
        assert t.exception() is None, f"expected a normal finish, got {t.exception()!r}"
    elif exc_type is asyncio.CancelledError:
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
        # Safe to reuse at once: the writer has copied everything it will ever read out of ba.
        ba[:4] = b"ZZZZ"
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

        outcome = run_bounded(go)
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

    assert run_bounded(go) is True, "async error raised with the export still held"


def test_success_releases_the_export_before_resolving(work_dir):
    ba = bytearray(b"s" * 300_000)

    async def go():
        await s3dlio.put_bytes_async(f"file://{work_dir}/ok.bin", ba, borrow=True)
        return resizable(ba)

    assert run_bounded(go) is True
    s3dlio.put_bytes(f"file://{work_dir}/ok2.bin", ba)
    assert resizable(ba)


def test_no_quiesce_timeouts_on_the_normal_paths(work_dir):
    before = s3dlio._put_export_stats()["quiesce_timeouts"]
    ba = bytearray(b"q" * 100_000)
    s3dlio.put_bytes(f"file://{work_dir}/q.bin", ba)

    async def go():
        await s3dlio.put_bytes_async(f"file://{work_dir}/q2.bin", ba, borrow=True)

    run_bounded(go)
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


# ---------------------------------------------------------------------------------------------
# R1 / R3 / P3: a borrowed put's awaitable is NEVER terminal (result, exception, cancelled) while a
# native reader remains or before the export has really been released. At the quiesce deadline it
# warns, counts, and keeps waiting. Deterministic: the hidden hook _put_test_retain_next_clone()
# keeps a native clone of the next borrowed payload (like a backend that keeps reading after the
# put ended) until _put_test_release_retained().
# ---------------------------------------------------------------------------------------------

DEADLINE = "S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS"


async def awaiting(fut):
    return await fut


def borrowed_put_with_retained_clone(path, ba):
    """Start a borrowed put to a regular file whose native clone is retained past the put's end."""
    s3dlio._put_test_retain_next_clone()
    fut = s3dlio.put_bytes_async(f"file://{path}", ba, borrow=True)
    assert s3dlio._put_test_retained() >= 1
    return fut


async def put_is_over(fut, seconds=30):
    """Wait until the put ITSELF has finished while its export barrier is still pending (the
    retained clone). The guard's bridged put future (`_inner`) is the one reliable signal: the
    file being fully written says nothing about the future being done. After this, a
    `cancel()` of the guard is refused (the outcome is fixed), deterministically."""
    assert type(fut).__name__ == "GuardedFuture", (
        f"a borrowed put must return the guard future, got {type(fut)!r}"
    )
    inner = fut._inner
    deadline = time.monotonic() + seconds
    while not inner.done():
        assert time.monotonic() < deadline, "the put itself never finished"
        await asyncio.sleep(0.005)
    assert not fut.done(), "the guard ended although a native clone is retained"


async def assert_pending(awaitables, seconds, what):
    done, _ = await asyncio.wait(set(awaitables), timeout=seconds)
    assert not done, f"{what}: terminal state observed while a native reader remained"


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setenv(DEADLINE, "0.4")
    yield
    s3dlio._put_test_release_retained()  # never leave a retained clone behind


def run_past_deadline_case(
    work_dir, wrapper, expected, cancel_again=True, deadline_clock=None
):
    """The inner put finishes at once; the retained clone outlives two quiesce deadlines; the
    awaiting task is cancelled (twice) / times out; nothing terminal may be seen until release.
    `deadline_clock`: a dict the wrapper fills with `started` (monotonic time) and `deadline`
    (seconds) when it starts a deadline of its own; the clone is then released only once that
    deadline has certainly fired, so the version-specific outcome is deterministic."""
    original = b"retained-clone" * 8000
    path = os.path.join(work_dir, "o.bin")
    ba = bytearray(original)
    seen = {}

    async def go():
        before = s3dlio._put_export_stats()["quiesce_timeouts"]
        fut = borrowed_put_with_retained_clone(path, ba)
        task = asyncio.ensure_future(wrapper(fut))
        await asyncio.sleep(0.25)  # the put itself is over; only the barrier is pending
        assert not fut.done()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await assert_pending([task, fut], 1.3, "after two deadlines")
            seen["warnings"] = [str(w.message) for w in caught]
        seen["timeouts"] = s3dlio._put_export_stats()["quiesce_timeouts"] - before
        seen["pinned"] = not resizable(ba)
        if cancel_again:
            task.cancel()  # a further cancellation on top of the first one
            await assert_pending([task, fut], 0.3, "after repeated cancellation")
        if deadline_clock is not None:
            assert "started" in deadline_clock, "the wrapper never started its deadline"
            fires_at = deadline_clock["started"] + deadline_clock["deadline"]
            wait_more = fires_at + 0.1 - time.monotonic()
            if wait_more > 0:
                await assert_pending([task, fut], wait_more, "up to the deadline")
            seen["deadline_fired_before_release"] = (
                time.monotonic() >= fires_at + 0.1 and not fut.done()
            )
        assert s3dlio._put_test_release_retained() >= 1
        await finished_with(task, expected)
        seen["fut_done"] = fut.done()
        seen["released"] = resizable(ba)
        # The buffer is safe to reuse at once: changing it now cannot reach the written object.
        ba[:4] = b"ZZZZ"

    run_bounded(go)
    assert seen["pinned"] is True
    assert seen["timeouts"] >= 2, f"each missed deadline must be counted: {seen}"
    assert any("400ms" in w and "borrowed" in w for w in seen["warnings"]), seen[
        "warnings"
    ]
    assert seen["fut_done"] is True
    if deadline_clock is not None:
        assert seen["deadline_fired_before_release"] is True, (
            "wait_for's deadline had not fired when the clone was released"
        )
    assert seen["released"] is True, "terminal before the export was released"
    with open(path, "rb") as f:
        assert f.read() == original


def test_task_cancel_after_finishing_is_not_terminal_past_the_deadline(
    work_dir, short_deadline
):
    async def cancelled_after_start(fut):
        task = asyncio.ensure_future(awaiting(fut))
        await asyncio.sleep(0.3)
        task.cancel()
        await task

    run_past_deadline_case(work_dir, cancelled_after_start, asyncio.CancelledError)


def test_asyncio_timeout_after_finishing_is_not_terminal_past_the_deadline(
    work_dir, short_deadline
):
    async def with_timeout(fut):
        async with asyncio.timeout(0.4):
            await fut

    # (no extra cancel: asyncio.timeout turns only ITS OWN cancellation into TimeoutError)
    run_past_deadline_case(work_dir, with_timeout, TimeoutError, cancel_again=False)


def test_wait_for_after_finishing_is_not_terminal_past_the_deadline(
    work_dir, short_deadline
):
    """`wait_for` expires after the put itself finished, so the guard refuses the cancellation
    (its outcome is fixed) and completes only after the export is released. What `wait_for`
    then reports is asyncio's own, version-specific contract (#278): 3.12+ (built on
    `asyncio.timeout`) raises TimeoutError; 3.11 returns the inner result when its
    cancellation was refused. On every version nothing is terminal before the release."""

    async def with_wait_for(fut):
        # The deadline must fire AFTER the put itself finished, or the cancellation is accepted
        # (inner not done yet) and 3.11 raises TimeoutError instead of returning the result.
        # So start the clock only once the inner put future is done.
        await put_is_over(fut)
        clock.update(started=time.monotonic(), deadline=0.4)
        await asyncio.wait_for(fut, 0.4)

    clock = {}
    expected = TimeoutError if sys.version_info >= (3, 12) else None
    run_past_deadline_case(
        work_dir, with_wait_for, expected, cancel_again=False, deadline_clock=clock
    )


# ---- #288: cancelling a task that is INSIDE wait_for ------------------------------------------
#
# On 3.11 `wait_for` cancels the inner awaitable and then waits for it with `_cancel_and_wait`,
# which a further cancellation deliberately interrupts: the task ends (CancelledError) while the
# guard future is still pending and the buffer is still exported. That is the standard library's
# behaviour; the guard cannot prevent it. 3.12+ (wait_for = asyncio.timeout + await) keeps the
# task pending until the guard ends. `asyncio.timeout()` is safe on EVERY version. The safe
# pattern on 3.11 is to await the guard future itself (never the task) before touching the
# buffer.

PY311 = sys.version_info < (3, 12)


def wait_for_30(fut):
    return asyncio.wait_for(fut, 30)


def wait_for_half_second(fut):
    return asyncio.wait_for(fut, 0.5)


async def timeout_30(fut):
    async with asyncio.timeout(30):
        await fut


async def timeout_half_second(fut):
    async with asyncio.timeout(0.5):
        await fut


async def wait_for_guard(fut):
    """Recovery after a task ended early (3.11 wait_for, #288): keep waiting for the guard
    future ITSELF until it is done, even if this waiting task is cancelled again. A cancellation
    received meanwhile is not lost: it is raised once the guard has ended."""
    cancelled = False
    while not fut.done():
        try:
            await asyncio.wait({fut})
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


def run_cancels_with_a_stuck_writer(work_dir, wrap, early, cancel_at=(0.3, 0.6)):
    """Stuck native writer; the awaiting task is cancelled at each time of `cancel_at` (seconds
    after it started). `early`: the task is expected to END while the guard is pending and the
    buffer exported (3.11 wait_for only); otherwise it must stay pending until the guard has
    ended."""
    original = b"stuck-writer-second-cancel" * 6000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        fut = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        task = asyncio.ensure_future(wrap(fut))
        started = time.monotonic()
        # The blocking writer is stuck opening the FIFO the whole time.
        for at in cancel_at:
            await asyncio.sleep(max(0.0, started + at - time.monotonic()))
            task.cancel()
        await asyncio.sleep(0.5)
        seen["task_done"] = task.done()
        seen["fut_done"] = fut.done()
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()  # lets the stuck write finish
        # The safe pattern: wait for the guard future itself, never for the task.
        await wait_for_guard(fut)
        await finished_with(fut, asyncio.CancelledError)
        seen["released"] = resizable(ba)
        await finished_with(task, asyncio.CancelledError)
        seen["data"] = reader.join()

    run_bounded(go, fifo)
    assert seen["fut_done"] is False and seen["pinned"] is True, seen
    if early:
        assert seen["task_done"] is True, (
            "3.11 wait_for was expected to end under the pending guard (#288); "
            "if this fails the interpreter's behaviour changed: update the docs"
        )
    else:
        assert seen["task_done"] is False, "the task ended under a running native write"
    assert seen["released"] is True, "the guard ended before the export was released"
    assert seen["data"] == original, "the write did not read the caller's data"


def test_wait_for_second_task_cancel_with_a_stuck_native_writer(work_dir):
    """#288. 3.12+: the task stays pending until the write finished and the export is released.
    3.11: the second Task.cancel() ends the task while the guard is pending and the buffer is
    still exported (documented limitation); awaiting the guard is the safe pattern."""
    run_cancels_with_a_stuck_writer(work_dir, wait_for_30, early=PY311)


def test_asyncio_timeout_second_task_cancel_with_a_stuck_native_writer(work_dir):
    """The recommended alternative: safe on every version, including 3.11."""
    run_cancels_with_a_stuck_writer(work_dir, timeout_30, early=False)


def test_wait_for_one_task_cancel_before_the_deadline_with_a_stuck_native_writer(
    work_dir,
):
    """ONE Task.cancel() before wait_for's deadline is safe on every version: wait_for cancels
    the put and waits for it, the task stays pending until the write finished and the export
    is released."""
    run_cancels_with_a_stuck_writer(
        work_dir, wait_for_30, early=False, cancel_at=(0.3,)
    )


def test_wait_for_task_cancel_after_the_deadline_fired_with_a_stuck_native_writer(
    work_dir,
):
    """#288 with a still-running writer: wait_for's 0.5 s deadline fires and cancels the put
    (accepted, the writer is stuck), then ONE Task.cancel() at 0.7 s. 3.12+: the task stays
    pending until the write finished and the export is released. 3.11: the cancellation
    interrupts wait_for's wait, the task ends under the running writer (documented limitation)."""
    run_cancels_with_a_stuck_writer(
        work_dir, wait_for_half_second, early=PY311, cancel_at=(0.7,)
    )


def test_asyncio_timeout_task_cancel_after_the_deadline_fired_with_a_stuck_native_writer(
    work_dir,
):
    """The recommended alternative, same schedule: safe on every version, including 3.11."""
    run_cancels_with_a_stuck_writer(
        work_dir, timeout_half_second, early=False, cancel_at=(0.7,)
    )


def test_recovery_wait_survives_a_further_cancel_until_the_guard_ends(work_dir):
    """The recovery helper for a task that ended early on 3.11 (#288): `wait_for_guard` keeps
    waiting for the guard even when the waiting task is cancelled again (a THIRD cancellation
    here, and a fourth), and raises that cancellation only after the guard ended."""
    original = b"recovery-third-cancel" * 6000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        fut = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        task = asyncio.ensure_future(wait_for_30(fut))
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.sleep(0.3)
        task.cancel()  # 3.11: the task ends here with the guard pending
        await asyncio.sleep(0.3)
        seen["task_done"] = task.done()
        recovery = asyncio.ensure_future(wait_for_guard(fut))
        await asyncio.sleep(0.2)
        recovery.cancel()  # a further cancellation of the recovery wait
        await asyncio.sleep(0.2)
        recovery.cancel()  # and another one
        await asyncio.sleep(0.3)
        seen["recovery_done"] = recovery.done()
        seen["fut_done"] = fut.done()
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()
        await finished_with(recovery, asyncio.CancelledError)
        seen["fut_done_at_recovery_end"] = fut.done()
        seen["released"] = resizable(ba)
        await finished_with(task, asyncio.CancelledError)
        seen["data"] = reader.join()

    run_bounded(go, fifo)
    if PY311:
        assert seen["task_done"] is True, (
            "3.11 wait_for was expected to end early (#288)"
        )
    assert seen["recovery_done"] is False, "the recovery wait ended before the guard"
    assert seen["fut_done"] is False and seen["pinned"] is True, seen
    assert seen["fut_done_at_recovery_end"] is True
    assert seen["released"] is True, "the recovery ended before the export was released"
    assert seen["data"] == original


def run_cancel_after_the_deadline_fired(work_dir, wrap, early):
    """The put finished at once, a retained native clone keeps its barrier pending; the 0.5 s
    deadline of `wrap` fires (the guard refuses the cancellation: the outcome is fixed), then the
    task is cancelled once at 0.7 s. The user's cancellation wins over the deadline, so the task
    ends with CancelledError - when the guard ends (3.12+ wait_for, asyncio.timeout on every
    version) or at once (3.11 wait_for, #288)."""
    original = b"cancel-after-the-deadline" * 8000
    path = os.path.join(work_dir, "o.bin")
    ba = bytearray(original)
    seen = {}

    async def go():
        fut = borrowed_put_with_retained_clone(path, ba)
        await put_is_over(fut)  # the deadline below fires after the put itself finished
        task = asyncio.ensure_future(wrap(fut))
        # the 0.5 s deadline has fired; only the barrier is pending
        await asyncio.sleep(0.7)
        assert not fut.done()
        task.cancel()
        await asyncio.sleep(0.5)
        seen["task_done"] = task.done()
        seen["fut_done"] = fut.done()
        seen["pinned"] = not resizable(ba)
        assert s3dlio._put_test_release_retained() >= 1
        # The safe pattern: wait for the guard future itself, never for the task. The put had
        # finished before any cancellation, so the guard keeps its outcome (a normal finish).
        await wait_for_guard(fut)
        await finished_with(fut, None)
        seen["released"] = resizable(ba)
        await finished_with(task, asyncio.CancelledError)

    run_bounded(go)
    assert seen["fut_done"] is False and seen["pinned"] is True, seen
    if early:
        assert seen["task_done"] is True, (
            "3.11 wait_for was expected to end under the pending guard (#288); "
            "if this fails the interpreter's behaviour changed: update the docs"
        )
    else:
        assert seen["task_done"] is False, "the task ended while the export was held"
    assert seen["released"] is True, "the guard ended before the export was released"
    # nothing wrote into the buffer: the object is the original
    with open(path, "rb") as f:
        assert f.read() == original


def test_wait_for_task_cancel_after_the_deadline_fired(work_dir):
    """#288. 3.12+: the task stays pending until the export is released, then ends
    CancelledError. 3.11: the cancellation after the deadline interrupts wait_for's wait for the
    cancelled inner awaitable, so the task ends while the guard is pending and the buffer is
    still exported (documented limitation)."""
    run_cancel_after_the_deadline_fired(work_dir, wait_for_half_second, early=PY311)


def test_asyncio_timeout_task_cancel_after_the_deadline_fired(work_dir):
    """The recommended alternative: safe on every version, including 3.11."""
    run_cancel_after_the_deadline_fired(work_dir, timeout_half_second, early=False)


def test_stuck_native_writer_past_the_deadline_warns_counts_and_keeps_waiting(
    work_dir, short_deadline
):
    """Real stuck writer (FIFO nobody reads) instead of the hook: cancel, outlive the deadline."""
    original = b"x" * 100_000
    fifo, ba = stuck_put(work_dir, original)
    seen = {}

    async def go():
        before = s3dlio._put_export_stats()["quiesce_timeouts"]
        task = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=True)
        await asyncio.sleep(0.3)
        task.cancel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await assert_pending([task], 1.3, "stuck writer past the deadline")
            seen["warnings"] = [str(w.message) for w in caught]
        seen["timeouts"] = s3dlio._put_export_stats()["quiesce_timeouts"] - before
        seen["pinned"] = not resizable(ba)
        reader = Reader(fifo)
        reader.start()
        await finished_with(task, asyncio.CancelledError)
        reader.join()
        seen["released"] = resizable(ba)

    run_bounded(go, fifo)
    assert seen["pinned"] is True
    assert seen["timeouts"] >= 1
    assert any("borrowed" in w for w in seen["warnings"]), seen["warnings"]
    assert seen["released"] is True


def test_blocking_put_keeps_the_distinct_error_when_a_clone_is_retained(
    work_dir, short_deadline
):
    """put_bytes raises ExportStillReferencedError directly in the caller's own stack (no Future
    or Task sits between, so cancellation machinery cannot rewrite it); the buffer stays pinned."""
    ba = bytearray(b"sync" * 20_000)
    s3dlio._put_test_retain_next_clone()
    with pytest.raises(s3dlio.ExportStillReferencedError) as info:
        s3dlio.put_bytes(f"file://{work_dir}/sync.bin", ba)
    assert "400ms" in str(info.value)
    assert not resizable(ba), "buffer must stay pinned after the error"
    s3dlio._put_test_release_retained()
    deadline = time.monotonic() + 10
    while not resizable(
        ba
    ):  # released by the autonomous releaser once the clone is gone
        assert time.monotonic() < deadline, "export never released after the clone died"
        time.sleep(0.01)


class SlowRelease:
    """A buffer exporter whose release callback blocks (without the GIL) until told to go on:
    it opens the schedule in which another drainer's private batch is still being released."""

    def __init__(self, data, entered, proceed):
        self.data = bytearray(data)
        self._entered = entered
        self._proceed = proceed

    def __buffer__(self, flags):
        return memoryview(self.data)

    def __release_buffer__(self, view):
        self._entered.set()
        self._proceed.wait(30)
        view.release()


needs_python_buffer_protocol = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="needs a Python-level buffer exporter (__buffer__ / __release_buffer__, PEP 688), "
    "which exists only from Python 3.12 (#278); the same schedule is covered on every version "
    "by the finalizer-based tests below",
)


@needs_python_buffer_protocol
def test_async_completion_waits_for_the_actual_release_of_a_batched_export(work_dir):
    """R3: exports A and B sit in one drainer's private batch; A's release callback yields the
    GIL. Neither awaitable may complete until its own export is really released."""
    entered, proceed = threading.Event(), threading.Event()
    a = SlowRelease(b"a" * 50_000, entered, proceed)
    b = bytearray(b"b" * 50_000)
    seen = {}

    async def go():
        loop = asyncio.get_running_loop()
        fa = borrowed_put_with_retained_clone(os.path.join(work_dir, "a.bin"), a)
        fb = borrowed_put_with_retained_clone(os.path.join(work_dir, "b.bin"), b)
        await asyncio.sleep(
            0.4
        )  # both writes are over; both barriers wait for the clones
        assert not fa.done() and not fb.done()
        assert (
            s3dlio._put_test_release_retained() == 2
        )  # queue [A, B], GIL held throughout
        assert await loop.run_in_executor(None, entered.wait, 15), (
            "no release callback ran"
        )
        # a drainer is now blocked inside A's release callback, B in its private batch
        await assert_pending([fa, fb], 0.7, "while A's release callback is blocked")
        seen["b_pinned"] = not resizable(b)
        proceed.set()
        done, _ = await asyncio.wait({fa, fb}, timeout=20)
        assert len(done) == 2, "never completed after the release callback finished"
        seen["b_released"] = resizable(b)
        seen["a_released"] = resizable(a.data)

    try:
        run_bounded(go)
    finally:
        proceed.set()
    assert seen["b_pinned"] is True
    assert seen["b_released"] is True and seen["a_released"] is True


@needs_python_buffer_protocol
def test_blocking_completion_waits_for_the_actual_release_of_a_batched_export(work_dir):
    """R3 for put_bytes: the second caller finds the public queue empty (another drainer owns the
    private batch) and must wait, GIL released, until its export is really released."""
    entered, proceed = threading.Event(), threading.Event()
    a = SlowRelease(b"a" * 50_000, entered, proceed)
    b = bytearray(b"b" * 50_000)
    finished = {}

    def call(name, buf):
        s3dlio.put_bytes(f"file://{work_dir}/{name}.bin", buf)
        finished[name] = time.monotonic()

    s3dlio._put_test_retain_next_clone()
    ta = threading.Thread(target=call, args=("a", a), daemon=True)
    ta.start()
    deadline = time.monotonic() + 15
    while s3dlio._put_test_retained() < 1:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    s3dlio._put_test_retain_next_clone()
    tb = threading.Thread(target=call, args=("b", b), daemon=True)
    tb.start()
    while s3dlio._put_test_retained() < 2:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    time.sleep(0.3)  # both calls now wait (GIL released) for their retained clone
    try:
        assert not finished
        assert (
            s3dlio._put_test_release_retained() == 2
        )  # queue [A, B], GIL held throughout
        assert entered.wait(15), "no release callback ran"
        time.sleep(0.7)
        assert finished == {}, (
            f"a call returned before its export was released: {finished}"
        )
        assert not resizable(b)
    finally:
        proceed.set()
    ta.join(20)
    tb.join(20)
    assert not ta.is_alive() and not tb.is_alive()
    assert resizable(b) and resizable(a.data)


class FinalizingBuffer(bytearray):
    """A buffer exporter whose FINALIZER blocks (without the GIL) until told to go on. It works
    on every supported interpreter (Python 3.11 has no Python-level `__release_buffer__`).

    The export of a native payload holds a reference to its exporter, and releasing the export
    drops that reference (`PyBuffer_Release` = the exporter's release callback, then the DECREF).
    When the export holds the LAST reference, the DECREF runs `__del__` inside the release, on the
    thread that drains the release queue - the same schedule as a release callback that yields
    the GIL: the drainer sits inside A's release while B's export waits in its private batch.
    The test must therefore drop every reference of its own before the export is released."""

    def __init__(self, data, entered, proceed):
        super().__init__(data)
        self._entered = entered
        self._proceed = proceed

    def __del__(self):
        self._entered.set()
        self._proceed.wait(30)


def test_async_completion_waits_for_a_batched_export_whose_release_yields_the_gil(
    work_dir,
):
    """R3 on every interpreter: exports A and B sit in one drainer's private batch; releasing A
    (its exporter's finalizer runs inside the release) yields the GIL. Neither awaitable may
    complete until its own export is really released."""
    entered, proceed = threading.Event(), threading.Event()
    b = bytearray(b"b" * 50_000)
    seen = {}

    async def go():
        loop = asyncio.get_running_loop()
        a = FinalizingBuffer(b"a" * 50_000, entered, proceed)
        fa = borrowed_put_with_retained_clone(os.path.join(work_dir, "a.bin"), a)
        del a  # the export now holds the only reference to A's exporter
        fb = borrowed_put_with_retained_clone(os.path.join(work_dir, "b.bin"), b)
        await asyncio.sleep(
            0.4
        )  # both writes are over; both barriers wait for the clones
        assert not fa.done() and not fb.done()
        assert (
            s3dlio._put_test_release_retained() == 2
        )  # queue [A, B], GIL held throughout
        assert await loop.run_in_executor(None, entered.wait, 15), (
            "A's exporter was never finalized inside its release"
        )
        # a drainer is now blocked inside A's release, B in its private batch
        await assert_pending([fa, fb], 0.7, "while A's release is blocked")
        seen["b_pinned"] = not resizable(b)
        proceed.set()
        done, _ = await asyncio.wait({fa, fb}, timeout=20)
        assert len(done) == 2, "never completed after the release finished"
        seen["b_released"] = resizable(b)
        for f in (fa, fb):
            assert f.exception() is None and not f.cancelled()

    try:
        run_bounded(go)
    finally:
        proceed.set()
    assert seen["b_pinned"] is True
    assert seen["b_released"] is True
    with open(os.path.join(work_dir, "a.bin"), "rb") as f:
        assert f.read() == b"a" * 50_000
    with open(os.path.join(work_dir, "b.bin"), "rb") as f:
        assert f.read() == b"b" * 50_000


def test_blocking_completion_waits_for_a_batched_export_whose_release_yields_the_gil(
    work_dir,
):
    """R3 for put_bytes on every interpreter. A blocking caller keeps a reference to its own
    exporter until the call returns, so a finalizer cannot run inside ITS release; export A is
    therefore an async borrowed put (no reference left in Python) and export B a blocking
    `put_bytes` from another thread. Queue [A, B]: the drainer blocks inside A's release, and
    the blocking caller of B (queue empty, another drainer owns the batch) must wait, GIL
    released, until B's export is really released."""
    entered, proceed = threading.Event(), threading.Event()
    b = bytearray(b"b" * 50_000)
    finished = {}
    seen = {}

    def call_b():
        s3dlio.put_bytes(f"file://{work_dir}/b.bin", b)
        finished["b"] = time.monotonic()

    async def go():
        a = FinalizingBuffer(b"a" * 50_000, entered, proceed)
        fa = borrowed_put_with_retained_clone(os.path.join(work_dir, "a.bin"), a)
        del a
        s3dlio._put_test_retain_next_clone()
        tb = threading.Thread(target=call_b, daemon=True)
        tb.start()
        deadline = time.monotonic() + 15
        while s3dlio._put_test_retained() < 2:
            assert time.monotonic() < deadline, "put_bytes never took its payload"
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.3)  # both are over; both wait for their retained clone
        assert not finished and not fa.done()
        assert (
            s3dlio._put_test_release_retained() == 2
        )  # queue [A, B], GIL held throughout
        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(None, entered.wait, 15), (
            "A's exporter was never finalized inside its release"
        )
        await asyncio.sleep(0.7)
        seen["finished_while_blocked"] = dict(finished)
        seen["b_pinned"] = not resizable(b)
        proceed.set()
        await finished_with(fa, None)
        await loop.run_in_executor(None, tb.join, 20)
        seen["b_alive"] = tb.is_alive()

    try:
        run_bounded(go)
    finally:
        proceed.set()
    assert seen["finished_while_blocked"] == {}, (
        f"put_bytes returned before its export was released: {seen['finished_while_blocked']}"
    )
    assert seen["b_pinned"] is True
    assert seen["b_alive"] is False and "b" in finished
    assert resizable(b)


# ---- GuardedFuture with stand-in barrier parts (internal failures, P3) ----------------------


def guarded_future_class(work_dir):
    async def go():
        fut = s3dlio.put_bytes_async(
            f"file://{work_dir}/g.bin", bytearray(b"x" * 1000), borrow=True
        )
        await asyncio.wait_for(fut, 30)
        return type(fut)

    # Bounded: a put that never completes fails this test instead of hanging the run (#278).
    return run_bounded(go)


def make_guard(cls, loop, ready, quiesce, block=None, poll=0.02):
    """A GuardedFuture over a controllable inner future. `ready` is a one-element list read by the
    stand-in latch_ready; `events` records the (kind, detail) pairs the guard reports."""
    events = []

    def event(kind, detail):
        events.append((kind, detail))
        return f"fake {kind}"

    inner = loop.create_future()
    fut = cls(inner, lambda: ready[0], quiesce, event, block or (lambda: None), poll)
    return inner, fut, events


async def completes_only_after_ready(inner, fut, ready, events, expect_kind):
    inner.set_result("uploaded")
    await asyncio.sleep(0.3)
    assert not fut.done(), "terminal although the barrier is not satisfied"
    assert any(k == expect_kind for k, _ in events), events
    ready[0] = True
    done, _ = await asyncio.wait({fut}, timeout=10)
    assert fut in done and fut.result() == "uploaded"


def test_cancel_before_the_put_ends_is_deferred_until_the_barrier(work_dir):
    cls = guarded_future_class(work_dir)

    async def go():
        loop = asyncio.get_running_loop()
        barrier = loop.create_future()
        ready = [False]
        inner, fut, _ = make_guard(cls, loop, ready, lambda: barrier)
        assert fut.cancel() is True
        assert inner.cancelled(), "the put was not asked to cancel"
        await asyncio.sleep(0.05)
        assert not fut.done(), "terminal before the barrier"
        ready[0] = True
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
        barrier = loop.create_future()
        ready = [False]
        inner, fut, _ = make_guard(cls, loop, ready, lambda: barrier)
        inner.set_result("uploaded")
        await asyncio.sleep(0.05)  # _inner_done ran; waiting for the barrier
        seen["cancel_returned"] = fut.cancel()

        task = asyncio.ensure_future(awaiting(fut))
        await asyncio.sleep(0.05)
        task.cancel()
        ready[0] = True
        barrier.set_result(False)
        await finished_with(task, asyncio.CancelledError)
        seen["future_result"] = fut.result()

    run_bounded(go)
    assert seen["cancel_returned"] is False
    assert seen["future_result"] == "uploaded", "the put's own outcome must be kept"


def test_add_done_callback_failing_never_completes_unsafely(work_dir):
    """P3: waiter.add_done_callback raises (MemoryError): the guard falls back to polling and
    completes only when the latch is really ready."""
    cls = guarded_future_class(work_dir)

    class BadWaiter:
        def add_done_callback(self, cb):
            raise MemoryError("callback registration")

    async def go():
        loop = asyncio.get_running_loop()
        ready = [False]
        inner, fut, events = make_guard(cls, loop, ready, lambda: BadWaiter())
        await completes_only_after_ready(inner, fut, ready, events, "internal")

    run_bounded(go)


def test_quiesce_raising_or_waiter_failing_never_completes_unsafely(work_dir):
    cls = guarded_future_class(work_dir)

    def raising():
        raise RuntimeError("quiesce broke")

    async def go():
        loop = asyncio.get_running_loop()
        ready = [False]
        inner, fut, events = make_guard(cls, loop, ready, raising)
        await completes_only_after_ready(inner, fut, ready, events, "internal")

        ready = [False]
        broken = loop.create_future()
        broken.set_exception(RuntimeError("waiter failed"))
        inner, fut, events = make_guard(cls, loop, ready, lambda: broken)
        await completes_only_after_ready(inner, fut, ready, events, "internal")

    run_bounded(go)


def test_latch_ready_raising_keeps_polling_without_completing(work_dir):
    cls = guarded_future_class(work_dir)

    async def go():
        loop = asyncio.get_running_loop()
        state = {"calls": 0}

        def latch_ready():
            state["calls"] += 1
            if state["calls"] < 6:
                raise RuntimeError("latch_ready broke")
            return True

        events = []
        inner = loop.create_future()
        fut = cls(
            inner,
            latch_ready,
            lambda: loop.create_future(),
            lambda kind, detail: events.append((kind, detail)) or "x",
            lambda: None,
            0.02,
        )
        inner.set_result("uploaded")
        done, _ = await asyncio.wait({fut}, timeout=10)
        assert fut in done and fut.result() == "uploaded"
        assert state["calls"] >= 6, "completed before latch_ready said ready"

    run_bounded(go)


def test_a_missed_deadline_warns_and_waits_again_instead_of_completing(work_dir):
    cls = guarded_future_class(work_dir)

    async def go():
        loop = asyncio.get_running_loop()
        ready = [False]
        calls = []

        def quiesce():
            f = loop.create_future()
            calls.append(f)
            loop.call_later(0.02, f.set_result, True)  # True: the wait TIMED OUT
            return f

        inner, fut, events = make_guard(cls, loop, ready, quiesce)
        inner.set_result("uploaded")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await asyncio.sleep(0.4)
        assert not fut.done(), "completed at a missed deadline"
        assert len(calls) >= 3, "did not wait again after a missed deadline"
        assert sum(1 for k, _ in events if k == "timeout") >= 3
        assert len(caught) >= 3, "one warning per missed deadline"
        ready[0] = True
        done, _ = await asyncio.wait({fut}, timeout=10)
        assert fut in done and fut.result() == "uploaded"

    run_bounded(go)


def test_failing_to_schedule_a_poll_falls_back_to_a_blocking_wait(work_dir):
    cls = guarded_future_class(work_dir)

    async def go():
        loop = asyncio.get_running_loop()
        ready = [False]
        blocked = []

        def block():
            blocked.append(True)
            ready[0] = (
                True  # the blocking wait returns only once the export is released
            )

        real_call_later = loop.call_later

        def no_poll_call_later(delay, callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "_poll":
                raise MemoryError("call_later")
            return real_call_later(delay, callback, *args, **kwargs)

        loop.call_later = no_poll_call_later  # instance attribute shadows the method
        try:
            inner, fut, _events = make_guard(
                cls,
                loop,
                ready,
                lambda: (_ for _ in ()).throw(RuntimeError("q")),
                block,
            )
            inner.set_result("uploaded")
            done, _ = await asyncio.wait({fut}, timeout=10)
            assert fut in done and fut.result() == "uploaded"
            assert blocked, "the blocking fallback was not used"
        finally:
            del loop.call_later

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


def test_unsupported_forced_completion_cleanup_stops_polling(work_dir):
    """UNSUPPORTED-USE CLEANUP, not a supported scenario. The guarantee covers the consumer
    operations (await, cancel, Task.cancel, timeout, wait_for); calling a producer method on
    the future (here the base-class ``asyncio.Future.set_result``) is outside it and ends the
    future at once, whether or not a native reader remains. This only checks that, after such
    a forced completion, the guard stops polling (and reporting failures) instead of looping
    forever."""
    cls = guarded_future_class(work_dir)

    async def go():
        loop = asyncio.get_running_loop()
        calls = []

        def latch_ready():
            calls.append(1)
            raise RuntimeError("never ready")

        inner = loop.create_future()
        fut = cls(
            inner,
            latch_ready,
            lambda: loop.create_future(),
            lambda k, d: "x",
            lambda: None,
            0.02,
        )
        inner.set_result("uploaded")
        await asyncio.sleep(0.2)
        assert calls, "the guard never polled"
        asyncio.Future.set_result(fut, "external")  # UNSUPPORTED: forces the base class
        await asyncio.sleep(0.1)  # let a poll that was already scheduled run
        settled = len(calls)
        await asyncio.sleep(0.3)
        assert len(calls) == settled, "kept polling after the future was finished"
        assert fut.result() == "external"

    run_bounded(go)
