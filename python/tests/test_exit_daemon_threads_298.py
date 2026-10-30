"""Issue #298: interpreter exit must not abort while daemon threads are inside s3dlio calls.

Every scenario runs in a fresh SUBPROCESS (a crash must not take pytest down) with a timeout, and
asserts: exit code 0, no "FATAL" on stderr, no hang, and that the scenario really was exercised
(every worker thread was inside, or had completed, an s3dlio call when the main thread returned,
and no worker raised).

Background: on CPython < 3.13.8 (3.11, 3.12, 3.13.0-3.13.7) a daemon thread that re-takes the GIL
after a blocking call, once finalization has begun, is ended with pthread_exit().  On glibc that is a
forced unwind; with PyO3 < 0.29 it crosses the Rust trampoline and the process dies with SIGABRT
(exit 134, sometimes "FATAL: exception not rethrown" on stderr).  PyO3 0.29.3 parks such threads
instead.  CPython 3.13.8+ and 3.14 park them themselves, so this file passes there for any PyO3.

Do NOT rely on the "FATAL" check alone: release builds can abort silently inside pthread_exit;
the exit code is the load-bearing assertion.
"""
import subprocess
import sys
import textwrap

import pytest

NWORKERS = 8

SCRIPT = textwrap.dedent('''
    import asyncio, atexit, os, socket, sys, threading, time
    import s3dlio

    mode, d = sys.argv[1], sys.argv[2]
    NW = int(sys.argv[3])
    p = os.path.join(d, "obj.bin")
    with open(p, "wb") as f:
        f.write(os.urandom(4 << 20))
    uri = "file://" + p

    # A listener that accepts and never answers: an s3:// call against it blocks (GIL released).
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(128)
    os.environ.update(AWS_ENDPOINT_URL="http://127.0.0.1:%d" % srv.getsockname()[1],
                      AWS_ACCESS_KEY_ID="x", AWS_SECRET_ACCESS_KEY="y", AWS_REGION="us-east-1")
    def accept_forever():
        held = []
        while True:
            held.append(srv.accept()[0])
    threading.Thread(target=accept_forever, daemon=True).start()

    lock = threading.Lock()
    started, completed, errors = set(), [0], []
    def tid():
        return threading.get_ident() % 100000

    def note_start():
        with lock:
            started.add(threading.get_ident())
    def note_done():
        with lock:
            completed[0] += 1

    # ---- scenario bodies: one call each -------------------------------------------------------
    def writer_once(i):
        w = s3dlio.create_filesystem_writer("file://%s/w%d_%d.bin" % (d, tid(), i % 3),
                                            s3dlio.PyWriterOptions())
        for _ in range(8):
            w.write_chunk(b"z" * (256 << 10))
        w.finalize()

    SYNC = {
        "get": lambda i: s3dlio.get(uri),
        "put_bytes": lambda i: s3dlio.put_bytes("file://%s/p%d_%d.bin" % (d, tid(), i % 4), b"x" * (1 << 20)),
        "list": lambda i: s3dlio.list("file://" + d + "/"),
        "stat": lambda i: s3dlio.stat(uri),
        "get_many": lambda i: s3dlio.get_many([uri] * 4),
        "get_range": lambda i: s3dlio.get_range(uri, 0, 1 << 20),
        "writer": writer_once,
        "sync_blackhole": lambda i: s3dlio.get("s3://bkt/key"),
        # op-log active at exit (daemon threads write op-log rows)
        "oplog_get": lambda i: s3dlio.get(uri),
        "oplog_put": lambda i: s3dlio.put_bytes("file://%s/o%d_%d.bin" % (d, tid(), i % 3), b"x" * (1 << 20)),
        # main-thread atexit handler that calls s3dlio while the daemons are still busy
        "atexit_get": lambda i: s3dlio.get(uri),
        "atexit_oplog_finalize": lambda i: s3dlio.get(uri),
    }
    def gi_async():
        return s3dlio.get_into_async(uri, bytearray(4 << 20), expected_len=4 << 20)
    ASYNC = {
        "async_get": lambda: s3dlio.get_async(uri),
        "async_put_bytes": lambda: s3dlio.put_bytes_async("file://%s/a%d.bin" % (d, tid()), b"y" * (1 << 20)),
        "async_pending": lambda: s3dlio.get_async("s3://bkt/key"),   # never completes
        "get_into_async": gi_async,                                  # sync detach on the loop thread
    }
    BLOCKS = {"sync_blackhole", "async_pending"}   # never complete a call

    def sync_worker(fn):
        def run():
            i = 0
            try:
                while True:
                    i += 1
                    note_start()
                    fn(i)
                    note_done()
            except BaseException as e:      # a dead worker would make the test vacuous
                errors.append(repr(e))
        return run

    def async_worker(factory):
        def run():
            async def main():
                while True:
                    note_start()
                    await asyncio.gather(*[factory() for _ in range(4)])
                    note_done()
            try:
                asyncio.run(main())
            except BaseException as e:
                errors.append(repr(e))
        return run

    if mode.startswith("oplog") or mode == "atexit_oplog_finalize":
        s3dlio.init_op_log(os.path.join(d, "ops.tsv.zst"))
    if mode == "atexit_get":
        atexit.register(lambda: s3dlio.get(uri))
    if mode == "atexit_oplog_finalize":
        atexit.register(lambda: (s3dlio.get(uri), s3dlio.finalize_op_log()))

    if mode == "nondaemon":
        done = []
        def w():
            for _ in range(200):
                s3dlio.get(uri)
            done.append(1)
        ts = [threading.Thread(target=w) for _ in range(NW)]
        for t in ts:
            t.start()
        # Python must wait for non-daemon threads; atexit runs after they are joined.
        atexit.register(lambda: print("DONE", len(done), flush=True))
        sys.exit(0)

    daemon_target = sync_worker(SYNC[mode]) if mode in SYNC else async_worker(ASYNC[mode])
    for _ in range(NW):
        threading.Thread(target=daemon_target, daemon=True).start()

    deadline = time.time() + 30
    while len(started) < NW and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(1)
    if errors or len(started) < NW or (mode not in BLOCKS and completed[0] == 0):
        print("SCENARIO NOT EXERCISED: started=%d completed=%d errors=%r"
              % (len(started), completed[0], errors[:3]), file=sys.stderr, flush=True)
        os._exit(3)
    if mode.startswith("oplog") or mode.startswith("atexit_oplog"):
        assert s3dlio.is_op_log_active()
    print("READY", len(started), completed[0], flush=True)
    # main returns here: interpreter finalization starts with the daemons inside s3dlio calls
''')

DAEMON_MODES = [
    "get", "put_bytes", "list", "stat", "get_many", "get_range", "writer", "sync_blackhole",
    "async_get", "async_put_bytes", "async_pending", "get_into_async",
    "oplog_get", "oplog_put", "atexit_get", "atexit_oplog_finalize",
]


def _run(mode, tmp_path):
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, mode, str(tmp_path), str(NWORKERS)],
        capture_output=True, text=True, timeout=60,
    )


# Known, separate problem (issue #302; NOT #298, NOT fixed by PyO3 0.29.3): on CPython 3.14 a daemon thread's event
# loop with s3dlio `get_async` calls in flight at exit crashes ~90% of the time ("Fatal Python error:
# PyInterpreterState_Delete: remaining threads", SIGSEGV/SIGABRT), with PyO3 0.27.2 (25/30) and 0.29.3
# (28/30) alike. The sync calls, `get_into_async`, `put_bytes_async` and a never-completing `get_async`
# are clean on 3.14. xfail (not skip) so the case still runs and an upstream fix shows up as XPASS.
_PY314_ASYNC_GET = pytest.mark.xfail(
    sys.version_info >= (3, 14), strict=False,
    reason="CPython 3.14 get_async on a daemon loop at exit: separate from #298, pre-existing, issue #302")
MODE_PARAMS = [pytest.param(m, marks=_PY314_ASYNC_GET) if m == "async_get" else m for m in DAEMON_MODES]


@pytest.mark.parametrize("attempt", range(3))
@pytest.mark.parametrize("mode", MODE_PARAMS)
def test_exit_with_daemon_threads_in_s3dlio_calls(mode, attempt, tmp_path):
    r = _run(mode, tmp_path)
    assert r.returncode == 0, "exit code %r\nstdout:\n%s\nstderr:\n%s" % (
        r.returncode, r.stdout[-500:], r.stderr[-2000:])
    assert "FATAL" not in r.stderr, r.stderr[-2000:]
    assert "READY" in r.stdout, "scenario was not exercised:\n%s\n%s" % (r.stdout[-500:], r.stderr[-1000:])


@pytest.mark.parametrize("attempt", range(3))
def test_exit_with_non_daemon_threads_still_waits_and_exits_cleanly(attempt, tmp_path):
    """Non-daemon threads are joined by Python before finalization: no abort, all work completes."""
    r = _run("nondaemon", tmp_path)
    assert r.returncode == 0, "exit code %r\nstderr:\n%s" % (r.returncode, r.stderr[-2000:])
    assert "FATAL" not in r.stderr, r.stderr[-2000:]
    assert "DONE %d" % NWORKERS in r.stdout, r.stdout[-500:]
