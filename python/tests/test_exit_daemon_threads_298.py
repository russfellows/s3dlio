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

Issue #302 (CPython 3.14): a tokio worker that attached to Python itself to build its result (a bare
`Python::attach` inside a `future_into_py` body) creates a thread state during interpreter finalization and
corrupts it (SIGSEGV/SIGABRT, "PyInterpreterState_Delete: remaining threads").  The `async_*`, `mes_*`,
`ds_get_item` and `async_loader` scenarios below cover every async entry point that used to do that.

Do NOT rely on the "FATAL" check alone: release builds can abort silently inside pthread_exit;
the exit code is the load-bearing assertion.
"""
import os
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

    # The S3-only calls (the "many" calls, the S3 loader) talk to a tiny S3 stand-in that runs in its
    # OWN process (the S3 loader's constructor blocks while holding the GIL, so an in-thread server
    # would starve). It serves 3 objects of 1 MiB under s3://bkt/p/ (HEAD, GET, ranged GET, list) and
    # exits when its stdin closes, i.e. when this process dies.
    S3_URI = "s3://bkt/p/o0.bin"
    S3_PREFIX = "s3://bkt/p/"
    if mode in ("async_get_many", "async_get_many_stats", "async_stat_many", "async_s3_loader"):
        import subprocess
        fake = subprocess.Popen([sys.executable, os.path.join(d, "fake_s3.py")], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        os.environ["AWS_ENDPOINT_URL"] = "http://127.0.0.1:%d" % int(fake.stdout.readline())

    # ---- fixtures for the #302 entry points ---------------------------------------------------
    if mode.startswith("mes_"):
        eps = ["file://%s/ep%d/" % (d, i) for i in range(2)]
        for i in range(2):
            os.makedirs("%s/ep%d" % (d, i), exist_ok=True)
            with open("%s/ep%d/obj.bin" % (d, i), "wb") as f:
                f.write(os.urandom(1 << 20))
        mes = s3dlio.create_multi_endpoint_store(uris=eps, strategy="round_robin")
    if mode in ("ds_get_item", "async_loader"):
        os.makedirs(d + "/ds", exist_ok=True)
        for i in range(8):
            with open("%s/ds/f%d.bin" % (d, i), "wb") as f:
                f.write(os.urandom(256 << 10))
        dsuri = "file://" + d + "/ds/"
        ds = s3dlio.create_dataset(dsuri)

    async def numeric_loader_once():
        n = 0
        async for _ in s3dlio.PyAsyncDataLoader(s3dlio.PyVecDataset(list(range(200))), {"batch_size": 10}):
            n += 1
            if n >= 3:
                break

    async def s3_loader_once():
        # `s3dlio.PyS3AsyncDataLoader` is the deprecated shim (PyS3AsyncDataLoaderCompat): its __aiter__
        # returns a PyBytesAsyncDataLoader, which is not itself an async iterator, so `async for` over the
        # shim raises TypeError (pre-existing). Its real iterator is one __aiter__ further.
        it = s3dlio.PyS3AsyncDataLoader(S3_PREFIX, {"batch_size": 2}).__aiter__().__aiter__()
        for _ in range(2):
            await it.__anext__()

    async def loader_once():
        n = 0
        async for _ in s3dlio.create_async_loader(dsuri, {"batch_size": 2}):
            n += 1
            if n >= 3:
                break

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
        # issue #302: every entry point whose result used to be built by an attaching tokio worker
        "async_get_range": lambda: s3dlio.get_range_async(uri, 0, 1 << 20),
        "async_get_many": lambda: s3dlio.get_many_async([S3_URI] * 4),
        "async_get_many_stats": lambda: s3dlio.get_many_stats_async([S3_URI] * 4),
        "async_stat": lambda: s3dlio.stat_async(uri),
        "async_stat_many": lambda: s3dlio.stat_many_async([S3_URI] * 4),
        "async_put": lambda: s3dlio.put_async("file://%s/pa%d/" % (d, tid()), 2, size=1 << 20,
                                              should_create_bucket=True),
        "mes_get": lambda: mes.get(eps[0] + "obj.bin"),
        "mes_get_range": lambda: mes.get_range(eps[0] + "obj.bin", 0, 1 << 19),
        "mes_get_from_endpoint": lambda: mes.get_from_endpoint(1, eps[1] + "obj.bin"),
        "ds_get_item": lambda: ds.get_item(3),
        "async_loader": lambda: loader_once(),
        "async_numeric_loader": lambda: numeric_loader_once(),
        "async_s3_loader": lambda: s3_loader_once(),
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
    # Give the slowest scenario (an S3 listing through a stand-in server) time to finish one call on a
    # loaded machine, then let the workers run for 1 s before the main thread returns.
    deadline = time.time() + 20
    while mode not in BLOCKS and completed[0] == 0 and not errors and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(1)
    if errors or len(started) < NW or (mode not in BLOCKS and completed[0] == 0):
        import faulthandler
        faulthandler.dump_traceback(all_threads=True)      # what were the workers waiting for?
        print("SCENARIO NOT EXERCISED: started=%d completed=%d errors=%r"
              % (len(started), completed[0], errors[:3]), file=sys.stderr, flush=True)
        os._exit(3)
    if mode.startswith("oplog") or mode.startswith("atexit_oplog"):
        assert s3dlio.is_op_log_active()
    print("READY", len(started), completed[0], flush=True)
    # main returns here: interpreter finalization starts with the daemons inside s3dlio calls
''')

# Stand-in S3 service for the scenarios that need a completing S3 call (run as its own process).
FAKE_S3 = textwrap.dedent('''
    import http.server, os, socketserver, sys, threading
    DATA = os.urandom(1 << 20)
    LIST = ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            '<Name>bkt</Name><Prefix>p/</Prefix><KeyCount>3</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>'
            + "".join('<Contents><Key>p/o%d.bin</Key><LastModified>2015-10-21T07:28:00.000Z</LastModified>'
                      '<ETag>"abc"</ETag><Size>%d</Size><StorageClass>STANDARD</StorageClass></Contents>' % (i, len(DATA))
                      for i in range(3))
            + '</ListBucketResult>').encode()
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a):
            pass
        def _head(self, code, n, extra=None):
            self.send_response(code)
            for k, v in {"Content-Length": str(n), "ETag": '"abc"', "Accept-Ranges": "bytes",
                         "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
                         "Content-Type": "application/octet-stream", **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
        def do_HEAD(self):
            self._head(200, len(DATA))
        def do_GET(self):
            if "list-type" in self.path:
                self._head(200, len(LIST), {"Content-Type": "application/xml"})
                self.wfile.write(LIST)
                return
            r = self.headers.get("Range")
            if r and r.startswith("bytes="):
                a, _, b = r[6:].partition("-")
                a = int(a)
                b = min(int(b) if b else len(DATA) - 1, len(DATA) - 1)
                body = DATA[a:b + 1]
                self._head(206, len(body), {"Content-Range": "bytes %d-%d/%d" % (a, b, len(DATA))})
            else:
                body = DATA
                self._head(200, len(body))
            self.wfile.write(body)
    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        # The default listen backlog is 5. 8 loops x 4 calls x 4 requests open about 100 connections at once;
        # the overflow drops the final ACK and the client sits in SYN retransmit back-off (1, 2, 4, 8 s),
        # which showed up as a 1-3% 'scenario not exercised' flake.
        request_queue_size = 512
    srv = S(("127.0.0.1", 0), H)
    print(srv.server_address[1], flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sys.stdin.read()          # parent gone (stdin closed): exit
    os._exit(0)
''')

# Attempts per scenario. A crash at exit is probabilistic (1/30 to 30/30 depending on the call), so CI
# raises this with S3DLIO_EXIT_ATTEMPTS; the default keeps a local run short.
ATTEMPTS = int(os.environ.get("S3DLIO_EXIT_ATTEMPTS", "3"))

DAEMON_MODES = [
    "get", "put_bytes", "list", "stat", "get_many", "get_range", "writer", "sync_blackhole",
    "async_get", "async_put_bytes", "async_pending", "get_into_async",
    "async_get_range", "async_get_many", "async_get_many_stats", "async_stat", "async_stat_many", "async_put",
    "mes_get", "mes_get_range", "mes_get_from_endpoint", "ds_get_item", "async_loader",
    "async_numeric_loader", "async_s3_loader",
    "oplog_get", "oplog_put", "atexit_get", "atexit_oplog_finalize",
]


def _run(mode, tmp_path):
    (tmp_path / "fake_s3.py").write_text(FAKE_S3)
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, mode, str(tmp_path), str(NWORKERS)],
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("attempt", range(ATTEMPTS))
@pytest.mark.parametrize("mode", DAEMON_MODES)
def test_exit_with_daemon_threads_in_s3dlio_calls(mode, attempt, tmp_path):
    r = _run(mode, tmp_path)
    assert r.returncode == 0, "exit code %r\nstdout:\n%s\nstderr:\n%s" % (
        r.returncode, r.stdout[-500:], r.stderr[-2000:])
    assert "FATAL" not in r.stderr, r.stderr[-2000:]
    assert "READY" in r.stdout, "scenario was not exercised:\n%s\n%s" % (r.stdout[-500:], r.stderr[-1000:])


@pytest.mark.parametrize("attempt", range(ATTEMPTS))
def test_exit_with_non_daemon_threads_still_waits_and_exits_cleanly(attempt, tmp_path):
    """Non-daemon threads are joined by Python before finalization: no abort, all work completes."""
    r = _run("nondaemon", tmp_path)
    assert r.returncode == 0, "exit code %r\nstderr:\n%s" % (r.returncode, r.stderr[-2000:])
    assert "FATAL" not in r.stderr, r.stderr[-2000:]
    assert "DONE %d" % NWORKERS in r.stdout, r.stdout[-500:]


# Issue #304: once the exit gate has closed (the s3dlio atexit drain ran), an async entry point is
# REFUSED with a catchable RuntimeError before it calls into Python, instead of entering Python while
# the interpreter finalizes (an abort on CPython < 3.13.8). An atexit handler registered BEFORE
# `import s3dlio` runs after the drain, so it sees the closed gate.
REFUSAL_SCRIPT = textwrap.dedent('''
    import atexit, asyncio, os, sys, tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "o.bin")
    open(p, "wb").write(b"x" * 4096)

    class UserBytes:                 # only __bytes__: the last-resort conversion runs the user's bytecode
        def __bytes__(self):
            return b"zz"

    def late():
        async def go():
            calls = {
                "get_async": lambda: s3dlio.get_async("file://" + p),
                "put_bytes_async": lambda: s3dlio.put_bytes_async("file://" + d + "/w.bin", b"y" * 1024),
                "put_bytes_async_bytearray": lambda: s3dlio.put_bytes_async("file://" + d + "/w2.bin", bytearray(1024)),
                "get_into_async": lambda: s3dlio.get_into_async("file://" + p, bytearray(4096), expected_len=4096),
                "stat_async": lambda: s3dlio.stat_async("file://" + p),
                "put_bytes_async_user_bytes": lambda: s3dlio.put_bytes_async("file://" + d + "/w3.bin", UserBytes()),
            }
            for name, call in calls.items():
                try:
                    call()
                    print("NOT-REFUSED", name, flush=True)
                except RuntimeError as e:
                    print("REFUSED", name, "exiting" in str(e), flush=True)
        asyncio.run(go())

    atexit.register(late)           # BEFORE the import: runs after s3dlio's drain
    import s3dlio
    print("READY", flush=True)
''')


def test_async_calls_after_the_exit_gate_closed_raise_runtime_error():
    r = subprocess.run([sys.executable, "-c", REFUSAL_SCRIPT], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, "exit code %r\nstdout:\n%s\nstderr:\n%s" % (r.returncode, r.stdout, r.stderr[-2000:])
    assert "NOT-REFUSED" not in r.stdout, r.stdout
    for name in ("get_async", "put_bytes_async", "put_bytes_async_bytearray", "get_into_async", "stat_async",
                 "put_bytes_async_user_bytes"):
        assert "REFUSED %s True" % name in r.stdout, (name, r.stdout, r.stderr[-1000:])


# Issue #304, round 2: ONE exit-gate scope covers a guarded entry point from its top to its end. If the
# gate closes after the native put was spawned (hidden hook `_test_close_exit_gate_after_next_spawn`,
# which closes it right after the spawn), the rest of the start (guard construction, and the cancel of
# every failure path) must NOT be refused: the call returns its awaitable. Before, `instantiate` was
# refused, the cancel was skipped, and the call blocked in the export barrier and raised.
CLOSE_AFTER_SPAWN_SCRIPT = textwrap.dedent('''
    import asyncio, os, tempfile, time
    import s3dlio
    d = tempfile.mkdtemp()

    async def go():
        buf = bytearray(1 << 20)               # a mutable buffer: borrowed, so the put is guarded
        s3dlio._test_close_exit_gate_after_next_spawn()
        t0 = time.time()
        aw = s3dlio.put_bytes_async("file://" + d + "/w.bin", buf)
        print("RETURNED %.2f" % (time.time() - t0), flush=True)
        aw.cancel()

    asyncio.run(go())
    print("DONE", flush=True)
''')


def test_gate_closing_after_the_spawn_does_not_refuse_the_rest_of_the_start():
    # The hook exists only in a wheel built with the `test-hooks` cargo feature (never in a release wheel).
    probe = subprocess.run([sys.executable, "-c", "import s3dlio,sys; sys.exit(0 if getattr(s3dlio, '_test_close_exit_gate_after_next_spawn', None) else 3)"])
    if probe.returncode != 0:
        msg = "this wheel was built without the `test-hooks` cargo feature (no s3dlio._test_close_exit_gate_after_next_spawn)"
        if os.environ.get("S3DLIO_REQUIRE_TEST_HOOKS") == "1":
            pytest.fail(msg + "; S3DLIO_REQUIRE_TEST_HOOKS=1 forbids skipping")
        pytest.skip(msg)
    r = subprocess.run([sys.executable, "-c", CLOSE_AFTER_SPAWN_SCRIPT], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, "exit code %r\nstdout:\n%s\nstderr:\n%s" % (r.returncode, r.stdout, r.stderr[-2000:])
    assert "RETURNED" in r.stdout and "DONE" in r.stdout, (r.stdout, r.stderr[-2000:])
