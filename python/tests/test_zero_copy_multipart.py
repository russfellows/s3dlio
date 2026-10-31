# python/tests/test_zero_copy_multipart.py
"""Stage G (Rule #1, zero copy everywhere): ``MultipartUploadWriter.write``.

* ``write(buf)`` no longer copies a mutable buffer into the sink's part buffers: the buffer's
  export is held (locked against resizing) and the parts are uploaded straight out of it. The
  export lives until ``close()``/``abort()`` returns (W9).
* ``write(BytesView)`` of any size shares the view's allocation; it used to copy views under
  8 MiB (W6).
* The sink's tail buffering no longer copies owned data twice (W3).

Proof:
* the buffer is locked (``BufferError`` on resize) until ``close()`` returns, and is free again
  afterwards;
* an allocation bound: peak RSS growth while writing a large buffer to an in-process server
  that DISCARDS what it receives (so the server holds nothing). A copy of the parts shows up
  as hundreds of MiB; the zero-copy path as a few MiB;
* the stored object is byte-exact, including parts that straddle two ``write()`` calls.

The process-wide S3 client binds to one endpoint for the life of the process, so each scenario
runs in an ISOLATED CHILD process (this file run as a script) against an in-process server.
No live endpoint is used.
"""

import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
# The default listen backlog of these stand-in servers is 5; concurrent connects beyond it wait for a SYN
# retransmit (1, 2, 4 s). A bigger backlog removes that source of flakiness (issue #304 round 2).
ThreadingHTTPServer.request_queue_size = 512
HTTPServer.request_queue_size = 512
from importlib import import_module
from urllib.parse import parse_qs, urlsplit

import pytest
import s3dlio

np = pytest.importorskip("numpy")
CORE = import_module(
    "s3dlio._pymod"
)  # the compiled module: hidden test hooks live here

MIB = 1 << 20
CHILD_ENV = "S3DLIO_ZC_MPU_CHILD"
CLIENT_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_CA_BUNDLE",
    "S3DLIO_TEST_BUCKET",
)


class DiscardingS3:
    """An in-process S3 multipart endpoint that reads each part and THROWS IT AWAY (it only
    counts bytes), after a short delay so that many parts are in flight at once. It holds no
    payload, so any memory growth of the process is the client's."""

    def __init__(self, part_delay=0.4, threaded=True):
        self.received = 0
        self.parts = 0
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def _drain(self):
                n = int(self.headers.get("Content-Length", "0") or 0)
                left = n
                while left:
                    got = len(self.rfile.read(min(left, 64 * 1024)))
                    if not got:
                        break
                    left -= got
                return n

            def _reply(self, status, body=b"", headers=None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def _handle(self):
                n = self._drain()
                q = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if self.command == "POST" and "uploads" in q:
                    self._reply(
                        200,
                        b"<InitiateMultipartUploadResult><Bucket>b</Bucket><Key>k</Key>"
                        b"<UploadId>up-1</UploadId></InitiateMultipartUploadResult>",
                    )
                elif self.command == "PUT" and "partNumber" in q:
                    with outer.lock:
                        outer.received += n
                        outer.parts += 1
                    threading.Event().wait(part_delay)
                    self._reply(
                        200, headers={"ETag": '"etag-{}"'.format(q["partNumber"][0])}
                    )
                elif self.command == "POST" and "uploadId" in q:
                    self._reply(
                        200,
                        b'<CompleteMultipartUploadResult><ETag>"done"</ETag>'
                        b"</CompleteMultipartUploadResult>",
                    )
                elif self.command == "DELETE":
                    self._reply(204)
                else:
                    self._reply(404)

            do_GET = do_HEAD = do_POST = do_DELETE = do_PUT = _handle

        server_cls = ThreadingHTTPServer if threaded else HTTPServer
        self.server = server_cls(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


def point_client_at(url):
    for name in CLIENT_ENV:
        os.environ.pop(name, None)
    os.environ["AWS_ACCESS_KEY_ID"] = "test-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret-key"
    os.environ["AWS_ENDPOINT_URL"] = url
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"


def peak_rss_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM not found")


def reset_peak_rss():
    # Linux 4.0+: writing 5 resets the peak (high-water mark) of the resident set.
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def touch(buf):
    """Fault in every page of a mutable buffer (so it counts in the baseline RSS)."""
    mv = memoryview(buf).cast("B")
    for i in range(0, len(mv), 4096):
        mv[i] = 1


# ---------------------------------------------------------------------------------------------
# Child scenarios
# ---------------------------------------------------------------------------------------------


def child_rss_bytearray():
    """256 MiB written in 16 parts of 16 MiB: peak RSS growth must stay far below one copy."""
    server = DiscardingS3()
    try:
        point_client_at(server.url)
        n = 256 * MIB
        buf = bytearray(n)
        touch(buf)
        reset_peak_rss()
        base = peak_rss_kib()
        w = s3dlio.MultipartUploadWriter(
            "bkt", "obj", part_size=16 * MIB, max_in_flight=32
        )
        assert w.write(buf) == n
        try:
            buf.extend(b"x")
            raise AssertionError(
                "the buffer must be locked while parts may still read it"
            )
        except BufferError:
            pass
        w.close()
        buf.extend(b"x")  # released again once close() has returned
        grew = (peak_rss_kib() - base) * 1024
        assert server.received == n, (server.received, n)
        assert grew < 64 * MIB, (
            f"peak RSS grew by {grew >> 20} MiB writing {n >> 20} MiB: parts were copied"
        )
        print(f"rss_bytearray: grew {grew >> 20} MiB")
    finally:
        server.close()


def child_rss_small_bytesview():
    """Fifty 5 MiB BytesViews (each < 8 MiB, each exactly one part) with many parts in flight:
    every part must be a slice of the view. The < 8 MiB branch used to copy each one (W6)."""
    server = DiscardingS3()
    try:
        point_client_at(server.url)
        view = s3dlio.generate_data(5 * MIB)
        reset_peak_rss()
        base = peak_rss_kib()
        w = s3dlio.MultipartUploadWriter(
            "bkt", "obj", part_size=5 * MIB, max_in_flight=64
        )
        for _ in range(50):
            assert w.write(view) == 5 * MIB
        w.close()
        grew = (peak_rss_kib() - base) * 1024
        assert server.received == 250 * MIB, (server.received, 250 * MIB)
        assert grew < 48 * MIB, (
            f"peak RSS grew by {grew >> 20} MiB writing BytesViews: parts were copied"
        )
        print(f"rss_small_bytesview: grew {grew >> 20} MiB")
    finally:
        server.close()


def vm_size_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmSize not found")


def child_reservation_lifetime():
    """8 x reserve(32 MiB) / commit(5 MiB = one part) with the memoryviews KEPT: after close()
    every view must still be intact (the reservation is owned by its exporter, not by the writer
    or the upload), and the 8 x 32 MiB stay allocated while the views exist (a short commit pins
    the whole reservation, as documented) and are given back when the last view is gone."""
    # One server thread: a thread per request would add a malloc arena (64 MiB of address space)
    # per concurrent part, which swamps the lifetime being measured.
    server = DiscardingS3(part_delay=0.1, threaded=False)
    try:
        point_client_at(server.url)
        # Warm up: the first upload starts the runtime's threads, whose malloc arenas reserve
        # GiBs of virtual address space that have nothing to do with the reservations.
        warm = s3dlio.MultipartUploadWriter("bkt", "warm", part_size=5 * MIB)
        warm.write(bytes(6 * MIB))
        warm.close()
        before = vm_size_bytes()
        w = s3dlio.MultipartUploadWriter(
            "bkt", "obj", part_size=5 * MIB, max_in_flight=8
        )
        views = []
        for i in range(8):
            mv = w.reserve(32 * MIB)
            mv[:8] = bytes([i + 1]) * 8
            w.commit(5 * MIB)
            views.append(mv)
        del mv
        w.close()
        assert server.received == 6 * MIB + 8 * 5 * MIB, server.received
        for i, view in enumerate(views):
            assert bytes(view[:8]) == bytes([i + 1]) * 8, "a view changed after close()"
        held = vm_size_bytes() - before
        del view
        views.clear()
        freed = vm_size_bytes() - before
        print(
            f"reservation_lifetime: {held >> 20} MiB held while the views exist, "
            f"{freed >> 20} MiB after the last view is gone"
        )
        assert held >= 192 * MIB, (
            f"only {held >> 20} MiB held: the views do not own their memory"
        )
        assert freed < 64 * MIB, (
            f"{freed >> 20} MiB still allocated after the last view was dropped"
        )
    finally:
        server.close()


def child_immutable_view_twice():
    """`memoryview(bytes)` is immutable memory: passing the same view twice while the first
    part is still in flight (slow server) is not a reuse and must not raise."""
    server = DiscardingS3(part_delay=1.0)
    try:
        point_client_at(server.url)
        data = bytes(6 * MIB)
        view = memoryview(data)
        w = s3dlio.MultipartUploadWriter("bkt", "obj", part_size=5 * MIB)
        assert w.write(view) == 6 * MIB
        assert w.write(view) == 6 * MIB  # the first part is still being uploaded
        w.close()
        assert server.received == 12 * MIB, server.received
        print("immutable_view_twice: ok")
    finally:
        server.close()


def child_bench_small_writes():
    """Borrow (export + latch per write) vs copy (once, into the part buffer) by write size,
    16 MiB parts, 128 MiB per case, total time = write loop + close() (which includes the final
    gather and the uploads to a discarding server). Prints a table; asserts only that the
    borrow mode really locks the buffers. Run with S3DLIO_ZC_BENCH=1; set
    S3DLIO_UNSIGNED_PAYLOAD=1 for the unsigned table."""
    server = DiscardingS3(part_delay=0.0)
    try:
        point_client_at(server.url)
        total = 128 * MIB
        part = 16 * MIB
        rows = []
        for size in (
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            MIB,
            4 * MIB,
            16 * MIB,
            32 * MIB,
        ):
            n = total // size
            bufs = [bytearray(size) for _ in range(n)]
            for b in bufs:
                b[0] = 1
            res = {}
            for mode, threshold in (("borrow", 0), ("copy", 1 << 40)):
                best = None
                for _ in range(3):
                    CORE._mpu_coalesce_below(threshold)
                    w = s3dlio.MultipartUploadWriter(
                        "bkt", "obj", part_size=part, max_in_flight=32
                    )
                    t0 = time.perf_counter()
                    for b in bufs:
                        w.write(b)
                    t1 = time.perf_counter()
                    locked = False
                    try:
                        bufs[0].append(1)
                    except BufferError:
                        locked = True
                    if mode == "copy":
                        assert not locked, "copy mode must not lock the buffer"
                    w.close()
                    t2 = time.perf_counter()
                    cur = (t1 - t0, t2 - t0, locked)
                    if best is None or cur[1] < best[1]:
                        best = cur
                res[mode] = best
            rows.append((size, n, res))
        print(
            f"unsigned={os.environ.get('S3DLIO_UNSIGNED_PAYLOAD', '0')} part=16MiB total=128MiB"
        )
        print("size_KiB writes borrow_total_s copy_total_s borrow/copy locked(borrow)")
        for size, n, res in rows:
            b, c = res["borrow"], res["copy"]
            print(
                f"{size // 1024:8d} {n:6d} {b[1]:15.4f} {c[1]:12.4f} {b[1] / c[1]:11.2f} {b[2]}"
            )
    finally:
        CORE._mpu_coalesce_below(2**64 - 1)
        server.close()


def child_unsigned_threshold():
    """S3DLIO_UNSIGNED_PAYLOAD=1: a part spanning writes is sent without a gather, so writes of
    2 KiB or more are borrowed (locked) and smaller ones are copied (free at once)."""
    assert os.environ.get("S3DLIO_UNSIGNED_PAYLOAD") == "1"
    server = DiscardingS3(part_delay=0.0)
    try:
        point_client_at(server.url)
        w = s3dlio.MultipartUploadWriter("bkt", "obj", part_size=16 * MIB)
        small = bytearray(b"s" * 1024)
        w.write(small)
        small.extend(b"x")  # copied: not locked
        mid = bytearray(b"m" * 4096)
        w.write(mid)
        try:
            mid.extend(b"x")
            raise AssertionError("a 4 KiB write must be borrowed in an unsigned upload")
        except BufferError:
            pass
        w.close()
        mid.extend(b"x")  # released by close()
        assert server.received == 1024 + 4096, server.received
        print("unsigned_threshold: ok")
    finally:
        server.close()


CHILD_SCENARIOS = {
    "rss_bytearray": child_rss_bytearray,
    "rss_small_bytesview": child_rss_small_bytesview,
    "reservation_lifetime": child_reservation_lifetime,
    "bench_small": child_bench_small_writes,
    "immutable_view_twice": child_immutable_view_twice,
    "unsigned_threshold": child_unsigned_threshold,
}


def run_child(scenario, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in CLIENT_ENV}
    env.update(extra_env or {})
    env[CHILD_ENV] = scenario
    env["AWS_EC2_METADATA_DISABLED"] = "true"
    env.setdefault("RUST_LOG", "s3dlio=warn,aws_sdk_s3=warn")
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(s3dlio.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            package_root,
            os.path.dirname(os.path.abspath(__file__)),
            env.get("PYTHONPATH"),
        )
        if p
    )
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            check=False,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    return proc


# ---------------------------------------------------------------------------------------------
# Strict-server scenarios (byte-exact object, locking, straddling parts), run in a child too
# ---------------------------------------------------------------------------------------------


@unittest.skipUnless(
    os.environ.get(CHILD_ENV) == "strict", "runs only inside the child process"
)
class TestStrictServerScenarios(unittest.TestCase):
    """Run only inside the child process (see ``test_strict_scenarios_in_child_process``)."""

    @classmethod
    def setUpClass(cls):
        from test_multipart_writer import MiniS3, unique_bucket

        cls.server = MiniS3()
        point_client_at(cls.server.url)
        cls.bucket = unique_bucket("zc")
        s3dlio.create_bucket(cls.bucket)

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def stored(self, key):
        return self.server.objects[(self.bucket, key)]

    def writer(self, key, **kw):
        return s3dlio.MultipartUploadWriter(
            self.bucket, key, part_size=5 * MIB, max_in_flight=4, **kw
        )

    def test_bytearray_is_locked_until_close_and_uploaded_in_place(self):
        from test_multipart_writer import payload_of

        payload = payload_of(12 * MIB + 123, 11)
        buf = bytearray(payload)
        w = self.writer("locked.bin")
        self.assertEqual(w.write(buf), len(payload))
        with self.assertRaises(BufferError):
            buf.extend(b"x")
        with self.assertRaises(BufferError):
            del buf[:1]
        w.close()
        self.assertEqual(self.stored("locked.bin"), payload)
        buf.extend(b"x")  # unlocked again
        self.assertEqual(len(buf), len(payload) + 1)

    def test_straddling_writes_of_borrowed_buffers_are_exact(self):
        from test_multipart_writer import payload_of

        sizes = [7 * MIB, 7 * MIB + 1, 3 * MIB, 9 * MIB, 5 * MIB, 1]
        chunks = [payload_of(n, 20 + i) for i, n in enumerate(sizes)]
        bufs = [
            bytearray(chunks[0]),
            np.frombuffer(chunks[1], dtype=np.uint8).copy(),
            memoryview(bytearray(chunks[2])),
            chunks[3],  # exact bytes: immutable, no export
            s3dlio.generate_data(sizes[4]),  # BytesView
            bytearray(chunks[5]),
        ]
        expected = chunks[:4] + [bytes(memoryview(bufs[4]))] + chunks[5:]
        w = self.writer("straddle.bin")
        for b in bufs:
            w.write(b)
        w.close()
        self.assertEqual(self.stored("straddle.bin"), b"".join(expected))

    def test_copy_true_is_an_independent_snapshot(self):
        from test_multipart_writer import payload_of

        payload = payload_of(6 * MIB, 31)
        buf = bytearray(payload)
        w = self.writer("snapshot.bin")
        w.write(buf, copy=True)
        buf.extend(b"x")  # copy=True releases the export at once
        buf[: 1 * MIB] = bytes(1 * MIB)  # and the writer no longer reads it
        w.close()
        self.assertEqual(self.stored("snapshot.bin"), payload)

    def test_abort_releases_the_buffer(self):
        from test_multipart_writer import payload_of

        buf = bytearray(payload_of(11 * MIB, 41))
        w = self.writer("aborted.bin")
        w.write(buf)
        w.abort()
        buf.extend(b"x")  # abort() does not return while a part still reads the buffer
        self.assertNotIn((self.bucket, "aborted.bin"), self.server.objects)

    def test_context_manager_exit_releases_the_buffer(self):
        from test_multipart_writer import payload_of

        payload = payload_of(6 * MIB, 51)
        buf = bytearray(payload)
        with self.writer("with.bin") as w:
            w.write(buf)
        buf.extend(b"x")
        self.assertEqual(self.stored("with.bin"), payload)

    # -- reserve() / commit(): the reserved memory is owned by its exporter (G-F1) -----------

    def reserve_and_fill(self, w, size, data):
        mv = w.reserve(size)
        mv[: len(data)] = data
        return mv

    def test_reserve_returns_a_writable_memoryview_owned_by_an_exporter(self):
        w = self.writer("res-owner.bin")
        mv = w.reserve(4096)
        try:
            self.assertIsInstance(mv, memoryview)
            self.assertFalse(mv.readonly)
            self.assertEqual(len(mv), 4096)
            self.assertIsInstance(mv.obj, CORE.ReservedBuffer)
            self.assertIs(mv[10:20].obj, mv.obj, "a slice shares the exporter")
            self.assertEqual(
                bytes(mv[:4]), bytes(4), "a reservation starts zero-filled"
            )
            mv[:4] = b"abcd"
            self.assertEqual(bytes(memoryview(mv.obj)[:4]), b"abcd")
        finally:
            w.abort()

    def test_views_stay_readable_after_commit_and_close(self):
        from test_multipart_writer import payload_of

        data = payload_of(6 * MIB, 71)
        w = self.writer("res-close.bin")
        mv = self.reserve_and_fill(w, 8 * MIB, data)
        tail = mv[10:20]
        w.commit(len(data))
        w.close()
        self.assertEqual(self.stored("res-close.bin"), data)
        del mv
        self.assertEqual(bytes(tail), data[10:20])
        self.assertEqual(bytes(memoryview(tail.obj)[: len(data)]), data)

    def test_short_commit_uploads_the_prefix_and_leaves_the_view_whole(self):
        from test_multipart_writer import payload_of

        data = payload_of(100, 72)
        w = self.writer("res-short.bin")
        mv = self.reserve_and_fill(w, 8 * MIB, data)
        w.commit(len(data))
        w.close()
        self.assertEqual(self.stored("res-short.bin"), data)
        self.assertEqual(len(mv), 8 * MIB)
        self.assertEqual(bytes(mv[: len(data)]), data)

    def test_invalid_commit_length_leaves_the_reservation_pending(self):
        from test_multipart_writer import payload_of

        data = payload_of(6 * MIB, 73)
        w = self.writer("res-invalid.bin")
        mv = self.reserve_and_fill(w, 6 * MIB, data)
        with self.assertRaises(ValueError):
            w.commit(6 * MIB + 1)
        with self.assertRaises(RuntimeError):  # still pending: no second reserve
            w.reserve(16)
        self.assertEqual(bytes(mv[:64]), data[:64])
        w.commit(6 * MIB)
        w.close()
        self.assertEqual(self.stored("res-invalid.bin"), data)

    def test_views_stay_readable_after_abort(self):
        from test_multipart_writer import payload_of

        data = payload_of(6 * MIB, 74)
        w = self.writer("res-abort.bin")
        mv = self.reserve_and_fill(w, 8 * MIB, data)
        w.abort()  # frees the writer's claim, not the memory
        self.assertNotIn((self.bucket, "res-abort.bin"), self.server.objects)
        self.assertEqual(bytes(mv[: len(data)]), data)

    def test_views_stay_readable_after_the_writer_is_deleted(self):
        from test_multipart_writer import payload_of

        data = payload_of(2 * MIB, 75)
        w = self.writer("res-del.bin")
        mv = self.reserve_and_fill(w, 4 * MIB, data)
        tail = mv[100:200]
        del w
        del mv
        self.assertEqual(bytes(tail), data[100:200])
        self.assertEqual(bytes(memoryview(tail.obj)[: len(data)]), data)

    def test_views_stay_readable_after_leaving_a_with_block(self):
        from test_multipart_writer import payload_of

        data = payload_of(100, 76)
        with (
            self.assertRaises(KeyError),
            self.writer("res-with-error.bin") as w,
        ):
            mv = self.reserve_and_fill(w, 1 * MIB, data)
            raise KeyError("boom")  # leaves the block by an exception: abort
        self.assertEqual(bytes(mv[: len(data)]), data)
        self.assertNotIn((self.bucket, "res-with-error.bin"), self.server.objects)

        written = payload_of(6 * MIB, 77)
        with self.writer("res-with-ok.bin") as w:
            w.write(written)
            mv2 = self.reserve_and_fill(w, 1 * MIB, data)  # never committed
        self.assertEqual(self.stored("res-with-ok.bin"), written)
        self.assertEqual(bytes(mv2[: len(data)]), data)

    def test_a_numpy_array_over_the_view_keeps_the_memory_alive(self):
        from test_multipart_writer import payload_of

        data = payload_of(6 * MIB, 78)
        w = self.writer("res-numpy.bin")
        mv = w.reserve(6 * MIB)
        arr = np.frombuffer(mv, dtype=np.uint8)
        arr[:] = np.frombuffer(data, dtype=np.uint8)  # fill through numpy
        del mv
        w.commit(6 * MIB)
        w.close()
        del w
        self.assertEqual(self.stored("res-numpy.bin"), data)
        self.assertEqual(arr.tobytes(), data)

    def test_a_zero_byte_reservation_works(self):
        w = self.writer("res-zero.bin")
        mv = w.reserve(0)
        self.assertEqual(len(mv), 0)
        with self.assertRaises(ValueError):
            w.commit(1)
        w.commit(0)
        w.write(bytes(6 * MIB))
        w.close()
        self.assertEqual(len(self.stored("res-zero.bin")), 6 * MIB)

    def test_small_bytesview_write_shares_the_view(self):
        view = s3dlio.generate_data(2 * MIB)
        expected = bytes(memoryview(view))
        w = self.writer("view.bin")
        for _ in range(4):
            w.write(view)
        w.close()
        self.assertEqual(self.stored("view.bin"), expected * 4)

    def test_reusing_a_still_borrowed_buffer_raises_bufferror_and_can_be_aborted(self):
        from test_multipart_writer import payload_of

        size = 6 * MIB  # >= part_size, so it is borrowed; 1 MiB of it stays in the sink
        first, second = payload_of(size, 61), payload_of(size, 62)
        buf = bytearray(size)
        src = io.BytesIO(first + second)
        w = self.writer("reuse.bin")
        src.readinto(buf)  # the classic loop: readinto(buf); write(buf)
        self.assertEqual(w.write(buf), size)
        src.readinto(buf)  # overwrites memory the writer still reads
        with self.assertRaises(BufferError) as ctx:
            w.write(buf)
        self.assertIn("buffer reused while still borrowed", str(ctx.exception))
        self.assertIn("copy=True", str(ctx.exception))
        w.abort()  # the writer is still usable for abort
        buf.extend(b"x")  # and the buffer is released again
        self.assertNotIn((self.bucket, "reuse.bin"), self.server.objects)

    def test_overlapping_slices_are_refused_and_distinct_buffers_are_not(self):
        from test_multipart_writer import payload_of

        payload = payload_of(18 * MIB, 63)
        buf = bytearray(payload)
        view = memoryview(buf)
        w = self.writer("slices.bin")
        w.write(view[0 : 6 * MIB])
        with self.assertRaises(BufferError):
            w.write(view[3 * MIB : 9 * MIB])  # overlaps the first
        w.write(view[6 * MIB : 12 * MIB])  # disjoint: fine
        other = bytearray(payload_of(6 * MIB, 64))  # a distinct buffer: fine
        w.write(other)
        w.close()
        self.assertEqual(self.stored("slices.bin"), payload[: 12 * MIB] + bytes(other))

    def test_copy_true_and_a_fresh_writer_may_reuse_the_buffer(self):
        from test_multipart_writer import payload_of

        a, b = payload_of(MIB, 65), payload_of(MIB, 66)
        buf = bytearray(a)
        w = self.writer("reuse-copy.bin")
        w.write(buf, copy=True)
        buf[:] = b  # copy=True snapshotted: no borrow, no error
        w.write(buf, copy=True)
        w.close()
        self.assertEqual(self.stored("reuse-copy.bin"), a + b)
        buf[:] = a
        w2 = self.writer("reuse-after.bin")
        w2.write(buf)  # a new writer holds nothing from the old one
        w2.close()
        self.assertEqual(self.stored("reuse-after.bin"), a)

    def test_a_short_final_readinto_chunk_over_a_borrowed_buffer_is_refused(self):
        """readinto(buf); write(buf) ... the last readinto returns fewer bytes than the buffer:
        that short write overwrites memory the writer still reads and must be refused too."""
        from test_multipart_writer import payload_of

        size = 6 * MIB  # >= part_size, so it is borrowed; 1 MiB of it stays in the sink
        buf = bytearray(size)
        src = io.BytesIO(payload_of(size, 81) + payload_of(100, 82))
        w = self.writer("short-final.bin")
        n = src.readinto(buf)
        self.assertEqual(w.write(memoryview(buf)[:n]), size)
        n = src.readinto(buf)  # 100 bytes: the short final chunk
        self.assertEqual(n, 100)
        with self.assertRaises(BufferError) as ctx:
            w.write(memoryview(buf)[:n])
        self.assertIn("buffer reused while still borrowed", str(ctx.exception))
        w.abort()
        buf.extend(b"x")
        # A short write of unrelated memory is not a reuse.
        w2 = self.writer("short-other.bin")
        w2.write(bytearray(6 * MIB))
        w2.write(bytearray(b"z" * 100))
        w2.close()
        self.assertEqual(len(self.stored("short-other.bin")), 6 * MIB + 100)

    def test_a_memoryview_of_bytes_written_twice_is_byte_exact(self):
        from test_multipart_writer import payload_of

        data = payload_of(6 * MIB, 83)
        view = memoryview(data)
        w = self.writer("view-twice.bin")
        w.write(view)
        w.write(view)
        w.write(view[:100])
        w.close()
        self.assertEqual(self.stored("view-twice.bin"), data + data + data[:100])

    def test_sub_part_writes_are_copied_and_do_not_lock_the_buffer(self):
        """A write smaller than part_size is copied once (a part is made of several writes and
        must be gathered anyway): nothing is borrowed, so the buffer is free at once."""
        from test_multipart_writer import payload_of

        a, b = payload_of(MIB, 71), payload_of(MIB, 72)
        buf = bytearray(a)
        w = self.writer("subpart.bin")  # part_size = 5 MiB
        w.write(buf)
        buf.extend(b"x")  # no BufferError
        del buf[MIB:]
        buf[:] = b  # and rewriting it does not touch what was written
        w.write(buf)  # no BufferError for reuse either
        w.close()
        self.assertEqual(self.stored("subpart.bin"), a + b)

    def test_small_writes_are_copied_and_do_not_lock_the_buffer(self):
        buf = bytearray(b"a" * 4096)
        w = self.writer("small.bin")
        w.write(buf)
        buf.extend(b"x")  # nothing is borrowed below the threshold: no BufferError
        buf[:4096] = b"b" * 4096  # and a later change is not seen by the upload
        w.write(bytearray(b"c" * 10))
        w.close()
        self.assertEqual(self.stored("small.bin"), b"a" * 4096 + b"c" * 10)

    def test_many_small_writes_scan_nothing(self):
        """16384 writes of 256 B: bookkeeping must be O(1) per write (a scan of the held
        exports made this quadratic: 1.2 s against 0.01 s)."""
        n = 16384
        bufs = [bytearray([i & 255]) * 256 for i in range(n)]
        w = self.writer("many-small.bin")
        t0 = time.perf_counter()
        for b in bufs:
            w.write(b)
        elapsed = time.perf_counter() - t0
        w.close()
        self.assertEqual(
            self.stored("many-small.bin"), b"".join(bytes(b) for b in bufs)
        )
        self.assertLess(elapsed, 0.5, f"{n} small writes took {elapsed:.3f} s")

    def test_many_borrowed_writes_are_amortised_o1(self):
        """The same with every write borrowed (threshold 0): lookups are ordered-map O(log n)
        and released entries are swept in batches, not rescanned on every write."""
        if not hasattr(CORE, "_mpu_coalesce_below"):
            self.skipTest("hidden threshold hook missing")
        previous = CORE._mpu_coalesce_below(0)
        try:
            n = 16384
            bufs = [bytearray([i & 255]) * 256 for i in range(n)]
            w = self.writer("many-borrowed.bin")
            t0 = time.perf_counter()
            for b in bufs:
                w.write(b)
            elapsed = time.perf_counter() - t0
            w.close()
        finally:
            CORE._mpu_coalesce_below(previous)
        self.assertEqual(
            self.stored("many-borrowed.bin"), b"".join(bytes(b) for b in bufs)
        )
        self.assertLess(elapsed, 0.5, f"{n} borrowed writes took {elapsed:.3f} s")


# ---------------------------------------------------------------------------------------------
# pytest entry points (parent process): spawn the children
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get(CHILD_ENV) is not None, reason="already inside the child process"
)
class TestParent:
    def test_strict_scenarios_in_child_process(self):
        proc = run_child("strict")
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, "child failed:\n" + out
        ran = re.search(r"Ran (\d+) tests?", out)
        assert ran and int(ran.group(1)) >= 15, out
        assert "skipped" not in out, out

    @pytest.mark.skipif(
        not os.path.exists("/proc/self/clear_refs"), reason="needs Linux /proc"
    )
    def test_bytearray_write_makes_no_copy(self):
        proc = run_child("rss_bytearray")
        print(proc.stdout.strip())
        assert proc.returncode == 0, "child failed:\n" + proc.stdout + proc.stderr

    @pytest.mark.skipif(
        not os.path.exists("/proc/self/clear_refs"), reason="needs Linux /proc"
    )
    def test_small_bytesview_writes_make_no_copy(self):
        proc = run_child("rss_small_bytesview")
        print(proc.stdout.strip())
        assert proc.returncode == 0, "child failed:\n" + proc.stdout + proc.stderr

    @pytest.mark.skipif(
        not os.path.exists("/proc/self/status"), reason="needs Linux /proc"
    )
    def test_reserved_memory_lives_as_long_as_its_views(self):
        proc = run_child("reservation_lifetime")
        print(proc.stdout.strip())
        assert proc.returncode == 0, "child failed:\n" + proc.stdout + proc.stderr

    @pytest.mark.skipif(
        os.environ.get("S3DLIO_ZC_BENCH") != "1",
        reason="benchmark: set S3DLIO_ZC_BENCH=1 and run with -s",
    )
    def test_bench_small_write_crossover(self):
        proc = run_child("bench_small")
        print(proc.stdout)
        assert proc.returncode == 0, "child failed:\n" + proc.stdout + proc.stderr

    def test_the_same_immutable_view_twice_is_not_a_reuse(self):
        proc = run_child("immutable_view_twice")
        print(proc.stdout.strip())
        assert proc.returncode == 0, "child failed:\n" + proc.stdout + proc.stderr

    def test_unsigned_uploads_borrow_from_2_kib(self):
        proc = run_child("unsigned_threshold", {"S3DLIO_UNSIGNED_PAYLOAD": "1"})
        print(proc.stdout.strip())
        assert proc.returncode == 0, "child failed:\n" + proc.stdout + proc.stderr

    def test_close_and_exit_docstrings_state_the_precedence_of_the_export_error(self):
        for func in (
            s3dlio.MultipartUploadWriter.close,
            s3dlio.MultipartUploadWriter.__exit__,
        ):
            doc = func.__doc__ or ""
            assert "ExportStillReferencedError" in doc, doc
            assert "precedence" in doc, doc

    def test_reserve_and_commit_docstrings_state_the_ownership_contract(self):
        reserve_doc = s3dlio.MultipartUploadWriter.reserve.__doc__ or ""
        assert "ReservedBuffer" in reserve_doc, reserve_doc
        assert "short commit" in reserve_doc.lower(), reserve_doc
        assert "Do not write through a retained view after" in reserve_doc, reserve_doc
        commit_doc = s3dlio.MultipartUploadWriter.commit.__doc__ or ""
        assert "ValueError" in commit_doc and "pending" in commit_doc, commit_doc

    def test_write_docstring_states_the_borrow_contract(self):
        doc = s3dlio.MultipartUploadWriter.write.__doc__
        assert "must not modify the buffer until" in doc
        assert "copy=True" in doc


if __name__ == "__main__":
    scenario = os.environ.get(CHILD_ENV)
    if scenario == "strict":
        unittest.main(argv=[sys.argv[0], "-v", "TestStrictServerScenarios"])
    elif scenario in CHILD_SCENARIOS:
        CHILD_SCENARIOS[scenario]()
    else:
        sys.exit("run through pytest")
