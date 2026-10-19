"""Issue #254: the streaming writer's write_chunk()/write_owned_bytes() go through the same
payload conversion as put_bytes(): no copy made by the Python layer for any C-contiguous buffer,
and the GIL is released for the write.

Memory and GIL-stall measurements run in a fresh interpreter per case (peak RSS is a high-water
mark). Quiet-host bounds: measured stalls are 1-3 ms against 150-300 ms for a copy under the GIL.
(MultipartUploadWriter.write() needs an S3 endpoint to construct and is not run here; see the R7
section at the end for what is covered.)
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import s3dlio

np = pytest.importorskip("numpy")

MIB = 1024 * 1024
PAYLOAD_MIB = 256
MAX_STALL_SECONDS = 0.050
MAX_RSS_GROWTH_FRACTION = 0.25

CHILD = r"""
import json, resource, sys, threading, time
import numpy as np
import s3dlio

uri, method, kind, mib = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
size = mib * 1024 * 1024
if kind == "bytearray":
    payload = bytearray(b"\xa5") * size
elif kind == "np_float32":
    payload = np.full(size // 4, 1.5, dtype=np.float32)
elif kind == "memoryview":
    payload = memoryview(bytearray(b"\x5a") * size)
else:
    raise SystemExit("bad kind")

baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
writer = s3dlio.create_filesystem_writer(uri, None)
stop = threading.Event()
state = {"max_gap": 0.0, "ticks": 0}

def ticker():
    last = time.perf_counter()
    while not stop.is_set():
        time.sleep(0.001)
        now = time.perf_counter()
        state["max_gap"] = max(state["max_gap"], now - last)
        state["ticks"] += 1
        last = now

t = threading.Thread(target=ticker, daemon=True)
t.start()
time.sleep(0.05)
state["max_gap"] = 0.0
getattr(writer, method)(payload)
stop.set(); t.join()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
payload_len = len(memoryview(payload).cast("B")) if kind != "bytearray" else len(payload)
resizable = None
if kind == "bytearray":
    try:
        payload.append(0); del payload[-1:]; resizable = True
    except BufferError:
        resizable = False
writer.finalize()
print(json.dumps({"max_gap": state["max_gap"], "ticks": state["ticks"],
                  "rss_growth_kib": peak - baseline, "payload_kib": size // 1024,
                  "resizable": resizable}))
"""


@pytest.fixture
def work_dir():
    base = os.environ.get("S3DLIO_TEST_DIR") or None
    with tempfile.TemporaryDirectory(dir=base) as d:
        yield d


def run_child(work_dir, method, kind):
    uri = f"file://{work_dir}/w.bin"
    out = subprocess.run(
        [sys.executable, "-c", CHILD, uri, method, kind, str(PAYLOAD_MIB)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize("method", ["write_chunk", "write_owned_bytes"])
@pytest.mark.parametrize("kind", ["bytearray", "np_float32", "memoryview"])
def test_streaming_write_makes_no_copy_and_does_not_stall_other_threads(
    work_dir, method, kind
):
    r = run_child(work_dir, method, kind)
    limit = r["payload_kib"] * MAX_RSS_GROWTH_FRACTION
    assert r["rss_growth_kib"] < limit, (
        f"{method}({kind}): peak RSS grew {r['rss_growth_kib'] / 1024:.0f} MiB "
        f"(limit {limit / 1024:.0f} MiB): the payload was copied"
    )
    assert r["max_gap"] < MAX_STALL_SECONDS, (
        f"{method}({kind}): another thread stalled {r['max_gap'] * 1000:.0f} ms "
        "(the GIL was held for the write)"
    )
    if r["resizable"] is not None:
        assert r["resizable"] is True, (
            "the export was still held after the call returned"
        )
    assert os.path.getsize(f"{work_dir}/w.bin") == PAYLOAD_MIB * MIB


def test_streaming_write_roundtrip_every_kind(work_dir):
    raw = bytes((i * 131 + (i >> 8) * 7) & 0xFF for i in range(300_003))
    chunks = [
        raw[:100_000],
        bytearray(raw[100_000:200_000]),
        memoryview(bytearray(raw[200_000:300_000])),
        np.frombuffer(raw[300_000:300_003], dtype=np.uint8).copy(),
    ]
    for method in ("write_chunk", "write_owned_bytes"):
        path = f"{work_dir}/rt-{method}.bin"
        w = s3dlio.create_filesystem_writer(f"file://{path}", None)
        for c in chunks:
            getattr(w, method)(c)
        w.finalize()
        with open(path, "rb") as f:
            assert f.read() == raw, method


def test_streaming_write_strided_view_keeps_logical_order(work_dir):
    base = bytearray(range(256)) * 64
    view = memoryview(base)[::3]
    path = f"{work_dir}/strided.bin"
    w = s3dlio.create_filesystem_writer(f"file://{path}", None)
    w.write_chunk(view)
    w.finalize()
    with open(path, "rb") as f:
        assert f.read() == bytes(view)


def test_streaming_write_rejects_non_bytes_like(work_dir):
    w = s3dlio.create_filesystem_writer(f"file://{work_dir}/bad.bin", None)
    with pytest.raises(TypeError):
        w.write_chunk(object())
    with pytest.raises(TypeError):
        w.write_chunk((ctypes.c_void_p * 4)())
    w.finalize()


@pytest.mark.parametrize("method", ["write_chunk", "write_owned_bytes"])
@pytest.mark.parametrize("bad", [5, [1, 2, 3], (i for i in range(3)), "abc", None, 2.5])
def test_streaming_write_refuses_ints_and_plain_iterables(work_dir, method, bad):
    path = f"{work_dir}/strict.bin"
    w = s3dlio.create_filesystem_writer(f"file://{path}", None)
    with pytest.raises(TypeError):
        getattr(w, method)(bad)
    w.finalize()
    assert os.path.getsize(path) == 0, (
        "a refused value must write nothing (5 is not 5 zero bytes)"
    )


def test_streaming_write_accepts_objects_with_dunder_bytes(work_dir):
    class C:
        def __bytes__(self):
            return b"hello"

    path = f"{work_dir}/dunder.bin"
    w = s3dlio.create_filesystem_writer(f"file://{path}", None)
    w.write_chunk(C())
    w.finalize()
    with open(path, "rb") as f:
        assert f.read() == b"hello"


# ---------------------------------------------------------------------------------------------
# R7: the ownership contract of the streaming writers. write_chunk(), write_owned_bytes() and
# MultipartUploadWriter.write() BORROW the caller's buffer by default (zero copy, GIL released, no
# thread may modify the bytes until the call returns); write_owned_bytes() and
# MultipartUploadWriter.write() take a keyword-only copy=True that snapshots a mutable input under
# the GIL first. (MultipartUploadWriter.write() needs an S3 endpoint to construct, so only its
# Python-visible signature and docstring are exercised here; its copy=True logic shares
# snapshot_for_call() with write_owned_bytes(), which is tested end to end below.)
# ---------------------------------------------------------------------------------------------


class BlockedWrite:
    """A filesystem writer on a FIFO whose read end is open but not being read: a payload larger
    than the pipe buffer blocks inside the native write, which makes 'during the call'
    deterministic (no sleeps decide the outcome; the sleep only gives the thread time to enter). The payload must exceed what tokio's file buffer (2 MiB) absorbs before it blocks."""

    def __init__(self, work_dir, data, method, **kwargs):
        self.fifo = os.path.join(work_dir, "pipe")
        os.mkfifo(self.fifo)
        self.data = data
        self.rfd = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        os.set_blocking(self.rfd, True)
        self.writer = s3dlio.create_filesystem_writer(f"file://{self.fifo}", None)
        self.error = None
        self.thread = threading.Thread(
            target=self._run, args=(method, kwargs), daemon=True
        )
        self.thread.start()
        deadline = time.monotonic() + 30
        time.sleep(0.5)  # the write fills the pipe and blocks
        assert self.thread.is_alive(), f"the write did not block: {self.error!r}"
        assert time.monotonic() < deadline

    def _run(self, method, kwargs):
        try:
            getattr(self.writer, method)(self.data, **kwargs)
        except BaseException as e:  # noqa: BLE001 - reported to the test
            self.error = e

    def drain(self, size):
        got = bytearray()
        while len(got) < size:
            chunk = os.read(self.rfd, 1 << 20)
            assert chunk, "unexpected EOF"
            got += chunk
        self.thread.join(30)
        assert not self.thread.is_alive(), "the write never finished"
        assert self.error is None, self.error
        os.close(self.rfd)
        return bytes(got)


def _pattern(n):
    return bytes((i * 7 + (i >> 9)) & 0xFF for i in range(n))


@pytest.mark.parametrize("kind", ["bytearray", "memoryview", "np_uint8"])
def test_write_owned_bytes_copy_true_snapshots_before_the_write(work_dir, kind):
    original = _pattern(8 * MIB)
    ba = bytearray(original)
    arg = {
        "bytearray": ba,
        "memoryview": memoryview(ba),
        "np_uint8": np.frombuffer(ba, dtype=np.uint8),
    }[kind]
    w = BlockedWrite(work_dir, arg, "write_owned_bytes", copy=True)
    # The native write is blocked and still running. The input was snapshotted first, so the
    # export is gone (resizable) and changing the source cannot reach the written bytes.
    if kind == "np_uint8":
        arg[:] = 0x5A
    else:
        ba[:] = b"Z" * len(ba)
    if kind == "bytearray":
        ba.append(0)
        del ba[-1:]
    assert w.drain(len(original)) == original, "the write saw the later modification"


def test_write_owned_bytes_default_borrows_the_buffer_during_the_call(work_dir):
    """Default (copy=False) is zero copy: the buffer is exported while the call runs (a resize
    raises BufferError) and is free again once it returned. No modification happens here."""
    original = _pattern(8 * MIB)
    ba = bytearray(original)
    w = BlockedWrite(work_dir, ba, "write_owned_bytes")
    with pytest.raises(BufferError):
        ba.append(0)
    assert w.drain(len(original)) == original
    ba.append(0)  # released before the call returned
    del ba[-1:]


def test_write_chunk_borrows_and_has_no_copy_parameter(work_dir):
    original = _pattern(8 * MIB)
    ba = bytearray(original)
    w = BlockedWrite(work_dir, ba, "write_chunk")
    with pytest.raises(BufferError):
        ba.append(0)
    assert w.drain(len(original)) == original
    other = s3dlio.create_filesystem_writer(f"file://{work_dir}/c.bin", None)
    with pytest.raises(TypeError):
        other.write_chunk(b"x", copy=True)
    other.finalize()


@pytest.mark.parametrize("copy", [False, True])
def test_write_owned_bytes_roundtrip_in_both_modes(work_dir, copy):
    raw = _pattern(300_003)
    path = f"{work_dir}/rt-{copy}.bin"
    w = s3dlio.create_filesystem_writer(f"file://{path}", None)
    w.write_owned_bytes(raw[:100_000], copy=copy)
    w.write_owned_bytes(bytearray(raw[100_000:200_000]), copy=copy)
    w.write_owned_bytes(memoryview(bytearray(raw[200_000:])), copy=copy)
    w.finalize()
    with open(path, "rb") as f:
        assert f.read() == raw


def test_copy_true_keeps_the_strict_conversion(work_dir):
    w = s3dlio.create_filesystem_writer(f"file://{work_dir}/s.bin", None)
    for bad in (5, [1, 2], "abc", None):
        with pytest.raises(TypeError):
            w.write_owned_bytes(bad, copy=True)
    w.finalize()
    assert os.path.getsize(f"{work_dir}/s.bin") == 0


def test_copy_is_keyword_only(work_dir):
    w = s3dlio.create_filesystem_writer(f"file://{work_dir}/k.bin", None)
    with pytest.raises(TypeError):
        w.write_owned_bytes(b"x", True)
    w.finalize()


@pytest.mark.parametrize(
    "func_name,owner",
    [
        ("write_chunk", "PyObjectWriter"),
        ("write_owned_bytes", "PyObjectWriter"),
    ],
)
def test_python_visible_docs_state_the_borrow_contract(work_dir, func_name, owner):
    """The contract is in the docstring Python users see, not only in comments and the guide."""
    if owner == "PyObjectWriter":
        target = type(s3dlio.create_filesystem_writer(f"file://{work_dir}/d.bin", None))
    else:
        target = s3dlio.MultipartUploadWriter
    doc = getattr(target, func_name).__doc__ or ""
    low = doc.lower()
    assert "borrow" in low, doc
    assert "until the call returns" in low, doc
    assert "modify" in low, doc
    if func_name != "write_chunk":
        assert "copy=true" in low, doc
        assert "0.9.126" in doc, "the behaviour change must be stated"


def test_multipart_write_signature_has_keyword_only_copy():
    import inspect

    sig = inspect.signature(s3dlio.MultipartUploadWriter.write)
    assert "copy" in sig.parameters, sig
    p = sig.parameters["copy"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False, sig


def test_multipart_writer_write_docstring_states_the_zero_copy_contract():
    """`MultipartUploadWriter.write` (stage G W9): zero-copy by default, so the buffer is
    borrowed until close()/abort() returns; `copy=True` is the snapshot; BytesView is shared."""
    doc = s3dlio.MultipartUploadWriter.write.__doc__ or ""
    low = " ".join(doc.lower().split())
    assert "zero-copy by default" in low, doc
    assert "must not modify the buffer until close()" in low.replace("``", ""), doc
    assert "abort()" in low, doc
    assert "copy=true" in low, doc
    assert "bytesview" in low, doc
    assert "until the call returns" not in low, "stale pre-0.10.0 contract"
