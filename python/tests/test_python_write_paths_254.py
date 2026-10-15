"""Issue #254: the streaming writer's write_chunk()/write_owned_bytes() go through the same
payload conversion as put_bytes(): no copy made by the Python layer for any C-contiguous buffer,
and the GIL is released for the write.

Memory and GIL-stall measurements run in a fresh interpreter per case (peak RSS is a high-water
mark). Quiet-host bounds: measured stalls are 1-3 ms against 150-300 ms for a copy under the GIL.
(The S3 MultipartUploadWriter.write() path needs an S3 endpoint and is not exercised here.)
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile

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
