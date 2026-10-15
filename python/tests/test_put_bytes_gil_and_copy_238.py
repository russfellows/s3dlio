"""Regression tests for issue #238: put_bytes() / put_bytes_async() copy the payload
with the GIL held and (on file:// and direct://) make a second full copy.

The stall and memory measurements run in a FRESH interpreter per case so that the
interpreter's memory high-water mark and thread state are not shared between cases.
The thresholds are generous; the structural (pointer-identity) proof of "no copy"
lives in the Rust unit tests of src/python_api/put_payload.rs and in
tests/test_put_path_no_copy_238.rs.

Run: uv run --no-project --isolated --with <wheel> --with pytest --with numpy \
         python -m pytest python/tests/test_put_bytes_gil_and_copy_238.py
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

import s3dlio

MIB = 1024 * 1024
PAYLOAD_MIB = 256
# A memcpy of 256 MiB while holding the GIL stalls the other thread for ~150-300 ms.
# Without the copy the other thread is only delayed by scheduler noise.
MAX_STALL_SECONDS = 0.050
# Peak RSS may grow by at most this fraction of the payload during a blocking put
# (the bug made it grow by 1.0x for file://, one copy in put_bytes and one in the store).
MAX_RSS_GROWTH_FRACTION = 0.25

CHILD = r"""
import asyncio, json, os, resource, sys, threading, time
import s3dlio

uri, kind, mib, use_async = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
size = mib * 1024 * 1024
if kind == "bytearray":
    payload = bytearray(b"\xa5") * size
elif kind == "memoryview":
    payload = memoryview(bytearray(b"\x5a") * size)
elif kind == "bytesview":
    seed = os.path.join(os.path.dirname(uri[len("file://"):]), "seed.bin")
    s3dlio.put_bytes("file://" + seed, bytearray(b"\x3c") * size)
    payload = s3dlio.get("file://" + seed)
    os.unlink(seed)
else:
    raise SystemExit("bad kind")

def rss_peak_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Touch every page of the payload so it is resident before the baseline is taken.
baseline = rss_peak_kib()

stop = threading.Event()
state = {"max_gap": 0.0, "ticks": 0}

def ticker():
    last = time.perf_counter()
    while not stop.is_set():
        time.sleep(0.001)
        now = time.perf_counter()
        gap = now - last
        if gap > state["max_gap"]:
            state["max_gap"] = gap
        state["ticks"] += 1
        last = now

t = threading.Thread(target=ticker, daemon=True)
t.start()
time.sleep(0.05)
state["max_gap"] = 0.0

t0 = time.perf_counter()
if use_async:
    asyncio.run(s3dlio.put_bytes_async(uri, payload))
else:
    s3dlio.put_bytes(uri, payload)
elapsed = time.perf_counter() - t0
stop.set()
t.join()
peak = rss_peak_kib()
print(json.dumps({
    "max_gap": state["max_gap"], "ticks": state["ticks"], "elapsed": elapsed,
    "rss_growth_kib": peak - baseline, "payload_kib": size // 1024,
}))
"""


def run_child(uri, kind, use_async=False, mib=PAYLOAD_MIB):
    out = subprocess.run(
        [sys.executable, "-c", CHILD, uri, kind, str(mib), "1" if use_async else "0"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture
def work_dir():
    base = os.environ.get("S3DLIO_TEST_DIR") or None
    with tempfile.TemporaryDirectory(dir=base) as d:
        yield d


@pytest.mark.parametrize("kind", ["bytearray", "memoryview", "bytesview"])
def test_blocking_put_does_not_stall_other_threads(work_dir, kind):
    r = run_child(f"file://{work_dir}/obj.bin", kind)
    assert r["ticks"] > 0
    assert r["max_gap"] < MAX_STALL_SECONDS, (
        f"another Python thread stalled {r['max_gap'] * 1000:.0f} ms during put_bytes "
        f"of a {PAYLOAD_MIB} MiB {kind}: the payload is copied with the GIL held"
    )


# BytesView is not measured here: reading it with s3dlio.get() already raised this
# process's high-water mark, so the baseline hides a copy (its stall test above and the
# Rust pointer-identity test cover it).
@pytest.mark.parametrize("kind", ["bytearray", "memoryview"])
def test_blocking_put_makes_no_full_copy(work_dir, kind):
    r = run_child(f"file://{work_dir}/obj.bin", kind)
    limit = r["payload_kib"] * MAX_RSS_GROWTH_FRACTION
    assert r["rss_growth_kib"] < limit, (
        f"peak RSS grew by {r['rss_growth_kib'] / 1024:.0f} MiB during put_bytes of "
        f"{PAYLOAD_MIB} MiB {kind} (limit {limit / 1024:.0f} MiB): a full copy was made"
    )


@pytest.mark.parametrize("kind", ["bytearray", "bytesview"])
def test_async_put_does_not_stall_other_threads(work_dir, kind):
    r = run_child(f"file://{work_dir}/obj.bin", kind, use_async=True)
    assert r["max_gap"] < MAX_STALL_SECONDS, (
        f"another Python thread stalled {r['max_gap'] * 1000:.0f} ms during "
        f"put_bytes_async of a {PAYLOAD_MIB} MiB {kind}"
    )


# ---------------------------------------------------------------------------
# Round trip: the bytes read back are exactly the bytes written, for every
# buffer-protocol exporter, on file:// and direct://.
# ---------------------------------------------------------------------------

SIZES = [0, 1, 4095, 4096, 4097, MIB + 3]


def _pattern(n):
    # Not periodic with any power of two, so a shifted or truncated write is caught.
    return bytes((i * 131 + (i >> 8) * 7) & 0xFF for i in range(n))


def _exporters(raw):
    yield "bytes", raw
    yield "bytearray", bytearray(raw)
    yield "memoryview", memoryview(bytearray(raw))
    if raw:
        yield "memoryview-of-bytes", memoryview(raw)
    try:
        import numpy

        yield "numpy-uint8", numpy.frombuffer(raw, dtype=numpy.uint8).copy()
        if len(raw) % 4 == 0:
            yield "numpy-float32", numpy.frombuffer(raw, dtype=numpy.float32).copy()
    except ImportError:
        pass


def _readback(uri):
    return bytes(memoryview(s3dlio.get(uri)))


@pytest.mark.parametrize("scheme", ["file", "direct"])
@pytest.mark.parametrize("size", SIZES)
def test_roundtrip_every_exporter(work_dir, scheme, size):
    raw = _pattern(size)
    for name, payload in _exporters(raw):
        uri = f"{scheme}://{work_dir}/rt-{name}-{size}.bin"
        s3dlio.put_bytes(uri, payload)
        assert _readback(uri) == raw, f"{name} {scheme} size {size}"


@pytest.mark.parametrize("scheme", ["file", "direct"])
@pytest.mark.parametrize("size", SIZES)
def test_roundtrip_bytesview_and_async(work_dir, scheme, size):
    import asyncio

    raw = _pattern(size)
    seed = f"file://{work_dir}/seed-{size}.bin"
    s3dlio.put_bytes(seed, raw)
    view = s3dlio.get(seed)
    uri = f"{scheme}://{work_dir}/bv-{size}.bin"
    s3dlio.put_bytes(uri, view)
    assert _readback(uri) == raw

    uri2 = f"{scheme}://{work_dir}/bv-async-{size}.bin"
    asyncio.run(s3dlio.put_bytes_async(uri2, view))
    assert _readback(uri2) == raw

    for name, payload in _exporters(raw):
        uri3 = f"{scheme}://{work_dir}/async-{name}-{size}.bin"
        asyncio.run(s3dlio.put_bytes_async(uri3, payload))
        assert _readback(uri3) == raw, f"async {name} {scheme} size {size}"


def test_strided_memoryview_is_written_in_logical_order(work_dir):
    base = bytearray(range(256)) * 16
    view = memoryview(base)[::2]
    uri = f"file://{work_dir}/strided.bin"
    s3dlio.put_bytes(uri, view)
    assert _readback(uri) == bytes(view)


def test_borrowed_bytearray_can_be_resized_after_the_call(work_dir):
    # The buffer export taken for the call must be released when the call returns.
    ba = bytearray(b"x" * 100_000)
    s3dlio.put_bytes(f"file://{work_dir}/a.bin", ba)
    ba.extend(b"more")  # raises BufferError if the export leaked
    assert len(ba) == 100_004
