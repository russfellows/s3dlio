"""Zero-copy put_bytes() / put_bytes_async() for every contiguous buffer, whatever its item
type, and put_bytes_async(borrow=True) (issue #238, round 2).

Memory and GIL measurements run in a FRESH interpreter per case (peak RSS is a high-water
mark). Thresholds are generous; the pointer-identity proof of "no copy" is in the Rust unit
tests of src/python_api/put_payload.rs.

Run: S3DLIO_TEST_DIR=<dir on a real filesystem> uv run --no-project --isolated --with <wheel> \
         --with pytest --with numpy python -m pytest python/tests/test_put_bytes_zero_copy_238.py
"""

import array
import asyncio
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
# QUIET-HOST bound: the stall tests assert that another Python thread is never delayed by more
# than this while a 256 MiB put runs. Measured stalls are 1-3 ms; a copy under the GIL is
# 150-300 ms. On a busy host (other builds, CI neighbours) scheduler noise alone can exceed
# 50 ms, so run these on a quiet host; a failure that shows a stall of a few tens of ms with no
# RSS growth is host noise, a stall of 150 ms or more is the bug.
MAX_STALL_SECONDS = 0.050
MAX_RSS_GROWTH_FRACTION = 0.25
MIN_RSS_GROWTH_FRACTION = 0.8  # a snapshot copy is visible as ~1.0x

CHILD = r"""
import array, asyncio, ctypes, json, os, resource, sys, threading, time
import numpy as np
import s3dlio

uri, kind, mib, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
size = mib * 1024 * 1024
if kind == "bytearray":
    payload = bytearray(b"\xa5") * size
elif kind == "np_float32":
    payload = np.full(size // 4, 1.5, dtype=np.float32)
elif kind == "np_int8":
    payload = np.full(size, 3, dtype=np.int8)
elif kind == "np_bool":
    payload = np.ones(size, dtype=bool)
elif kind == "array_b":
    payload = array.array("b", b"\x01") * size
elif kind == "ctypes_mv":
    arr = (ctypes.c_ubyte * size)()
    ctypes.memset(ctypes.addressof(arr), 0xAB, size)
    payload = memoryview(arr)  # writable, format '<B'
else:
    raise SystemExit("bad kind")

def rss_peak_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

baseline = rss_peak_kib()
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

def try_resize():
    # True when the source can be resized (no export held). Only bytearray can be resized.
    if not isinstance(payload, bytearray):
        return None
    try:
        payload.extend(b"x")
        del payload[-1:]
        return True
    except BufferError:
        return False

pending_resizable = None
after_resizable = None
t0 = time.perf_counter()
if mode == "sync":
    s3dlio.put_bytes(uri, payload)
    after_resizable = try_resize()
else:
    async def go():
        global pending_resizable, after_resizable
        aw = s3dlio.put_bytes_async(uri, payload, borrow=(mode == "async_borrow"))
        pending_resizable = try_resize()
        await aw
        after_resizable = try_resize()
    asyncio.run(go())
elapsed = time.perf_counter() - t0
stop.set()
t.join()
print(json.dumps({
    "max_gap": state["max_gap"], "ticks": state["ticks"], "elapsed": elapsed,
    "rss_growth_kib": rss_peak_kib() - baseline, "payload_kib": size // 1024,
    "pending_resizable": pending_resizable, "after_resizable": after_resizable,
}))
"""


def run_child(uri, kind, mode="sync", mib=PAYLOAD_MIB):
    out = subprocess.run(
        [sys.executable, "-c", CHILD, uri, kind, str(mib), mode],
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


def _odirect_supported(directory):
    """True when `directory` is on a filesystem that accepts O_DIRECT (not tmpfs)."""
    flag = getattr(os, "O_DIRECT", None)
    if flag is None:
        return False
    path = os.path.join(directory, ".odirect_probe")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | flag, 0o600)
    except OSError:
        return False
    os.close(fd)
    os.unlink(path)
    return True


def require_odirect(directory):
    if not _odirect_supported(directory):
        pytest.skip(
            f"{directory} does not support O_DIRECT (tmpfs?); set S3DLIO_TEST_DIR to a "
            "directory on a real filesystem"
        )


# --- item 1: any item type is zero-copy and does not stall other threads -----------------

NON_BYTE_KINDS = ["np_float32", "np_int8", "np_bool", "array_b", "ctypes_mv"]


@pytest.mark.parametrize("kind", NON_BYTE_KINDS)
def test_blocking_put_of_any_item_type_makes_no_copy_and_does_not_stall(work_dir, kind):
    r = run_child(f"file://{work_dir}/obj.bin", kind)
    limit = r["payload_kib"] * MAX_RSS_GROWTH_FRACTION
    assert r["rss_growth_kib"] < limit, (
        f"peak RSS grew {r['rss_growth_kib'] / 1024:.0f} MiB putting a {PAYLOAD_MIB} MiB "
        f"{kind} (limit {limit / 1024:.0f} MiB): the payload was copied"
    )
    assert r["max_gap"] < MAX_STALL_SECONDS, (
        f"another thread stalled {r['max_gap'] * 1000:.0f} ms during put_bytes of {kind}"
    )
    assert os.path.getsize(f"{work_dir}/obj.bin") == PAYLOAD_MIB * MIB


# --- item 8: put_bytes_async(borrow=True) ------------------------------------------------


@pytest.mark.parametrize("kind", ["bytearray", "ctypes_mv", "np_float32"])
def test_async_borrow_makes_no_copy(work_dir, kind):
    r = run_child(f"file://{work_dir}/obj.bin", kind, mode="async_borrow")
    limit = r["payload_kib"] * MAX_RSS_GROWTH_FRACTION
    assert r["rss_growth_kib"] < limit, (
        f"borrow=True grew peak RSS {r['rss_growth_kib'] / 1024:.0f} MiB for a "
        f"{PAYLOAD_MIB} MiB {kind}: it copied"
    )
    assert r["max_gap"] < MAX_STALL_SECONDS


@pytest.mark.parametrize("kind", ["bytearray", "ctypes_mv"])
def test_async_default_still_snapshots(work_dir, kind):
    r = run_child(f"file://{work_dir}/obj.bin", kind, mode="async")
    assert r["rss_growth_kib"] > r["payload_kib"] * MIN_RSS_GROWTH_FRACTION, (
        "the default put_bytes_async is documented to take one snapshot copy"
    )
    assert r["max_gap"] < MAX_STALL_SECONDS, (
        "the snapshot is copied with the GIL released"
    )


def test_borrow_holds_the_export_until_the_await_completes(work_dir):
    r = run_child(f"file://{work_dir}/obj.bin", "bytearray", mode="async_borrow")
    assert r["pending_resizable"] is False, (
        "export must be held while the put is pending"
    )
    assert r["after_resizable"] is True, (
        "export must be released before the await returns"
    )


def test_default_async_does_not_hold_the_source(work_dir):
    r = run_child(f"file://{work_dir}/obj.bin", "bytearray", mode="async")
    assert r["pending_resizable"] is True, (
        "a snapshot must not block resizing the source"
    )
    assert r["after_resizable"] is True


def test_blocking_put_releases_the_export_before_returning(work_dir):
    r = run_child(f"file://{work_dir}/obj.bin", "bytearray", mode="sync")
    assert r["after_resizable"] is True


# --- round trips ---------------------------------------------------------------------------

SIZES = [0, 1, 4095, 4096, 4097, MIB + 3]


def _pattern(n):
    return bytes((i * 131 + (i >> 8) * 7) & 0xFF for i in range(n))


def _exporters(raw):
    """(name, payload, expected stored bytes)."""
    yield "bytes", raw, raw
    yield "bytearray", bytearray(raw), raw
    yield "memoryview", memoryview(bytearray(raw)), raw
    yield "numpy-uint8", np.frombuffer(raw, dtype=np.uint8).copy(), raw
    yield "numpy-int8", np.frombuffer(raw, dtype=np.int8).copy(), raw
    if len(raw) % 4 == 0:
        yield "numpy-float32", np.frombuffer(raw, dtype=np.float32).copy(), raw
    if len(raw) % 8 == 0:
        yield "array-q", array.array("q", raw), raw
    bools = np.frombuffer(raw, dtype=np.uint8) & 1
    yield "numpy-bool", bools.astype(bool), bools.tobytes()
    if raw:
        c = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        yield "ctypes-mv", memoryview(c), raw
        yield "array-b", array.array("b", raw), raw


def _readback(uri):
    return bytes(memoryview(s3dlio.get(uri)))


async def _aput(uri, payload, **kw):
    await s3dlio.put_bytes_async(uri, payload, **kw)


@pytest.mark.parametrize("scheme", ["file", "direct"])
@pytest.mark.parametrize("size", SIZES)
def test_roundtrip_every_exporter_and_mode(work_dir, scheme, size):
    if scheme == "direct":
        require_odirect(work_dir)
    raw = _pattern(size)
    for name, payload, expected in _exporters(raw):
        uri = f"{scheme}://{work_dir}/rt-{name}-{size}.bin"
        s3dlio.put_bytes(uri, payload)
        assert _readback(uri) == expected, f"sync {name} {scheme} {size}"
        for kw in ({}, {"borrow": True}):
            uri2 = f"{scheme}://{work_dir}/rta-{name}-{size}-{len(kw)}.bin"
            asyncio.run(_aput(uri2, payload, **kw))
            assert _readback(uri2) == expected, f"async {kw} {name} {scheme} {size}"


def test_non_contiguous_views_keep_logical_order(work_dir):
    base = np.arange(4096, dtype=np.float32).reshape(64, 64)
    for name, view, expect in [
        ("transposed", base.T, base.T.tobytes()),
        ("strided", memoryview(bytearray(range(256)) * 16)[::3], None),
        ("array-strided", memoryview(array.array("H", range(4096)))[::2], None),
    ]:
        want = expect if expect is not None else bytes(memoryview(view).tobytes())
        for kw in ({}, {"borrow": True}):
            uri = f"file://{work_dir}/nc-{name}-{len(kw)}.bin"
            asyncio.run(_aput(uri, view, **kw))
            assert _readback(uri) == want, name
        uri = f"file://{work_dir}/nc-{name}-sync.bin"
        s3dlio.put_bytes(uri, view)
        assert _readback(uri) == want, name


def test_dunder_bytes_only_objects(work_dir):
    class C:
        def __bytes__(self):
            return b"abc" * 1000  # noqa: PLE0308

    uri = f"file://{work_dir}/c.bin"
    s3dlio.put_bytes(uri, C())
    assert _readback(uri) == b"abc" * 1000
    with pytest.raises(TypeError):
        s3dlio.put_bytes(f"file://{work_dir}/bad.bin", object())


def test_borrow_is_keyword_only(work_dir):
    with pytest.raises(TypeError):

        async def go():
            await s3dlio.put_bytes_async(f"file://{work_dir}/x.bin", b"abc", True)

        asyncio.run(go())


# --- direct:// GET is page-aligned -------------------------------------------------------


def test_direct_get_result_is_page_aligned(work_dir):
    require_odirect(work_dir)
    raw = _pattern(8 * MIB + 5)
    uri = f"direct://{work_dir}/g.bin"
    s3dlio.put_bytes(uri, raw)
    view = s3dlio.get(uri)
    addr = np.frombuffer(view, dtype=np.uint8).ctypes.data
    assert addr % 4096 == 0, f"direct:// GET returned address {addr:#x}"
    assert bytes(memoryview(view)) == raw
    # and putting it back is the zero-copy single-write path
    uri2 = f"direct://{work_dir}/g2.bin"
    s3dlio.put_bytes(uri2, view)
    assert _readback(uri2) == raw


# --- buffers of Python objects / pointers are refused ------------------------------------


@pytest.mark.parametrize(
    "make", [lambda: (ctypes.c_void_p * 8)(), lambda: (ctypes.py_object * 2)()]
)
def test_arrays_of_pointers_or_objects_are_refused(work_dir, make):
    payload = make()
    uri = f"file://{work_dir}/p.bin"
    with pytest.raises(TypeError, match="Python objects or pointers"):
        s3dlio.put_bytes(uri, payload)
    for kw in ({}, {"borrow": True}):
        with pytest.raises(TypeError, match="Python objects or pointers"):
            asyncio.run(_aput(uri, payload, **kw))
    assert not os.path.exists(f"{work_dir}/p.bin")


def test_small_default_async_snapshot_is_independent(work_dir):
    # below the GIL-held copy threshold (256 KiB): still a true snapshot
    ba = bytearray(b"a" * 1000)

    async def go():
        aw = s3dlio.put_bytes_async(f"file://{work_dir}/s.bin", ba)
        ba[:] = b"b" * 1000
        ba.extend(b"more")
        await aw

    asyncio.run(go())
    assert _readback(f"file://{work_dir}/s.bin") == b"a" * 1000
