"""Stage G (RULE #1, zero copy): checkpoint saves, checkpoint loads, loader items.

W1  checkpoint saves accept ANY buffer-protocol payload and read it in place (no ``to_vec``).
R1  ``load_checkpoint_with_validation_view`` returns a ``BytesView`` (zero copies); the ``bytes``
    variant stays (one documented copy).
R8  ``PyObjectItem`` supports the buffer protocol and ``.data`` (``BytesView``).
R9  (NPZ read) has no Python-level test here: ``read_npz`` fetches through the S3-only client, and
    its zero-copy bound is proven by the Rust allocation test in tests/test_zero_copy_checkpoint.rs.

Pointer identity for the Rust side of W1 (the ``Bytes`` that reaches the ObjectStore lies inside
the caller's buffer) is proven in tests/test_zero_copy_checkpoint.rs. Here the Python half is
proven: the save holds the caller's buffer export (``BufferError`` on resize while it is
referenced) instead of taking a snapshot.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
import s3dlio

PAYLOAD = bytes((i * 7) % 251 for i in range(300_000))
DEADLINE = "S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS"


@pytest.fixture(autouse=True)
def no_leftover_retention():
    """A test that armed `_put_test_retain_next_clone()` must not leave a clone behind.

    The teardown first CONSUMES a pending arming with a throw-away export, then releases what was
    retained (same pattern as test_put_release_barrier.py)."""
    yield
    s3dlio._put_test_queue_export(bytearray(1))
    s3dlio._put_test_release_retained()


def uri(root):
    return f"file://{root}"


def address(buf):
    """Address of the first byte `buf` exposes through the buffer protocol."""
    return np.frombuffer(buf, dtype=np.uint8).__array_interface__["data"][0]


def resizable(ba):
    try:
        ba.append(0)
        del ba[-1:]
        return True
    except BufferError:
        return False


def only_shard_bytes(root):
    """Bytes of the only shard file below ``root/shards``."""
    found = []
    for dirpath, _, names in os.walk(os.path.join(root, "shards")):
        found += [os.path.join(dirpath, n) for n in names]
    assert len(found) == 1, found
    with open(found[0], "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------------------------
# W1: any buffer-protocol payload is accepted and written byte-exact.
# ---------------------------------------------------------------------------------------------


def _payloads():
    base = np.frombuffer(PAYLOAD, dtype=np.uint8)
    f32 = np.arange(40_000, dtype=np.float32)
    strided = np.arange(80_000, dtype=np.float32)[
        ::2
    ]  # non-contiguous: one gather copy
    return {
        "bytes": (lambda: PAYLOAD, PAYLOAD),
        "bytearray": (lambda: bytearray(PAYLOAD), PAYLOAD),
        "memoryview": (lambda: memoryview(bytearray(PAYLOAD)), PAYLOAD),
        "numpy_uint8": (lambda: base.copy(), PAYLOAD),
        "numpy_float32": (lambda: f32.copy(), f32.tobytes()),
        "numpy_noncontiguous": (lambda: strided, strided.tobytes()),
        "empty_bytearray": (lambda: bytearray(), b""),
    }


def _save_store(root, data):
    return s3dlio.PyCheckpointStore(uri(root)).save(1, 1, "test", data)


def _save_writer(root, data):
    writer = s3dlio.PyCheckpointStore(uri(root)).writer(world_size=1, rank=0)
    return writer.save_distributed_shard(1, 1, "test", data)


def _save_module(root, data):
    return s3dlio.save_checkpoint(uri(root), 1, 1, "test", data)


def _save_module_shard(root, data):
    return s3dlio.save_distributed_shard(uri(root), 1, 1, "test", data, 1, 0)


SAVERS = {
    "PyCheckpointStore.save": _save_store,
    "PyCheckpointWriter.save_distributed_shard": _save_writer,
    "save_checkpoint": _save_module,
    "save_distributed_shard": _save_module_shard,
}


@pytest.mark.parametrize("kind", list(_payloads()))
@pytest.mark.parametrize("saver", list(SAVERS))
def test_save_accepts_any_buffer_payload(saver, kind):
    make, expected = _payloads()[kind]
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        SAVERS[saver](root, make())
        assert only_shard_bytes(root) == expected


def test_save_accepts_a_bytesview():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        s3dlio.save_checkpoint(uri(src), 1, 1, "test", PAYLOAD)
        manifest = s3dlio.PyCheckpointStore(uri(src)).reader().load_latest_manifest()
        view = s3dlio.load_checkpoint_with_validation_view(
            f"{uri(src)}/{manifest['shards'][0]['key']}"
        )
        for name, saver in SAVERS.items():
            root = os.path.join(tmp, name.replace(".", "_"))
            saver(root, view)
            assert only_shard_bytes(root) == PAYLOAD, name


@pytest.mark.parametrize("saver", list(SAVERS))
@pytest.mark.parametrize("bad", [123, "text", [1, 2, 3], None], ids=repr)
def test_save_refuses_a_non_buffer_with_an_informative_error(saver, bad):
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(TypeError, match="bytes-like|bytes"),
    ):
        SAVERS[saver](os.path.join(tmp, "ckpt"), bad)


@pytest.mark.parametrize("saver", list(SAVERS))
def test_save_holds_the_callers_buffer_instead_of_copying_it(saver, monkeypatch):
    """The save reads the caller's bytearray IN PLACE: the export is held (resize -> BufferError)
    for as long as any native reader (here: a retained clone, like a backend that keeps reading)
    can still see it. A save that took a snapshot (`to_vec`) would leave the bytearray free and
    could not trip the barrier at all."""
    monkeypatch.setenv(DEADLINE, "0.4")
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        s3dlio._put_test_retain_next_clone()
        with pytest.raises(s3dlio.ExportStillReferencedError):
            SAVERS[saver](os.path.join(tmp, "ckpt"), ba)
        assert not resizable(ba), "the save must hold the caller's buffer export"
        assert s3dlio._put_test_release_retained() >= 1
        deadline = time.monotonic() + 10
        while not resizable(
            ba
        ):  # released by the autonomous releaser once the clone is gone
            assert time.monotonic() < deadline, (
                "export never released after the clone died"
            )
            time.sleep(0.01)


@pytest.mark.parametrize("saver", list(SAVERS))
def test_buffer_is_released_when_the_save_returns(saver):
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        SAVERS[saver](os.path.join(tmp, "ckpt"), ba)
    assert resizable(ba), "the export must be released before the save returns"


def test_buffer_is_released_when_the_save_fails():
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        blocker = os.path.join(tmp, "blocker")
        with open(blocker, "wb") as fh:
            fh.write(b"a regular file where the checkpoint directory should be")
        with pytest.raises(Exception):  # noqa: B017 - any backend error
            s3dlio.save_checkpoint(uri(os.path.join(blocker, "sub")), 1, 1, "test", ba)
    assert resizable(ba), "the export must be released when the save raises"


def test_compressed_save_from_a_bytearray_round_trips():
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(uri(root), compression_level=3)
        store.save(1, 1, "test", ba)
        assert bytes(store.load_latest()) == PAYLOAD
    assert resizable(ba)


# ---------------------------------------------------------------------------------------------
# R1: load_checkpoint_with_validation_view -> BytesView, zero copies.
# ---------------------------------------------------------------------------------------------


def _shard_uri(root):
    manifest = s3dlio.PyCheckpointStore(uri(root)).reader().load_latest_manifest()
    return f"{uri(root)}/{manifest['shards'][0]['key']}", manifest


@pytest.mark.parametrize("compression_level", [None, 3], ids=["plain", "zstd"])
def test_view_has_the_same_content_and_validation_as_the_bytes_variant(
    compression_level,
):
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.PyCheckpointStore(uri(root), compression_level=compression_level).save(
            1, 1, "test", PAYLOAD
        )
        shard_uri, _ = _shard_uri(root)
        for validate in (True, False):
            as_bytes = s3dlio.load_checkpoint_with_validation(shard_uri, validate)
            as_view = s3dlio.load_checkpoint_with_validation_view(shard_uri, validate)
            assert isinstance(as_bytes, bytes)
            assert isinstance(as_view, s3dlio.BytesView)
            assert bytes(memoryview(as_view)) == as_bytes
        # the validated result is the logical payload for both
        assert s3dlio.load_checkpoint_with_validation(shard_uri) == PAYLOAD
        assert bytes(s3dlio.load_checkpoint_with_validation_view(shard_uri)) == PAYLOAD
        # validate_integrity defaults to True for both
        assert (
            bytes(
                s3dlio.load_checkpoint_with_validation_view(
                    shard_uri, validate_integrity=True
                )
            )
            == PAYLOAD
        )


def test_view_buffer_is_stable_and_shared_not_copied():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(uri(root), 1, 1, "test", PAYLOAD)
        shard_uri, _ = _shard_uri(root)
        view = s3dlio.load_checkpoint_with_validation_view(shard_uri)
        mv1, mv2 = memoryview(view), memoryview(view)
        assert mv1.readonly and mv2.readonly
        assert address(mv1) == address(mv2) == address(view), "views must share memory"
        assert address(view.memoryview()) == address(view)
        # the memoryview keeps the buffer alive after the view name is gone
        addr = address(mv1)
        del view
        assert address(mv1) == addr
        assert bytes(mv1[:16]) == PAYLOAD[:16]
        # each `bytes` result is a fresh copy (documented), the view is not
        copy1 = s3dlio.load_checkpoint_with_validation(shard_uri)
        copy2 = s3dlio.load_checkpoint_with_validation(shard_uri)
        assert copy1 == copy2 == PAYLOAD


def test_view_errors_match_the_bytes_variant():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(uri(root), 1, 1, "test", PAYLOAD)
        shard_uri, manifest = _shard_uri(root)
        shard_path = os.path.join(root, manifest["shards"][0]["key"])
        with open(shard_path, "wb") as fh:
            fh.write(bytes([PAYLOAD[0] ^ 0xFF]) + PAYLOAD[1:])

        for fn in (
            s3dlio.load_checkpoint_with_validation,
            s3dlio.load_checkpoint_with_validation_view,
        ):
            with pytest.raises(Exception, match="Integrity validation failed"):
                fn(shard_uri, validate_integrity=True)
            with pytest.raises(Exception, match="Integrity validation failed"):
                fn(shard_uri)  # True is the default
            assert bytes(memoryview(fn(shard_uri, validate_integrity=False))) == (
                bytes([PAYLOAD[0] ^ 0xFF]) + PAYLOAD[1:]
            )

        missing = f"{uri(root)}/shards/does-not-exist.bin"
        with pytest.raises(Exception) as e_bytes:
            s3dlio.load_checkpoint_with_validation(missing, False)
        with pytest.raises(Exception) as e_view:
            s3dlio.load_checkpoint_with_validation_view(missing, False)
        assert type(e_bytes.value) is type(e_view.value)
        assert str(e_bytes.value) == str(e_view.value)


# ---------------------------------------------------------------------------------------------
# R8: PyObjectItem buffer protocol and .data.
# ---------------------------------------------------------------------------------------------


def test_loader_items_read_in_place():
    with tempfile.TemporaryDirectory() as tmp:
        expected = {}
        for i in range(3):
            path = os.path.join(tmp, f"obj{i}.bin")
            data = bytes((j + i) % 251 for j in range(50_000 + i))
            with open(path, "wb") as fh:
                fh.write(data)
            expected[f"file://{path}"] = data
        ds = s3dlio.PyDataset.from_uris(list(expected))
        loader = s3dlio.PyBytesAsyncDataLoader(ds, {"batch_size": 1})
        seen = 0
        for item in loader.items():
            want = expected[item.uri]
            mv = memoryview(item)
            assert mv.readonly
            assert bytes(mv) == want
            view = item.data
            assert isinstance(view, s3dlio.BytesView)
            # zero copy: memoryview(item), item.data and np.frombuffer(item) share one address
            arr = np.frombuffer(item, dtype=np.uint8)
            assert arr.__array_interface__["data"][0] == address(view) == address(mv)
            assert address(memoryview(item)) == address(mv)
            # bytes(item) stays, and is the documented copy
            assert bytes(item) == want
            assert len(item) == item.byte_count == len(want)
            # the buffer outlives the item name
            del item
            assert bytes(mv[:8]) == want[:8]
            seen += 1
        assert seen == len(expected)


# ---------------------------------------------------------------------------------------------
# PyCheckpointStream.write_chunk: any buffer-protocol payload, read in place.
# ---------------------------------------------------------------------------------------------


def _stream(root, compression_level=None):
    store = s3dlio.PyCheckpointStore(uri(root), compression_level=compression_level)
    writer = store.writer(world_size=1, rank=0)
    return store, writer.get_distributed_shard_stream(step=5, epoch=1, framework="test")


def _chunk_kinds():
    f32 = np.arange(5_000, dtype=np.float32)
    return {
        "bytes": (lambda: b"chunk-bytes" * 100, b"chunk-bytes" * 100),
        "bytearray": (
            lambda: bytearray(b"chunk-bytearray" * 100),
            b"chunk-bytearray" * 100,
        ),
        "memoryview": (
            lambda: memoryview(bytearray(b"chunk-mv" * 100)),
            b"chunk-mv" * 100,
        ),
        "numpy_uint8": (
            lambda: np.frombuffer(b"chunk-np" * 100, dtype=np.uint8).copy(),
            b"chunk-np" * 100,
        ),
        "numpy_float32": (lambda: f32.copy(), f32.tobytes()),
        "numpy_noncontiguous": (lambda: f32[::2], f32[::2].tobytes()),
        "empty": (lambda: bytearray(), b""),
    }


@pytest.mark.parametrize("kind", list(_chunk_kinds()))
def test_write_chunk_accepts_any_buffer_payload(kind):
    make, expected = _chunk_kinds()[kind]
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        _, stream = _stream(root)
        first = make()
        assert stream.write_chunk(first) == len(expected)
        assert stream.write_chunk(b"|tail") == 5
        stream.finalize()
        assert only_shard_bytes(root) == expected + b"|tail"


def test_write_chunk_accepts_a_bytesview():
    with tempfile.TemporaryDirectory() as tmp:
        s3dlio.save_checkpoint(uri(os.path.join(tmp, "src")), 1, 1, "test", PAYLOAD)
        manifest = (
            s3dlio.PyCheckpointStore(uri(os.path.join(tmp, "src")))
            .reader()
            .load_latest_manifest()
        )
        view = s3dlio.load_checkpoint_with_validation_view(
            f"{uri(os.path.join(tmp, 'src'))}/{manifest['shards'][0]['key']}"
        )
        root = os.path.join(tmp, "ckpt")
        _, stream = _stream(root)
        assert stream.write_chunk(view) == len(PAYLOAD)
        stream.finalize()
        assert only_shard_bytes(root) == PAYLOAD


def test_write_chunk_compressed_from_a_bytearray_round_trips():
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        _, stream = _stream(root, compression_level=3)
        stream.write_chunk(ba)
        meta = stream.finalize()
        assert meta["key"].endswith(".zst")
        assert resizable(ba)
        shard_uri = f"{uri(root)}/{meta['key']}"
        # the stored shard is zstd; reading without validation returns it as stored
        stored = s3dlio.load_checkpoint_with_validation(shard_uri, False)
        assert stored != PAYLOAD and len(stored) < len(PAYLOAD)


@pytest.mark.parametrize("bad", [123, "text", [1, 2, 3], None], ids=repr)
def test_write_chunk_refuses_a_non_buffer_with_an_informative_error(bad):
    with tempfile.TemporaryDirectory() as tmp:
        _, stream = _stream(os.path.join(tmp, "ckpt"))
        with pytest.raises(TypeError, match="bytes-like|bytes"):
            stream.write_chunk(bad)


def test_write_chunk_holds_the_callers_buffer_instead_of_copying_it(monkeypatch):
    """Like the saves: the chunk is read in place under a held export (a snapshot copy would
    leave the bytearray free and could not trip the barrier)."""
    monkeypatch.setenv(DEADLINE, "0.4")
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        _, stream = _stream(os.path.join(tmp, "ckpt"))
        s3dlio._put_test_retain_next_clone()
        with pytest.raises(s3dlio.ExportStillReferencedError):
            stream.write_chunk(ba)
        assert not resizable(ba), "write_chunk must hold the caller's buffer export"
        assert s3dlio._put_test_release_retained() >= 1
        deadline = time.monotonic() + 10
        while not resizable(ba):
            assert time.monotonic() < deadline, (
                "export never released after the clone died"
            )
            time.sleep(0.01)


def test_buffer_is_released_when_write_chunk_returns():
    ba = bytearray(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        _, stream = _stream(os.path.join(tmp, "ckpt"))
        stream.write_chunk(ba)
        assert resizable(ba), "the export must be released before write_chunk returns"
        stream.cancel()


def test_write_chunk_with_an_unsupported_payload_creates_no_shard():
    """The payload is converted BEFORE the lazy writer is created: a rejected chunk must not
    create (and truncate) a shard, and a later finalize() must still report that nothing was
    written."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        _, stream = _stream(root)
        with pytest.raises(TypeError):
            stream.write_chunk(123)
        created = [f for _, _, names in os.walk(root) for f in names]
        assert created == [], f"a rejected chunk created {created}"
        with pytest.raises(Exception, match="No data written"):
            stream.finalize()


# ---------------------------------------------------------------------------------------------
# Peak-memory bounds (child processes): a copy anywhere on the path shows up as +1x the payload.
# ---------------------------------------------------------------------------------------------


MIB = 1 << 20
RSS_N = 128 * MIB  # payload size of the memory tests

_CHILD_PRELUDE = """
import gc, json, sys
import numpy as np
import s3dlio
gc.collect()
def peak_kib():
    # VmHWM is this process's own resident high-water mark. (getrusage's ru_maxrss would also
    # carry the PARENT's high-water mark across exec on Linux and hide the growth.)
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM not found")
"""

_LOAD_CHILD = (
    _CHILD_PRELUDE
    + """
uri, which = sys.argv[1], sys.argv[2]
base = peak_kib()
if which == "view":
    out = s3dlio.load_checkpoint_with_validation_view(uri)
else:
    out = s3dlio.load_checkpoint_with_validation(uri)
n = len(memoryview(out))
print(json.dumps({"delta_kib": peak_kib() - base, "len": n}))
"""
)

_SAVE_CHILD = (
    _CHILD_PRELUDE
    + """
root, saver, kind, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
if kind == "bytearray":
    data = bytearray(b"\\x5a") * n
elif kind == "numpy":
    data = np.full(n, 0x5A, dtype=np.uint8)
else:
    data = b"\\x5a" * n
uri = "file://" + root
base = peak_kib()
if saver == "store.save":
    s3dlio.PyCheckpointStore(uri).save(1, 1, "t", data)
elif saver == "writer.save_distributed_shard":
    s3dlio.PyCheckpointStore(uri).writer(1, 0).save_distributed_shard(1, 1, "t", data)
elif saver == "save_checkpoint":
    s3dlio.save_checkpoint(uri, 1, 1, "t", data)
else:
    s3dlio.save_distributed_shard(uri, 1, 1, "t", data, 1, 0)
print(json.dumps({"delta_kib": peak_kib() - base}))
"""
)


def _child(code, *args):
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_view_load_peaks_at_one_copy_and_the_bytes_variant_at_two():
    """The view load holds the payload ONCE (the fetched buffer); the `bytes` compatibility API
    holds it twice (fetched buffer + the Python bytes): the documented single copy."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(uri(root), 1, 1, "t", b"\x5a" * RSS_N)
        shard = only_shard_path(root)
        shard_uri = f"{uri(root)}/{os.path.relpath(shard, root)}"
        view = _child(_LOAD_CHILD, shard_uri, "view")
        as_bytes = _child(_LOAD_CHILD, shard_uri, "bytes")
    assert view["len"] == as_bytes["len"] == RSS_N
    view_mib = view["delta_kib"] / 1024
    bytes_mib = as_bytes["delta_kib"] / 1024
    n_mib = RSS_N / MIB
    assert view_mib < 1.3 * n_mib + 16, (
        f"view load peaked +{view_mib:.0f} MiB for {n_mib:.0f}"
    )
    assert bytes_mib > 1.7 * n_mib, (
        f"the bytes variant should hold two copies (+{bytes_mib:.0f} MiB): "
        "the probe is not measuring what it should"
    )


def only_shard_path(root):
    found = []
    for dirpath, _, names in os.walk(os.path.join(root, "shards")):
        found += [os.path.join(dirpath, n) for n in names]
    assert len(found) == 1, found
    return found[0]


@pytest.mark.parametrize("kind", ["bytearray", "numpy", "bytes"])
@pytest.mark.parametrize(
    "saver",
    [
        "store.save",
        "writer.save_distributed_shard",
        "save_checkpoint",
        "save_distributed_shard",
    ],
)
def test_save_does_not_grow_peak_memory_by_a_payload_copy(saver, kind):
    """Saving a 128 MiB payload must not allocate payload-sized memory: the old path made two
    copies (+256 MiB), a single compat copy would show as +128 MiB."""
    with tempfile.TemporaryDirectory() as tmp:
        result = _child(_SAVE_CHILD, os.path.join(tmp, "ckpt"), saver, kind, RSS_N)
    grew_mib = result["delta_kib"] / 1024
    assert grew_mib < 24, (
        f"{saver} from {kind} peaked +{grew_mib:.0f} MiB for a 128 MiB payload"
    )
