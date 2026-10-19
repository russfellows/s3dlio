"""Stage G, zero copy everywhere: the Python write paths of the core API (W7, W8, W9).

* W9  `put_bytes_async`, `put_many_async` and the `MultiEndpointStore` puts BORROW the caller's
      buffer by default (`borrow=True`); `borrow=False` is the opt-in snapshot, and an immutable
      source (`memoryview(bytes)`, a `bytes` subclass) is not copied even then.
* W8  `put_many` (blocking) holds each item's buffer exactly like `put_bytes`; it makes no copy.
* W7  the compat and DLIO wrappers pass the caller's memory on (memoryview / getbuffer()) instead
      of building `bytes` copies.

Borrowing is proved without any allocator: a file:// write to a FIFO blocks inside the native
`open()` until a reader attaches, so the native writer has not read the buffer yet. The buffer is
modified while the writer is stuck. A borrowing put then writes the MODIFIED bytes (it reads the
caller's memory in place), a snapshot writes the ORIGINAL bytes. (Modifying a borrowed buffer is
the contract violation the docs forbid; the test does it from one thread, before any native
reader runs, precisely to observe where the bytes are read from.) Copies of immutable sources are
measured as peak-RSS growth in a fresh interpreter.

Run: uv run --no-project --isolated --python 3.13 --with <wheel> --with pytest --with numpy \
         python -m pytest -p no:cacheprovider -q python/tests/test_zero_copy_core.py
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types

import pytest
import s3dlio

np = pytest.importorskip("numpy")

MIB = 1024 * 1024


def _install_fake_dlio_benchmark():
    """Minimal fake `dlio_benchmark.*` (not a dependency of this repo); idempotent. Registers the
    full set every DLIO test file needs, whichever installer runs first."""
    if "dlio_benchmark" in sys.modules:
        return

    dlio_benchmark = types.ModuleType("dlio_benchmark")
    common = types.ModuleType("dlio_benchmark.common")
    constants = types.ModuleType("dlio_benchmark.common.constants")
    constants.MODULE_STORAGE = "storage"
    enumerations = types.ModuleType("dlio_benchmark.common.enumerations")

    class NamespaceType:
        FLAT = "flat"

    class MetadataType:
        FILE = "file"
        DIRECTORY = "directory"

    enumerations.NamespaceType = NamespaceType
    enumerations.MetadataType = MetadataType

    storage = types.ModuleType("dlio_benchmark.storage")
    storage_handler = types.ModuleType("dlio_benchmark.storage.storage_handler")

    class Namespace:
        def __init__(self, name, ns_type):
            self.name = name
            self.ns_type = ns_type

    class DataStorage:
        def __init__(self, framework=None):
            self.framework = framework
            self._args = types.SimpleNamespace()

    storage_handler.DataStorage = DataStorage
    storage_handler.Namespace = Namespace

    s3_storage_mod = types.ModuleType("dlio_benchmark.storage.s3_storage")

    class S3Storage(DataStorage):
        pass

    s3_storage_mod.S3Storage = S3Storage

    utils = types.ModuleType("dlio_benchmark.utils")
    utility = types.ModuleType("dlio_benchmark.utils.utility")

    class _NullProfileDecorator:
        def __call__(self, fn):
            return fn

    class Profile:
        def __init__(self, module):
            self.module = module
            self.log = _NullProfileDecorator()
            self.log_init = _NullProfileDecorator()

    utility.Profile = Profile

    sys.modules["dlio_benchmark"] = dlio_benchmark
    sys.modules["dlio_benchmark.common"] = common
    sys.modules["dlio_benchmark.common.constants"] = constants
    sys.modules["dlio_benchmark.common.enumerations"] = enumerations
    sys.modules["dlio_benchmark.storage"] = storage
    sys.modules["dlio_benchmark.storage.storage_handler"] = storage_handler
    sys.modules["dlio_benchmark.storage.s3_storage"] = s3_storage_mod
    sys.modules["dlio_benchmark.utils"] = utils
    sys.modules["dlio_benchmark.utils.utility"] = utility


_install_fake_dlio_benchmark()

from s3dlio.integrations.dlio import s3_torch_storage, s3dlio_storage


@pytest.fixture
def work_dir():
    base = os.environ.get("S3DLIO_TEST_DIR") or None
    with tempfile.TemporaryDirectory(dir=base) as d:
        yield d


def resizable(ba):
    """True when a bytearray can be resized (no export holds it)."""
    try:
        ba.extend(b"x")
    except BufferError:
        return False
    del ba[-1:]
    return True


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def read_fifo(path):
    with open(path, "rb") as f:
        return f.read()


class FifoReader:
    """Reads a FIFO to EOF in a thread (attaching it lets a native writer stuck in open() go)."""

    def __init__(self, path):
        self.data = None
        self._t = threading.Thread(target=self._run, args=(path,), daemon=True)

    def _run(self, path):
        self.data = read_fifo(path)

    def start(self):
        self._t.start()
        return self

    def join(self, timeout=30):
        self._t.join(timeout)
        assert not self._t.is_alive(), "the FIFO reader never finished"
        return self.data


def make_fifo(work_dir, name="pipe"):
    path = os.path.join(work_dir, name)
    os.mkfifo(path)
    return path


def run(coro_fn, timeout=60):
    """Run `coro_fn()` on a fresh loop with a bound, so a hang fails instead of blocking."""

    async def bounded():
        return await asyncio.wait_for(coro_fn(), timeout)

    return asyncio.run(bounded())


# --------------------------------------------------------------------------------------
# W9: put_bytes_async borrows by default
# --------------------------------------------------------------------------------------


def test_put_bytes_async_default_borrows_a_bytearray(work_dir):
    original = bytes(range(251)) * 4000
    fifo = make_fifo(work_dir)
    ba = bytearray(original)
    seen = {}

    async def go():
        fut = s3dlio.put_bytes_async(
            f"file://{fifo}", ba
        )  # no borrow argument: the default
        await asyncio.sleep(0.3)  # the native writer is stuck opening the FIFO
        seen["locked"] = not resizable(ba)
        ba[:4] = b"ZZZZ"
        reader = FifoReader(fifo).start()
        await fut
        seen["data"] = reader.join()
        seen["resizable_after"] = resizable(ba)

    run(go)
    assert seen["locked"], "the default must hold the buffer (BufferError on resize)"
    assert seen["data"][:4] == b"ZZZZ", (
        "the default must read the caller's memory in place"
    )
    assert seen["data"][4:] == original[4:]
    assert seen["resizable_after"], (
        "the export must be released when the awaitable completed"
    )


def test_put_bytes_async_default_borrows_a_numpy_array(work_dir):
    original = np.arange(200_000, dtype=np.uint8)
    fifo = make_fifo(work_dir)
    arr = original.copy()
    seen = {}

    async def go():
        fut = s3dlio.put_bytes_async(f"file://{fifo}", arr)
        await asyncio.sleep(0.3)
        try:
            arr.resize(arr.size * 2, refcheck=True)
            seen["locked"] = False
        except ValueError:
            seen["locked"] = True  # an exported array cannot be resized
        arr[:3] = 7
        reader = FifoReader(fifo).start()
        await fut
        seen["data"] = reader.join()

    run(go)
    assert seen["locked"], "a borrowed numpy array must not be resizable while borrowed"
    assert seen["data"][:3] == bytes([7, 7, 7]), "numpy memory must be read in place"
    assert seen["data"][3:] == original[3:].tobytes()
    arr.resize(arr.size * 2, refcheck=True)  # released: resizing works again


def test_put_bytes_async_borrow_false_snapshots_a_writable_buffer(work_dir):
    original = bytes(range(251)) * 4000
    fifo = make_fifo(work_dir)
    ba = bytearray(original)
    seen = {}

    async def go():
        fut = s3dlio.put_bytes_async(f"file://{fifo}", ba, borrow=False)
        seen["resizable_at_once"] = resizable(ba)  # the snapshot released the export
        ba[:4] = b"ZZZZ"
        await asyncio.sleep(0.2)
        reader = FifoReader(fifo).start()
        await fut
        seen["data"] = reader.join()

    run(go)
    assert seen["resizable_at_once"], "borrow=False must not keep the buffer exported"
    assert seen["data"] == original, (
        "borrow=False must write the snapshot, not later changes"
    )


# --------------------------------------------------------------------------------------
# W9: an immutable source is not copied by borrow=False; a mutable read-only view still is
# --------------------------------------------------------------------------------------

CHILD_IMMUTABLE = r"""
import asyncio, json, os, sys
import s3dlio

uri, kind, mib = sys.argv[1], sys.argv[2], int(sys.argv[3])
size = mib * 1024 * 1024

class Sub(bytes):
    pass

if kind == "memoryview_of_bytes":
    payload = memoryview(bytes(b"\xa5") * size)
elif kind == "bytes_subclass":
    payload = Sub(bytes(b"\xa5") * size)
elif kind == "readonly_view_of_bytearray":
    payload = memoryview(bytearray(b"\xa5") * size).toreadonly()
elif kind == "bytearray":
    payload = bytearray(b"\xa5") * size
else:
    raise SystemExit("bad kind")

def rss_peak_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM not found")

with open("/proc/self/clear_refs", "w") as f:
    f.write("5")
base = rss_peak_kib()

async def go():
    await s3dlio.put_bytes_async(uri, payload, borrow=False)

asyncio.run(go())
print(json.dumps({"growth_kib": rss_peak_kib() - base, "size_kib": size // 1024}))
"""


def child_growth(work_dir, kind, mib=128):
    uri = f"file://{work_dir}/{kind}.bin"
    out = subprocess.run(
        [sys.executable, "-I", "-c", CHILD_IMMUTABLE, uri, kind, str(mib)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert os.path.getsize(f"{work_dir}/{kind}.bin") == mib * MIB
    return res["growth_kib"] / res["size_kib"]


@pytest.mark.parametrize("kind", ["memoryview_of_bytes", "bytes_subclass"])
def test_borrow_false_does_not_copy_an_immutable_source(work_dir, kind):
    growth = child_growth(work_dir, kind)
    assert growth < 0.25, (
        f"{kind}: borrow=False copied an immutable source (RSS +{growth:.2f}x)"
    )


@pytest.mark.parametrize("kind", ["readonly_view_of_bytearray", "bytearray"])
def test_borrow_false_still_snapshots_a_source_that_can_change(work_dir, kind):
    """A read-only VIEW of a bytearray is not immutable (the bytearray can change): it keeps the
    documented snapshot copy, so the opt-in keeps its meaning."""
    growth = child_growth(work_dir, kind)
    assert growth > 0.8, f"{kind}: borrow=False must snapshot (RSS +{growth:.2f}x)"


# --------------------------------------------------------------------------------------
# W8: put_many (blocking) holds the buffers; no copy
# --------------------------------------------------------------------------------------


def test_put_many_holds_every_buffer_and_makes_no_copy(work_dir):
    orig_a = bytes(range(251)) * 3000
    orig_b = np.arange(150_000, dtype=np.uint8)
    fa, fb = make_fifo(work_dir, "a"), make_fifo(work_dir, "b")
    ba = bytearray(orig_a)
    arr = orig_b.copy()
    seen = {}

    def call():
        try:
            s3dlio.put_many([(f"file://{fa}", ba), (f"file://{fb}", arr)])
        except BaseException as e:  # noqa: BLE001 - reported to the main thread
            seen["error"] = e

    t = threading.Thread(target=call, daemon=True)
    t.start()
    time.sleep(0.4)  # both native writers are stuck opening their FIFOs
    assert t.is_alive(), "put_many returned before its writers could finish"
    seen["locked_bytearray"] = not resizable(ba)
    try:
        arr.resize(arr.size * 2, refcheck=True)
        seen["locked_numpy"] = False
    except ValueError:
        seen["locked_numpy"] = True
    ba[:4] = b"ZZZZ"
    arr[:2] = 9
    ra, rb = FifoReader(fa).start(), FifoReader(fb).start()
    t.join(30)
    assert not t.is_alive()
    assert "error" not in seen, seen.get("error")
    data_a, data_b = ra.join(), rb.join()

    assert seen["locked_bytearray"] and seen["locked_numpy"]
    assert data_a[:4] == b"ZZZZ" and data_a[4:] == orig_a[4:], (
        "put_many copied the bytearray"
    )
    assert data_b[:2] == bytes([9, 9]) and data_b[2:] == orig_b[2:].tobytes()
    assert resizable(ba), "put_many must release every export before it returns"


def test_put_many_rejected_item_releases_the_buffers_already_taken(work_dir):
    ba = bytearray(b"abc" * 1000)
    uri_a = f"file://{work_dir}/a.bin"
    uri_b = f"file://{work_dir}/b.bin"
    with pytest.raises(TypeError, match="item 1"):
        s3dlio.put_many([(uri_a, ba), (uri_b, 12345)])
    assert resizable(ba), (
        "an earlier item's export must be released when a later item is bad"
    )
    assert not os.path.exists(f"{work_dir}/a.bin"), (
        "nothing may be sent when an item is bad"
    )


def test_put_many_roundtrip_and_empty(work_dir):
    items = [
        (f"file://{work_dir}/o{i}.bin", bytearray(bytes([i]) * (1000 + i)))
        for i in range(5)
    ]
    items.append((f"file://{work_dir}/e.bin", bytearray()))
    s3dlio.put_many(items)
    for i in range(5):
        assert slurp(f"{work_dir}/o{i}.bin") == bytes([i]) * (1000 + i)
    assert slurp(f"{work_dir}/e.bin") == b""
    s3dlio.put_many([])


CHILD_PUT_MANY = r"""
import json, sys
import s3dlio

d, mib, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
size = mib * 1024 * 1024
items = [(f"file://{d}/m{i}.bin", bytearray(b"\x5a") * size) for i in range(n)]

def rss_peak_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM not found")

with open("/proc/self/clear_refs", "w") as f:
    f.write("5")
base = rss_peak_kib()
s3dlio.put_many(items)
print(json.dumps({"growth_kib": rss_peak_kib() - base, "total_kib": size * n // 1024}))
"""


def test_put_many_peak_memory_stays_below_a_copy(work_dir):
    n, mib = 4, 64
    out = subprocess.run(
        [sys.executable, "-I", "-c", CHILD_PUT_MANY, work_dir, str(mib), str(n)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["growth_kib"] / res["total_kib"] < 0.25, (
        f"put_many copied its payloads: {res}"
    )
    for i in range(n):
        assert os.path.getsize(f"{work_dir}/m{i}.bin") == mib * MIB


# --------------------------------------------------------------------------------------
# W9: put_many_async
# --------------------------------------------------------------------------------------


def test_put_many_async_default_borrows(work_dir):
    original = bytes(range(251)) * 3000
    fifo = make_fifo(work_dir)
    ba = bytearray(original)
    seen = {}

    async def go():
        fut = s3dlio.put_many_async([(f"file://{fifo}", ba)])  # default borrow
        await asyncio.sleep(0.3)
        seen["locked"] = not resizable(ba)
        ba[:4] = b"ZZZZ"
        reader = FifoReader(fifo).start()
        await fut
        seen["data"] = reader.join()
        seen["resizable_after"] = resizable(ba)

    run(go)
    assert seen["locked"]
    assert seen["data"][:4] == b"ZZZZ" and seen["data"][4:] == original[4:]
    assert seen["resizable_after"]


def test_put_many_async_borrow_false_snapshots(work_dir):
    original = bytes(range(251)) * 3000
    fifo = make_fifo(work_dir)
    ba = bytearray(original)
    seen = {}

    async def go():
        fut = s3dlio.put_many_async([(f"file://{fifo}", ba)], borrow=False)
        seen["resizable_at_once"] = resizable(ba)
        ba[:4] = b"ZZZZ"
        await asyncio.sleep(0.2)
        reader = FifoReader(fifo).start()
        await fut
        seen["data"] = reader.join()

    run(go)
    assert seen["resizable_at_once"]
    assert seen["data"] == original


def test_put_many_async_roundtrip_mixed_payloads(work_dir):
    items = [
        (f"file://{work_dir}/x0.bin", b"plain bytes"),
        (f"file://{work_dir}/x1.bin", bytearray(b"bytearray" * 100)),
        (f"file://{work_dir}/x2.bin", np.arange(1000, dtype=np.uint8)),
        (f"file://{work_dir}/x3.bin", memoryview(b"mv of bytes")),
    ]

    async def go():
        await s3dlio.put_many_async(items)
        await s3dlio.put_many_async([], max_in_flight=4)

    run(go)
    assert slurp(f"{work_dir}/x0.bin") == b"plain bytes"
    assert slurp(f"{work_dir}/x1.bin") == b"bytearray" * 100
    assert slurp(f"{work_dir}/x2.bin") == np.arange(1000, dtype=np.uint8).tobytes()
    assert slurp(f"{work_dir}/x3.bin") == b"mv of bytes"


def test_put_many_async_rejected_item_releases_taken_buffers(work_dir):
    ba = bytearray(b"abc" * 1000)

    async def go():
        with pytest.raises(TypeError, match="item 1"):
            s3dlio.put_many_async(
                [(f"file://{work_dir}/a.bin", ba), (f"file://{work_dir}/b.bin", 12345)]
            )

    run(go)
    assert resizable(ba)


# --------------------------------------------------------------------------------------
# W9: MultiEndpointStore puts
# --------------------------------------------------------------------------------------


def endpoint_store(work_dir, n=2):
    dirs = []
    for i in range(n):
        d = os.path.join(work_dir, f"ep{i}")
        os.mkdir(d)
        os.mkfifo(os.path.join(d, "pipe"))
        dirs.append(d)
    store = s3dlio.create_multi_endpoint_store(
        uris=[f"file://{d}" for d in dirs], strategy="round_robin"
    )
    return store, dirs


@pytest.mark.parametrize("method", ["put", "put_to_endpoint", "put_all_endpoints"])
def test_multi_endpoint_puts_borrow_by_default(work_dir, method):
    original = bytes(range(251)) * 3000
    store, dirs = endpoint_store(work_dir)
    ba = bytearray(original)
    seen = {}
    fifos = [os.path.join(d, "pipe") for d in dirs]

    def start():
        if method == "put":
            return store.put("pipe", ba)
        if method == "put_to_endpoint":
            return store.put_to_endpoint(0, "pipe", ba)
        return store.put_all_endpoints("pipe", ba)

    async def go():
        fut = start()  # default borrow
        await asyncio.sleep(0.3)
        seen["locked"] = not resizable(ba)
        ba[:4] = b"ZZZZ"
        # `put` goes to whichever endpoint the balancer picked, `put_to_endpoint` to endpoint 0,
        # `put_all_endpoints` to both: attach a reader to every FIFO. A reader of a FIFO nobody
        # writes just waits (daemon thread); only the readers that finished are inspected.
        readers = [FifoReader(f).start() for f in fifos]
        await fut
        await asyncio.sleep(0.2)
        seen["data"] = [r.data for r in readers if r.data is not None]
        seen["resizable_after"] = resizable(ba)

    run(go)
    assert seen["locked"], (
        f"{method}: the buffer must be locked while the put is pending"
    )
    assert seen["data"], f"{method}: nothing was written"
    for data in seen["data"]:
        assert data[:4] == b"ZZZZ", (
            f"{method}: the payload was copied, not read in place"
        )
        assert data[4:] == original[4:]
    assert len(seen["data"]) == (2 if method == "put_all_endpoints" else 1)
    assert seen["resizable_after"]


@pytest.mark.parametrize("method", ["put", "put_to_endpoint", "put_all_endpoints"])
def test_multi_endpoint_puts_borrow_false_snapshot(work_dir, method):
    original = bytes(range(251)) * 3000
    store, dirs = endpoint_store(work_dir)
    ba = bytearray(original)
    fifos = [os.path.join(d, "pipe") for d in dirs]
    seen = {}

    def start():
        if method == "put":
            return store.put("pipe", ba, borrow=False)
        if method == "put_to_endpoint":
            return store.put_to_endpoint(0, "pipe", ba, borrow=False)
        return store.put_all_endpoints("pipe", ba, borrow=False)

    async def go():
        fut = start()
        seen["resizable_at_once"] = resizable(ba)
        ba[:4] = b"ZZZZ"
        await asyncio.sleep(0.2)
        readers = [FifoReader(f).start() for f in fifos]
        await fut
        await asyncio.sleep(0.2)
        seen["data"] = [r.data for r in readers if r.data is not None]

    run(go)
    assert seen["resizable_at_once"]
    assert seen["data"]
    for data in seen["data"]:
        assert data == original


# --------------------------------------------------------------------------------------
# W7: the wrappers pass the caller's memory on
# --------------------------------------------------------------------------------------


@pytest.fixture
def torch_stub(monkeypatch):
    """`torch` itself, or a minimal stand-in when it is not installed (the wrappers under test
    import `torch.utils.data` at module level and use nothing else of it on these paths)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        fake = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")
        data.Dataset = type("Dataset", (), {})
        data.IterableDataset = type("IterableDataset", (), {})
        data.get_worker_info = lambda: None
        fake.utils = utils
        fake.frombuffer = lambda buf, dtype=None: np.frombuffer(buf, dtype=np.uint8)
        fake.uint8 = np.uint8
        utils.data = data
        monkeypatch.setitem(sys.modules, "torch", fake)
        monkeypatch.setitem(sys.modules, "torch.utils", utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", data)
        for name in ("s3dlio.torch", "s3dlio.compat.s3torchconnector"):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_compat_checkpoint_writer_does_not_copy_on_close(torch_stub, monkeypatch):
    from s3dlio.compat import s3torchconnector as stc

    seen = {}

    def fake_put_bytes(uri, data):
        seen["uri"] = uri
        seen["type"] = type(data)
        seen["obj"] = type(data.obj).__name__ if isinstance(data, memoryview) else None
        seen["bytes"] = bytes(data)

    monkeypatch.setattr(stc._core, "put_bytes", fake_put_bytes)
    writer = stc.S3Checkpoint(region="us-east-1").writer("s3://bucket/ckpt.pt")
    writer.write(b"abc" * 1000)
    writer.close()
    assert seen["type"] is memoryview, (
        "the BytesIO memory must be passed as a buffer, not bytes()"
    )
    assert seen["obj"] == "_BytesIOBuffer", "not a view of the BytesIO's own memory"
    assert seen["bytes"] == b"abc" * 1000


def test_compat_put_object_writer_does_not_copy_on_close(torch_stub, monkeypatch):
    from s3dlio.compat import s3torchconnector as stc

    seen = {}

    def fake_put_bytes(uri, data):
        seen["type"] = type(data)
        seen["obj"] = type(data.obj).__name__ if isinstance(data, memoryview) else None
        seen["bytes"] = bytes(data)

    monkeypatch.setattr(stc._core, "put_bytes", fake_put_bytes)
    client = stc.S3Client(region="us-east-1")
    writer = client.put_object("bucket", "key")
    writer.write(b"xyz" * 500)
    writer.close()
    assert seen["type"] is memoryview, "read() copies; getbuffer() does not"
    assert seen["obj"] == "_BytesIOBuffer", "not a view of the BytesIO's own memory"
    assert seen["bytes"] == b"xyz" * 500


def test_compat_writers_roundtrip_through_file_uri(torch_stub, work_dir):
    from s3dlio.compat import s3torchconnector as stc

    path = f"{work_dir}/w7.bin"
    payload = bytes(range(251)) * 1000
    with stc.S3Checkpoint(region="us-east-1").writer(f"file://{path}") as w:
        w.write(payload)
    assert slurp(path) == payload
    path2 = f"{work_dir}/w7b.bin"
    wr = stc.S3Client(region="us-east-1").put_object("unused", f"file://{path2}")
    wr.write(payload)
    wr.close()
    assert slurp(path2) == payload


class _Recorder:
    """Stands in for MultipartUploadWriter: records what write() was given."""

    def __init__(self):
        self.chunks = []
        self.closed = False

    def write(self, chunk):
        self.chunks.append(chunk)

    def close(self):
        self.closed = True

    def abort(self):
        raise AssertionError("abort must not be called on success")


def _storage_cases():
    s1 = object.__new__(s3dlio_storage.S3dlioStorage)
    s1.prefix = "s3://bucket"
    s2 = object.__new__(s3_torch_storage.S3PyTorchConnectorStorage)
    return [
        ("s3dlio_storage", s1, "s3://bucket/k"),
        ("s3_torch_storage", s2, "s3://bucket/k"),
    ]


@pytest.mark.parametrize("which", [0, 1])
def test_dlio_multipart_slices_are_views_not_bytes_copies(monkeypatch, which):
    name, store, uri = _storage_cases()[which]
    monkeypatch.setenv("S3DLIO_MULTIPART_THRESHOLD_MB", "0")
    monkeypatch.setenv("S3DLIO_MULTIPART_PART_SIZE_MB", "1")
    content = bytes(range(251)) * (5 * MIB // 251 + 1)
    content = content[: 3 * MIB + 17]
    rec = _Recorder()
    monkeypatch.setattr(
        s3dlio, "put_bytes", lambda *a, **k: pytest.fail("single PUT used")
    )
    monkeypatch.setattr(
        s3dlio,
        "MultipartUploadWriter",
        types.SimpleNamespace(from_uri=lambda *a, **k: rec),
    )
    store.put_data(uri, content)
    assert rec.closed
    assert len(rec.chunks) == 4
    for chunk in rec.chunks:
        assert isinstance(chunk, memoryview), (
            f"{name}: a part was copied into {type(chunk)}"
        )
        assert chunk.obj is content, (
            f"{name}: a part is not a view of the caller's bytes"
        )
    assert b"".join(bytes(c) for c in rec.chunks) == content


@pytest.mark.parametrize("which", [0, 1])
def test_dlio_bytesio_is_passed_as_a_buffer_not_a_getvalue_copy(monkeypatch, which):
    import io

    name, store, uri = _storage_cases()[which]
    monkeypatch.setenv("S3DLIO_MULTIPART_THRESHOLD_MB", "1024")
    bio = io.BytesIO()
    bio.write(b"npy-bytes" * 1000)
    seen = {}

    def fake_put_bytes(u, data):
        seen["type"] = type(data)
        seen["obj"] = type(data.obj).__name__ if isinstance(data, memoryview) else None
        try:
            bio.write(b"resize while the buffer is passed on")
            seen["locked"] = False
        except BufferError:
            seen["locked"] = True  # the BytesIO's own memory is exported
        seen["bytes"] = bytes(data)

    monkeypatch.setattr(s3dlio, "put_bytes", fake_put_bytes)
    store.put_data(uri, bio)
    assert seen["type"] is memoryview, f"{name}: getvalue() copied the BytesIO"
    assert seen["obj"] == "_BytesIOBuffer", "not a view of the BytesIO's own memory"
    assert seen["locked"], f"{name}: the memoryview is not over the BytesIO's memory"
    assert seen["bytes"] == b"npy-bytes" * 1000
    # The temporary view is released: the BytesIO is resizable again.
    bio.write(b"more")
    assert bio.getvalue().endswith(b"more")


@pytest.mark.parametrize("which", [0, 1])
def test_dlio_put_data_writes_a_file_uri(work_dir, monkeypatch, which):
    import io

    _name, store, _uri = _storage_cases()[which]
    path = f"{work_dir}/dlio{which}.bin"
    payload = bytes(range(251)) * 100
    monkeypatch.setenv("S3DLIO_MULTIPART_THRESHOLD_MB", "1024")
    store.put_data(f"file://{path}", io.BytesIO(payload))
    assert slurp(path) == payload


@pytest.fixture
def s3torch(torch_stub):
    """`s3dlio.torch` (with the torch stand-in when torch is not installed)."""
    import s3dlio.torch as module

    return module


def test_map_dataset_reader_return_type_does_not_copy_the_bytesview(work_dir, s3torch):
    """return_type='reader' used to build `_BytesReader(bytes(bytesview))`, a full copy; the
    reader wraps any buffer in a memoryview, so the BytesView itself is passed."""
    data = bytes(range(251)) * 100
    with open(f"{work_dir}/obj0.bin", "wb") as f:
        f.write(data)
    ds = s3torch.ObjectStoreMapDataset(f"file://{work_dir}/", return_type="reader")
    reader = ds[0]
    assert isinstance(reader._buf.obj, s3dlio.BytesView), (
        f"the reader copied the payload into {type(reader._buf.obj)}"
    )
    assert reader.read(10) == data[:10]
    assert reader.read() == data[10:]
    # return_type='bytes' explicitly returns bytes (a documented copy) and still works.
    as_bytes = s3torch.ObjectStoreMapDataset(
        f"file://{work_dir}/", return_type="bytes"
    )[0]
    assert as_bytes == data


# --------------------------------------------------------------------------------------
# Fix round, item 1: a failed or cancelled batch stops the rest of the batch
# --------------------------------------------------------------------------------------


def _batch_with_early_failure(work_dir, n=40, mib=4):
    """First item cannot be written (its parent is a regular file); the next `n` are fine."""
    blocker = os.path.join(work_dir, "afile")
    with open(blocker, "wb") as f:
        f.write(b"x")
    items = [(f"file://{blocker}/bad.bin", np.zeros(MIB, dtype=np.uint8))]
    items += [
        (f"file://{work_dir}/ok{i}.bin", np.full(mib * MIB, i % 251, dtype=np.uint8))
        for i in range(n)
    ]
    return items


def _written(work_dir, n=40):
    return [i for i in range(n) if os.path.exists(os.path.join(work_dir, f"ok{i}.bin"))]


def _is_real_error(exc):
    still = getattr(s3dlio, "ExportStillReferencedError", ())
    assert not isinstance(exc, still), f"the real error was replaced by {exc!r}"
    text = str(exc).lower()
    assert "file exists" in text or "director" in text, (
        f"not the item's own error: {exc!r}"
    )


def test_put_many_early_failure_raises_the_real_error_and_stops_the_rest(work_dir):
    items = _batch_with_early_failure(work_dir)
    with pytest.raises(Exception) as caught:
        s3dlio.put_many(items, max_in_flight=1)
    _is_real_error(caught.value)
    written = _written(work_dir)
    assert len(written) <= 3, (
        f"the batch kept uploading after its failure: {len(written)} items"
    )


def test_put_many_async_early_failure_raises_the_real_error_and_stops_the_rest(
    work_dir,
):
    items = _batch_with_early_failure(work_dir)

    async def go():
        with pytest.raises(Exception) as caught:
            await s3dlio.put_many_async(items, max_in_flight=1)
        return caught.value

    exc = run(go)
    _is_real_error(exc)
    written = _written(work_dir)
    assert len(written) <= 3, (
        f"the batch kept uploading after its failure: {len(written)} items"
    )


def test_cancelled_put_many_async_stops_the_remaining_uploads(work_dir):
    fifo = make_fifo(work_dir)
    ba = bytearray(b"f" * 4000)
    n = 20
    items = [(f"file://{fifo}", ba)] + [
        (f"file://{work_dir}/ok{i}.bin", bytearray(b"y" * MIB)) for i in range(n)
    ]
    seen = {}

    async def go():
        fut = s3dlio.put_many_async(items, max_in_flight=1)
        await asyncio.sleep(
            0.3
        )  # item 0 is stuck opening the FIFO; the others wait for it
        fut.cancel()
        reader = FifoReader(fifo).start()  # lets the in-flight write finish
        done, _ = await asyncio.wait({fut}, timeout=30)
        seen["done"] = bool(done)
        seen["cancelled"] = fut.cancelled()
        reader.join()

    run(go)
    assert seen["done"] and seen["cancelled"]
    written = _written(work_dir, n)
    assert len(written) <= 2, (
        f"a cancelled batch kept uploading: {len(written)} of {n} items"
    )
    assert resizable(ba), "the cancelled batch must release its exports"


# --------------------------------------------------------------------------------------
# Fix round, item 4: the compat writers keep the data when an upload fails
# --------------------------------------------------------------------------------------


def _checkpoint_writer(stc):
    return stc.S3Checkpoint(region="us-east-1").writer("s3://bucket/ckpt.pt")


def _put_object_writer(stc):
    return stc.S3Client(region="us-east-1").put_object("bucket", "key")


@pytest.mark.parametrize("make", [_checkpoint_writer, _put_object_writer])
def test_compat_writer_failed_upload_can_be_retried(torch_stub, monkeypatch, make):
    from s3dlio.compat import s3torchconnector as stc

    calls = {"n": 0, "payloads": []}

    def fake_put_bytes(uri, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient upload failure")
        calls["payloads"].append(bytes(data))

    monkeypatch.setattr(stc._core, "put_bytes", fake_put_bytes)
    writer = make(stc)
    writer.write(b"checkpoint" * 1000)
    with pytest.raises(RuntimeError, match="transient upload failure"):
        writer.close()
    writer.close()  # the data was not destroyed by the failure: the retry uploads it
    assert calls["payloads"] == [b"checkpoint" * 1000]


@pytest.mark.parametrize("make", [_checkpoint_writer, _put_object_writer])
def test_compat_writer_real_exception_survives_a_still_exported_buffer(
    torch_stub, monkeypatch, make
):
    """put_bytes can fail while something still holds an export of the buffer (for example
    ExportStillReferencedError): releasing the view then raises BufferError, which must neither
    mask the real exception nor skip the cleanup."""
    from s3dlio.compat import s3torchconnector as stc

    held = []

    def fake_put_bytes(uri, data):
        held.append(
            memoryview(data)
        )  # an extra export, still alive when close() releases
        raise RuntimeError("the real failure")

    monkeypatch.setattr(stc._core, "put_bytes", fake_put_bytes)
    writer = make(stc)
    writer.write(b"payload" * 100)
    with pytest.raises(RuntimeError, match="the real failure"):
        writer.close()
    for view in held:
        view.release()


# --------------------------------------------------------------------------------------
# Fix round 3: ordered admission and a huge max_in_flight
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [2**62, 2**63, 2**64 - 1])
def test_put_many_with_a_huge_max_in_flight_is_clamped_not_a_crash(work_dir, limit):
    items = [
        (f"file://{work_dir}/h{i}.bin", bytearray(bytes([i]) * 1000)) for i in range(5)
    ]
    s3dlio.put_many(items, max_in_flight=limit)
    for i in range(5):
        assert slurp(f"{work_dir}/h{i}.bin") == bytes([i]) * 1000


@pytest.mark.parametrize("limit", [2**62, 2**63, 2**64 - 1])
def test_put_many_async_with_a_huge_max_in_flight_is_clamped_not_a_crash(
    work_dir, limit
):
    items = [
        (f"file://{work_dir}/h{i}.bin", bytearray(bytes([i]) * 1000)) for i in range(5)
    ]

    async def go():
        await s3dlio.put_many_async(items, max_in_flight=limit)

    run(go)
    for i in range(5):
        assert slurp(f"{work_dir}/h{i}.bin") == bytes([i]) * 1000
