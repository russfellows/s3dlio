# s3dlio Python API Guide

**Version:** 0.10.0

**Last Updated:** October 18, 2026

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [S3 Connection Configuration](#s3-connection-configuration)
4. [Architecture](#architecture)
4. [Core Storage Operations](#core-storage-operations)
5. [Zero-Copy Data Flow](#zero-copy-data-flow)
6. [Batch Operations](#batch-operations)
7. [Multi-Backend Support](#multi-backend-support)
8. [Multi-Endpoint Load Balancing](#multi-endpoint-load-balancing)
9. [Streaming API](#streaming-api)
10. [AI/ML Integration](#aiml-integration)
11. [**Object Data Loader (General-Purpose)**](#object-data-loader-general-purpose) ✨ New in v0.9.100
12. [**Parquet DataLoader**](#parquet-dataloader) ✨ New in v0.9.98
13. [Data Generation (NPZ / NPY)](#data-generation-npz--npy)
14. [s3torchconnector Compatibility](#s3torchconnector-compatibility)
15. [Checkpoint System](#checkpoint-system)
16. [Performance & Threading](#performance--threading)
17. [Advanced Features](#advanced-features)
18. [API Reference](#api-reference)
19. [Migration Guide](#migration-guide)

---

## Installation

### From Wheel (Recommended)

```bash
# Build and install from source
cd s3dlio
./build_pyo3.sh
./install_pyo3_wheel.sh

# Verify installation
python -c "import s3dlio; print(s3dlio.__version__)"
```

### From PyPI

```bash
pip install s3dlio
```

### Requirements

- Python 3.11+ (release validation targets CPython 3.11–3.13)
- Rust 1.94.1+ (for building from source)
- Optional: PyTorch, JAX, or TensorFlow for ML integration

---

## Quick Start

```python
import s3dlio

# Initialize logging (optional)
s3dlio.init_logging("info")  # Options: trace, debug, info, warn, error

# Put data to storage — works with S3, Azure, GCS, local filesystem
s3dlio.put_bytes("s3://my-bucket/data.bin", b"Hello, World!")
s3dlio.put_bytes("file:///tmp/local.bin", b"Local data")

# Get data — returns BytesView (zero-copy from Rust memory)
data = s3dlio.get("s3://my-bucket/data.bin")
print(len(data))       # 13
print(bytes(data))     # b'Hello, World!'

# Range request — server-side, only fetches needed bytes
chunk = s3dlio.get_range("s3://my-bucket/data.bin", offset=0, length=5)

# List objects — returns full URIs
objects = s3dlio.list("s3://my-bucket/prefix/")

# Metadata
metadata = s3dlio.stat("s3://my-bucket/data.bin")
print(f"Size: {metadata['size']} bytes")

# Check existence
if s3dlio.exists("s3://my-bucket/data.bin"):
    s3dlio.delete("s3://my-bucket/data.bin")
```

**Supported URI Schemes:**

| Scheme | Example | Description |
|--------|---------|-------------|
| `s3://` | `s3://bucket/key` | Amazon S3, MinIO, Ceph, VAST |
| `gs://` | `gs://bucket/key` | Google Cloud Storage |
| `az://` | `az://account/container/key` | Azure Blob Storage |
| `file://` | `file:///path/to/file` | Local filesystem |
| `direct://` | `direct:///path/to/file` | Direct I/O (O_DIRECT) — bypasses the OS page cache. **v0.9.95:** correctly routed through `ConfigurableFileSystemObjectStore::with_direct_io()` in `get_many()` (previously silently used buffered I/O). |

---

## S3 Connection Configuration

`configure_s3()` is the Python equivalent of the CLI's `--endpoint-url`, `--region`, and
`--ca-bundle` global flags.  It sets the corresponding environment variables **and** clears
the internal store cache so the next operation creates a fresh connection with the new
settings.

### Signature

```python
s3dlio.configure_s3(
    endpoint_url: str | None = None,
    region:       str | None = None,
    ca_bundle:    str | None = None,
) -> None
```

| Parameter | Env var set | CLI equivalent | Description |
|-----------|-------------|----------------|-------------|
| `endpoint_url` | `AWS_ENDPOINT_URL` | `--endpoint-url` | Full URL of the S3-compatible server, e.g. `"https://minio.corp:9000"` |
| `region` | `AWS_DEFAULT_REGION` | `--region` | AWS region name, e.g. `"us-east-1"` |
| `ca_bundle` | `AWS_CA_BUNDLE` | `--ca-bundle` | Filesystem path to a PEM CA certificate bundle for TLS |

### Example

```python
import s3dlio

# Must be called BEFORE the first S3 operation (see critical note below)
s3dlio.configure_s3(
    endpoint_url="https://172.16.1.40:9000",
    region="us-east-1",
    ca_bundle="/etc/ssl/certs/my-ca.pem",
)

# All S3 operations now use the configured endpoint:
keys  = s3dlio.list("s3://my-bucket/", recursive=False)
data  = s3dlio.get("s3://my-bucket/file.bin")
s3dlio.put_bytes("s3://my-bucket/out.bin", data)
info  = s3dlio.stat("s3://my-bucket/file.bin")
s3dlio.delete("s3://my-bucket/file.bin")
```

### Scope

`configure_s3()` applies to **all** S3 operations: `list`, `get`, `get_range`, `get_many`,
`put_bytes`, `put_many`, `stat`, `exists`, `delete`, `upload`, `download`, `mp_get`,
`create_bucket`, `delete_bucket`, and all async variants.  It does **not** affect Azure or
GCS backends (those use their own environment variables — see [Authentication](#authentication)).

### ⚠️ Critical: Call Before the First S3 Operation

The underlying AWS SDK client is a **process-global singleton** (`OnceCell`) that is
initialised exactly once — on the first S3 call — and **cannot be reconfigured afterwards**.

```python
import s3dlio

# ✅ CORRECT — configure before any S3 operation
s3dlio.configure_s3(endpoint_url="https://minio:9000", region="us-east-1")
data = s3dlio.get("s3://bucket/key")   # uses the configured endpoint

# ❌ WRONG — SDK client already initialised; configure_s3() has no effect
data = s3dlio.get("s3://bucket/key")   # initialises client with default settings
s3dlio.configure_s3(endpoint_url="https://minio:9000")  # too late — ignored
```

`configure_s3()` **does** clear the store cache (`STORE_CACHE`), so if you only need to
switch endpoints between operations (and the AWS SDK client settings are compatible with
both), calling it mid-script will cause the next operation to build a new store against
the new endpoint.

### Relationship to Environment Variables

`configure_s3()` is syntactic sugar for `os.environ` assignments plus a cache flush.  Both
approaches are equivalent:

```python
# These two blocks are identical in effect:

# Option A — configure_s3()
s3dlio.configure_s3(endpoint_url="https://minio:9000", region="us-east-1")

# Option B — environment variables directly
import os
os.environ["AWS_ENDPOINT_URL"]    = "https://minio:9000"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
# (no cache flush needed if called before any S3 op)
```

Prefer `configure_s3()` when you want the cache flush guarantee.

---

## Architecture

### Runtime Model (v0.9.50)

s3dlio uses an **io_uring-style submit pattern** for all Python API calls:

```
Python Thread → spawn(async work) → channel.recv() → result
```

1. **SUBMIT**: The calling thread spawns the async future onto a dedicated global Tokio runtime
2. **PROCESS**: Runtime worker threads handle the async I/O
3. **COMPLETE**: Result flows back through `std::sync::mpsc` channel

**This design is fully thread-safe.** You can call any s3dlio function from:
- Python `ThreadPoolExecutor` (16, 64, 128+ threads)
- PyTorch `DataLoader` worker processes
- Any plain OS thread

The calling thread blocks on channel recv (NOT on `block_on`), so there are **no Tokio runtime conflicts**.

### Global Client Cache

s3dlio maintains a process-global `DashMap<StoreKey, Arc<dyn ObjectStore>>` cache:

- **Key**: `(scheme, endpoint, region)` — NOT bucket-specific
- **Hit rate**: >99% in typical workloads (<100ns lookup)
- **Thread-safe**: Lock-free concurrent read/write (DashMap sharded locking)
- **Automatic**: No manual client passing — first call creates, all subsequent calls reuse

```python
# These all reuse the SAME underlying Rust ObjectStore client:
s3dlio.put_bytes("s3://bucket-a/key1", data1)  # Creates client
s3dlio.put_bytes("s3://bucket-b/key2", data2)  # Reuses client (same endpoint)
s3dlio.get("s3://bucket-c/key3")               # Reuses client
```

### Configuration

```bash
# Control Tokio worker thread count (default: num_cpus)
export S3DLIO_WORKER_THREADS=16
```

---

## Core Storage Operations

### put_bytes() — Upload Data

```python
# Upload bytes to any backend
s3dlio.put_bytes("s3://bucket/key", b"data")

# Upload from file
with open("local_file.bin", "rb") as f:
    s3dlio.put_bytes("s3://bucket/key", f.read())

# Async version (for use with asyncio)
await s3dlio.put_bytes_async("s3://bucket/key", b"data")
```

**What is copied** (`put_bytes` / `put_bytes_async`), by input type. "Copies" counts copies made *by s3dlio*
between your memory and the storage client:

| Input | `put_bytes` | `put_bytes_async` (default) | `put_bytes_async(..., borrow=True)` |
|---|---|---|---|
| `s3dlio.BytesView`, exact `bytes` | 0 | 0 | 0 |
| any other **C-contiguous** buffer, any item type (`bytearray`, `memoryview`, numpy `uint8`/`float32`/`bool`/..., `array.array`, ctypes arrays, `mmap`) | 0 | 1 (a snapshot, taken before the call returns, GIL released) | 0 |
| non-contiguous view (strided, reversed, transposed) | 1 (gathered in logical order by Python, GIL held) | 1 | 1 |
| object with only `__bytes__()` | 1 (made by Python) | 1 | 1 |

The buffer is read **in place with the GIL released**, so other Python threads keep running during a large
upload (a 512 MiB PUT used to stall every other thread for about 0.3 s).

Per backend, after that: `s3://`, `az://`, `gs://` (official client) and `file://` hand the same bytes to the
storage client / `write` with no further s3dlio copy (what the SDKs do internally is outside s3dlio; the legacy
community GCS client, feature `gcs-community`, copies once). `direct://` (O_DIRECT) writes a **page-aligned**
source (address a multiple of 4096) straight from your memory in one `write` per at most 2 GiB; any other source
is copied **once** through a warm, reused, page-aligned bounce buffer (16 MiB by default, 4 pairs per NUMA node,
shared by PUTs and GETs, ping-pong so the copy of the next chunk overlaps the write of the current one; see
`docs/Environment_Variables.md`). The last partial block of a PUT is always bounced (a zero-padded 4 KiB block).
`direct://` GET returns page-aligned memory (by default read straight into one fresh aligned buffer), so for
objects of at least `min_io_size` (4 KiB) on a store with the range engine off (the default),
`s3dlio.get("direct://...")` followed by `put_bytes("direct://...", data)` needs no bounce copy of the body
(only that final padded block). Smaller objects use buffered I/O.

**Contract for buffers that are read in place** (`put_bytes`, and `put_bytes_async(borrow=True)` until the
awaitable completes):

* the source is *exported*: resizing it from another thread (`bytearray.extend()` / `clear()` / `del ba[:]`)
  raises `BufferError` until the call (or awaitable) is finished;
* you must **not modify the bytes** from another thread (or other code sharing the memory) meanwhile: nothing
  prevents it, and it is a **data race**: the bytes uploaded are undefined (and the behaviour is undefined in the
  Rust sense). The memory is never freed or resized meanwhile because the buffer stays exported, but that does not
  make a concurrent write safe;
* with `borrow=True` you additionally promise not to free the memory (for example ctypes/pinned host memory) until
  the awaitable has completed, successfully or not;
* **completion is a safe-reuse point, cancellation included, on every backend (exact statement).** The awaitable
  of `put_bytes_async(..., borrow=True)` is a `Future` that **never reaches a terminal state - result, exception
  or cancelled - while any native reader of the buffer remains or before the export has actually been released**
  (`PyBuffer_Release` has run, so a resize no longer raises `BufferError`). A native reader is any clone of the
  payload, wherever it lives: a store task, an SDK/HTTP task, a detached file write. The supported consumer
  operations cannot change that: `await`, `fut.cancel()`, `Task.cancel()` (once or repeatedly), `asyncio.timeout()`
  and `asyncio.wait_for()` all leave the awaiting task pending until the future ends - with one Python 3.11
  exception for `wait_for`, described below (#288). The guarantee covers
  those consumer operations only: the future belongs to s3dlio, so calling its producer methods
  (`set_result`, `set_exception`, or the base-class `asyncio.Future.set_result` / `cancel`) is unsupported and ends
  it at once, whether or not a native reader remains. `cancel()` cancels the put, but a running native write cannot
  be stopped, so the cancellation (or the timeout) reaches you only AFTER that write finished and the export was
  released. Once the `await` finishes - with a result, an error, `CancelledError` or `TimeoutError` - the buffer
  can be resized or reused. The same holds when the barrier machinery itself fails (an internal error, for
  example a callback that cannot be registered): the future then falls back to polling every 50 ms and still
  completes only after the release; it is counted in `_put_export_stats()["internal_errors"]`. The blocking
  `put_bytes` returns or raises only after the same condition;
* **a backend that never lets go keeps the awaitable pending - by design.** Every
  `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (default 30 s) that a native reader remains, s3dlio emits one
  `RuntimeWarning` (and a log line) that names the timeout, adds one to `_put_export_stats()["quiesce_timeouts"]`
  and keeps waiting. It does **not** complete, because an exception cannot carry the "do not reuse" warning past
  `Task` cancellation (an awaiting task whose cancellation was refused turns any exception into a plain
  `CancelledError`/`TimeoutError`). If your code must bound the wait, use the counter or the warning to detect it
  and treat the buffer as owned by s3dlio until the future ends; do not wrap it in `asyncio.shield`;
* **blocking `put_bytes` keeps `s3dlio.ExportStillReferencedError`.** Its wait is bounded by the same timeout and is
  not interruptible by signals / `KeyboardInterrupt` (the buffer must not be released while it can still be read).
  If a backend still holds the payload after that, the call raises `ExportStillReferencedError` (data may or may
  not have been written; the message names the timeout and what the put itself did, including the text of the put's own error if it failed). The exception is raised directly in the calling thread's own stack: no Future or Task
  stands between, so asyncio cancellation cannot rewrite it. The buffer stays pinned (a resize raises
  `BufferError`): **do not reuse it** after this error. The same class is raised by `put_bytes_async` itself, synchronously, if the put could
  not be started and the readers did not let go in time;
* **await the future itself.** `asyncio.shield(fut)` bypasses the guard: the shield's outer future is cancelled
  at once while the put still reads the buffer, and nothing waits for the barrier. Do not shield a borrowed put
  (if you must, await `fut` again after the cancellation before touching the buffer). `asyncio.wait_for(fut, t)` and
  `asyncio.timeout(t)` can take up to `t` plus however long a running native write needs;
* **`asyncio.wait_for` after the put itself finished (Python-version difference).** If the deadline expires after
  the put has finished but before its export is released, the cancellation is refused (the outcome is already
  fixed) and the future completes with the put's own result once the export is released. What `wait_for` then
  reports is asyncio's behaviour: on Python 3.12 and later it raises `TimeoutError`; on Python 3.11 it returns the
  result. In both cases the future itself completes only after the release. The same applies to `get_into_async`
  and `head_async`, which use the same guard;
* **Python 3.11 only: do not cancel a task that is inside `asyncio.wait_for` (#288).** In 3.11, once `wait_for`'s
  deadline has fired (or it has been cancelled once), it cancels the s3dlio future and waits for it in a way that a
  further `Task.cancel()` interrupts. The awaiting task then ends while the future is still pending and s3dlio may
  still read (or, for `get_into_async`, write) the buffer, so the buffer is **not** safe to reuse at that point
  (for `head_async`, the request may still be running). Python 3.12 and 3.13 (tested) are not affected. On 3.11,
  use `asyncio.timeout()` instead of `wait_for` (it has no such gap, however many times the task is cancelled), or,
  if a `wait_for`-wrapped task can be cancelled, keep a reference to the s3dlio future and wait for it in a loop
  that a further cancellation cannot cut short before touching the buffer:

  ```python
  async def wait_for_guard(fut):
      """Wait until the s3dlio future has ended, then re-raise any cancellation received meanwhile."""
      cancelled = False
      while not fut.done():
          try:
              await asyncio.wait({fut})
          except asyncio.CancelledError:
              cancelled = True
      if cancelled:
          raise asyncio.CancelledError
  ```

  Plain `await`, `asyncio.timeout()`, and a `wait_for` that is cancelled at most once and before its deadline
  fires are safe on every supported version;
* buffers that hold Python objects or pointers are refused with a `TypeError`: their bytes are addresses, not
  data. Since v0.10.0 (#282) this covers every pointer-bearing format code anywhere in the buffer format: `O`, `P`,
  `&` (ctypes `POINTER(T)`), `z` / `Z` (`c_char_p` / `c_wchar_p`) and `X` (function pointers), on every write
  path. NumPy complex formats (`Zf`, `Zd`, `Zg`) are accepted. A structured numpy dtype with padding is uploaded as its raw memory, padding bytes
  included (they are whatever was in memory).

The default `put_bytes_async` takes a snapshot instead, so the buffer can be reused or resized as soon as the call
returns. The one obligation: do not write to it from another thread *during that copy*. Up to 256 KiB the copy runs
with the GIL held, so Python threads cannot interleave; above that it runs with the GIL released (so large copies do
not stall other threads), and a Python thread, a C extension or a device that writes to the buffer at that moment is
a **data race** with the copy (the snapshot is then undefined, and the behaviour is undefined in the Rust sense).
Multi-threaded callers that mutate shared buffers must therefore serialise the call with their writers.
`borrow` is keyword-only: `await s3dlio.put_bytes_async(uri, buf, borrow=True)`.

### get() — Download Data

```python
# Returns BytesView — zero-copy wrapper around Rust Bytes
data = s3dlio.get("s3://bucket/key")

# BytesView supports Python buffer protocol:
mv = memoryview(data)                          # Zero-copy memoryview
arr = numpy.frombuffer(data, dtype=np.uint8)   # Zero-copy NumPy array
raw = bytes(data)                              # Creates a copy (only if needed)
print(len(data))                               # Size in bytes
```

### get_range() — Server-Side Range Request

```python
# Fetch only bytes 1024-5119 from the server (saves bandwidth)
chunk = s3dlio.get_range("s3://bucket/key", offset=1024, length=4096)

# Fetch from offset to end of object
tail = s3dlio.get_range("s3://bucket/key", offset=1024)
```

### get_into_async() / head_async() — Read Into Your Own Buffer ✨ New in v0.10.0

`get_into_async` reads a whole object into memory you own (a `bytearray`, a NumPy `uint8`
array, an `mmap`, a ctypes byte array, a writable `memoryview`, ...). Data is written into the
destination as it arrives: there is no whole-object staging buffer and no copy afterwards.
`head_async` returns the object's size and version identity without reading it.
`object_capabilities` says, from the URI and the build alone, whether both are available.

```python
import asyncio
import numpy as np
import s3dlio

async def load(uri):
    meta = await s3dlio.head_async(uri)        # ObjectMetadata or None (absent)
    if meta is None:
        raise FileNotFoundError(uri)
    dst = np.empty(meta.size, dtype=np.uint8)  # any writable, C-contiguous byte buffer
    n = await s3dlio.get_into_async(uri, dst, expected_len=meta.size)
    assert n == meta.size
    return dst

s3dlio.object_capabilities("az://account/container/key")
# {'provider': 'az', 'get_into': True, 'head': True}
```

| Call | Result |
|---|---|
| `object_capabilities(uri)` | `{"provider": "s3"\|"az"\|"gs"\|"file"\|"direct"\|"unknown", "get_into": bool, "head": bool}`. No store, credentials or network. `ValueError` for a malformed URI in a known object scheme; unknown schemes return `"unknown"` and both operations false. |
| `get_into_async(uri, dst, *, expected_len)` | Awaitable resolving to `expected_len` once `dst[:expected_len]` holds the whole object. |
| `head_async(uri)` | Awaitable resolving to `ObjectMetadata(size, etag, version_id)`, or `None` when the object does not exist. |

**Errors.**
- `ObjectNotFoundError` (a `FileNotFoundError`): `get_into_async` on a missing object.
- `LengthMismatchError` (a `ValueError`, with `.expected` and `.actual`, `None` if unknown):
  the object's size is not `expected_len`. Nothing is written past `expected_len`.
- `UnsupportedOperationError` (a `NotImplementedError`): raised at once when the build or
  provider lacks the operation. No fallback download is attempted.
- `ValueError`: a URI that is malformed for its scheme (checked before anything else).
- A refused buffer raises `BufferError` (read-only or not C-contiguous) or `TypeError` (item size or
  format); an `expected_len` that is not an integer (or is a `bool`) raises `TypeError`, and one
  that is negative or larger than the buffer raises `ValueError`. All of these are raised before
  any request is sent.
- Every other failure (403, throttling, transport, server errors, and an object that changed
  version during the read) raises `RuntimeError` with the full cause chain in its message.

**Buffer rules (important).**
- Until the awaitable finishes, s3dlio has exclusive use of `dst[:expected_len]`: do not read,
  modify, resize or reuse it. After a failure its contents are unspecified.
- The awaitable never finishes before every native writer has stopped and the buffer is
  released, even on an error, `cancel()`, `Task.cancel()`, `asyncio.timeout()` or `wait_for()`.
  The buffer is safe to reuse right after the `await` (Python 3.11 has one `wait_for` exception,
  below).
- `asyncio.wait_for` (Python-version difference): if its deadline expires after the read or HEAD
  has itself finished but before the buffer is released, the cancellation is refused and the call
  completes with its own result once released. Python 3.12+ `wait_for` then raises
  `TimeoutError`; Python 3.11 `wait_for` returns the result.
- **Python 3.11 only (#288):** a task inside `asyncio.wait_for` that is cancelled after the
  deadline fired, or cancelled twice, can end while the read still writes into `dst`. Use
  `asyncio.timeout()` on 3.11, or keep the future and wait for it with the cancellation-proof
  `wait_for_guard` loop shown in the `put_bytes_async` section before touching `dst`. Python 3.12
  and 3.13 are not affected.
- Await the returned object directly; do not wrap it in `asyncio.shield`, and do not call
  `set_result`/`set_exception` on it.
- Concurrent calls need their own destinations, or non-overlapping slices of one buffer.
- Accepted: writable, C-contiguous buffers with item size 1 (formats `B`, `b`, `c`, or none).
  Read-only, strided, pointer and object formats are refused before any request is sent.

**Semantics.**
- A read never mixes data from two object versions. The S3 range fan-out sends `If-Match` with
  the HEAD's ETag on every range; otherwise the first response's ETag (S3, Azure) or generation
  (GCS) pins every retry and repair. A response that violates the pinned identity fails the read;
  a concurrent overwrite can still permit a consistent read of the original version when that
  version remains available. If the first response has no usable ETag, a read that fails part-way
  fails instead of retrying. A separate `head_async` call does not condition the following GET:
  a same-size replacement between those calls can be the version the GET reads.
- `etag` is opaque (not an MD5). `version_id` is the S3 or Azure version ID (when versioning is
  on) or the GCS generation.
- On `s3://` and `gs://`, `head_async` also returns `None` for a missing bucket: the provider
  cannot tell it from a missing object. A missing Azure container raises.
- Supported: S3 and S3-compatible endpoints, Azure Blob, and native GCS in builds with the GCS
  backend (the PyPI wheels have S3 and Azure). Not supported: `file://`, `direct://`.
- Once native work has started, the op-log writes one GET row per `get_into_async` and one HEAD
  row per `head_async`, with the real outcome (also after an error or cancellation). A call
  refused up front writes no row.
- A call still running when the interpreter exits is not waited for.

### list() — List Objects

```python
# List all objects under prefix (returns full URIs)
uris = s3dlio.list("s3://bucket/prefix/")
# ['s3://bucket/prefix/file1.dat', 's3://bucket/prefix/file2.dat', ...]

# Recursive listing
uris = s3dlio.list("s3://bucket/prefix/", recursive=True)

# With an explicit regex filter (a Rust regex, NOT a glob; see below)
uris = s3dlio.list("s3://bucket/prefix/", recursive=True, pattern=r"\.npz$")
```

**For `s3://` URIs the key is a name, a directory or an explicit glob -- never a regex, never a bare
prefix (#213).** `list("s3://b/data/run.1", recursive=True)` returns the object `data/run.1` and the keys
under `data/run.1/` and nothing else: not `data/run.1.old`, `data/run_1` or `data/prerun.1`. A name
selects only itself (and its `name/` directory), so `list("s3://b/file.txt")` never returns
`file.txt.bak`. The characters `.`, `(`, `)`, `[`, `+`, `$`, `?` in the URI are ordinary characters and
never fail to parse. With `recursive=False` the object AND the "directory" (returned with a trailing
`/`) come back: `list("s3://b/foo")` returns `foo` and `foo/`, not `foobar`, `afoo` or unrelated
top-level folders. A trailing `/` (`s3://b/dl/`) selects everything under it, and the bucket root
(`s3://b/`) everything in the bucket. `*` makes an explicit glob over the FULL key: `*` matches any run
of characters except `/`, `**` matches any run including `/`, everything else is literal.
`list("s3://b/file.txt*")` returns `file.txt`, `file.txt.bak` and `file.txt2`;
`list("s3://b/data/**.npz")` every `.npz` under `data/`. With a wildcard an object is selected iff its
full key matches, so `recursive` makes no difference; a wildcard key ending in `/` raises
`RuntimeError`. `*` also matches the empty string, so `list("s3://b/dir/*")` includes a folder-marker
object `dir/` if one exists; `a/**/b` needs something between its slashes (it matches `a/x/b` and
`a//b`, not `a/b`). There is no escape for a literal `*` in a listing or `recursive=True` URI: a key
literally named `a*b` is reached by the exact, non-recursive calls (`get`, `stat`, `delete` without
`recursive=True`), which take the key as written. `download()` follows the same rule.
`s3dlio.compat.s3torchconnector` is the exception by design: its `prefix` arguments keep the
upstream key-PREFIX meaning (`train/img_` returns every key that starts with it). Earlier versions split the URI at its last `/` and
compiled the rest as an unanchored regex. Only `s3://` follows this rule; `file://`, `direct://`,
`gs://` and `az://` listings are unchanged (tracked in #220).

**`pattern=` is the one way to select by regular expression.** It is a Rust `regex` (not a glob),
applied AFTER the selection above, and matched, unanchored, against each returned URI *relative to the
listing base*: the key itself when it ends with `/` (`s3://b/dl/` and `s3://b/dl/sub/c.npz` ->
`sub/c.npz`), the directory when the key names one (`s3://b/dl` -> `sub/c.npz`), the parent directory
for the named object (`s3://b/data/run.1` -> `run.1`), and for a wildcard URI the directory part of the
text before the first `*` (`s3://b/data/file*` and `s3://b/data/file.txt.bak` -> `file.txt.bak`). For a
URI without a trailing `/` this differs from earlier versions, which matched an entry that merely
starts with the prefix against the whole URI. For a host-form URI
(`s3://host:port/bkt/dl/`) the base is taken from the key parts. An invalid pattern raises before
any request is made. `s3-cli ls --pattern` uses the same rule.

### stat() — Get Metadata

```python
meta = s3dlio.stat("s3://bucket/key")
print(meta['size'])           # int: object size in bytes
print(meta['last_modified'])  # str: timestamp
print(meta['etag'])           # str: ETag/hash

# Async version
meta = await s3dlio.stat_async("s3://bucket/key")

# Batch stat (async)
metas = await s3dlio.stat_many_async(["s3://bucket/a", "s3://bucket/b"])
```

### exists() — Check Existence

```python
if s3dlio.exists("s3://bucket/key"):
    print("Found")

# Async version
found = await s3dlio.exists_async("s3://bucket/key")
```

### delete() — Remove Object (or everything under a prefix)

```python
s3dlio.delete("s3://bucket/key")                       # exactly this object
s3dlio.delete("s3://bucket/logs/2026/", recursive=True)  # every object whose key starts with "logs/2026/"
```

`delete(uri, recursive=False)` follows the same contract as `s3-cli delete` (issues #210, #167, #212):

- **`delete(uri)` sends ONE delete request for exactly the key named by `uri`, and returns `None`.**
  There is no lookup first (no HEAD), no listing, and no parent "directory" is consulted, so it needs
  only delete permission.
- **An already-missing object is success.** S3's DELETE succeeds for a key that does not exist, so
  `delete()` cannot tell "deleted" from "was already gone", and success means only that the delete
  request succeeded. On other backends only the backend's own typed not-found error counts as
  success (file backends; Azure `BlobNotFound`; GCS NOT_FOUND, where a missing bucket gives the same
  status). On a **versioned bucket** a delete adds a delete marker and does not erase earlier versions.
- **Any actual failure raises `RuntimeError`** naming the key (`failed to delete s3://b/k: ...`): a
  403, a 5xx, a network error. A failure is never read as "already absent". A bucket-only URI
  (`s3://b/`) names no object and raises `RuntimeError` without sending anything.
- **`/`, `*` and `?` are ordinary key characters.** `delete("s3://b/dir/")` deletes the object named
  `dir/` (a folder marker) and nothing under it; `delete("s3://b/a*")` deletes the key `a*`, not every
  key starting with `a`. A trailing slash never means "recursive".
- **`delete(uri, recursive=True)` deletes the real objects the `s3://` URI selects (#213).** The
  listing uses no delimiter and only objects are deleted (never a synthetic "directory" entry).
  `recursive=True` on `s3://b/foo` removes the object `foo` and everything under `foo/`, and NEVER
  `foobar` or `foo.bak`; `s3://b/foo/` means the directory; `s3://b/foo*` is an explicit glob that
  selects `foo`, `foo.bak` and `foobar` (`*` stays in one directory level, `**` crosses `/`); `?` is
  literal. A wildcard key ending in `/` raises `RuntimeError` and deletes nothing. A selection with no
  objects is a no-op. It needs list permission. **Each selected object gets its own delete request** (up to
  10 in flight), as before; the batched DeleteObjects that `s3-cli delete -r` uses is deferred for the
  Python API. If a delete request or the listing fails, no further delete request is sent (requests
  already in flight finish) and `RuntimeError` reports how many requests were sent and how many
  selected keys were not attempted. S3 reports a key that does not exist as deleted, so the call never
  says how many objects it removed.
- **A directory is never deleted without `recursive=True`.** On `file://` and `direct://`,
  `delete("file:///d")` and `delete("file:///d/")` raise `RuntimeError` (`... is a directory; use -r /
  recursive=True ...`) and delete nothing; a regular file is deleted and a missing path is success.
  `recursive=True` with a trailing `/` deletes the files under it.
- **Other backends** (`file://`, `direct://`, `gs://`, `az://`) use the same two store operations;
  where a backend has no prefix listing, `recursive=True` requires a URI ending in `/` and
  raises `RuntimeError` otherwise (nothing is deleted).
- The op-log records an exact delete as one `DELETE` row (no `HEAD` row), and `recursive=True` as
  one `LIST` row plus one `DELETE` row per object (replayable; never `DELETE_BATCH`).

**Compatibility change (intentional).** Earlier versions deleted every listed entry for a URI ending
in `/` or containing `*` or `?` (including synthetic common-prefix entries) and treated the last
segment of a recursive URI as a regex. Code that relied on that must now pass `recursive=True` with a URI
that selects them (`name/`, or `name*`) or name the object. The lower-level `ObjectStore.delete`,
`delete_batch` and `delete_prefix` are unchanged.

### mkdir() — Create Directory

```python
s3dlio.mkdir("file:///data/output/subdir")      # creates the directory (and parents)
s3dlio.mkdir("direct:///data/output/subdir")    # same as file://
s3dlio.mkdir("s3://bucket/prefix/")             # no-op: returns None, sends no request
await s3dlio.mkdir_async("s3://bucket/prefix/")
```

`file://` and `direct://` create a real directory (`mkdir -p`; an existing directory is
success). On `s3://`, `az://` and `gs://` `mkdir()` **does nothing**: an object store has no
directories, a prefix exists as soon as an object is written under it, so there is nothing to
create. The URI is checked, no request is sent and no marker object is written. To remove the
objects under a prefix use `s3dlio.delete(uri, recursive=True)`. `mkdir` is never recorded in
the op-log.

---

## Zero-Copy Data Flow

s3dlio's `get()` returns `BytesView`, a Python object backed by Rust `Bytes` (Arc-counted reference). **No data is copied** when passing through the Rust runtime:

```
Rust async I/O → Bytes (Arc) → channel → BytesView (Python buffer protocol)
```

### Using BytesView

```python
data = s3dlio.get("s3://bucket/model_weights.bin")

# 1. Zero-copy memoryview (fastest — no allocation)
mv = memoryview(data)

# 2. Zero-copy NumPy array
import numpy as np
weights = np.frombuffer(data, dtype=np.float32)

# 3. Zero-copy PyTorch tensor
import torch
tensor = torch.frombuffer(data, dtype=torch.float32)

# 4. Convert to bytes (creates copy — only when needed)
raw = bytes(data)
```

### Performance Impact

| Operation | Copies | Notes |
|-----------|--------|-------|
| `s3dlio.get()` | 0 | Returns BytesView (Rust Bytes Arc) |
| `memoryview(data)` | 0 | Buffer protocol, no allocation |
| `np.frombuffer(data)` | 0 | Shares Rust memory |
| `torch.frombuffer(data)` | 0 | Shares Rust memory |
| `bytes(data)` | 1 | Explicit copy to Python heap |
| `s3dlio.put_bytes(uri, pydata)` | 0 | Any C-contiguous buffer is read in place with the GIL released (1 copy only for strided views and `__bytes__()`-only objects); see "put_bytes() — Upload Data" for per-backend details, including `direct://` |
| `s3dlio.put_bytes_async(uri, pydata)` | 0 or 1 | `bytes`/`BytesView`: 0. Other buffers: 1 snapshot copy (GIL released), or 0 with `borrow=True` |

---

## Batch Operations

### put_many() — Batch Upload (v0.9.50+)

Upload multiple objects in a single call with parallel execution. Each `data` may be `bytes`, any buffer-protocol
object (`bytearray`, `memoryview`, a NumPy array, ...) or an object with `__bytes__()`; since v0.10.0 (#272) an
`int`, a `bool`, a list of ints or any other iterable raises `TypeError` (wrap byte values in `bytes(...)`).
Every item is converted before the first request, so one bad item writes nothing, and the error names the item
(index and URI). Each payload is copied once when the call is made (exact `bytes` are not copied), so you may
reuse the buffers as soon as `put_many` returns or `put_many_async` is called. Both resolve to `None`.

```python
# List of (uri, data) tuples
items = [
    ("s3://bucket/file1.bin", b"data1"),
    ("s3://bucket/file2.bin", b"data2"),
    ("s3://bucket/file3.bin", b"data3"),
]
s3dlio.put_many(items)

# Async version
await s3dlio.put_many_async(items)
```

### get_many() — Batch Download

```python
uris = ["s3://bucket/file1", "s3://bucket/file2", "s3://bucket/file3"]
results = s3dlio.get_many(uris, max_in_flight=64)

# az:// and gs:// work too (all URIs in one call must use the same scheme)
results = s3dlio.get_many(["az://myaccount/container/a.bin", "az://myaccount/container/b.bin"])
results = s3dlio.get_many(["gs://bucket/a.bin", "gs://bucket/b.bin"])

# Async version (s3:// only: get_many_async() and get_many_stats() read through the S3 client)
results = await s3dlio.get_many_async(uris)

# O_DIRECT reads — bypass page cache (v0.9.95: now correctly uses O_DIRECT)
results = s3dlio.get_many(["direct:///data/file1.npz", "direct:///data/file2.npz"])
```

> **v0.9.95 fix:** `direct://` URIs in `get_many()` previously fell through to the standard buffered `tokio::fs::read()` path — O_DIRECT was never engaged. Fixed in v0.9.95: `direct://` reads now go through `ConfigurableFileSystemObjectStore::with_direct_io()`, correctly bypassing the OS page cache.

### upload() / download() — Bulk File Transfer

```python
# Upload entire directory
s3dlio.upload(
    src_uri="file:///data/files/",
    dest_uri="s3://bucket/uploads/",
)

# Download prefix to local directory
s3dlio.download(
    src_uri="s3://bucket/downloads/",
    dest_dir="/data/output/"
)

# On s3:// the source URI selects like list(): "s3://bucket/a.txt" is only a.txt (never
# a.txt.bak); "s3://bucket/a.txt*" is the explicit glob that also takes a.txt.bak
```

`download()` keeps each object's key path relative to the listing base (issue #284): downloading
`s3://bucket/downloads/` writes the key `downloads/a/x.bin` to `<dest_dir>/a/x.bin` and `downloads/b/x.bin` to
`<dest_dir>/b/x.bin`, creating directories as needed. A single exact object is written to
`<dest_dir>/<object name>`. The base is one for the whole listing: `s3://bucket/dl` selects the object `dl` (for
example a zero-byte folder marker) and the directory `dl/`, so both are placed relative to the parent
(`dl/a` goes to `<dest_dir>/dl/a`, never to `<dest_dir>/a`). A file system cannot hold the object `dl` and the
directory `dl/`, so **the directory wins**: only the single object `dl` is refused, as one failed object, with
"object 'dl' and directory 'dl/' cannot both exist locally; ... fetch the object alone (non-recursively), or use
's3://bucket/dl/' to download only the directory". The same holds for a glob that selects both. Use the
trailing slash (`s3://bucket/dl/`) for the directory only, or `recursive=False` for the object only.
Only a directory with at least one downloadable object counts: if EVERY child under `dl/` is refused (unsafe key,
collision), the object `dl` is downloaded as an ordinary file; in a chain (`p`, `p/q`, `p/q/r`) every object that
has a valid child (`p` and `p/q`) is refused and only `p/q/r` is downloaded.
`s3-cli download --pattern` (`download()` has no pattern argument) is applied BEFORE this decision: the layout base and the exact-object check see only the objects the
pattern kept, so a pattern that filters out `dl` itself makes the directory the base. `gs://`, `gcs://`,
`az://` and `https://<account>.blob.core.windows.net/...` sources all map correctly; for `gs://` and `az://`
the listing is by RAW KEY PREFIX (the `s3://` selection rule is not implemented there, tracked in #220), so a
sibling such as `dl.bak` can appear in a `gs://bucket/dl` listing and is written to `<dest_dir>/dl.bak`; use
the trailing slash to select the directory only. An object whose
relative key is empty, absolute, contains a `.` / `..` / empty (`a//b`) component, a NUL byte, a backslash-based
traversal or a component that is exactly a drive designator (`C:`) is refused (`a:b`, `x:y` and `v:1/x` are
ordinary names), as is one that would write through a symlink below `dest_dir`; two objects that map to the
same local path are never merged. `download()` refuses PER OBJECT: the other
objects are still written, and then it raises an error that lists the counts and the first three failures (the
object URI and the reason), `... failed=2: s3://b/dl/../x: ... refused; ... (and N more; see the log)`. Control
characters in a remote key (newline, ESC, bidi and other invisible format characters) appear escaped (`\n`,
`\u{1b}`, `\u{202e}`) in these messages, including the cause text of a failed fetch, and
credentials are redacted from every `scheme://` URI in that text (userinfo, signed query strings), not only from
the object URI. Documented limits of the redaction: a schemeless `user:password@host` in the text is not recognised;
a password that contains whitespace, or two or more `/`, is not fully removed (one `/` is handled in `http(s)` URIs only); everything up to the last `@` before the first whitespace or `/` after `://` is treated as userinfo, and when that removed part holds a `?` or `#` (a password with a raw `?`/`#`, or an `@` inside a query such as `https://h?e=a@b&sig=S`) the text after it is treated as a query too and removed (other schemes: when it holds a credential parameter), favouring redaction (`https://u:p?w@h/x` becomes `https://`); so a bare-host URI directly followed by text with an `@` and no whitespace (`https://h,user=a@b`) is over-redacted and `abfs://container@account` loses `container@`; a credential parameter is cut from `http(s)` URIs by dropping the whole query, but from other schemes (`s3://`, `gs://`, `az://`) only when it is one of the well-known names (`sig`, `signature`, `token`, `x-amz-*`, `x-goog-*`); and a cut query runs to the next whitespace, quote, backtick or angle bracket (`)`, `]`, `}`, `,` and `;` do not end it), the text after that boundary is kept, so compact JSON / XML keep their other fields (when the removed part holds another `scheme://`, the rest of the word is removed too, so a later URI glued to it never survives in part); a raw quote or angle bracket inside a query in front of a credential parameter therefore ends the removed part early (`?name=O'Brien&X-Amz-Signature=...` keeps the text after the quote; signed URLs percent-encode these characters). URIs JSON-escaped (`https:\/\/...`) or percent-encoded as a whole, and secrets outside a URI (an `Authorization:` header, an XML `<SignatureProvided>`), are not recognised.
(The Rust library helpers `s3dlio::s3_copy::download_objects*` are fail-fast instead: one bad key, or an object
and directory of one name, refuses the whole call
before anything is created or fetched.) A symlink below `dest_dir` is checked before the write and, for the
file itself on Unix, refused at open time (`O_NOFOLLOW`); an intermediate directory swapped for a symlink by
another local process in the instant between check and write is not detected. A write that fails after its
directories were created leaves those directories, empty, behind, and a file that was only partly written is
NOT removed (delete or overwrite it). Zero-length "folder" keys (ending in `/`) are
skipped.

**Scope of these protections.** They reject ordinary hostile remote KEYS (traversal, absolute and drive-letter
forms, collisions, a pre-existing symlink). They are not containment against a hostile LOCAL directory, and they
are not a Windows file-name mapping: Windows is not a supported or tested platform of the release (the release
matrix is Linux), and alternate data streams, reserved device names and path aliases are not handled. Download
only to a destination whose directory entries are not being modified by an untrusted actor.

### mp_get() — Multi-Process GET

For maximum throughput with very large datasets:

```python
result = s3dlio.mp_get(
    uri="s3://bucket/dataset/",
    procs=8,           # 8 worker processes
    jobs=128,          # 128 concurrent ops per process
    num=10000,         # 10,000 objects
    template="data_{}.bin"
)
print(f"Throughput: {result['throughput_mb_s']} MB/s")
```

---

## Multi-Backend Support

### Authentication

Settings can also come from a `.env` file, but the Python library does **not** read one by default
(since 0.9.128): set `S3DLIO_LOAD_DOTENV=1` (the working directory's `.env`) or
`S3DLIO_LOAD_DOTENV=/path/to/file` in the environment. Variables that are already set are never
overridden by the file. See [Environment Variables](Environment_Variables.md#env-files-s3dlio_load_dotenv).

**Amazon S3 / S3-Compatible:**
```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['AWS_ENDPOINT_URL'] = 'http://minio:9000'  # Optional: MinIO, Ceph, VAST
```

**Google Cloud Storage:**
```bash
gcloud auth application-default login
# Or: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
```

**Azure Blob Storage:**

The storage account comes from the URI (`az://account/container/key`). Set
`S3DLIO_AZURE_CREDENTIAL` to choose the credential (default `auto`): `workload-identity`, `client-secret`,
`client-certificate` (needs a build with the cargo feature `azure-client-certificate`), `managed-identity`,
`cli` (`az login` / `azd auth login`; the `az` / `azd` executable must be on `PATH`), `sas`, or `none`.
`auto` is a fallback chain over workload identity, client secret, client certificate, managed identity and `cli`
(members whose variables are not set are skipped; one that fails to give a token falls through to the next). Credentials
are created once per process, so rotating a secret needs a restart. A bearer-token mode needs an `https` endpoint. Account keys (Shared Key) and
connection strings are **not** supported (the Azure SDK s3dlio is built on, `azure_storage_blob` 1.0, signs
only with bearer tokens or a SAS). The full table of modes and the variables each reads is in
[Environment_Variables.md](Environment_Variables.md#azure-blob-storage-configuration).
```python
import os
# Service principal (the same variables the Azure SDKs use):
os.environ['AZURE_TENANT_ID'] = '<tenant>'
os.environ['AZURE_CLIENT_ID'] = '<app id>'
os.environ['AZURE_CLIENT_SECRET'] = '<secret>'          # auto -> client-secret
# Managed identity on an Azure VM / App Service / AKS node:
#   os.environ['S3DLIO_AZURE_CREDENTIAL'] = 'managed-identity'
# SAS token:
#   os.environ['S3DLIO_AZURE_CREDENTIAL'] = 'sas'
#   os.environ['S3DLIO_AZURE_SAS_TOKEN'] = 'sv=...&sig=...'
# Optional custom endpoint. Azurite (http://) needs no credential:
os.environ['AZURE_STORAGE_ENDPOINT'] = 'http://127.0.0.1:10000'  # Azurite
```

| Backend | Endpoint Variable | Alternative |
|---------|------------------|-------------|
| S3 | `AWS_ENDPOINT_URL` | — |
| Azure | `AZURE_STORAGE_ENDPOINT` | `AZURE_BLOB_ENDPOINT_URL` |
| GCS | `GCS_ENDPOINT_URL` | `STORAGE_EMULATOR_HOST` |

### S3 Bucket Management

```python
s3dlio.create_bucket("my-bucket")
s3dlio.delete_bucket("my-bucket")
```

---

## Multi-Endpoint Load Balancing

Create stores that distribute operations across multiple endpoints (v0.9.14+):

```python
import asyncio
import s3dlio

async def main():
    # From explicit URI list
    store = s3dlio.create_multi_endpoint_store(
        uris=["s3://bucket-1", "s3://bucket-2", "s3://bucket-3"],
        strategy="round_robin"  # or "least_connections"
    )
    
    # From template (expands {1...10})
    store = s3dlio.create_multi_endpoint_store_from_template(
        uri_template="s3://my-bucket-{1...10}",
        strategy="round_robin"
    )
    
    # From file (one URI per line)
    store = s3dlio.create_multi_endpoint_store_from_file(
        file_path="/path/to/endpoints.txt",
        strategy="least_connections"
    )
    
    # All operations are async
    await store.put("s3://bucket-1/data.bin", b"Hello")
    view = await store.get("s3://bucket-1/data.bin")      # BytesView
    view = await store.get_range("s3://bucket-1/x", 0, 1024)
    objects = await store.list("s3://bucket-1/", recursive=True)
    await store.delete("s3://bucket-1/data.bin")
    
    # Statistics
    print(store.endpoint_count())
    print(store.strategy())
    print(store.get_total_stats())
    print(store.get_endpoint_stats())

asyncio.run(main())
```

**Strategies:**
- `"round_robin"` — Even distribution, lowest overhead
- `"least_connections"` — Routes to endpoint with fewest active requests

---

## Streaming API

For large uploads with compression and chunking:

```python
options = s3dlio.PyWriterOptions()
options.compression = "zstd"       # none, zstd, gzip, lz4
options.compression_level = 3      # 1-22 for zstd
options.chunk_size = 4 * 1024 * 1024  # 4 MiB chunks

# Backend-specific writers
writer = s3dlio.create_s3_writer("s3://bucket/file.bin.zst", options)
writer = s3dlio.create_azure_writer("az://acct/ctr/file.bin.zst", options)
writer = s3dlio.create_filesystem_writer("file:///path/file.bin", options)
writer = s3dlio.create_direct_filesystem_writer("direct:///path/file.bin", options)

# Write in chunks, then finalize
writer.write(chunk1)
writer.write(chunk2)
stats = writer.finalize()
print(f"Wrote {stats['bytes_written']} bytes")
```

### Buffer ownership of the write calls

`write_chunk(data)`, `write_owned_bytes(data, *, copy=False)` and
`MultipartUploadWriter.write(data, *, copy=False)` **borrow your buffer without copying it** (zero copy is the
default for every C-contiguous buffer: `bytearray`, `memoryview`, NumPy of any dtype, `array.array`, ctypes, mmap):

* the call reads your memory in place, with the GIL **released**, so other Python threads keep running;
* the buffer is *exported* while the call runs (`bytearray.extend()` / `clear()` from another thread raises
  `BufferError`) and **no thread may modify its bytes until the call returns**: a concurrent write is a data race
  and the bytes written are undefined (undefined behaviour in the Rust sense, not just torn data);
* the export is released **before the call returns**, so the buffer is yours again right after;
* exact `bytes` and `s3dlio.BytesView` are immutable: they never race and are never copied, in any mode;
* **`copy=True`** (`write_owned_bytes` and `MultipartUploadWriter.write` only) first takes ONE snapshot of a mutable
  buffer with the GIL held and releases the export, so you may modify or reuse your buffer as soon as the call has
  started. Use it when another Python thread may write to the buffer during the call. The snapshot is atomic only
  against writers that **hold the GIL** (pure-Python code): a writer that releases the GIL (a large NumPy
  operation, `file.readinto(buf)`, `sock.recv_into(buf)`, a CUDA or torch copy into pinned memory) can still tear
  it, and you must serialise such writers yourself. It costs one copy of the data;
  `write_chunk` has no `copy` parameter (it always borrows).

**Behaviour change in 0.9.128.** In 0.9.126 and earlier `write_owned_bytes()` copied a mutable buffer under the GIL
before writing; it now borrows it exactly like `write_chunk()`. If your code relied on the old snapshot (for
example another thread refills the same `bytearray` while a write is in flight), pass `copy=True`.

---

## AI/ML Integration

### PyTorch Datasets (Native)

```python
from s3dlio import ObjectStoreMapDataset, ObjectStoreIterableDataset
from torch.utils.data import DataLoader

# Map-style (random access)
dataset = ObjectStoreMapDataset(uri="s3://bucket/train/", pattern="*.npz")
item = dataset[0]  # Fetch by index

# Iterable (streaming)
dataset = ObjectStoreIterableDataset(uri="s3://bucket/train/", shuffle=True)

# With DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4)
for batch in loader:
    pass  # Train
```

`ObjectStoreMapDataset` fetches every item by the exact object it listed. Keys: for `s3://` (also with
the server form `s3://host:port/bucket/...`) the key is the full key from the bucket root, prefix included
(`s3://bkt/dir/` lists `dir/x`); for `file://` and `direct://` it is the path below the dataset root
(`sub/b.bin`, or just the file name when the URI names one file). `s3dlio.create_dataset(uri).uris()` gives the URI
that fetches each item.

### JAX Integration

```python
from s3dlio import JaxIterable

jax_iter = JaxIterable(uri="gs://bucket/train/", batch_size=32)
for batch in jax_iter:
    pass  # JAX training
```

### TensorFlow Integration

```python
from s3dlio import make_tf_dataset

ds = make_tf_dataset(uri="s3://bucket/train/", batch_size=32, shuffle=True)
for batch in ds:
    pass  # TF training
```

### PyLoaderOptions — Builders and Readable Values

`PyLoaderOptions` is configured with chained builder methods. Each builder returns a **new**
options object and leaves the original unchanged, so you must use the returned value:
`opts.pin_memory(True)` with the result discarded has no effect.

```python
import s3dlio

opts = (
    s3dlio.PyLoaderOptions()
    .with_batch_size(128)
    .num_workers(8)
    .pin_memory(True)
    .shuffle(True, 42)
    .drop_last(True)
    .prefetch(4)
)
```

`opts.pin_memory`, `opts.num_workers`, `opts.shuffle`, `opts.drop_last`, `opts.prefetch`,
`opts.persistent_workers` and `opts.non_blocking` **remain builder methods** (a Python class
cannot have a method and a property with the same name). To read the current value, use the
read-only `*_value` properties:

| Builder method | Read-only property | Type |
|---|---|---|
| `pin_memory(bool)` | `pin_memory_value` | `bool` |
| `num_workers(int)` | `num_workers_value` | `int` |
| `shuffle(bool, seed)` | `shuffle_value` | `bool` |
| `drop_last(bool)` | `drop_last_value` | `bool` |
| `prefetch(int)` | `prefetch_value` | `int` |
| `persistent_workers(bool)` | `persistent_workers_value` | `bool` |
| `non_blocking(bool)` | `non_blocking_value` | `bool` |

`batch_size`, `seed`, `timeout_seconds`, `shard_rank` and `shard_world_size` have no
same-named builder and are plain read-only properties.

`opts.to_dict()` returns a plain `dict` of every option that `PyLoaderOptions` can set, with
its current value (keys: `batch_size`, `drop_last`, `shuffle`, `seed`, `num_workers`,
`prefetch`, `auto_tune`, `pin_memory`, `persistent_workers`, `timeout_seconds`,
`non_blocking`, `shard_rank`, `shard_world_size`, `generator_seed`, `enable_transforms`,
`collate_buffer_size`, `multiprocessing_context`, `memory_format`, `sampler_type`,
`sampler_replacement`, `sampler_rank`, `sampler_world_size`). Options that cannot be set from
Python (such as `loading_mode`, `reader_mode`, `part_size`, `max_inflight_parts`, `worker_id`)
are not included. `auto_tune`, `generator_seed`, `enable_transforms` and `collate_buffer_size`
have no `*_value` property and are readable only through `to_dict()`.

Enum-valued options are reported by lower-case name (for example `"channels_last"`,
`"forkserver"`). `sampler_type` is one of `"sequential"`, `"random"`, `"weighted_random"` or
`"distributed_random"`; its parameters are reported by separate keys:

| Key | Value |
|---|---|
| `sampler_replacement` | `bool` when `sampler_type` is `"random"` (the `random_sampling(replacement)` argument), otherwise `None` |
| `sampler_rank` | `int` when `sampler_type` is `"distributed_random"`, otherwise `None` |
| `sampler_world_size` | `int` when `sampler_type` is `"distributed_random"`, otherwise `None` |

```python
opts = s3dlio.PyLoaderOptions().num_workers(8).pin_memory(True)
assert opts.num_workers_value == 8 and opts.pin_memory_value is True
assert opts.to_dict()["num_workers"] == 8

a = s3dlio.PyLoaderOptions()
b = a.pin_memory(True)  # a new object; `a` is unchanged
assert a.pin_memory_value is False and b.pin_memory_value is True

d = a.distributed_sampling(1, 4).to_dict()
assert (d["sampler_type"], d["sampler_rank"], d["sampler_world_size"]) == ("distributed_random", 1, 4)
```

---

## Object Data Loader (General-Purpose)

**New in v0.9.100.** Sliding-window, URI-carrying object streaming built on Tokio's
`buffer_unordered`. Works with all s3dlio URI schemes without code changes.

Key APIs: `PyDataset.from_uris()` · `PyBytesAsyncDataLoader.items()` · `PyObjectDataLoaderSyncIter.collect_batch(n)` · `PyObjectItem`

📖 **[Python_Data-Loader.md — Object DataLoader section](Python_Data-Loader.md#object-data-loader-general-purpose)**

---

## Parquet DataLoader

**New in v0.9.98.** Epoch-aware, row-group-streaming Parquet loader for AI/ML training.
Works with all s3dlio backends. Epoch-2+ zero-refetch fast path; raw and Arrow IPC decode modes.

Key API: `create_async_loader(uri, {"format": "parquet", ...})`

📖 **[Python_Data-Loader.md — Parquet section](Python_Data-Loader.md#parquet-dataloader)** · **[Complete Parquet Guide](Parquet_Data-Loader.md)**

---

## Data Generation (NPZ / NPY)

**Added in v0.9.94.** Generate complete NumPy NPZ archives in Rust — single allocation, Rayon parallel fill, hardware-accelerated CRC32 — without holding the Python GIL. ~5× faster than `numpy.savez()` for large files.

### `generate_npz_bytes()` — Build NPZ in Rust

```python
import s3dlio

# Build a 140 MiB unet3d-style NPZ archive (~20 ms vs ~178 ms for numpy.savez)
npz = s3dlio.generate_npz_bytes(
    shape=[6053, 6053, 1],   # x.npy array shape
    dtype="<f4",             # float32 little-endian (default)
    num_samples=1,           # number of labels in y.npy (default)
)
print(type(npz))   # <class 's3dlio.BytesView'>
print(len(npz))    # ~146,800,000 bytes
```

**Arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `list[int]` | required | Array shape for `x.npy` |
| `dtype` | `str` | `"<f4"` | NumPy dtype string, e.g. `"<f4"` (float32), `"<f8"` (float64) |
| `num_samples` | `int` | `1` | Length of label array `y.npy` (int64 zeros) |

**Returns:** `BytesView` — zero-copy buffer supporting Python buffer protocol. Pass directly to `MultipartUploadWriter.write()`, `put_bytes()`, or any buffer-accepting API.

### Upload NPZ zero-copy

```python
import s3dlio, concurrent.futures

# Pre-generate once (shape is fixed per benchmark run)
buf = s3dlio.generate_npz_bytes(shape=[6053, 6053, 1])

# Upload 48 files concurrently — no GIL contention, no memcpy
def upload(i):
    with s3dlio.MultipartUploadWriter.from_uri(
        f"s3://my-bucket/train/sample_{i:06d}.npz"
    ) as w:
        w.write(buf)  # BytesView fast path: Arc clone only, GIL released

with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
    list(pool.map(upload, range(1000)))
```

**Performance (28-core machine, loopback fake S3):**

| Method | write() latency (140 MiB) | Throughput N=48 |
|--------|--------------------------|------------------|
| `numpy.savez()` + `bytes()` + `put_bytes()` | ~178 ms gen + ~109 ms write | ~1,250 MiB/s |
| `generate_npz_bytes()` + `write(BytesView)` | ~20 ms gen + ~6 ms write | ~2,440 MiB/s |

### NPZ file structure

The generated archive matches NumPy's format exactly:

```
file.npz
├── x.npy   — float32 (or dtype) array, shape as specified, Rayon-filled random data
└── y.npy   — int64 array of shape (num_samples,), all zeros (class labels)
```

Load with standard NumPy:

```python
import numpy as np
arrays = np.load("file.npz")
x = arrays["x"]   # float32 ndarray
y = arrays["y"]   # int64 ndarray
```

---

## s3torchconnector Compatibility

s3dlio provides a **drop-in replacement** for AWS `s3torchconnector`. Change one import line:

```python
# Before (s3torchconnector):
from s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint

# After (s3dlio — zero code changes needed):
from s3dlio.compat.s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint
```

### S3IterableDataset

```python
from s3dlio.compat.s3torchconnector import S3IterableDataset
from torch.utils.data import DataLoader

dataset = S3IterableDataset.from_prefix("s3://bucket/train/", region="us-east-1")

for item in dataset:
    print(item.bucket, item.key)
    data = item.read()  # Returns BytesView (zero-copy)
    # Use with torch.frombuffer(data, dtype=torch.uint8)
```

### S3MapDataset

```python
from s3dlio.compat.s3torchconnector import S3MapDataset

dataset = S3MapDataset.from_prefix("s3://bucket/train/", region="us-east-1")
item = dataset[0]       # Random access
item = dataset[-1]      # Negative indexing supported
print(len(dataset))     # Number of objects
```

### S3Checkpoint

```python
from s3dlio.compat.s3torchconnector import S3Checkpoint
import torch

checkpoint = S3Checkpoint(region="us-east-1")

# Save — torch.save() writes to in-memory buffer, then uploads via put_bytes()
with checkpoint.writer("s3://bucket/model.pt") as writer:
    torch.save(model.state_dict(), writer)

# Load — downloads to BytesView, wrapped in seekable BufferedReader
with checkpoint.reader("s3://bucket/model.pt") as reader:
    state_dict = torch.load(reader, weights_only=True)
```

### S3Client (Low-Level)

```python
from s3dlio.compat.s3torchconnector import S3Client, S3ClientConfig

client = S3Client(
    region="us-east-1",
    endpoint="http://minio:9000",
    s3client_config=S3ClientConfig(force_path_style=True)
)

# Upload
writer = client.put_object("my-bucket", "key.bin")
writer.write(b"data")
writer.close()

# Download (full object)
reader = client.get_object("my-bucket", "key.bin")
data = reader.read()  # BytesView (zero-copy)

# Download (range — server-side, saves bandwidth)
reader = client.get_object("my-bucket", "key.bin", start=0, end=1023)
chunk = reader.read()  # 1024 bytes

# List
for result in client.list_objects("my-bucket", "prefix/"):
    for info in result.object_info:
        print(info.key)
```

Keys are used exactly as written, as in AWS's `s3torchconnector`: only the `/` after the bucket is a
separator, so `S3Item("s3://b//k", ...)` has `key == "/k"`, and `/k`, `k` and `a//b`, `a/b` are different
objects. The DLIO storage classes use a full-URI name exactly the same way, but keep their 0.9.124
normalisation of a name relative to the configured namespace (`/train/f.npz` under `s3://b/base` is the
key `base/train/f.npz`).

### Advantages over s3torchconnector

| Feature | s3torchconnector | s3dlio compat |
|---------|------------------|---------------|
| S3 | Yes | Yes |
| Azure / GCS / Local | No | Yes |
| Zero-copy reads | No | Yes (BytesView) |
| Server-side range requests | No | Yes (get_range) |
| Global client cache | No | Yes (DashMap) |
| Multi-endpoint load balancing | No | Yes |

For complete migration instructions, see [S3TORCHCONNECTOR_MIGRATION.md](S3TORCHCONNECTOR_MIGRATION.md).

---

## Checkpoint System

A checkpoint lives under a base URI (`s3://`, `gs://`, `az://`, `file://`, ...). Data is
always `bytes` (serialize your state first, e.g. `torch.save` into a `BytesIO`), and every
checkpoint is identified by `step`, `epoch` and a `framework` tag.

> **`user_meta` is optional** (default `None`) in `save_checkpoint`, `PyCheckpointStore.save`
> and `finalize_distributed_checkpoint` (module function and writer method). User metadata is
> accepted but not yet persisted.

### Save / Load

```python
import s3dlio

# Single-rank checkpoint: save_checkpoint(uri, step, epoch, framework, data, user_meta=None)
key = s3dlio.save_checkpoint(
    "s3://bucket/checkpoints/run1", 10, 10, "pytorch", state_bytes
)  # returns the manifest key

# Load the latest checkpoint under that base URI -> BytesView (or None if there is none)
view = s3dlio.load_checkpoint("s3://bucket/checkpoints/run1")
state_bytes = bytes(view)   # or memoryview(view) for zero-copy access

# Read one checkpoint shard by URI -> bytes. There is no `expected_keys` argument.
# With validate_integrity=True (the default) the bytes are checked against the checksum the
# checkpoint's manifest recorded when the shard was written: a mismatch raises
# ("Integrity validation failed ..."), and so does a shard that no manifest under the
# checkpoint base lists, a manifest that recorded no checksum for it, or a URI that is not a
# shard URI (`<checkpoint base>/shards/<key>`) -- the check is never skipped silently.
# A `.zst` shard is decoded first. validate_integrity=False reads the object as stored,
# unverified.
# Only manifests whose status is "complete" count. Precedence: the manifest named after the
# shard directory (`manifests/ckpt-<step>-<ts>.json`) is used when it is complete and lists
# the shard (no other manifest is compared with it); only otherwise is the newest complete
# manifest of the same step that lists the shard used (newest by parsed wall_time, compared
# as an instant), and only after that a scan of all the other manifests. A manifest that
# cannot be read or parsed raises that error rather than being reported as an unlisted shard.
# A store may list an object in a different spelling from the base you gave (an Azure
# `https://<account>.blob.core.windows.net/...` base lists as `az://`, `gcs://` as `gs://`,
# `s3://host:port/bucket/...` as bucket-form `s3://`, a relative `file://rel/ck` as `rel/ck/...`).
# Listed manifests are matched to the base by provider identity (bucket, or account and
# container; the server when both name one) and then by key, with the base followed by exactly
# one `/` (`ckpt` does not own `ckpt2/...`). A listed `.json` manifest that cannot be placed
# under the base (another bucket, container, account or server, a sibling prefix, or a
# spelling no rule maps) raises an error naming it (credentials redacted) -- in this fallback
# AND in `PyCheckpointStore.list_checkpoints` and `delete_checkpoint` (and the Rust by-step
# lookup) -- it is never skipped silently.
# `delete_checkpoint(step)` checks the whole listing, reads the manifest and validates every
# shard key it lists before deleting anything, then deletes the manifest first and its shards
# after (best effort). See "Deleting a checkpoint" below.
data = s3dlio.load_checkpoint_with_validation(
    "s3://bucket/checkpoints/run1/shards/<ckpt-id>/rank-0.bin", validate_integrity=True
)
```

For more control use `PyCheckpointStore`:

```python
store = s3dlio.PyCheckpointStore(
    "s3://bucket/checkpoints/run1",
    strategy=None, multipart_threshold=None, compression_level=None,
)
store.save(step, epoch, framework, data, user_meta=None)   # -> manifest key
store.load_latest()                                   # -> BytesView | None
store.list_checkpoints()                              # -> [PyCheckpointInfo]
store.delete_checkpoint(step)                         # -> bool
```

#### Deleting a checkpoint

`store.delete_checkpoint(step)` returns `True` when a checkpoint of `step` was found and its
manifest deleted, `False` when there is none.

**Coordinate saves and deletes.** Before deleting, quiesce checkpoint saves, finalizations and
other manifest/shard mutations under the same base, and prevent new ones until the delete
returns. The caller must coordinate this across all participating processes; s3dlio does not
provide a transaction or distributed lock. Shared-shard protection uses a listing snapshot:
a checkpoint published after that snapshot, or still incomplete when inspected, can reference
a shard that the delete then removes. Re-reading `latest.json` does not close this race.
The shared-shard and manifest-preservation guarantees below assume this coordination.

- **One checkpoint per call.** If two checkpoints share a step (saved twice at different
  times), a call deletes exactly one -- the first the store lists, whatever its status; the
  listing order is the store's and is not specified. Call again to delete the next; the result
  is `False` once none is left. `latest.json` changes only when it names the manifest that was
  deleted or is corrupt.
- **Validated before anything is deleted.** Every listed manifest must be placeable under the
  base, the manifest must parse and validate (a complete manifest that fails validation, for
  example one missing a rank, cannot be deleted by step), and every shard key it lists must be
  a safe key: relative, with a non-empty final component (never a directory or prefix), no
  leading `/`, no backslash, no `?` or `#`, no percent-encoded dot (`%2e`, any case; a
  URL-based store could normalise it to `..`), no empty / `.` / `..` component, no component
  named `manifests`, `latest`, `latest.json` or `latest.json.tmp` at any position, no final
  component `shards`, and no `ckpt-<step>-<ts>` directory of a different step at any position
  (a different timestamp of the same step is fine; a hand-built manifest that reuses another
  step's shard directory therefore cannot be deleted by step). On a `file://` or `direct://`
  base every target is also checked and a directory is refused. Keys outside `shards/`
  (written with a caller-chosen key) are fine. A manifest that fails the check makes the call
  raise with nothing deleted.
- **Shared shards are kept.** Every other manifest of the base is read; a shard key that
  another *complete* manifest also lists belongs to that checkpoint as well, so it is NOT
  deleted. The call deletes this checkpoint's manifest and only the shards no other complete
  manifest lists; the shared shards are kept (reported in an info-level log line per kept
  shard) and the other checkpoint stays whole and loadable. A manifest that does not parse or
  is not complete is ignored.
- **Cost, and all-or-nothing on unreadable manifests.** Each call makes one listing of
  `manifests/` plus one GET of each other manifest (a bounded number at a time). Deleting the
  checkpoint `latest.json` names also scans the remaining manifests once more to choose the new
  latest. If any other manifest cannot be read (a failure other than not-found), the call
  raises and NOTHING is deleted, because that manifest could list a shard of this checkpoint.
- **Order and failures.** The manifest is deleted first. If that delete fails the call raises
  and nothing was deleted. Shards are deleted after it, best effort: a shard that cannot be
  deleted is ignored and the call still returns `True`; it is then an unreferenced object
  that a rerun cannot find (the step now reports `False`), but a complete manifest never
  points at deleted shards.
- **`latest.json` follows the delete.** If `latest.json` named the deleted checkpoint (or is
  corrupt) it is pointed at the newest remaining complete, *valid* checkpoint (a complete
  manifest that fails validation is skipped, as is one whose key is not
  `ckpt-<step>-<ts>.json` with `<step>` equal to its recorded step), or removed when none
  remains, so
  `load_latest()` returns the previous checkpoint. A pointer is only written when it is
  certain: if any remaining `ckpt-<step>-<ts>.json` manifest cannot be read or parsed it could
  be the newest, so `latest.json` is left untouched and the call raises. If updating it fails
  the shards are still deleted and the call raises an error that says the checkpoint WAS
  deleted and `latest.json` was not updated. The pointer is re-read just before it is written
  and left alone if a concurrent save changed it; a save that publishes in that instant can
  still be overwritten, because `latest.json` has no conditional write.
- **Reading follows the same rule.** `load_latest()` / `load_latest_manifest()` scan the
  manifests (newest complete, valid one, by the same key rule) when `latest.json` is missing, unreadable or
  corrupt, or names a manifest that no longer exists, so a stale pointer never hides older
  checkpoints. Tradeoff: if the GET of the named manifest fails with something other than
  not-found and a successful listing of `manifests/` does not show it (for example a
  transient failure on a store whose listing lags), the newest remaining complete checkpoint
  is loaded, which can be older than the one `latest.json` names. A named manifest that
  exists but cannot be read or parsed is still an error.

### Distributed Checkpointing

Each rank writes its own shard and returns a shard-metadata dict (`rank`, `key`, `size`,
and `etag` / `checksum` when available). Rank 0 gathers the dicts (for example with
`torch.distributed.gather_object`) and publishes the manifest.

```python
store = s3dlio.PyCheckpointStore("s3://bucket/ckpt")
writer = store.writer(world_size, rank)                  # writer(world_size, rank)

# Every rank: save_distributed_shard(step, epoch, framework, data) -> shard-meta dict
meta = writer.save_distributed_shard(step, epoch, "pytorch", local_state_bytes)

# Rank 0 only: finalize_distributed_checkpoint(step, epoch, framework, shard_metas, user_meta=None)
if rank == 0:
    manifest_key = writer.finalize_distributed_checkpoint(
        step, epoch, "pytorch", all_shard_metas
    )
```

Streaming variant for large shards: `writer.get_distributed_shard_stream(step, epoch,
framework)` returns a `PyCheckpointStream` with `write_chunk(data)`, `bytes_written()`,
`finalize()` (returns the shard-meta dict) and `cancel()`.

The module-level one-shot functions open a store at `uri` on every call and are
equivalent to the writer methods:

```python
meta = s3dlio.save_distributed_shard(uri, step, epoch, framework, data, world_size, rank)
key = s3dlio.finalize_distributed_checkpoint(
    uri, step, epoch, framework, shard_metas, world_size, rank, None
)
```

### Reading a Distributed Checkpoint

`reader.load_latest_manifest()` returns the newest manifest as a dict (or `None`);
`reader.read_shard_by_rank(manifest, rank)` returns that rank's shard as a `BytesView`. When
the shard entry of the manifest dict you pass carries a `crc32c:` checksum the read is validated
against it and a mismatch raises; an entry without a checksum is read unvalidated (it is not
refused, and no stored manifest is consulted). Use `load_checkpoint_with_validation` to refuse a
shard that has no recorded checksum.

Runnable end-to-end example (all four "ranks" in one process, `file://` URI):

```python
import tempfile
import s3dlio

with tempfile.TemporaryDirectory() as tmp:
    base = f"file://{tmp}/ckpt"
    world_size, step, epoch = 4, 100, 3

    # Write: every rank saves a distinct shard
    shard_metas = []
    for rank in range(world_size):
        writer = s3dlio.PyCheckpointStore(base).writer(world_size, rank)
        shard_metas.append(
            writer.save_distributed_shard(step, epoch, "pytorch", bytes([rank]) * 1024)
        )

    # Rank 0 publishes the manifest
    store = s3dlio.PyCheckpointStore(base)
    store.writer(world_size, 0).finalize_distributed_checkpoint(
        step, epoch, "pytorch", shard_metas, None
    )

    # Read: latest manifest, then each rank's shard
    reader = store.reader()
    manifest = reader.load_latest_manifest()
    assert manifest["framework"] == "pytorch"
    assert (manifest["global_step"], manifest["epoch"], manifest["world_size"]) == (100, 3, 4)
    for rank in range(world_size):
        shard = reader.read_shard_by_rank(manifest, rank)   # BytesView
        assert bytes(shard) == bytes([rank]) * 1024
```

The manifest dict has the keys `format_version`, `framework`, `global_step`, `epoch`,
`wall_time`, `world_size`, `status` and `shards` (a list of shard-meta dicts).

---

## Performance & Threading

### Thread Safety (v0.9.50)

All s3dlio functions are **fully thread-safe**. Use `ThreadPoolExecutor` freely:

```python
from concurrent.futures import ThreadPoolExecutor

def upload_one(i):
    s3dlio.put_bytes(f"s3://bucket/obj_{i}.bin", data)

# 16 threads uploading concurrently — no runtime conflicts
with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(upload_one, range(1000)))
```

**Fixed in v0.9.50:**
- v0.9.27: Per-call `Runtime::new()` caused "dispatch failure" after ~40 objects
- v0.9.40: `GLOBAL_RUNTIME.block_on()` caused "Cannot start a runtime from within a runtime"
- v0.9.50: io_uring-style submit pattern — works from ANY thread context

### Pre-Stat Size Caching

```python
uris = s3dlio.list("s3://bucket/dataset/")
s3dlio.get_many_stats(uris, max_in_flight=100)

# Subsequent gets use cached sizes (2.5x faster)
for uri in uris:
    data = s3dlio.get(uri)
```

### Performance Tips

1. **Use `put_many()` for batch uploads** — single round-trip, parallel execution
2. **Use `get_range()` instead of `get()` + slicing** — server-side range saves bandwidth
3. **Use `memoryview(data)` not `bytes(data)`** — avoid unnecessary copies
4. **Pre-stat with `get_many_stats()`** — eliminates per-get stat overhead
5. **Use `ThreadPoolExecutor`** — s3dlio handles concurrency internally via global runtime
6. **Set `S3DLIO_WORKER_THREADS`** — tune Tokio worker count for your workload
7. **Enable range optimization for large objects** — see below

### Large Object Download Optimization

For workloads with large S3 objects (> 100 MB), enable parallel range downloads:

```python
import os
os.environ['S3DLIO_ENABLE_RANGE_OPTIMIZATION'] = '1'
os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '64'  # Objects ≥ 64 MB use parallel ranges

import s3dlio

# Large objects automatically use parallel range downloads
data = s3dlio.get("s3://bucket/large-checkpoint.bin")  # 148 MB → 25-50% faster
```

**How it works:**
- **Disabled by default** to avoid HEAD request overhead on small objects
- When enabled, objects ≥ threshold are downloaded using parallel range requests
- HEAD request determines size, then concurrent GET ranges fetch chunks in parallel
- Best for: ML checkpoints, large datasets, multi-GB files
- Not recommended for: Many small objects, latency-sensitive workloads

**Tuning parameters:**
```python
import os

# Conservative (recommended for mixed workloads)
os.environ['S3DLIO_ENABLE_RANGE_OPTIMIZATION'] = '1'
os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '64'  # Default

# Aggressive (for very large objects > 500 MB)
os.environ['S3DLIO_ENABLE_RANGE_OPTIMIZATION'] = '1'
os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '128'
os.environ['S3DLIO_RANGE_CONCURRENCY'] = '32'  # More parallel requests
os.environ['S3DLIO_CHUNK_SIZE'] = '16777216'  # 16 MB chunks
```

**Actual benchmark results (16x 148 MB objects, MinIO):**

| Threshold | Time | Throughput | Speedup |
|-----------|------|------------|---------|
| Disabled (baseline) | 5.52s | 429 MB/s (0.42 GB/s) | 1.00x |
| 8 MB | 3.50s | 676 MB/s (0.66 GB/s) | **1.58x** (58% faster) |
| 16 MB | 3.27s | 725 MB/s (0.71 GB/s) | **1.69x** (69% faster) |
| 32 MB | 3.23s | 732 MB/s (0.71 GB/s) | **1.71x** (71% faster) |
| **64 MB (default)** | **3.14s** | **755 MB/s (0.74 GB/s)** | **1.76x (76% faster)** 🏆 |
| 128 MB | 3.22s | 735 MB/s (0.72 GB/s) | **1.71x** (71% faster) |

**Summary:**
- **64 MB threshold (default) is optimal**, achieving 76% improvement
- 16-64 MB range all perform excellently (69-76% faster)
- Even aggressive 8 MB threshold delivers 58% improvement with minimal downside
- HEAD overhead: ~10-20ms (well amortized over parallel download)

---

## Advanced Features

### NPZ Files

```python
# Read NumPy .npz files directly from any backend
data = s3dlio.get("s3://bucket/data.npz")
import numpy as np, io
arrays = np.load(io.BytesIO(bytes(data)))
```

### TFRecord Index

```python
s3dlio.create_tfrecord_index(
    input_path="s3://bucket/dataset.tfrecord",
    output_path="file:///tmp/dataset.idx"
)
```

### Operation Logging

```python
s3dlio.init_op_log("/tmp/operations.tsv.zst")
# ... operations are automatically logged ...
s3dlio.finalize_op_log()
```

The op-log is written to the local filesystem as a zstd-compressed TSV. The
path may be a plain path (`/tmp/operations.tsv.zst`) or a `file://` destination
(`file:///tmp/operations.tsv.zst`); the parent directory must already exist.
`file://` follows **s3dlio's `file://` convention** (the same as the `file://`
object store); it is **not** full RFC 8089. Only the lowercase prefix `file://`
is recognised: it is removed and the rest is the path, used literally, so
`file:///tmp/x` is `/tmp/x`, but `file://localhost/x` is the *relative* path
`localhost/x`, `file://relative/x` is the relative path `relative/x`, and
percent sequences are not decoded. `file://` with nothing after it is an error.
Every other scheme is rejected as unsupported, shown as written: `s3://`,
`direct://` and also `FILE://` (uppercase). Text before `://` that is not a valid
URI scheme (for example `dir/a://b.log`) is a plain path.

`file:` followed by anything but `//` (`file:x`, `file:/tmp/x`, also `FILE:x`) is
rejected with an error that shows the `file://` spelling; a local file really
named `file:x` is written `./file:x`. A one-letter scheme (`c://x`) is rejected
like any other scheme, on every platform; the error says to write a Windows drive
as `C:/dir/file` or `C:\dir\file` (`C:/x` and `C:\x` are plain paths).

**The op-log destination must be a local path or a `file://` URI.** Cloud
destinations (`s3://`, `gs://`, `az://`, `direct://`, etc.) are not supported. A
string like `s3:bucket/key` (no `//`) is not a URI here: it is treated as a literal
local file name.

A process supports **one** op-log session: `init_op_log` -> `finalize_op_log`.
`init_op_log` while already active is a no-op; after `finalize_op_log` it raises
`RuntimeError` and touches no file. `finalize_op_log` is safe to repeat. The
module-level functions (`get`, `put_bytes`, `stat`, `list`, `delete`, `get_async`,
...) check the op-log on every call: a call made while it is active is recorded
once, a call made before `init_op_log` or after `finalize_op_log` is not,
whichever call first used the scheme. Objects you construct yourself (stores,
readers, writers) capture the logger at construction: those created before
`init_op_log` are never logged, and those created before `finalize_op_log` stop
recording when it runs.

The op-log is **lossy by default**: if the writer cannot keep up and the queue
(`S3DLIO_OPLOG_BUF`, default 8192 entries) fills, entries are dropped rather than
slowing your code. The background writer can also fail (disk full, or a panic in
the writer thread) with entries still queued. Detect both with `op_log_stats()`
**after** `finalize_op_log()`:

```python
s3dlio.finalize_op_log()        # stop your worker threads first
stats = s3dlio.op_log_stats()   # None before init_op_log(); else a dict
# {'accepted': 4, 'written': 4, 'dropped_full': 0, 'rejected_closed': 0,
#  'writer_failed': 0, 'writer_error': None, 'lossless': False,
#  'rejected_unformattable': 0, 'rejected_unsafe_field': 0,
#  'normalized_error_text': 0}
complete = (stats["accepted"] == stats["written"] and stats["dropped_full"] == 0
            and stats["writer_failed"] == 0 and stats["writer_error"] is None
            and stats["rejected_unformattable"] == 0
            and stats["rejected_unsafe_field"] == 0)
```

No counted loss happened exactly when `accepted == written` and `dropped_full ==
0` and `writer_failed == 0` and `writer_error is None` and
`rejected_unformattable == 0` and `rejected_unsafe_field == 0`. `rejected_unformattable` counts accepted rows the
writer did not write because a timestamp (after the clock offset is applied) is
before 1970-01-01 or at or after year 10000 and has no RFC 3339 text; such a row
is absent from the file, later rows are written, and the trace is incomplete (#190).
`rejected_unsafe_field` counts accepted records the writer did not write because the
operation name, client id, endpoint or file holds a TAB, LF or CR, which the
tab-separated row cannot carry; the whole record is absent (lost, not encoded), later
records are written, and the trace is incomplete (#186). `normalized_error_text`
counts written rows whose error text had a TAB, LF or CR replaced by a space; the
operation is recorded, so this does **not** make the log incomplete, and finalize
prints one line stating the count. `writer_error` is the
first error the writer met (a failed write, a failed finish or flush of the
compressed stream, or a writer panic, which is a defect rather than an I/O error)
as text; `written` is an upper bound of the rows in the file. `writer_failed` only counts calls made *after* the writer
stopped, so queued entries lost to a writer error show as `accepted > written`.

What the stats count: the calls that reached the logger between `init_op_log` and
`finalize_op_log`. "Written" counts the rows the zstd compressor accepted (the
bytes may still be buffered at that moment, and a later write, finish or flush can
still fail, which shows as `writer_error`). A successful `finalize_op_log()` with
the complete condition above true means every accepted row reached the compressor
and the stream was finished and flushed to the file without error: that is
completion, not durability (there is no fsync). The stats do **not** cover:

- **Process exit without `finalize_op_log()`.** There is no `atexit` hook:
  entries still queued and the zstd frame end are lost, and the stats are never
  read. Always call `finalize_op_log()` before the process ends.
- **Stores created before `init_op_log()`**: they are unlogged and uncounted.
- **Errors reported only when the file is closed** (for example NFS or quota
  errors at `close(2)`): they are not observable.

Calls made after `finalize_op_log()` began (a store created before finalize that
a worker still uses) are counted as `rejected_closed` and are **not in the file**;
they are not part of "complete", but a program should finalize only after its
workers stop and check `rejected_closed == 0` too.

`op_log_stats()` works while logging is active and after `finalize_op_log()`
(while active, `accepted > written` is normal). `finalize_op_log()` prints one
warning on stderr if any record was lost, naming only the non-zero counts (the
exact number of queue-full drops included). In the default mode the first
queue-full drop also prints one stderr note per session; later drops are only
counted. To avoid queue-full drops set `S3DLIO_OPLOG_LOSSLESS=1` (callers wait
instead of dropping) and/or raise `S3DLIO_OPLOG_BUF`.

### Python Convenience Helpers

```python
# list_keys — returns relative keys (not full URIs)
keys = s3dlio.list_keys("s3://bucket/prefix/")  # ['file1.bin', 'file2.bin']

# list_full_uris — returns full URIs (alias for list())
uris = s3dlio.list_full_uris("s3://bucket/prefix/")

# get_object — same as get()
data = s3dlio.get_object("s3://bucket/key")

# stat_object — same as stat()
meta = s3dlio.stat_object("s3://bucket/key")
```

---

## API Reference

### Core Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `configure_s3(endpoint_url, region, ca_bundle)` | Configure S3 endpoint, region, and CA bundle; clears store cache — **call before first S3 op** | None |
| `put_bytes(uri, data)` | Upload bytes | None |
| `put_bytes_async(uri, data)` | Async upload | Coroutine |
| `put_many(items)` | Batch upload `[(uri, data), ...]` | None |
| `put_many_async(items)` | Async batch upload | Coroutine |
| `get(uri)` | Download object | BytesView |
| `get_range(uri, offset, length=None)` | Server-side range request | BytesView |
| `get_many(uris, max_in_flight=64)` | Parallel download (s3://, az://, gs://, file://, direct://) | List[(uri, BytesView)] |
| `get_many_async(uris, max_in_flight=64)` | Async parallel download (s3:// only) | Coroutine |
| `list(uri, recursive=False, pattern=None)` | List objects | List[str] |
| `stat(uri)` | Get metadata | dict |
| `stat_async(uri)` | Async metadata | Coroutine |
| `stat_many_async(uris)` | Batch metadata | Coroutine |
| `exists(uri)` | Check existence | bool |
| `exists_async(uri)` | Async existence check | Coroutine |
| `delete(uri)` | Delete object | None |
| `mkdir(uri)` | Create directory (`file://`, `direct://`); no-op on object stores | None |
| `mkdir_async(uri)` | Async mkdir | Coroutine |
| `upload(src_uri, dest_uri)` | Bulk upload files | None |
| `download(src_uri, dest_dir)` | Bulk download files | None |
| `mp_get(uri, procs, jobs, num, template)` | Multi-process GET | dict |
| `create_bucket(name)` | Create S3 bucket | None |
| `delete_bucket(name)` | Delete S3 bucket | None |

### Data Generation Functions (v0.9.94+)

| Function | Description | Returns |
|----------|-------------|---------|
| `generate_npz_bytes(shape, dtype="<f4", num_samples=1)` | Build NPZ archive in Rust (GIL-free, Rayon parallel fill) | BytesView |

### BytesView

| Property/Method | Description | Zero-Copy |
|----------------|-------------|-----------|
| `len(view)` | Size in bytes | Yes |
| `memoryview(view)` | Python memoryview | Yes |
| `bytes(view)` | Convert to bytes | No (copy) |
| `np.frombuffer(view)` | NumPy array | Yes |
| `torch.frombuffer(view)` | PyTorch tensor | Yes |

### Multi-Endpoint API

| Function | Description |
|----------|-------------|
| `create_multi_endpoint_store(uris, strategy)` | Create from URI list |
| `create_multi_endpoint_store_from_template(template, strategy)` | Create from template |
| `create_multi_endpoint_store_from_file(path, strategy)` | Create from file |

**MultiEndpointStore Methods** (all async). Since v0.10.0 the write methods (`put`, `put_to_endpoint`,
`put_all_endpoints`) accept the same payloads as `put_bytes` (#281), copied once when the call is made, and every
method returning `Coroutine[None]` resolves to `None`, not `()` (#271):

| Method | Returns |
|--------|---------|
| `put(uri, data)` | Coroutine[None] |
| `get(uri)` | Coroutine[BytesView] |
| `get_range(uri, offset, length)` | Coroutine[BytesView] |
| `list(uri, recursive)` | Coroutine[List[Dict]] |
| `delete(uri)` | Coroutine[None] |
| `get_endpoint_stats()` | List[Dict] |
| `get_total_stats()` | Dict |
| `endpoint_count()` | int |
| `strategy()` | str |

### Streaming Writers

| Function | Description |
|----------|-------------|
| `create_s3_writer(uri, opts)` | S3 streaming writer |
| `create_azure_writer(uri, opts)` | Azure streaming writer |
| `create_filesystem_writer(uri, opts)` | Local filesystem writer |
| `create_direct_filesystem_writer(uri, opts)` | Direct I/O writer |

The write calls borrow the buffer for the duration of the call; see "Buffer ownership of the write calls" in the
Streaming API section (`copy=True` for a snapshot).

### s3torchconnector Compat Classes

| Class | Description |
|-------|-------------|
| `S3IterableDataset.from_prefix(uri, region)` | Streaming dataset |
| `S3MapDataset.from_prefix(uri, region)` | Random access dataset |
| `S3Checkpoint(region)` | Checkpoint save/load |
| `S3Client(region, endpoint, config)` | Low-level client |
| `S3ClientConfig(force_path_style, max_attempts)` | Client configuration |
| `S3Item` | Item with `.bucket`, `.key`, `.read()` |

### AI/ML Integration

| Class/Function | Framework |
|----------------|-----------|
| `ObjectStoreMapDataset(uri, pattern)` | PyTorch |
| `ObjectStoreIterableDataset(uri, shuffle)` | PyTorch |
| `JaxIterable(uri, batch_size)` | JAX |
| `make_tf_dataset(uri, batch_size)` | TensorFlow |

### Checkpoint System

| Function | Description |
|----------|-------------|
| `save_checkpoint(uri, step, epoch, framework, data, user_meta=None)` | Save single-rank checkpoint; returns manifest key |
| `load_checkpoint(uri)` | Load latest checkpoint; returns `BytesView` or `None` |
| `load_checkpoint_with_validation(uri, validate_integrity=True)` | Read one checkpoint shard by URI; returns `bytes`. `validate_integrity=True` verifies the returned bytes against the checksum the manifest recorded at write time and raises on a mismatch, an unlisted shard, a missing recorded checksum, or a non-shard URI; `False` reads unverified |
| `save_distributed_shard(uri, step, epoch, framework, data, world_size, rank)` | Save this rank's shard; returns shard-meta dict |
| `finalize_distributed_checkpoint(uri, step, epoch, framework, shard_metas, world_size, rank, user_meta=None)` | Rank 0 publishes the manifest; returns manifest key |
| `PyCheckpointStore(uri, strategy=None, multipart_threshold=None, compression_level=None)` | Store: `save`, `load_latest`, `list_checkpoints`, `delete_checkpoint`, `writer(world_size, rank)`, `reader()` |
| `PyCheckpointWriter` (`store.writer(world_size, rank)`) | `save_distributed_shard(step, epoch, framework, data)`, `get_distributed_shard_stream(step, epoch, framework)`, `finalize_distributed_checkpoint(step, epoch, framework, shard_metas, user_meta=None)` |
| `PyCheckpointReader` (`store.reader()`) | `load_latest_manifest()` -> dict or `None`; `read_shard_by_rank(manifest, rank)` -> `BytesView` (checks the shard against the `checksum` in the manifest dict you pass it; reads unvalidated when that entry has none, and does not consult the stored manifests) |

### Logging

| Function | Description |
|----------|-------------|
| `init_logging(level)` | Set log level: trace, debug, info, warn, error |
| `init_op_log(path)` | Start the process's single op-log session on a local file (plain path or `file://` destination, s3dlio's file-store convention, not full RFC 8089); `RuntimeError` after finalize |
| `finalize_op_log()` | Write all accepted records (unless the writer fails: an I/O error or a writer-thread panic) and close the log; **call it before the process exits, nothing does it for you**; releases the GIL while draining; terminal and idempotent |
| `is_op_log_active()` | True between init and finalize only |
| `op_log_stats()` | `None` before `init_op_log`; else a dict `accepted` / `written` / `dropped_full` / `rejected_closed` / `writer_failed` / `writer_error` / `lossless` / `rejected_unformattable` / `rejected_unsafe_field` / `normalized_error_text`, available while active and after finalize, to detect lost records (complete = `accepted == written`, no drops, no refusals, `writer_error is None`, `rejected_unformattable == 0`, `rejected_unsafe_field == 0`; `normalized_error_text` is a disclosure, not a loss) |

---

## Migration Guide

### From v0.9.40

v0.9.50 is a **non-breaking** upgrade. The only change is the internal runtime architecture:

| v0.9.40 | v0.9.50 |
|---------|---------|
| `GLOBAL_RUNTIME.block_on()` | io_uring submit via `run_on_global_rt()` |
| Panics in multi-threaded Python | Fully thread-safe from any context |
| No `put_many()` | `put_many()` and `put_many_async()` added |

### From v0.8.x

| Removed (v0.8.x) | Replacement (v0.9.x+) |
|------------------|----------------------|
| `list_objects(bucket, prefix)` | `list(uri)` |
| `get_object(bucket, key)` | `get(uri)` or `get_range(uri, offset, length)` |

### From s3torchconnector

See [s3torchconnector Migration Guide](S3TORCHCONNECTOR_MIGRATION.md).

---

## Support

- **GitHub**: https://github.com/russfellows/s3dlio
- **Issues**: https://github.com/russfellows/s3dlio/issues
- **License**: Apache-2.0
