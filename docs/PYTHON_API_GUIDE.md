# s3dlio Python API Guide

**Version:** 0.10.0

**Last Updated:** October 19, 2026

> **Upgrading to 0.10.0 (read this first): asynchronous writes now BORROW your buffer.**
> `put_bytes_async`, `put_many_async` and the `MultiEndpointStore` writes (`put`, `put_to_endpoint`,
> `put_all_endpoints`) no longer take a private copy of a mutable buffer (`bytearray`, `memoryview`,
> NumPy array, ...). They read your memory in place (`borrow=True`, the new default) until the
> awaitable completes, so **do not modify, resize or free the buffer until the `await` has finished**:
> a `bytearray` raises `BufferError` if you resize it meanwhile, and changing its contents is a data
> race (the uploaded bytes are undefined). To keep the old behaviour (reuse the buffer the moment the
> call returns) pass `borrow=False`: it costs one copy. `bytes`, `bytes` subclasses, `memoryview` of
> `bytes` and `BytesView` are immutable and are never copied in either mode. Details:
> [put_bytes()](#put_bytes--upload-data), [put_many()](#put_many--batch-upload-v0950),
> [Multi-Endpoint](#multi-endpoint-load-balancing). Every copy that remains anywhere in the library is
> listed in [Remaining copies (Rule #1)](#remaining-copies-rule-1).

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [S3 Connection Configuration](#s3-connection-configuration)
4. [Architecture](#architecture)
4. [Core Storage Operations](#core-storage-operations)
5. [Zero-Copy Data Flow](#zero-copy-data-flow) (includes [Remaining copies (Rule #1)](#remaining-copies-rule-1))
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

- Python 3.11+ (release validation targets CPython 3.11–3.13). Exit with daemon threads inside s3dlio calls is
  fixed since the PyO3 0.29.3 upgrade (#298) plus the #302 and #304 fixes for async calls; see
  [Interpreter exit with daemon threads](#interpreter-exit-with-daemon-threads-298)
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

### Interpreter exit with daemon threads (#298)

If the main thread returns while daemon threads are inside s3dlio calls (sync calls, a streaming writer, or
`get_into_async` on a daemon thread's event loop), the interpreter finalizes and the daemon threads are abandoned: the
process exits with code 0 and `atexit` handlers run. A daemon thread that returns from its blocking call after
finalization began is parked forever; it never runs again, so its remaining Python code and Rust destructors do not run.

- **Before the fix** (wheels built with PyO3 0.27 or older, including `s3dlio==0.10.0`): on CPython **before 3.13.8**
  (3.11, 3.12 and 3.13.0 to 3.13.7) this was a SIGABRT (exit code 134, often `FATAL: exception not rethrown`). Python
  3.13.8+ and 3.14 were never affected. Workarounds for such a wheel: use Python 3.13.8+, or signal your workers to stop
  and `join()` them (for example in a `try/finally` around `main()`) before the main thread returns.
- **Now** (PyO3 0.29.3 or later 0.29.x, with the #302 and #304 fixes): fixed for daemon threads in sync calls and in every
  async call the changelog lists, on Python 3.11 to 3.13 (all patch levels) and 3.14. Two later fixes are part of that
  claim: [#302](https://github.com/russfellows/s3dlio/issues/302) (CPython 3.14: a daemon loop with `get_async` and the
  other awaitables in flight at exit crashed) and [#304](https://github.com/russfellows/s3dlio/issues/304) (Python before
  3.13.8: an async call that was STARTING while the interpreter shut down aborted, about 3 to 6% of runs for
  `put_bytes_async`). Wheels from before those fixes still have those two crashes. An async call made after s3dlio's exit
  drain (from an `atexit` handler registered before `import s3dlio`) raises `RuntimeError` ("the interpreter is exiting").
  A residual window remains (a call still starting when the 2 s drain timeout expires; Python code that Rust runs outside
  the async machinery, such as a user-defined `__bytes__` / `__buffer__`, argument extraction from user objects, `__del__`
  of released user buffers); see the changelog entry for #304.
- Non-daemon threads are unaffected: Python waits for them before finalizing.
- Parking a thread is not a clean shutdown. Work a daemon thread had in flight is lost; stop and join your workers (and
  call `finalize_op_log()`) if you need it to complete.

## Core Storage Operations

### put_bytes() — Upload Data

```python
# Upload bytes to any backend
s3dlio.put_bytes("s3://bucket/key", b"data")

# Upload from file
with open("local_file.bin", "rb") as f:
    s3dlio.put_bytes("s3://bucket/key", f.read())

# Async version (for use with asyncio). Since 0.10.0 it BORROWS a mutable buffer (borrow=True):
# do not modify, resize or free `buf` until this await has finished.
await s3dlio.put_bytes_async("s3://bucket/key", buf)

# Opt-in snapshot: one copy now, `buf` is yours again as soon as the call returns
await s3dlio.put_bytes_async("s3://bucket/key", buf, borrow=False)
```

**BEHAVIOUR CHANGE in 0.10.0 (zero copy is Rule #1).** `put_bytes_async` used to take a private snapshot of a
mutable buffer by default; it now **borrows** it by default (`borrow=True`). The same default applies to
`put_many_async` and the `MultiEndpointStore` writes (`put`, `put_to_endpoint`, `put_all_endpoints`), which used to
copy each mutable payload once with the GIL held. Code that modifies, resizes or frees the buffer right after
`put_bytes_async(...)` returns, before the awaitable has completed, must either wait for the awaitable or pass
`borrow=False`. See "Contract for buffers that are read in place" below.

**What is copied** (`put_bytes` / `put_bytes_async`), by input type. "Copies" counts copies made *by s3dlio*
between your memory and the storage client:

| Input | `put_bytes` | `put_bytes_async` (default, `borrow=True`) | `put_bytes_async(..., borrow=False)` |
|---|---|---|---|
| `s3dlio.BytesView`, exact `bytes`, a `bytes` subclass, a `memoryview` of `bytes` (immutable) | 0 | 0 | 0 |
| any other **C-contiguous** buffer, any item type (`bytearray`, `memoryview`, numpy `uint8`/`float32`/`bool`/..., `array.array`, ctypes arrays, `mmap`) | 0 | 0 (read in place until the awaitable completes) | 1 (a snapshot, taken before the call returns, GIL released above 256 KiB) |
| non-contiguous view (strided, reversed, transposed) | 1 (gathered in logical order by Python, GIL held) | 1 | 1 |
| object with only `__bytes__()` | 1 (made by Python) | 1 | 1 |

A read-only view does not count as immutable unless its base is a `bytes` object: `memoryview(bytearray).toreadonly()`,
a read-only `mmap` and a non-writeable NumPy view can still change through another handle, so `borrow=False` still
snapshots them.

The buffer is read **in place with the GIL released**, so other Python threads keep running during a large
upload (a 512 MiB PUT used to stall every other thread for about 0.3 s).

Per backend, after that: `s3://`, `az://`, `gs://` (official client) and `file://` hand the same bytes to the
storage client / `write` with no further s3dlio copy (what the SDKs do internally is outside s3dlio). `direct://` (O_DIRECT) writes a **page-aligned**
source (address a multiple of 4096) straight from your memory in one `write` per at most 2 GiB; any other source
is copied **once** through a warm, reused, page-aligned bounce buffer (16 MiB by default, 4 pairs per NUMA node,
shared by PUTs and GETs, ping-pong so the copy of the next chunk overlaps the write of the current one; see
`docs/Environment_Variables.md`). The last partial block of a PUT is always bounced (a zero-padded 4 KiB block).
`direct://` GET returns page-aligned memory (by default read straight into one fresh aligned buffer), so for
objects of at least `min_io_size` (4 KiB) on a store with the range engine off (the default),
`s3dlio.get("direct://...")` followed by `put_bytes("direct://...", data)` needs no bounce copy of the body
(only that final padded block). Smaller objects use buffered I/O.

**Contract for buffers that are read in place** (`put_bytes` and `put_many` for the whole call; `put_bytes_async`,
`put_many_async` and the `MultiEndpointStore` writes by default, until the awaitable completes):

* the source is *exported*: resizing it from another thread (`bytearray.extend()` / `clear()` / `del ba[:]`)
  raises `BufferError` until the call (or awaitable) is finished;
* you must **not modify the bytes** from another thread (or other code sharing the memory) meanwhile: nothing
  prevents it, and it is a **data race**: the bytes uploaded are undefined (and the behaviour is undefined in the
  Rust sense). The memory is never freed or resized meanwhile because the buffer stays exported, but that does not
  make a concurrent write safe;
* with a borrowing async write (the default) you additionally promise not to free the memory (for example
  ctypes/pinned host memory) until the awaitable has completed, successfully or not;
* **completion is a safe-reuse point, cancellation included, on every backend (exact statement).** The awaitable
  of a borrowing async write (`put_bytes_async`, `put_many_async`, the `MultiEndpointStore` writes, with
  `borrow=True`, the default) is a `Future` that **never reaches a terminal state - result, exception
  or cancelled - while any native reader of the buffer remains or before the export has actually been released**
  (`PyBuffer_Release` has run, so a resize no longer raises `BufferError`). A native reader is any clone of the
  payload, wherever it lives: a store task, an SDK/HTTP task, a detached file write. The supported consumer
  operations cannot change that: `await`, `fut.cancel()`, `Task.cancel()` (once or repeatedly), `asyncio.timeout()`
  and `asyncio.wait_for()` all leave the awaiting task pending until the future ends - with one Python 3.11
  exception for `wait_for`, described below (#288). For `put_many_async` the barrier covers every buffer of the batch
  at once. The guarantee covers
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
* **blocking `put_bytes` keeps `s3dlio.ExportStillReferencedError`** (the blocking `put_many` and the checkpoint saves
  behave the same). Its wait is bounded by the same timeout and is
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
  (for `head_async`, the request may still be running). This now applies to the DEFAULT `put_bytes_async`,
  `put_many_async` and `MultiEndpointStore` writes as well, because they borrow by default (pass `borrow=False`, or
  use the loop below). Python 3.12 and 3.13 (tested) are not affected. On 3.11,
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

`borrow=False` (the opt-in snapshot) takes ONE copy of a mutable buffer instead, so the buffer can be reused or
resized as soon as the call returns. This copy is the documented price of not borrowing. The one obligation: do not
write to the buffer from another thread *during that copy*. Up to 256 KiB the copy runs with the GIL held, so Python
threads cannot interleave; above that it runs with the GIL released (so large copies do not stall other threads),
and a Python thread, a C extension or a device that writes to the buffer at that moment is a **data race** with the
copy (the snapshot is then undefined, and the behaviour is undefined in the Rust sense). Multi-threaded callers that
mutate shared buffers must therefore serialise the call with their writers. Immutable sources (`bytes`, `bytes`
subclasses, `memoryview` of `bytes`, `BytesView`) are never copied, in either mode.
`borrow` is keyword-only: `await s3dlio.put_bytes_async(uri, buf, borrow=False)`.

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

#### What a read copies (0.10.0: zero copy is Rule #1)

Every byte received from the network is copied exactly once, into the buffer that `get()` returns; nothing
copies it again on the way to Python. The bytes you get back are a `BytesView` over that buffer.

* **S3 `get` / `get_range`.** The response body is received into ONE buffer allocated up front from the
  response's `Content-Length`, and each network chunk is freed as soon as it is copied in. Peak memory is about
  1x the object (it was about 2x: every chunk was kept and then joined into a second buffer). A body longer or
  shorter than its `Content-Length` is an error. A response with no `Content-Length` (chunked) cannot be
  pre-sized: its chunks are kept and joined once, a 2x peak that S3 itself does not produce.
* **Large objects fetched as parallel ranges** (S3, Azure, the official GCS client, `file://`, buffered
  `direct://`) are received range by range straight into that range's window of ONE final buffer; the windows are
  joined without copying. There is no per-range buffer and no copy into the result.
* **Azure large blobs.** `get` of a blob larger than one partition receives the first response into the front of one
  buffer of the blob's size and every other partition straight into its own slice, pinned to the first response's
  ETag with `If-Match`: exactly one copy of each byte and no second buffer. Dropping or cancelling the
  call stops every request. A blob whose first response has no strong ETag cannot be pinned, so it is read again as
  one request. A first response with no `Content-Range` that is exactly as long as the partition asked for is an
  error, because it cannot be told from the first partition of a larger blob whose header was stripped.
* **`file://` and `direct://`.** Reads use positional reads straight into the destination. They no longer pass
  through tokio's internal buffer, which copied every byte (up to 2 MiB at a time) before the caller's buffer.
* **`direct://` `get_range`** returns a slice of the aligned buffer the kernel filled (no copy) when that buffer is
  private to the read. A buffer taken from the shared bounce pool must go back to the pool, so that range is
  copied out once (see "Remaining copies"). `get_range(uri, offset, 2**64 - 1)` reads to the end of the file, as
  on `file://`; an offset at or past the end returns an empty result.

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
- Supported: S3 and S3-compatible endpoints, Azure Blob, native GCS in builds with the GCS
  backend (the PyPI wheels have S3 and Azure), and the local `file://` and `direct://` stores in
  every build (unreleased, #295, #267; see "Local files" below).
- Once native work has started, the op-log writes one GET row per `get_into_async` and one HEAD
  row per `head_async`, with the real outcome (also after an error or cancellation). A call
  refused up front writes no row.
- A call still running when the interpreter exits is not waited for.

**Local files: `file://` and `direct://`.** Both calls work with the same contract as the network
stores, with these local specifics:
- `get_into_async` on `file://` is a `pread` straight into `dst`; no other buffer of the object's size
  is allocated. On `direct://` a destination whose **address** is a multiple of the store alignment
  (the page size, 4096) is read with O_DIRECT straight into `dst`, with no copy: an aligned NumPy
  array, an `mmap`, a page-aligned ctypes buffer. The last, sub-alignment part of an object whose
  size is not a multiple of the alignment, and every byte of a destination at an unaligned address
  (a `bytearray`, or a NumPy slice at an odd offset), is read through the shared bounce pool: the
  kernel fills an aligned bounce buffer and the bytes are copied once into `dst`. That copy cannot be
  avoided (O_DIRECT forbids unaligned memory); it is bounded by the pool's own buffers, never by the
  size of the object. A file smaller than the store's minimum direct size, and a filesystem that
  refuses O_DIRECT, are read with ordinary positional reads (still straight into `dst`).
- A Python `bytearray` (and `bytes`-like buffers from `bytearray(n)`) is only 16-byte aligned, so on `direct://` it
  ALWAYS takes the bounce copy. To read with no copy, give `get_into_async` page-aligned memory: an anonymous `mmap`
  (`buf = mmap.mmap(-1, n)`), or a NumPy view cut at a page boundary:
  ```python
  raw = np.empty(n + 4096, dtype=np.uint8)
  dst = raw[(-raw.ctypes.data) % 4096:][:n]      # dst.ctypes.data % 4096 == 0
  ```
  `n` need not be a multiple of 4096 (the last, sub-page part is bounced). The bounce pool is shared by every unaligned
  `direct://` transfer and is tuned by `S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB`, `S3DLIO_DIRECT_IO_BOUNCE_PAIRS_PER_NODE` and
  `S3DLIO_DIRECT_IO_BOUNCE_WHEN_BUSY`; `get_into_async` ignores `S3DLIO_DIRECT_IO_GET_MODE` (its `direct` mode allocates
  a buffer of the object's size, which a caller-buffer GET must not do).
- The size is checked first: a file whose size is not `expected_len` raises `LengthMismatchError`
  (`.actual` is the file's size) and nothing is written to `dst`. The read is also re-checked at the
  end, so a file that changed size during the read is an error too (`LengthMismatchError` with the size it has now).
- A file modified in place while it is read, with its size unchanged (a same-size overwrite, truncate-then-regrow, a
  replacement by `rename`, a deletion), raises `RuntimeError` ("version changed"), never a successful read of mixed
  data: the open file's modification and status-change times are compared before and after the read. A metadata-only
  change (`chmod`, `chown`, a hard link) moves the status-change time and is reported too; retry. Limits: a write
  in the same kernel timestamp tick (1-10 ms on ext4) as the last change before the read opened the file goes unseen;
  filesystems with 1-2 s timestamps widen that window; a write through a shared `mmap` is stamped only at write-back.
  The check cannot protect against a writer that is still writing when the read ends successfully.
  **Network and cluster filesystems (NFS, Lustre, GPFS, BeeGFS, CephFS, SMB):** the timestamps are the ones the
  client has cached. Attribute caching (`actimeo`, the revalidation interval), server-to-client clock or mtime lag and
  coarse timestamp granularity mean a same-size overwrite made by ANOTHER node can go unseen for seconds, so the
  version check is best effort there (the size check is only as current as the client's cached size). If another
  node may rewrite a file you are reading, coordinate with a lock or a handshake, or use an object store.
- A missing file (or a path below a regular file) raises `ObjectNotFoundError`; `head_async` returns `None`.
  A directory raises `RuntimeError` from both. A file you may not read raises `RuntimeError` naming the
  permission error (`head_async`, which only needs `stat`, still succeeds).
- `head_async` is a `stat`: `size`, `etag` = `file-<size>-<mtime seconds>` (the identity
  `ObjectStore.stat` reports; it is not a content hash) and `version_id` = `None`. The etag has second
  granularity (a historical format, unchanged): two same-size rewrites within one second share it. The
  read-time version check above uses nanoseconds and is not affected.
- Embedding in Rust? The file stores hand a runtime worker off around each read chunk only on threads marked with
  `s3dlio::runtime_support::mark_tokio_worker_thread` (s3dlio's own runtimes, including everything the Python
  module starts, are marked); an unmarked embedder runtime runs `direct://` / `file://` reads, writes, fsyncs and
  finalize inline on the polling worker (the remedy is that one `on_thread_start` line; a marked `spawn_blocking`
  thread that drives a `LocalSet`, on any runtime, needs `inline_blocking_io()`). See the Rust section of the
  changelog and the `runtime_support` rustdoc.
- Cost of the hand-off, for the runtimes s3dlio starts for Python (1 GiB `get_into_async`, warm page cache, one reader;
  3 repetitions, noisy, 2-socket box): `file://` 3.8-6.9 GiB/s against 5.6-7.7 GiB/s without the hand-off,
  `direct://` 4.4-4.8 against 4.9-5.5. In exchange a neighbouring task stalls about 2 ms instead of about 10 ms
  (4 readers on 4 workers). Both are far above the allocating `get` (1.5-2.0 GiB/s).
- The read runs without the GIL, in chunks of at most 16 MiB. Cancelling the awaitable stops it at the next
  chunk boundary; as everywhere, the awaitable ends only after the last write into `dst`.

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
| `s3dlio.put_bytes_async(uri, pydata)` | 0 | Borrows the buffer until the awaitable completes (`borrow=True`, the default since 0.10.0); `borrow=False` takes 1 snapshot copy of a mutable buffer; `bytes`/`BytesView` and other immutable sources: 0 either way |
| `s3dlio.put_many(items)` / `put_many_async(items)` | 0 | Every buffer is read in place (held for the call / until the awaitable completes); `put_many_async(borrow=False)`: 1 snapshot per mutable buffer |
| `MultiEndpointStore.put` / `put_to_endpoint` / `put_all_endpoints` | 0 | Borrow by default, `borrow=False` for 1 snapshot; `put_all_endpoints` reads the one buffer for all endpoints |
| `s3dlio.save_checkpoint(...)`, `PyCheckpointStore.save`, `save_distributed_shard`, `PyCheckpointStream.write_chunk` | 0 | Any buffer (`bytes`, `bytearray`, `memoryview`, NumPy, `BytesView`) is read in place; the call blocks until written (compression is a transform) |
| `s3dlio.load_checkpoint_with_validation_view(uri)` | 0 | Returns a `BytesView`; `load_checkpoint_with_validation(uri)` returns `bytes`: 1 (documented) |
| `memoryview(item)`, `np.frombuffer(item)`, `item.data` on `loader.items()` / Parquet items | 0 | `bytes(item)`: 1 (documented) |

### Remaining copies (Rule #1)

s3dlio is zero copy everywhere unless this section says otherwise: **a copy of payload bytes that is not listed
here is a defect** (please report it). Every copy below is also commented at the place in the code where it is
made. "Copy" means a copy made by s3dlio of the payload bytes; what the SDKs and the operating system do inside
themselves is outside s3dlio.

| # | Copy | When it applies | How to avoid it |
|---|------|-----------------|-----------------|
| 1 | **Network / SDK chunk into the destination.** Each chunk the transport hands up is copied once into its final place (the buffer `get()` returns, a window of the range-engine buffer, or your buffer in `get_into_async`) | Every read from `s3://`, `az://`, `gs://`. This is the one unavoidable copy: the chunk belongs to the transport. s3dlio makes it exactly once per byte and never again | Cannot be avoided. `get_into_async` makes it straight into memory you own |
| 1a | The same, for an S3 response **without `Content-Length`** (chunked body): the chunks are kept and joined once, so the peak is 2x the body | Rare (S3 sends `Content-Length` on every GET) | Nothing to do; a body with a `Content-Length` peaks at 1x |
| 2 | **Transforms.** zstd compression when writing a shard or stream with a compression option; zstd decoding of a `.zst` checkpoint shard; the little-endian `float32` decode of `read_npz` (the stored member is sliced out of the archive in place and CRC-checked, then each element is decoded into the aligned array NumPy takes over, because the member's bytes are only 1-byte aligned inside the archive); inflating a compressed NPZ member | Only when you use compression or `read_npz` | Write checkpoints uncompressed to skip the encoder and decoder. The zstd decoder pre-sizes its output when the frame header records the size (frames written with a known size); frames from the streaming encoder carry no size, so their output grows as it is read |
| 3 | **O_DIRECT bounce for unaligned memory.** `direct://` cannot use unaligned memory with `O_DIRECT`, so (a) a PUT from a source whose address is not page-aligned is copied once through a reused aligned bounce buffer, and the last partial block is always bounced; (a2) a `get_into_async` into an unaligned destination (an odd address, or the sub-alignment tail of an object whose size is not a multiple of the alignment) is read through the bounce pool and copied once into your buffer; an aligned destination is read straight into it; (b) a range-engine read of a large `direct://` object copies each range once from the aligned read buffer into its window of the (unaligned) result; (c) a `direct://` `get_range` served from the shared bounce pool is copied out once, because the pool buffer is reused | `direct://` with O_DIRECT (what `direct://` uses unless configured otherwise) | Use page-aligned memory for PUTs and for `get_into_async` destinations (a `BytesView` from `s3dlio.get("direct://...")` is aligned; a NumPy array whose `ctypes.data` is a multiple of 4096 is too). A `get_range` whose read buffer is private to the call is returned as a slice with no copy. Buffered `direct://` (O_DIRECT off) has none of these |
| 4 | **Non-contiguous source.** A strided, reversed or transposed array/view must be gathered into logical order | Any write that is given such an array | Pass a C-contiguous array (make it contiguous once, up front, with `np.ascontiguousarray`) |
| 5 | **Object with only `__bytes__()`.** Python builds one `bytes` for it | A payload that is neither a buffer nor `bytes` | Expose the buffer protocol (or pass a `bytes`/`BytesView`) |
| 6 | **`bytes`-returning compatibility APIs.** A Python `bytes` owns its memory, so building one from a fetched buffer is one copy: `load_checkpoint_with_validation` (returns `bytes`), `bytes(item)` / `bytes(view)`, `ObjectStoreMapDataset(return_type="bytes")`, and `read()` of the file-like reader of `return_type="reader"` (it returns `bytes` slices, as file objects do; the reader itself holds the fetched buffer without copying it) | Only when you call them | `load_checkpoint_with_validation_view` (a `BytesView`), `memoryview(item)` / `np.frombuffer(item)` / `item.data`, `return_type="tensor"` |
| 7 | **Copies you opt into.** `put_bytes_async(..., borrow=False)`, `put_many_async(..., borrow=False)` and the `MultiEndpointStore` `borrow=False` writes (one snapshot per mutable buffer); `write_owned_bytes(..., copy=True)`; `ObjectStoreMapDataset(..., writable=True)` (one copy per item) | Only when you pass them | Leave them at their defaults and keep the buffer unchanged until the call or awaitable completes |
| 8 | **Whole-object buffered streaming writers.** `create_s3_writer` and `create_azure_writer` (and the Rust GCS writer) keep the whole object in one buffer until `finalize()` (then the allocation is handed to the store, with no copy): each `write_chunk()` is copied once into that buffer. `write_owned_bytes()` from Python goes through the same path, so it does not avoid it. These writers are not redesigned in this release | Writing through those three writers | `put_bytes` (the object is already in one buffer: 0 copies) or `MultipartUploadWriter` (borrows large writes, see "MultipartUploadWriter.write()" under the Streaming API). `create_filesystem_writer` and `create_direct_filesystem_writer` write straight from your buffer (the O_DIRECT bounce, #3, still applies to the latter) |
| 9 | **Rust API compatibility shapes.** `Writer` / `CheckpointStore::save` take `impl IntoShardBytes`: `Bytes` and `Vec<u8>` are zero copy, while `&[u8]` and the other borrowed forms are copied once, only when the object is stored uncompressed (the encoder reads a slice without a copy). `RangeEngine::download` (closures that return `Bytes`) copies each range into the master buffer; `RangeEngine::download_into_master` has no such copy. `ObjectStore::load_checkpoint_with_validation` returns `Vec<u8>` (taken over from the fetched buffer without a copy when it is the sole owner, otherwise moved or copied once) | Rust callers only | Pass `Bytes`; use `download_into_master`; use `get_with_validation` for a `Bytes` result |
| 10 | **Multipart writer: signed straddling part.** In a signed upload (the default) a part that spans two `MultipartUploadWriter.write()` calls, and the final short part of an object that spans writes, must be one contiguous request body, so it is gathered once: one copy of at most `part_size` bytes. Writes that are multiples of the part size never copy | `MultipartUploadWriter.write()` of writes that are not multiples of the part size, signed upload | Write multiples of the part size, use `reserve()` / `commit()` with part-sized buffers, or set `S3DLIO_UNSIGNED_PAYLOAD` (the part is then sent as a multi-chunk body with no gather) |
| 11 | **Multipart writer: sub-threshold coalescing.** A `write()` smaller than the threshold (the part size for a signed upload, 2 KiB for an unsigned one) is copied once into the part buffer instead of being borrowed; so is a `bytes`, `BytesView` or `commit()` smaller than the threshold. This copy is bounded and was measured: for signed sub-part writes borrowing gave no gain beyond run-to-run noise (it was slower at 1 KiB and below), and coalescing needs no lock and no pinned buffer | Small writes through `MultipartUploadWriter` | Write buffers of at least the threshold (borrowed, no copy), or use `reserve()` / `commit()` |


Notes:

* **Azure SDK.** The Azure SDK's own partition buffers are outside s3dlio, which is why s3dlio drives the large-blob
  ranged reads itself (see "What a read copies"). One SDK defect is worth knowing about: a hostile or broken server
  reply with a `Content-Range` whose end is `usize::MAX` overflows `end + 1` inside the SDK's parser
  (`azure_storage_blob` 1.1.0, `models/http_ranges.rs`). That panics in a debug build of the library only; a
  release build (the PyPI wheels) wraps instead, and s3dlio checks every length it derives from such headers with
  checked arithmetic.
* **Proof.** Each removed copy has a test that fails with the copy present: a pointer-identity or owner check,
  or an allocation / peak-memory bound.

---

## Batch Operations

### put_many() — Batch Upload (v0.9.50+)

Upload multiple objects in a single call with parallel execution. Each `data` may be `bytes`, any buffer-protocol
object (`bytearray`, `memoryview`, a NumPy array, ...) or an object with `__bytes__()`; since v0.10.0 (#272) an
`int`, a `bool`, a list of ints or any other iterable raises `TypeError` (wrap byte values in `bytes(...)`).
Every item is converted before the first request, so one bad item writes nothing, and the error names the item
(index and URI). Both resolve to `None`.

**Buffers are borrowed, not copied (0.10.0, BEHAVIOUR CHANGE).** Both calls used to copy each mutable payload once
when the call was made. They now read every buffer in place, like `put_bytes`:

* **`put_many(items)`** blocks until every object is written and holds every buffer for the whole call: a
  `bytearray` cannot be resized meanwhile (`BufferError`) and no thread may modify any buffer (a data race: the
  uploaded bytes are undefined). It returns, or raises, only after every native reader of every buffer is done and
  the buffers are released, so each buffer is yours again as soon as the call returns. If a backend still holds a
  buffer after `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (30 s by default) the call raises `ExportStillReferencedError`
  and the buffers stay pinned: do not modify, resize or free them. No copy is made.
* **`put_many_async(items, max_in_flight=64, *, borrow=True)`** borrows by default: do not modify, resize or free any
  buffer until the awaitable has completed (successfully or not). It never completes (result, error or
  cancellation) while any buffer is still read or before every export is released, so the usual `await`,
  `Task.cancel()` and `asyncio.timeout()` are safe; do not wrap it in `asyncio.shield` (see "Contract for buffers
  that are read in place" under `put_bytes()`). `borrow=False` is the opt-in snapshot: one copy of each MUTABLE
  buffer, taken when the call is made, so you may reuse the buffers immediately.
* `bytes`, `bytes` subclasses, `memoryview` of `bytes` and `BytesView` are never copied in either mode.

**A failure stops the rest.** Items are admitted in list order (first in, first out), so the first `max_in_flight`
items start first; with `max_in_flight` above 1 that is the order they *start* in, not the order they finish in. The
first item that fails ends the batch: items that have not started never start, items still queued are dropped at once
(releasing their buffers), every item still in flight is **aborted** (its payload is dropped at once, so its request
may never complete), and that item's own error is raised, not a generic "batch stopped" error. Only a write that
already runs on a blocking thread cannot be stopped; the call (or awaitable) waits for it and for any other native
reader of your buffers before it returns. The batch is **not transactional**: objects whose put completed before the
failure stay stored; an item aborted in flight may be absent, complete or (on a store whose put is not atomic, such
as `file://`) partial, and the error does not say which; items that never started were not written. After an error,
re-check or re-put every object that is not known to be complete. Cancelling the `put_many_async` awaitable stops
the batch the same way: nothing that has not started is uploaded, items in flight are aborted, and the cancellation
is delivered once the native readers that survive cancellation are done.

```python
# List of (uri, data) tuples
items = [
    ("s3://bucket/file1.bin", b"data1"),
    ("s3://bucket/file2.bin", b"data2"),
    ("s3://bucket/file3.bin", b"data3"),
]
s3dlio.put_many(items)

# Async version: leave the buffers alone until the await finishes
await s3dlio.put_many_async(items)

# Opt-in snapshot: reuse the buffers right away at the cost of one copy per mutable buffer
await s3dlio.put_many_async(items, borrow=False)
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
connection strings are **not** supported (the Azure SDK s3dlio is built on, `azure_storage_blob` 1.1, signs
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
| GCS (builds with the GCS backend) | `GCS_ENDPOINT_URL` | `STORAGE_EMULATOR_HOST` (either one: anonymous credentials, official client) |

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

**Writes borrow your buffer (0.10.0, BEHAVIOUR CHANGE).** `put(uri, data, *, borrow=True)`,
`put_to_endpoint(index, uri, data, *, borrow=True)` and `put_all_endpoints(uri, data, *, borrow=True)` accept the
same payloads as `put_bytes`. By default a mutable buffer (`bytearray`, `memoryview`, NumPy array, ...) is read in
place, with no copy, until the awaitable completes: do not modify, resize or free it before then (a `bytearray`
raises `BufferError` on resize; a write to the bytes is a data race and the uploaded bytes are undefined). The
awaitable never completes while a native reader of the buffer remains (see "Contract for buffers that are read in
place" under `put_bytes()`). `put_all_endpoints` reads the one buffer for every endpoint, with no per-endpoint copy.
`borrow=False` is the opt-in snapshot: one copy of a mutable buffer, taken when the call is made, so the buffer can
be reused immediately. `bytes`, `bytes` subclasses, `memoryview` of `bytes` and `BytesView` are never copied in
either mode. Before 0.10.0 each mutable payload was copied once with the GIL held.

```python
buf = bytearray(payload)
await store.put("s3://bucket-1/data.bin", buf)                 # borrows buf: leave it alone until done
await store.put("s3://bucket-1/data.bin", buf, borrow=False)   # snapshot: reuse buf immediately
```

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

`write_chunk(data)` and `write_owned_bytes(data, *, copy=False)` **borrow your buffer for the duration of the call,
without copying it** (zero copy is the default for every C-contiguous buffer: `bytearray`, `memoryview`, NumPy of any
dtype, `array.array`, ctypes, mmap). `MultipartUploadWriter.write()` borrows for much longer; it has its own contract,
see "MultipartUploadWriter.write(): the buffer is borrowed until close()" below.

* the call reads your memory in place, with the GIL **released**, so other Python threads keep running;
* the buffer is *exported* while the call runs (`bytearray.extend()` / `clear()` from another thread raises
  `BufferError`) and **no thread may modify its bytes until the call returns**: a concurrent write is a data race
  and the bytes written are undefined (undefined behaviour in the Rust sense, not just torn data);
* the export is released **before the call returns**, so the buffer is yours again right after;
* exact `bytes` and `s3dlio.BytesView` are immutable: they never race and are never copied, in any mode;
* **`copy=True`** (`write_owned_bytes`; `MultipartUploadWriter.write` has it too) first takes ONE snapshot of a mutable
  buffer with the GIL held and releases the export, so you may modify or reuse your buffer as soon as the call has
  started. Use it when another Python thread may write to the buffer during the call. The snapshot is atomic only
  against writers that **hold the GIL** (pure-Python code): a writer that releases the GIL (a large NumPy
  operation, `file.readinto(buf)`, `sock.recv_into(buf)`, a CUDA or torch copy into pinned memory) can still tear
  it, and you must serialise such writers yourself. It costs one copy of the data;
  `write_chunk` has no `copy` parameter (it always borrows).

**Behaviour change in 0.9.128.** In 0.9.126 and earlier `write_owned_bytes()` copied a mutable buffer under the GIL
before writing; it now borrows it exactly like `write_chunk()`. If your code relied on the old snapshot (for
example another thread refills the same `bytearray` while a write is in flight), pass `copy=True`.

**Which streaming writers copy (0.10.0).** `create_filesystem_writer` (`file://`) writes straight from your buffer
with plain `write(2)` calls: it no longer copies each chunk into tokio's internal buffer first. The same holds for
`create_direct_filesystem_writer` in buffered mode; with O_DIRECT the bounce rules of `put_bytes()` apply.
Compression (`options.compression`) is a transform: the encoder reads your buffer and its output is new data.
`create_s3_writer` and `create_azure_writer` keep the whole object in one buffer until `finalize()`, so each
`write_chunk()` is copied once into it (and `write_owned_bytes()` from Python takes the same path); at `finalize()`
that buffer is handed to the store without a further copy (it used to be duplicated for Azure). To upload an object
with no copy at all, use `put_bytes` or `MultipartUploadWriter`. This whole-object buffering is a documented exception, not redesigned in this
release; see "Remaining copies (Rule #1)" under "Zero-Copy Data Flow".

#### MultipartUploadWriter.write(): the buffer is borrowed until close()

**BEHAVIOUR CHANGE in 0.10.0.** `MultipartUploadWriter.write(data, *, copy=False)` used to hold your buffer only for
the duration of the call and copy it into its part buffers. A write that is large enough is now **borrowed**: the
parts are uploaded straight out of your memory, asynchronously, after `write()` has returned, and the buffer stays
borrowed until `close()` (or `abort()`, or leaving the `with` block) has returned. **Do not modify, resize or free a
buffer you passed to `write()` until then.** A `bytearray` raises `BufferError` if you resize it meanwhile; changing
its contents is a data race and the object holds undefined bytes. Allocate a fresh buffer for each `write()`, or pass
`copy=True`.

```python
with s3dlio.MultipartUploadWriter.from_uri("s3://bucket/big.bin") as w:      # part_size defaults to 16 MiB
    for _ in range(n_chunks):
        chunk = bytearray(chunk_size)          # a FRESH buffer per write
        f.readinto(chunk)
        w.write(chunk)                         # borrowed until the with-block exits
# every borrowed buffer is released when the block has exited

# Reusing ONE buffer for a readinto loop: take a snapshot of each write instead
buf = bytearray(chunk_size)
with s3dlio.MultipartUploadWriter.from_uri("s3://bucket/big.bin") as w:
    while f.readinto(buf):
        w.write(buf, copy=True)                # one copy, buf is yours again at once
```

* **Which writes are borrowed (the threshold).** A write of at least the threshold is borrowed; a smaller write is
  copied ONCE into the part buffer and nothing is borrowed (no lock, no `BufferError`, you may reuse the buffer at
  once). The threshold is the **part size for a signed upload** (the default) and **2 KiB for an unsigned upload**
  (`S3DLIO_UNSIGNED_PAYLOAD`). Why: a signed part must be one contiguous request body, and a part is made of many
  writes, so borrowing a write smaller than a part would not avoid the copy; in the benchmark (16 MiB parts,
  128 MiB per case, best of 3, write loop plus `close()`, time with borrowing / time with copying) borrowing was
  slower by up to 1.25x at 1 KiB and below and faster by at most about 7% from 2 KiB up, which is about the
  run-to-run noise, so the lock and the pinned buffers were not worth it. An unsigned upload sends a part that spans
  writes as a multi-chunk body with no gather (below), so borrowing small writes really is zero copy there: it
  measured 0.72-0.93x the copying time for 2-32 KiB writes, parity from 64 KiB up and slower below 2 KiB, which is
  why unsigned uploads switch at 2 KiB.
* **Reusing a buffer that is still borrowed is refused.** `write()` raises `BufferError` ("buffer reused while still
  borrowed; pass copy=True or use a fresh buffer") when the new buffer overlaps memory of an earlier `write()` whose
  parts have not been uploaded yet, because overwriting it would corrupt the earlier parts. The writer stays usable
  and can still be aborted. The classic loop `readinto(buf); write(buf)` therefore needs `copy=True` (or a fresh
  buffer per write, as above). Distinct buffers, and non-overlapping slices of one buffer, are fine.
* **`copy=True`** takes one independent snapshot of a mutable buffer (GIL held, export released before the call
  returns). `bytes` and `BytesView` are immutable: they are shared with the parts, never held under a lock, and (when
  they are at least the threshold) never copied.
* **Straddling parts.** Parts are cut from consecutive writes. A part that lies inside one write is a plain slice (no
  copy); writes that are multiples of the part size never copy. A part that **spans two writes** (the unfinished tail
  of the previous write plus the head of this one) is, in a **signed** upload, gathered once into one contiguous body:
  one copy of at most `part_size` bytes (the final short part of an object that spans writes is gathered the same
  way). In an **unsigned** upload (`S3DLIO_UNSIGNED_PAYLOAD`) it is sent as a multi-chunk body of known length, with
  no copy: the HTTP connector streams the segments instead of gathering them (an SDK or connector retry resends the
  same segments).
* **`close()`, `abort()` and `__exit__`.** They return only after every part that reads one of your buffers is done and
  the buffers are released. If a storage backend still references a buffer after
  `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (30 s) they raise `s3dlio.ExportStillReferencedError` and the buffers stay
  pinned (do not modify, resize or free them). That error **takes precedence** over an upload failure (in `close()`)
  and over an exception propagating out of the `with` block (in `__exit__`): the upload error is quoted in its message
  ("the upload itself failed: ..."). `abort()` cancels the parts still in flight before waiting. Leaving a `with` block
  because of an exception aborts the upload.
* **`reserve(size)` / `commit(nbytes)`** (the highest-throughput path) hands you a Rust-owned buffer to fill. The
  memoryview it returns is backed by an exporter object (`mv.obj`, an `s3dlio.ReservedBuffer`) that owns the memory:
  the memory never moves, and it stays allocated while ANY view, slice or NumPy array over it exists, while the writer
  holds the reservation, and while the upload still reads it. `close()`, `abort()`, leaving a `with` block, deleting
  the writer and `commit()` therefore never leave a view you retain pointing at freed memory; reading it stays safe.
  `commit(nbytes)` validates `nbytes` first: a length larger than the reservation raises `ValueError` and the
  reservation stays pending (still committable). On success the upload takes the first `nbytes` bytes straight out of
  the reserved memory, with no copy (a commit smaller than the threshold is copied once into the part buffer instead,
  like any small write). Two consequences, both by design: a **short commit keeps the whole reservation allocated**
  until its part is uploaded and every view is gone (it is not shrunk: shrinking can relocate the memory), so reserve
  what you will fill; and **writing through a retained view after `commit()` changes the payload in flight**, while the
  upload reads it: the stored object may contain a mix of old and new bytes, and a signed part may (but is not
  guaranteed to) fail its payload checksum; the same contract as a borrowed `write()`, so do not write after
  `commit()`. Nothing of yours is borrowed, so `close()` has no buffer of yours to wait for.

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

**What the items copy (0.10.0).** `return_type="tensor"` (the default) is a read-only `uint8` tensor over the fetched
buffer, with no copy (`writable=True` makes one copy per item). `return_type="reader"` wraps the fetched buffer in a
file-like object without copying it (it used to call `bytes(...)` first); its `read()` returns `bytes` slices, as file
objects do. `return_type="bytes"` returns a Python `bytes`, which is one copy by definition (a documented
compatibility copy).

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

**Reading an item without copying it (0.10.0).** A `PyObjectItem` (from `.items()` and `collect_batch`) and a
Parquet item (`PyParquetItem`, with `file_uri`, `rg_idx` and `byte_count`) expose the fetched object through the
buffer protocol and through `.data`. They are read-only (a writable request raises `BufferError`), and the buffer
lives as long as any view of it:

```python
mv  = memoryview(item)            # zero copy, read-only
arr = np.frombuffer(item, dtype=np.uint8)   # zero copy
view = item.data                  # a read-only BytesView over the same buffer, zero copy
raw = bytes(item)                 # a Python bytes: ONE copy, kept for compatibility
```

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

### `read_npz()` — Read an NPZ array

```python
arr = s3dlio.read_npz("s3://bucket/train/sample_000001.npz", array_name="x")   # numpy float32 array
```

`read_npz(uri, array_name=None)` fetches the object and returns the named array (the first one when
`array_name` is omitted) as a NumPy `float32` array (only little-endian `<f4`, C-order arrays are supported; any other dtype is an error). Since 0.10.0 a stored (uncompressed) member is sliced out of the
fetched archive in place and checked against the CRC-32 the archive recorded, so it is never copied out of the
archive; the one copy is the format decode (the little-endian `float32` payload is decoded into the aligned array
NumPy takes over, because the member's bytes are only 1-byte aligned inside the archive). A compressed member is
inflated into a new buffer (a transform). A malformed or hostile array header (an out-of-range shape, a misplaced
parenthesis) is an error, not a crash.

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
        w.write(buf)  # BytesView: shared with the parts, never copied (140 MiB >= the 16 MiB part size)

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

# Save — torch.save() writes to an in-memory buffer, then close() uploads that buffer via
# put_bytes() read in place (no copy; before 0.10.0 close() copied it with bytes(...)).
# If the upload fails, close() raises and the data stays in the writer, so close() can be retried.
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

# Upload: close() uploads the accumulated data read in place (no copy); if the upload fails,
# close() raises and the data stays in the writer so close() can be retried
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

### DLIO storage integration

`s3dlio.integrations.dlio.s3dlio_storage.S3dlioStorage` and
`s3dlio.integrations.dlio.s3_torch_storage.S3PyTorchConnectorStorage` (`put_data`) no longer copy the payload
(0.10.0). A `BytesIO` (what `numpy.save` and `torch.save` write into) is passed on as `getbuffer()`, a view of its own
memory, instead of `getvalue()`, which copied it into a new `bytes`; objects uploaded in parts are sliced as
`memoryview` views, not as `bytes` slices (slicing `bytes` copies). A generic stream that only offers `read()` can
hand its bytes over only by reading them, which is one copy (stream to memory); an object that only offers
`getvalue()` copies as that method does.

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
any bytes-like object (serialize your state first, e.g. `torch.save` into a `BytesIO`, then pass
`bio.getbuffer()`; see "Checkpoint payloads and copies" below), and every
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
)   # bytes: ONE documented copy (a Python bytes owns its memory)

# The same read, same validation, same errors, but ZERO copies: returns a BytesView
view = s3dlio.load_checkpoint_with_validation_view(
    "s3://bucket/checkpoints/run1/shards/<ckpt-id>/rank-0.bin", validate_integrity=True
)
mv = memoryview(view)                       # or np.frombuffer(view, dtype=...)
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

#### Checkpoint payloads and copies (0.10.0)

`save_checkpoint`, `save_distributed_shard`, `PyCheckpointStore.save`, `PyCheckpointWriter.save_distributed_shard`
and `PyCheckpointStream.write_chunk` accept `bytes`, `bytearray`, `memoryview`, NumPy arrays, `BytesView` (and any
other buffer-protocol object) and **read the payload in place: no copy**. They used to require `bytes` and copied it
once or twice.

* **The call blocks until the data is written**, with the GIL released. The buffer is held (exported) for the whole call:
  a `bytearray` cannot be resized meanwhile (`BufferError`), and no thread may modify its bytes until the call returns
  (a data race: the stored bytes are undefined). The call does not return, normally or with an error, until every
  native reader of the buffer is done and the export is released, so the buffer is yours again as soon as it returns.
  If a storage backend still holds the buffer after `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (30 s by default) the call
  raises `ExportStillReferencedError` (the data may or may not have been written); the buffer stays pinned, so do not
  modify, resize or free it.
* A **non-contiguous** array (strided, transposed, ...) is gathered once into logical order, with the GIL held; pass a
  C-contiguous array to avoid that copy. An object that only has `__bytes__()` costs one copy made by Python.
* **Compression is a transform.** With `compression_level` set the encoder reads your buffer and writes new data; it
  is not counted as a copy of your payload.
* **A rejected payload creates nothing.** An `int`, a `bool`, an arbitrary iterable or a pointer-bearing buffer raises
  `TypeError`; for `PyCheckpointStream.write_chunk` that happens before the shard is created, so a later `finalize()`
  still reports that no data was written.
* **Reading.** `load_checkpoint_with_validation_view` returns a `BytesView` (zero copies);
  `load_checkpoint_with_validation` keeps returning `bytes` for compatibility (one documented copy). A `.zst` shard is
  decoded by a transform either way, into a buffer pre-sized from the frame header when it records the size.

The Rust API follows the same rule: `Writer::{put_shard, put_shard_with_key, save_checkpoint,
save_distributed_shard}` and `CheckpointStore::save` take `impl IntoShardBytes` (see the Changelog).

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
  base, the manifest (selected by its file name `ckpt-<step>-<ts>.json`) must record the same step
  in its `global_step` (otherwise the call raises with nothing deleted), it must parse and
  validate (a complete manifest that fails validation, for example one missing a rank, cannot
  be deleted by step), and every shard key it lists must be
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

A process runs **one op-log session at a time**: `init_op_log` -> `finalize_op_log`,
and after `finalize_op_log` a new `init_op_log` starts a new session (issue #294),
any number of times. `init_op_log` while already active is a no-op when it names the
file being written (any spelling: `x`, `./x`, an absolute path, `file:///abs/x`,
`dir/../x`, a symlink or hard link to it) and raises `RuntimeError` for any other file:
the message names the file being written and the requested path and says to call
`finalize_op_log()` first; the requested file is not created or truncated and the
session carries on. Of several threads calling `init_op_log` at once exactly one
creates the file (same file: every call returns; different files: the rest raise). After
`os.fork()` the child does not share the parent's session (its writer thread is not
copied): `is_op_log_active()` is `False` in the child, which can start its own session,
but never on a file an ancestor process was still writing at the fork, however often
the child starts and finishes sessions of its own (a `RuntimeError`: initialising it
would truncate a live log; a log the ancestor had already finalized is not protected
and may be initialised). Stores
created before the fork stay silent in the child (one line on stderr says why; see
"Fork safety" below for everything else a forked child can do). If the
active file was renamed or deleted behind the session's back (log rotation), a request
for the old name is a `RuntimeError` saying the session is writing a file that was
renamed or deleted (it keeps writing the open file); a request that is a directory,
`/dev/null`, a FIFO or a dangling symlink is described as such. Files are compared by
`(device, inode)`: on a filesystem without stable inode numbers (FUSE without `use_ino`,
CIFS `noserverino`, overlayfs without `xino`) the same path can look like another file
and raise while active; finalize first. A re-init waits
for the previous session's writer to finish (with the GIL released), then creates
its file exactly as a first init does: the same path replaces the old log, a
different path leaves it untouched. `finalize_op_log` is safe to repeat. The
module-level functions (`get`, `put_bytes`, `stat`, `list`, `delete`, `get_async`,
...) check the op-log on every call: a call made while it is active is recorded
once, a call made before `init_op_log` or after `finalize_op_log` is not,
whichever call first used the scheme.
An `*_async` call is checked when it is made, not when the event loop first runs it: an
awaitable created in one session never writes into a later one. Datasets and loaders over
`s3://` (`create_dataset`, `create_async_loader`, ...) and `MultiEndpointStore` objects
(`create_multi_endpoint_store*`) follow the rule for objects you construct: they capture the logger when they are constructed and use it for every fetch,
so one built before `init_op_log` is never logged and one built in session 1 never writes
into session 2. The synchronous `get_many` on `s3://` takes the logger once, at entry.
Objects you construct yourself (stores, readers, writers) capture the logger at construction: those created before
`init_op_log` are never logged, and those created before `finalize_op_log` stop
recording when it runs and stay silent after a later `init_op_log` (they never
write into the new session). A call that is still running when `finalize_op_log` runs writes
no row, even if a new session has started by the time it completes, and it is not
counted in the new session's `op_log_stats()`; the call itself is unaffected.

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
workers stop and check `rejected_closed == 0` too. Such stores stay rejected after a
re-init: their calls never reach the new session's file.

`op_log_stats()` works while logging is active and after `finalize_op_log()`
(while active, `accepted > written` is normal). `finalize_op_log()` prints one
warning on stderr if any record was lost, naming only the non-zero counts (the
exact number of queue-full drops included). In the default mode the first
queue-full drop also prints one stderr note per session; later drops are only
counted. To avoid queue-full drops set `S3DLIO_OPLOG_LOSSLESS=1` (callers wait
instead of dropping) and/or raise `S3DLIO_OPLOG_BUF`.

### Fork safety (`os.fork`, `multiprocessing`, `DataLoader` workers)

`fork()` copies only the calling thread. Before the fix for issue #297 a forked child that called s3dlio after
its parent had used it waited for ever on runtime threads that do not exist (issue #297). Now a
child **works**: it notices the fork and builds its own runtimes, thread pools, store cache and
clients on first use, and prints ONE line on stderr the first time it abandons inherited state:

```
s3dlio: this process was forked after s3dlio had already been used; the inherited runtime, ...
```

Set `S3DLIO_FORK_WARN=0` to silence it. A child whose parent never used s3dlio prints nothing.

What works in a child, whatever the parent did: every function (`get`, `put_bytes`, `list`, `stat`,
`get_many`, `get_async`, `get_into_async`, ..., `init_op_log`, `generate_data`,
`generate_data_with_threads`, `generate_into_buffer`, `generate_npz_bytes`), and every object the
**child itself creates** (`PyCheckpointStore`, writers, `MultiEndpointStore`, datasets, loaders,
`Generator`). The child never reuses a connection of the parent, and the parent is not affected by
what the child does.

What does **not** work: an object **created in the parent** that owns a runtime and/or
connections, used in the child. It raises a catchable `RuntimeError` ("created before fork();
create it in the child process"), it never hangs and never raises `PanicException`:

```python
store = s3dlio.PyCheckpointStore("s3://bucket/ckpt")      # parent
pid = os.fork()
if pid == 0:
    store.save(...)                                        # RuntimeError: created before fork()
    store = s3dlio.PyCheckpointStore("s3://bucket/ckpt")   # create it in the child instead
```

This covers the checkpoint objects, `create_*_writer` / `MultipartUploadWriter`,
`MultiEndpointStore`, datasets, loaders and their iterators, parquet stream objects and `Generator`.
Deleting such an object, garbage collection and interpreter exit in the child are safe: its state
is leaked, never dropped (dropping would join threads that do not exist, or abort the parent's
multipart upload).

Practical advice:

- PyTorch `DataLoader(num_workers>0)` and `multiprocessing.Pool` (fork) work. Create s3dlio
  objects inside the worker (the dataset's `__getitem__` or `worker_init_fn`), not in the main process.
- `spawn` and `forkserver` remain the most robust start methods (they never inherit anything, so
  they never print the line above); Python 3.14 makes `forkserver` the default on Linux.
- Each forked process builds its own pools. On a many-core machine set `S3DLIO_RT_THREADS`
  (and `configure_tokio_threads(n)`) so N workers do not each start one thread per core.
- A process that execs after fork (`subprocess`, `os.execv`) is a new program: nothing applies.

Known limits (none observed as a hang):

- s3dlio never drops a Python reference without the GIL and its op-log lock is protected by fork
  handlers, so a busy parent can fork safely: measured 0 hangs in several thousand forked children
  with 4 and 8 parent threads calling s3dlio in a tight loop (see the Changelog). Earlier
  builds hung 1 to 9% of such children; that was s3dlio's own off-GIL drops, not a PyO3 limit.
- A streaming `Generator` created in a child whose ancestor already created one runs with one
  thread (dgen-data's pool is not fork-aware): correct, slower.
- Fork from inside a thread-local destructor is not protected; one-time `OnceLock`/`Lazy`
  initialisation that is running on another thread at the fork, and third-party tables (dashmap,
  tracing), are theoretical windows.

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
| `put_bytes_async(uri, data, *, borrow=True)` | Async upload; reads a mutable buffer in place until the awaitable completes (`borrow=False`: one copy now) | Coroutine[None] |
| `put_many(items)` | Batch upload `[(uri, data), ...]`; stops at the first failure | None |
| `put_many_async(items, max_in_flight=64, *, borrow=True)` | Async batch upload; stops at the first failure | Coroutine[None] |
| `get(uri)` | Download object | BytesView |
| `get_into_async(uri, dst, *, expected_len)` | Read a whole object into your own writable byte buffer (v0.10.0) | Coroutine[int] (`expected_len`) |
| `head_async(uri)` | Size and version identity without reading the object (v0.10.0) | Coroutine[`ObjectMetadata` or `None`] |
| `object_capabilities(uri)` | Whether this build supports `get_into_async` / `head_async` for the URI; no network (v0.10.0) | dict |
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
| `mkdir_async(uri)` | Async mkdir | Coroutine[None] |
| `upload(src_uri, dest_uri)` | Bulk upload files | None |
| `download(src_uri, dest_dir)` | Bulk download files | None |
| `mp_get(uri, procs, jobs, num, template)` | Multi-process GET | dict |
| `create_bucket(name)` | Create S3 bucket | None |
| `delete_bucket(name)` | Delete S3 bucket | None |

### Types and exceptions added in v0.10.0

| Name | Description |
|------|-------------|
| `ObjectMetadata` | Frozen value returned by `head_async`: `size`, `etag` (opaque, not an MD5), `version_id` (S3 or Azure version ID when versioning is on, the GCS generation) |
| `ObjectNotFoundError` | `FileNotFoundError` subclass: `get_into_async` on a missing object (or a missing `file://` / `direct://` file) |
| `LengthMismatchError` | `ValueError` subclass with `.expected` and `.actual` (`None` when unknown): the object's size is not `expected_len` |
| `UnsupportedOperationError` | `NotImplementedError` subclass: the build or provider lacks the operation (no fallback download is attempted) |
| `ExportStillReferencedError` | A backend still references your buffer after `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`; the buffer stays pinned |
| `ReservedBuffer` | Owner of the memory behind a `MultipartUploadWriter.reserve()` memoryview (`mv.obj`) |

See [get_into_async() / head_async()](#get_into_async--head_async--read-into-your-own-buffer--new-in-v0100) for the contracts.

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
`put_all_endpoints`) accept the same payloads as `put_bytes` (#281), take a keyword-only `borrow=True` (the buffer is
read in place until the awaitable completes; `borrow=False` copies a mutable buffer once, now), and every
method returning `Coroutine[None]` resolves to `None`, not `()` (#271):

| Method | Returns |
|--------|---------|
| `put(uri, data, *, borrow=True)` | Coroutine[None] |
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

`write_chunk` and `write_owned_bytes` borrow the buffer for the duration of the call; `MultipartUploadWriter.write`
borrows a large buffer until `close()` / `abort()` returns. See "Buffer ownership of the write calls" and
"MultipartUploadWriter.write(): the buffer is borrowed until close()" in the Streaming API section (`copy=True` for a
snapshot).

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
| `save_checkpoint(uri, step, epoch, framework, data, user_meta=None)` | Save single-rank checkpoint; returns manifest key. `data` is any bytes-like object, read in place (zero copy); the call blocks until written |
| `load_checkpoint(uri)` | Load latest checkpoint; returns `BytesView` or `None` |
| `load_checkpoint_with_validation_view(uri, validate_integrity=True)` | Same read, validation and errors as `load_checkpoint_with_validation`, but returns a `BytesView` (zero copies) |
| `load_checkpoint_with_validation(uri, validate_integrity=True)` | Read one checkpoint shard by URI; returns `bytes` (one documented copy; use the `_view` form for none). `validate_integrity=True` verifies the returned bytes against the checksum the manifest recorded at write time and raises on a mismatch, an unlisted shard, a missing recorded checksum, or a non-shard URI; `False` reads unverified |
| `save_distributed_shard(uri, step, epoch, framework, data, world_size, rank)` | Save this rank's shard; returns shard-meta dict. `data` as for `save_checkpoint` |
| `finalize_distributed_checkpoint(uri, step, epoch, framework, shard_metas, world_size, rank, user_meta=None)` | Rank 0 publishes the manifest; returns manifest key |
| `PyCheckpointStore(uri, strategy=None, multipart_threshold=None, compression_level=None)` | Store: `save`, `load_latest`, `list_checkpoints`, `delete_checkpoint`, `writer(world_size, rank)`, `reader()` |
| `PyCheckpointWriter` (`store.writer(world_size, rank)`) | `save_distributed_shard(step, epoch, framework, data)`, `get_distributed_shard_stream(step, epoch, framework)`, `finalize_distributed_checkpoint(step, epoch, framework, shard_metas, user_meta=None)` |
| `PyCheckpointReader` (`store.reader()`) | `load_latest_manifest()` -> dict or `None`; `read_shard_by_rank(manifest, rank)` -> `BytesView` (checks the shard against the `checksum` in the manifest dict you pass it; reads unvalidated when that entry has none, and does not consult the stored manifests) |

### Logging

| Function | Description |
|----------|-------------|
| `init_logging(level)` | Set log level: trace, debug, info, warn, error |
| `init_op_log(path)` | Start an op-log session on a local file (plain path or `file://` destination, s3dlio's file-store convention, not full RFC 8089). Allowed again after `finalize_op_log()` (new session, fresh `op_log_stats()`; the same path is truncated, #294); while a session is active: a no-op for the same file, `RuntimeError` for another file |
| `finalize_op_log()` | Write all accepted records (unless the writer fails: an I/O error or a writer-thread panic) and close the log; **call it before the process exits, nothing does it for you**; releases the GIL while draining; idempotent; ends the session, `init_op_log` can start a new one (#294) |
| `is_op_log_active()` | True between init and finalize only |
| `op_log_stats()` | `None` before `init_op_log`; else a dict `accepted` / `written` / `dropped_full` / `rejected_closed` / `writer_failed` / `writer_error` / `lossless` / `rejected_unformattable` / `rejected_unsafe_field` / `normalized_error_text`, available while active and after finalize, to detect lost records (complete = `accepted == written`, no drops, no refusals, `writer_error is None`, `rejected_unformattable == 0`, `rejected_unsafe_field == 0`; `normalized_error_text` is a disclosure, not a loss) |

---

## Migration Guide

### From v0.9.128 to v0.10.0

The full list is in [docs/Changelog.md](Changelog.md); the changes that can break existing code:

| Change | What to do |
|--------|-----------|
| `put_bytes_async`, `put_many_async` and the `MultiEndpointStore` writes borrow a mutable buffer by default | Do not modify, resize or free the buffer until the `await` has finished, or pass `borrow=False` (one copy) |
| `MultipartUploadWriter.write()` borrows a large buffer until `close()` | Use a fresh buffer per write, or `copy=True`; see [the write contract](#multipartuploadwriterwrite-the-buffer-is-borrowed-until-close) |
| `put_bytes`, `put_bytes_async`, `put_many`, `put_many_async` and the `MultiEndpointStore` writes raise `TypeError` for an `int`, a `bool` or an arbitrary iterable | Wrap a list of byte values in `bytes(...)` |
| `put_many` and `put_many_async` stop at the first failed item | Catch that item's own error; queued items are not started |
| `mkdir_async`, `put_bytes_async`, `put_many_async` and the `MultiEndpointStore` writes resolve to `None`, not `()` | Do not compare the result with `()` |
| Recursive `download` writes `<dest>/<key path relative to the listing base>` | Expect nested directories instead of one flat directory |
| `load_checkpoint_with_validation(..., validate_integrity=True)` now validates | A shard with no recorded checksum (written before 0.10.0) raises; pass `validate_integrity=False`, or use `load_checkpoint` / `load_latest` |
| `PyDataset.from_uris(uris, opts=None)` | The ignored keyword `_opts` is renamed `opts` |
| `S3DLIO_CHUNK_SIZE=0` and `S3DLIO_RANGE_CONCURRENCY=0` are errors on the S3 range paths | Unset them or set a positive value |
| Rust: the `gcs-community` feature is removed | Use `backend-gcs`; `GCS_ENDPOINT_URL` and `STORAGE_EMULATOR_HOST` now configure it |

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
