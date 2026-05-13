# s3dlio Python Data Loaders

**Last Updated:** May 12, 2026

This guide covers both Python data loaders shipped with s3dlio. Both loaders work with
every s3dlio URI scheme (`s3://`, `file://`, `direct://`, `az://`, `gs://`) without
code changes.

↩ [Back to Python API Guide](PYTHON_API_GUIDE.md)

## Table of Contents

1. [**Object Data Loader (General-Purpose)**](#object-data-loader-general-purpose) ✨ New in v0.9.100  
   `PyDataset.from_uris()` · `PyBytesAsyncDataLoader.items()` · `collect_batch(n)`
2. [**Parquet DataLoader**](#parquet-dataloader) ✨ New in v0.9.98  
   `create_async_loader()` with row-group streaming, epoch-2 fast path, Arrow IPC decode

---

## Object Data Loader (General-Purpose)

**New in v0.9.100.** A backend-agnostic, sliding-window object streaming API built on
Rust/Tokio.  Works with every s3dlio URI scheme — `s3://`, `file://`, `direct://`,
`az://`, `gs://` — without changing any Python code.  Designed as a general-purpose
replacement for PyTorch's default dataloader when the bottleneck is storage I/O rather
than CPU preprocessing.

> **DLIO benchmark integration**: `_s3dlio` storage path in dlio_benchmark uses these
> APIs to deliver a true sliding window with no chunk boundaries — see
> [`dlio_benchmark/docs/Batch-API-Design-Analysis.md`](../../dlio_benchmark/docs/Batch-API-Design-Analysis.md).

### `PyDataset.from_uris(uris)` — dataset from a pre-built URI list

```python
uris = [
    "s3://bucket/train/img_000001.jpg",
    "s3://bucket/train/img_000002.jpg",
    # ... any number of objects ...
]
ds = s3dlio.PyDataset.from_uris(uris)
print(len(ds))   # == len(uris)
```

Creates a map-style dataset directly from a `list[str]` of full URIs. **No network
listing occurs** — the caller provides the manifest. The storage backend is inferred
from the first URI.

| Constraint | Detail |
|------------|--------|
| All URIs same scheme | Backend selected from `uris[0]`; mixing schemes in one call is not supported |
| Non-empty list | `from_uris([])` raises `RuntimeError` |
| Any s3dlio scheme | `s3://`, `file://`, `direct://`, `az://`, `gs://` all work |

Contrast with `PyDataset(prefix_uri)`, which performs a `list_objects()` call to
discover files at runtime. Use `from_uris()` when you already have a file manifest,
a per-worker shard, or epoch-shuffled ordering computed in Python.

### `PyBytesAsyncDataLoader` — sliding-window streaming loader

```python
ds     = s3dlio.PyDataset.from_uris(uris)
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 64})
```

`PyBytesAsyncDataLoader` wraps any `PyDataset` and drives up to `prefetch` concurrent
GET requests in flight at all times via Tokio's `buffer_unordered`.  Each item Python
consumes frees one Tokio slot and immediately triggers one new GET — a true sliding
window.

**Constructor options** (`opts` dict):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"prefetch"` | `int` | `32` | Concurrent in-flight GETs (channel capacity = backpressure depth) |
| `"skip_head"` | `bool` | `True` | Skip the per-object HEAD request; see [HEAD-skip optimisation](#head-skip-optimisation) below |

### HEAD-skip optimisation (`skip_head`)

By default, s3dlio's range-optimisation path issues one **HEAD request** per object on
first access to learn the object's size.  Once the size is known it decides whether to
use a single streaming GET (small objects) or parallel range GETs (large objects ≥ 32 MiB).

`skip_head=True` (the default) bypasses this HEAD and goes straight to a streaming GET.
After the GET completes, the actual size — taken from the response body length — is
stored in the in-process `ObjectSizeCache`.  From that point on the cached size is used
for routing decisions, so **range splitting fires correctly from epoch 2 onwards** with
zero extra network requests.

```
Epoch 1 (skip_head=True):   GET → cache size            (no HEAD ever)
Epoch 2+:                   cache hit → route decision   (range split if >= 32 MiB)
```

**When to pass `skip_head=False`:** only when you need range splitting to be active
from the very first epoch — typically large objects (video frames, large HDF5 shards,
LLM checkpoints) in a **single-epoch job** where there is no epoch 2 to benefit from
the cached sizes.

```python
# Small objects (images, NPZ, CSV) — default, no HEAD ever
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 64})

# Large objects, single-epoch job — HEAD fires on epoch 1 to enable range GETs
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 8, "skip_head": False})
```

The `skip_head` flag is backed by a **process-wide latch**: the first loader that
passes `skip_head=True` sets the latch for all subsequent loaders in the same process
(the latch is irreversible within a process lifetime).  Alternatively, set
`S3DLIO_SKIP_HEAD=1` in the environment before startup for the same effect without
any code change.

> **Rule of thumb**: leave `skip_head` at its default (`True`) unless you have ≥ 32 MiB
> objects **and** either (a) only one epoch, or (b) you need peak throughput on epoch 1.
> For all other workloads — images, NPZ, HDF5 row-groups, CSVs — the HEAD is pure
> overhead and skipping it is strictly better.

### `loader.__iter__()` — bytes-only iterator

```python
for view in loader:          # synchronous; GIL released while waiting
    arr = numpy.frombuffer(view, dtype=numpy.uint8)  # zero-copy
```

Returns `PyBytesView` items — zero-copy wrappers around Rust `Bytes`.  Use when you
do not need to know which URI produced each item (e.g. when processing all objects
identically in submission order).

### `loader.items()` → `PyObjectDataLoaderSyncIter` — URI-carrying iterator

```python
for item in loader.items():          # GIL released per item while waiting
    print(item.uri, len(item))       # which object, how many bytes
    raw = bytes(item)                # one copy to Python heap (only if needed)
```

Returns `PyObjectItem` values in **network-completion order** (which may differ from
submission order).  Each item carries the source URI so Python never needs a parallel
index.

**`PyObjectItem` attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `item.uri` | `str` | Full URI of the completed GET (`s3://…`, `file://…`, …) |
| `item.byte_count` | `int` | Bytes transferred |
| `len(item)` | `int` | Alias for `byte_count` |
| `bytes(item)` | `bytes` | Raw data — creates one Python-heap copy |

### `item_iter.collect_batch(n)` → `list[PyObjectItem]`

```python
item_iter = loader.items()
while batch := item_iter.collect_batch(16):   # empty list = end of stream
    for item in batch:                         # plain Python list — no __next__ overhead
        process(item.uri, bytes(item))
```

Drains up to `n` completed items from the Rust channel with **one GIL crossing** for
the whole batch, then returns a plain Python `list[PyObjectItem]`.  The inner
`for item in batch` loop iterates a native Python list — no per-item `__next__()`
dispatch.

**Choosing `n`:** A natural default is `max(1, batch_size // num_samples_per_file)` —
one training batch worth of files.  Returns an empty list when the stream is exhausted
(use as the `while` stop condition).

**GIL behaviour:**
- `items().__next__()` — 1 GIL release + reacquire per item
- `collect_batch(n)` — 1 GIL release + reacquire for `n` items

At ~1,600 completions/sec per subprocess worker (loopback latency) the per-item GIL
cost is ~0.33% of wall time, so `collect_batch()` is primarily useful for reducing
Python object-allocation churn rather than GIL contention (DLIO workers are separate
OS processes with independent GILs).

### Full example — DLIO-style training loop

```python
import s3dlio

# Called from each PyTorch DataLoader worker subprocess
def stream_epoch(obj_keys, batch_size, num_spf=1, prefetch=64):
    """
    obj_keys  — list of storage keys (relative to bucket/prefix)
    batch_size — training batch size in samples
    num_spf    — samples per file (1 for JPEG/PNG, 4 for NPZ, etc.)
    """
    uris = [f"s3://my-bucket/data/{k}" for k in obj_keys]
    uri_to_key = {u: k for k, u in zip(obj_keys, uris)}

    ds = s3dlio.PyDataset.from_uris(uris)
    loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": prefetch})
    item_iter = loader.items()

    collect_n = max(1, batch_size // num_spf)   # files per batch
    sample_buf = 0
    dummy_batch = b"\x00"  # placeholder — real code would yield actual tensors

    while batch := item_iter.collect_batch(collect_n):
        for item in batch:
            key = uri_to_key.get(item.uri, item.uri)
            record_telemetry(key, len(item))     # byte count — no copy
            for _ in range(num_spf):
                sample_buf += 1
                if sample_buf >= batch_size:
                    yield dummy_batch
                    sample_buf -= batch_size
```

---

## Parquet DataLoader

**New in v0.9.98.** A production-ready, epoch-aware Parquet DataLoader for AI/ML training
loops. Works with **any s3dlio storage backend** — S3, Azure Blob Storage, GCS, local
`file://` paths, and `direct://` (O_DIRECT). Each dataset item is one Parquet **row group**
— the natural I/O unit for distributed training with large columnar datasets.

> **Feature flags**: Both `parquet` and `parquet-arrow` are **enabled by default** — no
> build flags required. See the [complete guide](Parquet_Data-Loader.md) for full details.

> **Backend-agnostic**: Only the URI prefix changes when switching storage backends. No
> code changes are needed — `s3://`, `az://`, `gs://`, `file://`, and `direct://` all work
> with the same `create_async_loader()` call.

### Quick start

```python
import s3dlio

# Raw mode (default) — Python receives compressed Parquet column-chunk bytes
loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "prefetch": 32}
)

for item in loader:
    # item["data"]: bytes  — the row-group payload
    # item["uri"]:  str   — URI of the source file (any s3dlio scheme)
    # item["rg_idx"]: int — row-group index within the file
    import pyarrow.parquet as pq, io
    table = pq.read_table(io.BytesIO(item["data"]))
    # ... feed to PyTorch / JAX / TensorFlow ...
```

### Decode modes

| Mode | `decode=` value | Python receives | When to use |
|------|-----------------|-----------------|-------------|
| **Raw** (default) | `"raw"` or omit | Compressed Parquet column-chunk `bytes` | Pure I/O benchmark, or Python controls decoding |
| **ArrowIpc** | `"arrow"` | Arrow IPC stream `bytes` (Rust-decoded) | CPU-bound training; eliminates Python decode overhead |

```python
# Arrow IPC mode — Rust decodes to Arrow RecordBatch, serialises to IPC bytes
loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "decode": "arrow", "prefetch": 32}
)

for item in loader:
    import pyarrow as pa
    batch = pa.ipc.open_stream(pa.py_buffer(item["data"])).read_next_batch()
    # batch: pyarrow.RecordBatch — fully decoded, in Arrow memory format
```

### `create_async_loader()` — Parquet options

```python
loader = s3dlio.create_async_loader(
    uri,   # any s3dlio URI prefix: "s3://bucket/train/", "az://container/train/",
           #   "gs://bucket/train/", "file:///data/train/", "direct:///mnt/nvme/train/"
    opts   # dict of options
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `"format"` | `str` | — | Must be `"parquet"` to activate Parquet mode |
| `"decode"` | `str` | `"raw"` | `"raw"` or `"arrow"` |
| `"columns"` | `list[int]` \| `None` | `None` | Column subset (0-based indices); `None` = all columns |
| `"footer_cap"` | `int` | `4194304` | Bytes to fetch from file tail for footer parsing (4 MiB default covers all known workloads including DLRM at ~2.66 MiB) |
| `"prefetch"` | `int` | `32` | Concurrent in-flight row-group GETs (bounded channel capacity) |

### Epoch-2+ fast path — zero re-fetches

After the first epoch, row-group byte ranges are cached in a process-global index. Epoch 2+
construction issues **zero storage footer reads** — only a directory listing still hits the
network or filesystem. This applies to all backends (S3, Azure, GCS, file, direct).

```python
for epoch in range(num_epochs):
    loader = s3dlio.create_async_loader(
        "s3://bucket/train/",
        {"format": "parquet", "prefetch": 32}
    )
    for item in loader:
        train(item["data"])
    # Index persists across loop iterations — epoch 2+ is 2.5× faster to construct
```

**Measured speedup (MinIO, 2 files / 4 row groups):**

| Epoch | Construction time | What happened |
|-------|-----------------|---------------|
| 1 | 20.4 ms | `list_objects` + 2 footer GETs |
| 2+ | 8.3 ms | `list_objects` only; DashMap lookup ≈ 0 ms |
| Speedup | **2.5×** | Scales to 10×+ for 64+ files |

### Memory model

s3dlio holds **only metadata** in RAM — no row-group bulk data is buffered between `get()`
calls. 8 concurrent workers sharing the same process share the process-global caches:

| Component | RAM | Shared? |
|-----------|-----|--------|
| `extents` (byte ranges per RG) | ~48 B × N_rgs | Per instance |
| `parquet_file_cache` (parsed footer) | few MB per file | Process-global, Arc-shared |
| `parquet_index` (DashMap) | ~80 B × N_rgs | Process-global singleton |
| Active `get()` payload | 1 row-group (1–100 MB) | Freed on Python release |

**8 workers, 100 MB RGs, prefetch=2:** peak ≈ 1.6 GB data + ~20 MB shared metadata.

### dlio_benchmark integration

```python
# In a DLIO plugin / reader callback:
loader = s3dlio.create_async_loader(
    config.data_folder,   # any s3dlio URI: "s3://…", "az://…", "gs://…", "file://…", "direct://…"
    {
        "format":   "parquet",
        "decode":   "raw",     # decode_mode=none equivalent
        "prefetch": config.prefetch_size,
    }
)
for item in loader:
    yield item["data"]  # return bytes to DLIO without decoding
```

📖 **[Complete Parquet DataLoader Guide](Parquet_Data-Loader.md)** — full API reference,
Rust examples, architecture deep dive, and all test documentation.

---

↩ [Back to Python API Guide](PYTHON_API_GUIDE.md)
