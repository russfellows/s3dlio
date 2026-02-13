# s3dlio Python API Guide

**Version:** 0.9.50  
**Last Updated:** February 13, 2026

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Core Storage Operations](#core-storage-operations)
5. [Zero-Copy Data Flow](#zero-copy-data-flow)
6. [Batch Operations](#batch-operations)
7. [Multi-Backend Support](#multi-backend-support)
8. [Multi-Endpoint Load Balancing](#multi-endpoint-load-balancing)
9. [Streaming API](#streaming-api)
10. [AI/ML Integration](#aiml-integration)
11. [s3torchconnector Compatibility](#s3torchconnector-compatibility)
12. [Checkpoint System](#checkpoint-system)
13. [Performance & Threading](#performance--threading)
14. [Advanced Features](#advanced-features)
15. [API Reference](#api-reference)
16. [Migration Guide](#migration-guide)

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

- Python 3.12+
- Rust 1.90+ (for building from source)
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
| `direct://` | `direct:///path/to/file` | Direct I/O (O_DIRECT) |

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

**Data path**: `Python bytes` → `Bytes::copy_from_slice` (one unavoidable copy from Python heap) → `Bytes` (Arc-counted, zero-copy through upload pipeline).

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

### list() — List Objects

```python
# List all objects under prefix (returns full URIs)
uris = s3dlio.list("s3://bucket/prefix/")
# ['s3://bucket/prefix/file1.dat', 's3://bucket/prefix/file2.dat', ...]

# Recursive listing
uris = s3dlio.list("s3://bucket/prefix/", recursive=True)

# With glob pattern filter
uris = s3dlio.list("s3://bucket/prefix/", pattern="*.npz")
```

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

### delete() — Remove Object

```python
s3dlio.delete("s3://bucket/key")
```

### mkdir() — Create Directory / Prefix

```python
s3dlio.mkdir("file:///data/output/subdir")
await s3dlio.mkdir_async("s3://bucket/prefix/")
```

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
| `s3dlio.put_bytes(uri, pydata)` | 1 | Unavoidable Python→Rust copy |

---

## Batch Operations

### put_many() — Batch Upload (v0.9.50+)

Upload multiple objects in a single call with parallel execution:

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
results = s3dlio.get_many(uris, workers=64)

# Async version
results = await s3dlio.get_many_async(uris)
```

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
```

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
```python
import os
os.environ['AZURE_STORAGE_ACCOUNT'] = 'myaccount'
os.environ['AZURE_STORAGE_KEY'] = 'mykey'
# Optional custom endpoint:
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

### Save / Load

```python
# Save checkpoint to any backend
s3dlio.save_checkpoint(
    uri="s3://bucket/checkpoints/epoch_10.bin",
    data={"epoch": 10, "model": model.state_dict()},
    compress=True
)

# Load checkpoint
ckpt = s3dlio.load_checkpoint("s3://bucket/checkpoints/epoch_10.bin")

# Load with validation
ckpt = s3dlio.load_checkpoint_with_validation(
    uri="s3://bucket/checkpoints/epoch_10.bin",
    expected_keys=["epoch", "model"]
)
model.load_state_dict(ckpt["model"])
```

### Distributed Checkpointing

```python
# Each rank saves its shard
s3dlio.save_distributed_shard(
    uri=f"s3://bucket/ckpt/shard_{rank}.bin",
    shard_data=local_state,
    rank=rank,
    world_size=world_size
)

# Rank 0 finalizes
if rank == 0:
    s3dlio.finalize_distributed_checkpoint(
        base_uri="s3://bucket/ckpt/",
        world_size=world_size
    )
```

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
s3dlio.get_many_stats(uris, concurrency=100)

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
s3dlio.init_op_log("file:///tmp/operations.log")
# ... operations are automatically logged ...
s3dlio.finalize_op_log()
```

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
| `put_bytes(uri, data)` | Upload bytes | None |
| `put_bytes_async(uri, data)` | Async upload | Coroutine |
| `put_many(items)` | Batch upload `[(uri, data), ...]` | None |
| `put_many_async(items)` | Async batch upload | Coroutine |
| `get(uri)` | Download object | BytesView |
| `get_range(uri, offset, length=None)` | Server-side range request | BytesView |
| `get_many(uris, workers)` | Parallel download | List[BytesView] |
| `get_many_async(uris)` | Async parallel download | Coroutine |
| `list(uri, recursive=False, pattern=None)` | List objects | List[str] |
| `stat(uri)` | Get metadata | dict |
| `stat_async(uri)` | Async metadata | Coroutine |
| `stat_many_async(uris)` | Batch metadata | Coroutine |
| `exists(uri)` | Check existence | bool |
| `exists_async(uri)` | Async existence check | Coroutine |
| `delete(uri)` | Delete object | None |
| `mkdir(uri)` | Create directory/prefix | None |
| `mkdir_async(uri)` | Async mkdir | Coroutine |
| `upload(src_uri, dest_uri)` | Bulk upload files | None |
| `download(src_uri, dest_dir)` | Bulk download files | None |
| `mp_get(uri, procs, jobs, num, template)` | Multi-process GET | dict |
| `create_bucket(name)` | Create S3 bucket | None |
| `delete_bucket(name)` | Delete S3 bucket | None |

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

**MultiEndpointStore Methods** (all async):

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
| `save_checkpoint(uri, data, compress)` | Save checkpoint |
| `load_checkpoint(uri)` | Load checkpoint |
| `load_checkpoint_with_validation(uri, expected_keys)` | Load with validation |
| `save_distributed_shard(uri, data, rank, world_size)` | Save distributed shard |
| `finalize_distributed_checkpoint(base_uri, world_size)` | Finalize distributed checkpoint |

### Logging

| Function | Description |
|----------|-------------|
| `init_logging(level)` | Set log level: trace, debug, info, warn, error |
| `init_op_log(path)` | Enable operation logging to file |
| `finalize_op_log()` | Flush and close operation log |
| `is_op_log_active()` | Check if operation logging is active |

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
