# s3dlio Python API Guide

**Version:** 0.9.12  
**Last Updated:** November 3, 2025

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Storage Operations](#core-storage-operations)
4. [Multi-Backend Support](#multi-backend-support)
5. [Streaming API](#streaming-api)
6. [AI/ML Integration](#aiml-integration)
7. [Checkpoint System](#checkpoint-system)
8. [Advanced Features](#advanced-features)
9. [Performance Optimization](#performance-optimization)
10. [API Reference](#api-reference)

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

### Requirements

- Python 3.8+
- Rust 1.90+ (for building from source)
- Optional: PyTorch, JAX, or TensorFlow for ML integration

---

## Quick Start

### Basic Operations

```python
import s3dlio

# Initialize logging (optional)
s3dlio.init_logging("info")  # Options: trace, debug, info, warn, error

# Put data to storage
s3dlio.put("s3://my-bucket/data.bin", b"Hello, World!")
s3dlio.put("file:///tmp/local.bin", b"Local data")
s3dlio.put("gs://my-bucket/cloud.bin", b"GCS data")
s3dlio.put("az://account/container/blob.bin", b"Azure data")

# Get data from storage
data = s3dlio.get("s3://my-bucket/data.bin")
print(data)  # b'Hello, World!'

# List objects
objects = s3dlio.list("s3://my-bucket/prefix/")
for uri in objects:
    print(uri)

# Get metadata
metadata = s3dlio.stat("s3://my-bucket/data.bin")
print(f"Size: {metadata['size']} bytes")
print(f"Last modified: {metadata['last_modified']}")

# Delete objects
s3dlio.delete("s3://my-bucket/data.bin")
```

---

## Core Storage Operations

### put() - Upload Data

Upload bytes to any supported storage backend:

```python
# Basic upload
s3dlio.put("s3://bucket/key", b"data")

# Upload from file
with open("local_file.bin", "rb") as f:
    data = f.read()
    s3dlio.put("s3://bucket/key", data)

# Async version (returns coroutine)
await s3dlio.put_async("s3://bucket/key", b"data")
```

**Supported URIs:**
- `s3://bucket/key` - Amazon S3
- `gs://bucket/key` - Google Cloud Storage
- `az://account/container/key` - Azure Blob Storage
- `file:///path/to/file` - Local filesystem
- `direct:///path/to/file` - Direct I/O (O_DIRECT)

### get() - Download Data

Download bytes from storage:

```python
# Basic download
data = s3dlio.get("s3://bucket/key")

# Download with range request
data = s3dlio.get_range("s3://bucket/key", offset=1024, length=4096)

# Async download
data = await s3dlio.get_many_async(["s3://bucket/key1", "s3://bucket/key2"])
```

### list() - List Objects

List objects under a prefix:

```python
# List all objects
objects = s3dlio.list("s3://bucket/prefix/")

# Returns list of full URIs
for uri in objects:
    print(uri)  # s3://bucket/prefix/file1.dat, ...

# List local files
files = s3dlio.list("file:///data/directory/")
```

### stat() - Get Metadata

Get object metadata without downloading:

```python
# Single object
metadata = s3dlio.stat("s3://bucket/key")
print(metadata['size'])           # Bytes
print(metadata['last_modified'])  # Timestamp
print(metadata['etag'])          # ETag/hash

# Multiple objects (async)
stats = await s3dlio.stat_many_async([
    "s3://bucket/key1",
    "s3://bucket/key2"
])
```

### delete() - Remove Objects

Delete objects from storage:

```python
# Single object
s3dlio.delete("s3://bucket/key")

# Multiple objects (use list comprehension)
objects = s3dlio.list("s3://bucket/prefix/")
for uri in objects:
    s3dlio.delete(uri)
```

---

## Multi-Backend Support

### Supported Backends

s3dlio provides **unified API** across all storage backends:

```python
# Amazon S3 / MinIO
s3dlio.put("s3://bucket/data.bin", data)

# Google Cloud Storage
s3dlio.put("gs://bucket/data.bin", data)

# Azure Blob Storage  
s3dlio.put("az://account/container/data.bin", data)

# Local Filesystem
s3dlio.put("file:///tmp/data.bin", data)

# Direct I/O (O_DIRECT)
s3dlio.put("direct:///nvme/data.bin", data)
```

### Authentication

**Amazon S3:**
```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'
os.environ['AWS_REGION'] = 'us-east-1'
```

**Google Cloud Storage:**
```bash
# Use gcloud CLI
gcloud auth application-default login
```

**Azure Blob Storage:**
```python
import os
os.environ['AZURE_STORAGE_ACCOUNT'] = 'myaccount'
os.environ['AZURE_STORAGE_KEY'] = 'mykey'
```

### Backend-Specific Features

```python
# S3 bucket management
s3dlio.create_bucket("my-bucket")
s3dlio.delete_bucket("my-bucket")

# S3-specific list (returns keys only, not full URIs)
keys = s3dlio.list_objects("s3://bucket/prefix/")  # DEPRECATED

# Multi-process GET for maximum throughput
result = s3dlio.mp_get(
    uri="s3://bucket/prefix/",
    procs=4,        # Number of worker processes
    jobs=64,        # Concurrent operations per process
    num=1000,       # Number of objects
    template="obj_{}.dat"
)
print(f"Throughput: {result['throughput_mb_s']} MB/s")
```

---

## Streaming API

### Streaming Writers

For large uploads with compression and chunking:

```python
# Create writer for S3
writer = s3dlio.create_s3_writer(
    "s3://bucket/large_file.bin.zst",
    options=s3dlio.PyWriterOptions()
)

# Write data in chunks
writer.write(chunk1)
writer.write(chunk2)
writer.write(chunk3)

# Finalize and get stats
stats = writer.finalize()
print(f"Wrote {stats['bytes_written']} bytes")
print(f"Compressed size: {stats['compressed_size']}")
```

### Writer Options

```python
options = s3dlio.PyWriterOptions()
options.compression = "zstd"  # Options: none, zstd, gzip, lz4
options.compression_level = 3  # 1-22 for zstd
options.chunk_size = 4194304  # 4 MiB chunks

# Backend-specific writers
s3_writer = s3dlio.create_s3_writer(uri, options)
azure_writer = s3dlio.create_azure_writer(uri, options)
fs_writer = s3dlio.create_filesystem_writer(uri, options)
direct_writer = s3dlio.create_direct_filesystem_writer(uri, options)
```

### Zero-Copy Access

Use `BytesView` for zero-copy memory access:

```python
# Get data as BytesView (zero-copy)
view = s3dlio.get("s3://bucket/key")  # Returns BytesView

# Access as memoryview (zero-copy)
memview = view.memoryview()

# Convert to bytes (copy)
data = bytes(view)
```

---

## AI/ML Integration

### PyTorch Datasets

```python
from s3dlio import S3MapDataset, S3IterableDataset

# Map-style dataset (random access)
dataset = S3MapDataset(
    uri="s3://bucket/training-data/",
    pattern="*.npz",        # Filter pattern
    transform=my_transform   # Optional transform function
)

# Iterable dataset (streaming)
dataset = S3IterableDataset(
    uri="s3://bucket/training-data/",
    pattern="*.npz",
    shuffle=True,
    num_workers=4
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    # Train your model
    pass
```

### JAX Integration

```python
from s3dlio import S3JaxIterable

# Create JAX-compatible iterable
jax_iter = S3JaxIterable(
    uri="gs://bucket/training-data/",
    pattern="*.npz",
    batch_size=32
)

# Iterate over batches
for batch in jax_iter:
    # JAX training code
    pass
```

### TensorFlow Integration

```python
from s3dlio import make_tf_dataset

# Create TF Dataset
ds = make_tf_dataset(
    uri="s3://bucket/training-data/",
    pattern="*.tfrecord",
    batch_size=32,
    shuffle=True
)

# Use in training
for batch in ds:
    # TensorFlow training code
    pass
```

### Generic Data Loaders

```python
# Universal dataset (works with any backend)
dataset = s3dlio.create_dataset("gs://bucket/data/")

# Async data loader
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    options=s3dlio.PyLoaderOptions()
)

# Iterate asynchronously
async for batch in loader:
    process(batch)
```

---

## Checkpoint System

### Save Checkpoints

```python
# Save checkpoint to any backend
checkpoint_data = {
    "epoch": 10,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}

s3dlio.save_checkpoint(
    uri="s3://bucket/checkpoints/ckpt_epoch_10.bin",
    data=checkpoint_data,
    compress=True
)
```

### Load Checkpoints

```python
# Load checkpoint
checkpoint = s3dlio.load_checkpoint("s3://bucket/checkpoints/ckpt_epoch_10.bin")

# Load with validation
checkpoint = s3dlio.load_checkpoint_with_validation(
    uri="s3://bucket/checkpoints/ckpt_epoch_10.bin",
    expected_keys=["epoch", "model_state", "optimizer_state"]
)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
```

### Distributed Checkpointing

```python
# Save distributed shards (multi-rank)
s3dlio.save_distributed_shard(
    uri=f"s3://bucket/checkpoints/shard_{rank}.bin",
    shard_data=local_state,
    rank=rank,
    world_size=world_size
)

# Finalize checkpoint (rank 0 only)
if rank == 0:
    s3dlio.finalize_distributed_checkpoint(
        base_uri="s3://bucket/checkpoints/",
        world_size=world_size
    )
```

---

## Advanced Features

### NPZ Files

```python
# Read NumPy .npz files directly
arrays = s3dlio.read_npz("s3://bucket/data.npz")
print(arrays.keys())  # ['arr_0', 'arr_1', ...]
```

### TFRecord Index

```python
# Create TFRecord index for NVIDIA DALI
s3dlio.create_tfrecord_index(
    input_path="s3://bucket/dataset.tfrecord",
    output_path="file:///tmp/dataset.idx"
)

# Read index entries
entries = s3dlio.get_tfrecord_index_entries("/tmp/dataset.idx")
for entry in entries:
    print(f"Offset: {entry['offset']}, Size: {entry['size']}")
```

### Operation Logging

```python
# Enable operation logging
s3dlio.init_op_log("file:///tmp/operations.log")

# Perform operations (automatically logged)
s3dlio.put("s3://bucket/key", data)
s3dlio.get("s3://bucket/key")

# Finalize log
s3dlio.finalize_op_log()

# Check if logging is active
if s3dlio.is_op_log_active():
    print("Logging enabled")
```

### Batch Operations

```python
# Upload multiple files
s3dlio.upload(
    local_dir="/data/files/",
    remote_uri="s3://bucket/uploads/",
    pattern="*.dat"
)

# Download multiple files
s3dlio.download(
    remote_uri="s3://bucket/downloads/",
    local_dir="/data/output/",
    pattern="*.dat"
)

# Get multiple objects async
uris = ["s3://bucket/file1", "s3://bucket/file2", "s3://bucket/file3"]
results = await s3dlio.get_many_async(uris)
```

---

## Performance Optimization

### Pre-Stat Size Caching (v0.9.10+)

For benchmarking workloads, pre-stat objects to eliminate stat overhead:

```python
# Pre-stat all objects once
uris = s3dlio.list("s3://bucket/dataset/")
s3dlio.get_many_stats(uris, concurrency=100)

# Subsequent gets skip stat operations (2.5x faster)
for uri in uris:
    data = s3dlio.get(uri)  # Uses cached size
```

### Parallel Downloads

```python
# Download many objects in parallel
uris = [f"s3://bucket/obj_{i}.dat" for i in range(1000)]
results = s3dlio.get_many(uris, workers=64)
```

### Multi-Process GET

For maximum throughput with very large datasets:

```python
result = s3dlio.mp_get(
    uri="s3://bucket/dataset/",
    procs=8,           # Use 8 processes
    jobs=128,          # 128 concurrent operations per process
    num=10000,         # Download 10,000 objects
    template="data_{}.bin"
)

print(f"Total: {result['total_bytes']} bytes")
print(f"Time: {result['duration_seconds']}s")
print(f"Throughput: {result['throughput_mb_s']} MB/s")
print(f"Operations/sec: {result['ops_per_sec']}")
```

---

## API Reference

### Core Functions

| Function | Description | Backends |
|----------|-------------|----------|
| `put(uri, data)` | Upload bytes | All |
| `put_async(uri, data)` | Async upload | All |
| `get(uri)` | Download bytes | All |
| `get_range(uri, offset, length)` | Range request | All |
| `list(uri)` | List objects | All |
| `stat(uri)` | Get metadata | All |
| `stat_async(uri)` | Async metadata | All |
| `delete(uri)` | Delete object | All |
| `upload(local, remote)` | Bulk upload | All |
| `download(remote, local)` | Bulk download | All |

### Streaming API

| Class/Function | Description |
|----------------|-------------|
| `PyWriterOptions` | Writer configuration |
| `PyObjectWriter` | Streaming writer |
| `create_s3_writer(uri, opts)` | Create S3 writer |
| `create_azure_writer(uri, opts)` | Create Azure writer |
| `create_filesystem_writer(uri, opts)` | Create filesystem writer |
| `create_direct_filesystem_writer(uri, opts)` | Create Direct I/O writer |

### AI/ML Integration

| Class/Function | Description | Framework |
|----------------|-------------|-----------|
| `S3MapDataset` | Random access dataset | PyTorch |
| `S3IterableDataset` | Streaming dataset | PyTorch |
| `S3JaxIterable` | JAX-compatible iterable | JAX |
| `make_tf_dataset(uri)` | TensorFlow dataset | TensorFlow |
| `create_dataset(uri)` | Generic dataset | Any |
| `create_async_loader(uri)` | Async data loader | Any |

### Checkpoint System

| Function | Description |
|----------|-------------|
| `save_checkpoint(uri, data)` | Save checkpoint |
| `load_checkpoint(uri)` | Load checkpoint |
| `load_checkpoint_with_validation(uri, keys)` | Load with validation |
| `save_distributed_shard(uri, data, rank)` | Save distributed shard |
| `finalize_distributed_checkpoint(uri, world_size)` | Finalize distributed checkpoint |

### Logging & Debugging

| Function | Description |
|----------|-------------|
| `init_logging(level)` | Initialize logging (trace/debug/info/warn/error) |
| `init_op_log(path)` | Enable operation logging |
| `finalize_op_log()` | Finalize operation log |
| `is_op_log_active()` | Check if logging enabled |

---

## Common Patterns

### Training Loop with Checkpoints

```python
import s3dlio

# Load dataset
dataset = s3dlio.create_dataset("s3://bucket/training-data/")

# Training loop
for epoch in range(num_epochs):
    for batch in dataset:
        # Training step
        loss = train_step(model, batch)
    
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        s3dlio.save_checkpoint(
            f"s3://bucket/checkpoints/epoch_{epoch}.bin",
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "loss": loss
            }
        )
```

### Distributed Data Loading

```python
from torch.utils.data import DataLoader
from s3dlio import S3IterableDataset

# Each rank loads different shard
dataset = S3IterableDataset(
    uri=f"s3://bucket/shard_{rank}/",
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4
)

for batch in loader:
    # Distributed training
    pass
```

---

## Migration from v0.8.x

### Deprecated Functions

| Old (v0.8.x) | New (v0.9.x) | Notes |
|--------------|--------------|-------|
| `list_objects(uri)` | `list(uri)` | Returns full URIs |
| `get_object(uri)` | `get(uri)` | Unified interface |
| `stat_object(uri)` | `stat(uri)` | Unified interface |

### Backward Compatibility

v0.9.x maintains backward compatibility. Old functions still work but are deprecated.

---

## Examples

See `python/tests/` directory for comprehensive examples:
- `test_comprehensive_api.py` - All storage operations
- `bench_s3-torch_v8.py` - Performance benchmarking
- `test_checkpoint_framework_integration.py` - Checkpoint examples
- `pytorch_smoke.py` - PyTorch integration
- `jax_smoke.py` - JAX integration
- `tf_smoke.py` - TensorFlow integration

---

## Support & Contributing

- **GitHub**: https://github.com/russfellows/s3dlio
- **Issues**: https://github.com/russfellows/s3dlio/issues
- **Documentation**: https://github.com/russfellows/s3dlio/tree/main/docs

**License:** AGPL-3.0
