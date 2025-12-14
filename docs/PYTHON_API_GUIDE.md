# s3dlio Python API Guide

**Version:** 0.9.25  
**Last Updated:** December 9, 2025

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Storage Operations](#core-storage-operations)
4. [Multi-Backend Support](#multi-backend-support)
5. [Multi-Endpoint Load Balancing](#multi-endpoint-load-balancing)
6. [Streaming API](#streaming-api)
7. [AI/ML Integration](#aiml-integration)
8. [Checkpoint System](#checkpoint-system)
9. [Advanced Features](#advanced-features)
10. [Performance Optimization](#performance-optimization)
11. [API Reference](#api-reference)

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

### exists() - Check Object Existence

Check if an object exists without downloading it:

```python
# Check single object
if s3dlio.exists("s3://bucket/key"):
    print("Object exists")
    
# Check before operation
if not s3dlio.exists("s3://bucket/config.json"):
    s3dlio.put("s3://bucket/config.json", b'{}')

# Async version (returns coroutine)
exists = await s3dlio.exists_async("s3://bucket/key")
```

**Note:** `exists()` uses `stat()` internally, so it's efficient for single checks.
For bulk existence checks, consider using `stat_many_async()`.

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

### Custom Endpoints (S3-Compatible, Emulators, Proxies)

s3dlio supports custom endpoints for all three cloud backends, enabling use with:
- **S3-compatible systems**: MinIO, Ceph, VAST
- **Local emulators**: Azurite (Azure), fake-gcs-server (GCS)
- **Multi-protocol proxies**

**Amazon S3 / S3-Compatible:**
```python
import os
# Path-style addressing is automatic when endpoint is set
os.environ['AWS_ENDPOINT_URL'] = 'http://localhost:9000'  # MinIO
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

data = s3dlio.get("s3://mybucket/mykey")
```

**Azure Blob (Azurite/Custom):**
```python
import os
os.environ['AZURE_STORAGE_ENDPOINT'] = 'http://127.0.0.1:10000'  # Azurite
os.environ['AZURE_STORAGE_ACCOUNT'] = 'devstoreaccount1'
os.environ['AZURE_STORAGE_KEY'] = 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=='

data = s3dlio.get("az://devstoreaccount1/container/blob")
```

**Google Cloud Storage (fake-gcs-server/Custom):**
```python
import os
os.environ['GCS_ENDPOINT_URL'] = 'http://localhost:4443'  # fake-gcs-server
# OR use the standard GCS emulator convention:
os.environ['STORAGE_EMULATOR_HOST'] = 'localhost:4443'

data = s3dlio.get("gs://testbucket/testkey")
```

| Backend | Environment Variable | Alternative |
|---------|---------------------|-------------|
| S3 | `AWS_ENDPOINT_URL` | - |
| Azure | `AZURE_STORAGE_ENDPOINT` | `AZURE_BLOB_ENDPOINT_URL` |
| GCS | `GCS_ENDPOINT_URL` | `STORAGE_EMULATOR_HOST` |

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

## Multi-Endpoint Load Balancing

### Overview

Multi-endpoint storage (v0.9.14+) enables **parallel operations across multiple storage locations** with automatic load balancing. Use this for:
- **Distributed storage**: Multiple S3 buckets/regions/endpoints
- **High throughput**: Aggregate bandwidth across endpoints  
- **Load distribution**: Balance requests across storage systems
- **Fault tolerance**: Continue operating if endpoints fail

### Creating Multi-Endpoint Stores

#### From URI List

```python
import s3dlio

# Create store from explicit list of URIs
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://us-east-1-bucket",
        "s3://us-west-2-bucket", 
        "s3://eu-west-1-bucket"
    ],
    strategy="round_robin"  # or "least_connections"
)
```

#### From URI Template

Use `{start...end}` syntax for range expansion:

```python
# Expands to endpoint1, endpoint2, ..., endpoint10
store = s3dlio.create_multi_endpoint_store_from_template(
    uri_template="s3://my-bucket-{1...10}",
    strategy="round_robin"
)

# Multiple ranges with zero-padding
store = s3dlio.create_multi_endpoint_store_from_template(
    uri_template="s3://region-{1...3}/shard-{01...99}",
    strategy="least_connections"
)
```

#### From File

Load URIs from a text file (one per line):

```python
# endpoints.txt:
# s3://bucket-1
# s3://bucket-2
# s3://bucket-3

store = s3dlio.create_multi_endpoint_store_from_file(
    file_path="/path/to/endpoints.txt",
    strategy="round_robin"
)
```

### Load Balancing Strategies

#### Round-Robin (Default)

Distributes requests evenly across all endpoints in rotation:

```python
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://endpoint1", "s3://endpoint2", "s3://endpoint3"],
    strategy="round_robin"
)

# Requests distributed: endpoint1 → endpoint2 → endpoint3 → endpoint1 → ...
```

**Best for:**
- Uniform workloads where all endpoints have similar performance
- Simple load distribution without overhead
- Predictable request patterns

#### Least-Connections

Routes requests to the endpoint with fewest active operations:

```python
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://endpoint1", "s3://endpoint2"],
    strategy="least_connections"
)

# Requests routed to endpoint with lowest active_requests count
```

**Best for:**
- Variable request latencies
- Heterogeneous endpoint performance
- Adaptive load balancing under varying conditions

### Async Operations

All multi-endpoint operations are **async** and must be awaited:

```python
import asyncio
import s3dlio

async def main():
    store = s3dlio.create_multi_endpoint_store(
        uris=["s3://bucket-1", "s3://bucket-2"],
        strategy="round_robin"
    )
    
    # Put operation (async)
    await store.put("s3://bucket-1/data.bin", b"Hello, World!")
    
    # Get operation (async)
    result = await store.get("s3://bucket-1/data.bin")
    print(bytes(result))  # Convert BytesView to bytes
    
    # List operation (async)
    objects = await store.list("s3://bucket-1/prefix/", recursive=True)
    for obj in objects:
        print(f"URI: {obj['uri']}, Size: {obj['size']}")
    
    # Delete operation (async)
    await store.delete("s3://bucket-1/data.bin")

# Run async code
asyncio.run(main())
```

### Zero-Copy Data Access

Multi-endpoint stores return `BytesView` for zero-copy memory access:

```python
import asyncio
import numpy as np

async def process_data():
    store = s3dlio.create_multi_endpoint_store(
        uris=["s3://bucket-1", "s3://bucket-2"],
        strategy="round_robin"
    )
    
    # Get data (returns BytesView)
    view = await store.get("s3://bucket-1/array.bin")
    
    # Zero-copy access via memoryview
    mv = view.memoryview()
    
    # Use with NumPy (zero-copy)
    array = np.frombuffer(mv, dtype=np.float32)
    print(f"Array shape: {array.shape}")
    
    # Convert to bytes if needed (copies data)
    data = bytes(view)

asyncio.run(process_data())
```

### Range Requests

```python
async def get_ranges():
    store = s3dlio.create_multi_endpoint_store(
        uris=["s3://bucket-1", "s3://bucket-2"],
        strategy="least_connections"
    )
    
    # Get bytes 1000-1999 (offset=1000, length=1000)
    view = await store.get_range(
        "s3://bucket-1/large_file.bin",
        offset=1000,
        length=1000
    )
    
    data = bytes(view)  # 1000 bytes
    print(f"Retrieved {len(data)} bytes")

asyncio.run(get_ranges())
```

### Statistics and Monitoring

Track per-endpoint and total statistics:

```python
import asyncio

async def monitor_stats():
    store = s3dlio.create_multi_endpoint_store(
        uris=["s3://bucket-1", "s3://bucket-2", "s3://bucket-3"],
        strategy="round_robin"
    )
    
    # Perform operations
    for i in range(100):
        await store.put(f"s3://bucket-{(i % 3) + 1}/file{i}.bin", b"data")
    
    # Get per-endpoint statistics
    endpoint_stats = store.get_endpoint_stats()
    for stat in endpoint_stats:
        print(f"Endpoint: {stat['uri']}")
        print(f"  Total requests: {stat['total_requests']}")
        print(f"  Bytes read: {stat['bytes_read']}")
        print(f"  Bytes written: {stat['bytes_written']}")
        print(f"  Active requests: {stat['active_requests']}")
        print(f"  Errors: {stat['error_count']}")
    
    # Get total aggregated statistics
    total_stats = store.get_total_stats()
    print(f"\nTotal Statistics:")
    print(f"  Total requests: {total_stats['total_requests']}")
    print(f"  Total bytes read: {total_stats['bytes_read']}")
    print(f"  Total bytes written: {total_stats['bytes_written']}")
    print(f"  Active requests: {total_stats['active_requests']}")
    print(f"  Total errors: {total_stats['error_count']}")

asyncio.run(monitor_stats())
```

### Store Properties

Query store configuration:

```python
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://bucket-1", "s3://bucket-2"],
    strategy="least_connections"
)

# Get number of endpoints
count = store.endpoint_count()
print(f"Endpoints: {count}")

# Get current strategy
strategy = store.strategy()
print(f"Strategy: {strategy}")  # "least_connections"
```

### Complete Example

```python
import asyncio
import s3dlio

async def distributed_upload():
    """Upload 1000 files across 10 S3 buckets"""
    
    # Create multi-endpoint store with template expansion
    store = s3dlio.create_multi_endpoint_store_from_template(
        uri_template="s3://my-bucket-{1...10}",
        strategy="round_robin"
    )
    
    print(f"Created store with {store.endpoint_count()} endpoints")
    print(f"Using strategy: {store.strategy()}")
    
    # Upload files in parallel
    tasks = []
    for i in range(1000):
        bucket_num = (i % 10) + 1
        uri = f"s3://my-bucket-{bucket_num}/file{i:04d}.dat"
        data = f"File {i} data".encode()
        tasks.append(store.put(uri, data))
    
    # Wait for all uploads
    await asyncio.gather(*tasks)
    
    # Check statistics
    total_stats = store.get_total_stats()
    print(f"\nUploaded {total_stats['total_requests']} files")
    print(f"Total bytes written: {total_stats['bytes_written']:,}")
    print(f"Errors: {total_stats['error_count']}")
    
    # List all objects from first bucket
    objects = await store.list("s3://my-bucket-1/", recursive=False)
    print(f"\nBucket 1 contains {len(objects)} objects")

# Run
asyncio.run(distributed_upload())
```

### Testing with pytest-asyncio

```python
import pytest
import s3dlio

@pytest.mark.asyncio
async def test_multi_endpoint_operations():
    """Test multi-endpoint store with pytest"""
    store = s3dlio.create_multi_endpoint_store(
        uris=["file:///tmp/endpoint1", "file:///tmp/endpoint2"],
        strategy="round_robin"
    )
    
    # Test put/get
    test_data = b"test data"
    await store.put("file:///tmp/endpoint1/test.bin", test_data)
    
    result = await store.get("file:///tmp/endpoint1/test.bin")
    assert bytes(result) == test_data
    
    # Test statistics
    stats = store.get_total_stats()
    assert stats['total_requests'] >= 2  # put + get
```

### Best Practices

1. **Use templates for many endpoints**: `{1...100}` is cleaner than listing 100 URIs
2. **Choose strategy based on workload**: 
   - Round-robin for uniform access patterns
   - Least-connections for variable latencies
3. **Monitor statistics**: Check error counts and per-endpoint load
4. **Handle BytesView properly**: Call `.memoryview()` for zero-copy or `bytes()` to convert
5. **Use asyncio.gather() for parallel operations**: Maximize throughput with concurrent requests
6. **Test with pytest-asyncio**: Use `@pytest.mark.asyncio` decorator for async tests

### Performance Tips

- **Parallel uploads/downloads**: Use `asyncio.gather()` with 100+ concurrent operations
- **Round-robin for speed**: Lowest overhead when endpoints have similar performance  
- **Least-connections for mixed latency**: Adapts to slow endpoints automatically
- **Zero-copy with NumPy/PyTorch**: Use `.memoryview()` to avoid copying large arrays
- **Monitor endpoint health**: Check `error_count` in statistics to detect failing endpoints

For more details, see [Multi-Endpoint Storage Guide](MULTI_ENDPOINT_GUIDE.md).

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

#### TFRecord Index Generation (v0.9.17+)

**NEW**: Generate TFRecord files with accompanying index files for TensorFlow Data Service compatibility.

**Rust API:**
```rust
use s3dlio::data_formats::{build_tfrecord_with_index, TfRecordWithIndex};

// Generate synthetic data
let raw_data = s3dlio::generate_controlled_data(102400, 1, 1);

// Create TFRecord with index in single pass
let result = build_tfrecord_with_index(
    100,    // num_records
    1024,   // record_size_bytes
    &raw_data
)?;

// result.data: Bytes containing TFRecord file
// result.index: Bytes containing index file

// Write both files to storage
use s3dlio::object_store::store_for_uri;
let store = store_for_uri("s3://bucket/tfrecords/")?;

// Write TFRecord data
store.put("train_00000.tfrecord", &result.data).await?;

// Write index file (16 bytes per record)
store.put("train_00000.tfrecord.index", &result.index).await?;
```

**Index Format:**
- **16 bytes per record**: Little-endian format
  - 8 bytes: Record offset in TFRecord file (u64)
  - 8 bytes: Record length in bytes (u64)
- **TensorFlow compatible**: Standard format for TensorFlow Data Service
- **Zero overhead**: Generated during TFRecord creation (single pass)

**Python Index Parsing:**
```python
import struct
import s3dlio

# Fetch index from S3
index_bytes = s3dlio.get("s3://bucket/train_00000.tfrecord.index")

# Parse index (16 bytes per record)
num_records = len(index_bytes) // 16
records_info = []

for i in range(num_records):
    # Unpack little-endian u64 offset and u64 length
    offset, length = struct.unpack('<QQ', index_bytes[i*16:(i+1)*16])
    records_info.append((offset, length))
    print(f"Record {i}: offset={offset}, length={length}")

# Use index for random access
import tensorflow as tf

def indexed_read(tfrecord_uri, index_info, record_idx):
    """Read specific record using index."""
    offset, length = index_info[record_idx]
    
    # Fetch specific byte range from S3
    tfrecord_bytes = s3dlio.get_range(
        tfrecord_uri, 
        offset, 
        offset + length
    )
    
    # Parse TFRecord
    return tf.train.Example.FromString(tfrecord_bytes)
```

**TensorFlow Data Service Integration:**
```python
import tensorflow as tf

# TensorFlow Data Service can use index files for efficient sharding
data_service_config = tf.data.experimental.service.DispatcherConfig(
    work_dir="/tmp/dispatcher_work"
)

# Dataset with index-aware sharding
dataset = tf.data.Dataset.list_files("s3://bucket/*.tfrecord")
dataset = dataset.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Data Service automatically uses .index files if present
dataset = dataset.apply(
    tf.data.experimental.service.distribute(
        processing_mode="distributed_epoch",
        service="grpc://localhost:5000"
    )
)
```

**Use Cases:**
- **Distributed training**: Efficient dataset sharding across workers
- **Random access**: Seek to specific records without scanning
- **Large-scale datasets**: Minimize I/O for distributed TensorFlow training
- **Data Service**: Enable TensorFlow Data Service optimizations
- **Cloud storage**: Works with S3, GCS, Azure Blob for remote datasets

**Performance Benefits:**
- **No scanning overhead**: Direct access to any record via offset
- **Worker coordination**: Data Service uses indices for optimal shard distribution
- **Bandwidth efficiency**: Fetch only required byte ranges from cloud storage
- **Single-pass generation**: Index computed during TFRecord creation (no extra cost)

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

#### Multi-Array NPZ Creation (v0.9.17+)

**NEW**: Create NPZ archives with multiple named arrays for PyTorch/JAX-style datasets.

**Rust API:**
```rust
use s3dlio::data_formats::npz::build_multi_npz;
use ndarray::ArrayD;

// Create arrays
let data = ArrayD::zeros(vec![100, 224, 224, 3]);     // Images
let labels = ArrayD::ones(vec![100]);                  // Labels
let metadata = ArrayD::from_elem(vec![100, 5], 1.0);   // Metadata

// Package into multi-array NPZ
let arrays = vec![
    ("data", &data),
    ("labels", &labels),
    ("metadata", &metadata),
];

let npz_bytes = build_multi_npz(arrays)?;

// Write to storage
use s3dlio::object_store::store_for_uri;
let store = store_for_uri("s3://bucket/")?;
store.put("dataset.npz", &npz_bytes).await?;
```

**Python Loading:**
```python
import numpy as np
import s3dlio

# Load from cloud storage
data_bytes = s3dlio.get("s3://bucket/dataset.npz")

# Save locally and load with NumPy
with open("dataset.npz", "wb") as f:
    f.write(data_bytes)

data = np.load("dataset.npz")
print(data.files)  # ['data', 'labels', 'metadata']

images = data['data']      # Shape: (100, 224, 224, 3)
labels = data['labels']    # Shape: (100,)
metadata = data['metadata'] # Shape: (100, 5)
```

**PyTorch DataLoader:**
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import s3dlio
import io

class S3NPZDataset(Dataset):
    def __init__(self, s3_uris):
        """Dataset loading NPZ files from S3.
        
        Args:
            s3_uris: List of S3 URIs to NPZ files
        """
        self.uris = s3_uris
    
    def __len__(self):
        return len(self.uris)
    
    def __getitem__(self, idx):
        # Fetch from S3
        npz_bytes = s3dlio.get(self.uris[idx])
        
        # Load into memory
        with io.BytesIO(npz_bytes) as f:
            data = np.load(f)
            return {
                'data': torch.from_numpy(data['data']),
                'labels': torch.from_numpy(data['labels']),
                'metadata': torch.from_numpy(data['metadata'])
            }

# Use with DataLoader
uris = [f"s3://bucket/train_{i:06d}.npz" for i in range(1000)]
dataset = S3NPZDataset(uris)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

for batch in loader:
    images = batch['data']    # Tensor: (32, 224, 224, 3)
    labels = batch['labels']  # Tensor: (32,)
    # Train your model
```

**Key Features:**
- **Zero-copy design**: Efficient memory handling with `Bytes`
- **Standard format**: 100% compatible with NumPy's `np.load()`
- **Named arrays**: Organize data, labels, and metadata logically
- **Cloud-native**: Works seamlessly with S3, GCS, Azure Blob
- **Framework agnostic**: Use with PyTorch, JAX, TensorFlow, or pure NumPy

**Use Cases:**
- **Image classification**: Store images + labels + augmentation metadata
- **Scientific ML**: Store simulation results + parameters + timestamps
- **NLP datasets**: Store embeddings + labels + text metadata
- **Distributed training**: Each worker loads different NPZ shards

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

### Multi-Endpoint API (v0.9.14+)

| Function | Description |
|----------|-------------|
| `create_multi_endpoint_store(uris, strategy)` | Create store from URI list |
| `create_multi_endpoint_store_from_template(uri_template, strategy)` | Create store from template with range expansion |
| `create_multi_endpoint_store_from_file(file_path, strategy)` | Create store from file containing URIs |

**MultiEndpointStore Methods** (all async):

| Method | Description | Returns |
|--------|-------------|---------|
| `put(uri, data)` | Upload bytes | Coroutine[None] |
| `get(uri)` | Download bytes | Coroutine[BytesView] |
| `get_range(uri, offset, length)` | Range request | Coroutine[BytesView] |
| `list(uri, recursive)` | List objects | Coroutine[List[Dict]] |
| `delete(uri)` | Delete object | Coroutine[None] |
| `get_endpoint_stats()` | Per-endpoint statistics | List[Dict] |
| `get_total_stats()` | Aggregated statistics | Dict |
| `endpoint_count()` | Number of endpoints | int |
| `strategy()` | Current load balancing strategy | str |

**Load Balancing Strategies:**
- `"round_robin"` - Distribute requests evenly in rotation
- `"least_connections"` - Route to endpoint with fewest active requests

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
