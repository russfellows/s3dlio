# s3dlio - Universal Storage I/O Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/russfellows/s3dlio)
[![Tests](https://img.shields.io/badge/tests-175%20passing-brightgreen)](docs/Changelog.md)
[![Rust Tests](https://img.shields.io/badge/rust%20tests-175%2F175-brightgreen)](docs/Changelog.md)
[![Version](https://img.shields.io/badge/version-0.9.26-blue)](https://github.com/russfellows/s3dlio/releases)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)

High-performance, multi-protocol storage library for AI/ML workloads with universal copy operations across S3, Azure, GCS, local file systems, and DirectIO.

## 🌟 Latest Release

### v0.9.26 - DLIO Benchmark Integration (December 2025)

**🆕 New Features:**

**DLIO Benchmark Integration**
- Added comprehensive integration support for [Argonne DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark)
- Two installation options:
  - **Option 1 (Recommended):** New `storage_type: s3dlio` with explicit configuration
  - **Option 2:** Drop-in replacement for existing S3 configurations
- Enables DLIO to use all s3dlio backends: S3, Azure, GCS, file://, direct://

**Zero-Copy Write Functions**
```python
import s3dlio

# Write bytes directly to any backend
s3dlio.put_bytes("s3://bucket/file.bin", data)

# Create directories/prefixes
s3dlio.mkdir("s3://bucket/my-prefix/")
```

**Azure SDK Update**
- Updated from Azure SDK 0.4.0 to 0.7.0 API

See [Changelog](docs/Changelog.md) for complete details and [DLIO Integration Guide](docs/integration/DLIO_BENCHMARK_INTEGRATION.md) for setup instructions.

---

### v0.9.23 - Azure Blob & GCS Custom Endpoint Support (December 3, 2025)

**🆕 New Features:**

**Custom Endpoint Support for Azure Blob Storage**
- Added environment variable support for custom Azure endpoints
- Primary: `AZURE_STORAGE_ENDPOINT` (e.g., `http://localhost:10000`)
- Alternative: `AZURE_BLOB_ENDPOINT_URL`
- Enables use with Azurite or other Azure-compatible emulators/proxies

```bash
# Azurite (local emulator)
export AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000
sai3-bench util ls az://devstoreaccount1/testcontainer/

# Multi-protocol proxy
export AZURE_STORAGE_ENDPOINT=http://localhost:9001
sai3-bench util ls az://myaccount/mycontainer/
```

**Custom Endpoint Support for Google Cloud Storage**
- Primary: `GCS_ENDPOINT_URL` (e.g., `http://localhost:4443`)
- Alternative: `STORAGE_EMULATOR_HOST` (GCS emulator convention)
- Enables use with fake-gcs-server or other GCS-compatible emulators

```bash
# fake-gcs-server (local emulator)
export GCS_ENDPOINT_URL=http://localhost:4443
sai3-bench util ls gs://testbucket/

# Using STORAGE_EMULATOR_HOST convention
export STORAGE_EMULATOR_HOST=localhost:4443
sai3-bench util ls gs://testbucket/
```

**Related Issue:** [sai3-bench#56](https://github.com/russfellows/sai3-bench/issues/56)

See [Changelog](docs/Changelog.md) for complete details.

---

### v0.9.22 - Client ID & First Byte Tracking (November 25, 2025)

**🆕 New Features:**

**Client ID Support for Multi-Agent Operation Logging**
- Enable identification of which client/agent performed each operation
- Thread-safe implementation with minimal overhead (~10ns per log entry)
- Essential for distributed benchmarking with merged oplogs

```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Set client identifier (agent ID, hostname, etc.)
s3dlio::set_client_id("agent-1")?;

// Set offset for all future log entries
s3dlio::set_clock_offset(offset)?;
```

**Pseudo-Random Data Generation Method**
- Added `generate_controlled_data_prand()` for CPU-efficient data generation
- Performance: prand ~3-4 GB/s (consistent), random ~1-7 GB/s (variable)
- Choose based on needs: `prand` for speed, `random` for true incompressibility

**Issues Resolved:**
- #100: Clock offset support implemented
- #98: Old data generation code now serves as "prand" method
- #95: Range Engine messages confirmed at trace level (already fixed)

See [Changelog](docs/Changelog.md) for complete details.

---

### v0.9.20 - High-Performance List & Delete Optimizations (November 22, 2025)

**🚀 Major Performance Improvements:**

Optimized for workloads with 100K-1M+ objects, targeting **5x faster deletion** (28 min → 5-8 min for 1M objects).

**Key Features:**
- **Batch Delete API**: 1000 objects/request (S3), efficient batching for all backends
- **Streaming List**: Memory-efficient iteration with progress indicators (`-c/--count-only`)
- **Concurrent Pipeline**: Overlapped list+delete operations for maximum throughput

```bash
# Count objects with streaming progress
s3-cli ls -rc s3://bucket/prefix/
# Output: Total objects: 1,234,567 (12.3s, rate: 100,000 objects/s)

# Fast deletion with automatic pipeline
s3-cli delete s3://bucket/prefix/
# Uses concurrent list+delete (1000-object batches, 10K in-flight)
```

**Benefits:**
- Works across all 7 storage backends (S3, Azure, GCS, file://, direct://)
- Python API automatically uses optimizations
- Proper op-log integration for workload replay
- Clean abstractions maintained throughout

See [Changelog](docs/Changelog.md) for complete details.

---

### v0.9.17 - NPY/NPZ Enhancements & TFRecord Index API (November 16, 2025)

**🎯 Multi-Array NPZ Support:**

```rust
use s3dlio::data_formats::npz::build_multi_npz;

// Create NPZ with multiple arrays (data, labels, metadata)
let arrays = vec![
    ("data", &data_array),
    ("labels", &labels_array),
    ("metadata", &metadata_array),
];
let npz_bytes = build_multi_npz(arrays)?;
```

**🔧 TFRecord Index Generation:**

```rust
use s3dlio::data_formats::{build_tfrecord_with_index, TfRecordWithIndex};

// Generate TFRecord with index in single pass
let result = build_tfrecord_with_index(100, 1024, &raw_data)?;
// result.data: TFRecord file, result.index: Index file (16 bytes/record)
```

**Key Features:**
- Multi-array NPZ archives (PyTorch/JAX dataset pattern)
- TFRecord index generation (TensorFlow Data Service compatible)
- Zero-copy `Bytes`-based design
- Full NumPy/Python interoperability
- 11 comprehensive tests (6 NPY + 5 multi-NPZ)

See [Changelog](docs/Changelog.md#version-0917) for complete details.

---

### v0.9.16 - Optional Op-Log Sorting (November 7, 2025)

**📊 Configurable operation log sorting for chronological analysis:**

```bash
# Default: Fast streaming write (unsorted)
sai3-bench run --op-log /tmp/ops.tsv --config test.yaml

# Opt-in: Auto-sort at shutdown (sorted output)
S3DLIO_OPLOG_SORT=1 sai3-bench run --op-log /tmp/ops.tsv --config test.yaml
```

Adds optional automatic sorting of operation logs by start timestamp (controlled via `S3DLIO_OPLOG_SORT` environment variable). Default streaming mode has zero overhead; opt-in sorting adds ~1.2μs per entry. See [Changelog](docs/Changelog.md#version-0916) for details.

---

### v0.9.15 - S3 URI Endpoint Parsing (November 6, 2025)

**🔧 Enhanced URI parsing for multi-endpoint testing:**

```python
import s3dlio

# Parse URI with custom endpoint
result = s3dlio.parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")
# {'endpoint': '192.168.100.1:9001', 'bucket': 'mybucket', 'key': 'data.bin'}
```

Enables tools like sai3-bench and dl-driver to parse MinIO/Ceph endpoints from config files for multi-process testing scenarios. See [Changelog](docs/Changelog.md#version-0915) for details.

---

### v0.9.14 - Multi-Endpoint Storage (November 6, 2025)

**🎯 Load Balancing for High-Throughput Workloads:**

Distribute I/O operations across multiple storage endpoints with intelligent load balancing:

```python
import s3dlio

# Create multi-endpoint store with 3 S3 buckets
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://bucket-1/data",
        "s3://bucket-2/data",
        "s3://bucket-3/data",
    ],
    strategy="least_connections"  # or "round_robin"
)

# Zero-copy data access (buffer protocol)
data = store.get("s3://bucket-1/large-file.bin")
array = np.frombuffer(memoryview(data), dtype=np.float32)

# Monitor load distribution
stats = store.get_endpoint_stats()
```

**Key Features:**
- **Load Balancing**: RoundRobin (simple, predictable) or LeastConnections (adaptive)
- **URI Templates**: `"s3://bucket-{1...10}/data"` → 10 endpoints automatically
- **Zero-Copy Python**: `BytesView` with buffer protocol for numpy/torch
- **Statistics**: Per-endpoint and total metrics (requests, bytes, errors, active connections)
- **Thread Control**: Configure concurrency per endpoint for optimal performance

**Performance:**
- 2-4× throughput scaling with multiple endpoints
- < 0.01% overhead (10-50ns per request)
- Lock-free atomic statistics (zero contention)

📖 **[Complete Multi-Endpoint Guide](docs/MULTI_ENDPOINT_GUIDE.md)** | [Full Details v0.9.14](docs/Changelog.md#version-0914)

---

### v0.9.12 - GCS Factory Fixes (November 3, 2025)

**🔧 Fixed GCS Factory Functions:**

Resolved 4 "not implemented" errors in GCS factory functions that were blocking dl-driver v0.8.4 checkpoint operations:

```rust
// Previously broken, now working:
let store = store_for_uri("gs://bucket/path").await?;
let store = store_for_uri_with_options("gs://bucket/path", options).await?;

// New high-performance cloud factory:
let store = store_for_uri_with_high_performance_cloud("gs://bucket/").await?;
```

**What Changed:**
- Fixed `store_for_uri()` for GCS URIs (was returning "not implemented")
- Fixed `store_for_uri_with_options()` for GCS URIs
- Added `store_for_uri_with_high_performance_cloud()` function family
- Ensures consistent behavior across S3, Azure, and GCS

**Integration:** Required for dl-driver v0.8.4+ checkpoint reload across all backends  
**Compatibility:** Zero breaking changes - fixes previously broken functions

📖 [Full Details v0.9.12](docs/Changelog.md#version-0912) | [Previous Release v0.9.11](docs/Changelog.md#version-0911)

---

### v0.9.11 - Directory Operations (November 2, 2025)

**🎯 Unified Directory Management:**

New `mkdir` and `rmdir` operations enable consistent directory handling across all backends:

```rust
// Create directory (POSIX) or prefix marker (cloud)
store.mkdir("file:///data/test/").await?;
store.mkdir("s3://bucket/prefix/").await?;

// Remove directory - recursive or empty-only
store.rmdir("file:///data/test/", false).await?;  // Must be empty
store.rmdir("s3://bucket/prefix/", true).await?;  // Delete all objects
```

**Backend Support:**
- **file://, direct://** - Creates/removes actual POSIX directories
- **s3://, az://, gs://** - Manages prefix markers and object deletion

**Integration:** Required for sai3-bench v0.7.0+ directory tree workloads  
**Compatibility:** Zero breaking changes - default implementations for unsupported backends

📖 [Full Details v0.9.11](docs/Changelog.md#version-0911) | [Previous Release v0.9.10](docs/Changelog.md#version-0910)

---

### v0.9.10 - Pre-Stat Size Cache for Benchmarking (October 19, 2024)

**🚀 2.5x Faster Multi-Object Downloads:**

New `pre_stat_and_cache()` API eliminates stat overhead in benchmarking workloads:

```rust
// Pre-stat all objects once (200ms for 1000 objects)
store.pre_stat_and_cache(&object_uris, 100).await?;

// Downloads now skip stat operations - 2.5x faster!
for uri in &object_uris {
    let data = store.get(uri).await?;  // Uses cached size
}
```

**Performance:** 32.8s → 13.0s for 1000-object benchmark (60% reduction in total time)  
**Use Cases:** Benchmarking tools, dataset pre-loading, batch processing  
**Backends:** S3, GCS, Azure (file:// uses TTL=0 for immediate freshness)

📖 [Full Details v0.9.10](docs/Changelog.md#version-0910) | [Usage Examples](docs/Changelog.md#api-usage---after-v0910---pre-stat-optimization)

---

### v0.9.9 - Buffer Pool Optimization for DirectIO (18, October 2025)

**🚀 15-20% Faster DirectIO:**

Eliminated allocation churn in DirectIO range reads through intelligent buffer pool integration:

**What Changed:**
- **Buffer Pool Infrastructure**: Wired existing buffer pool into DirectIO hot path
- **Smart Borrow/Return Pattern**: Replaces fresh allocation + full copy with pool borrow + small copy + pool return
- **90% Fewer Allocations**: Reused 64MB aligned buffers dramatically reduce allocator overhead
- **30-50% Fewer Page Faults**: Less memory churn reduces kernel page fault activity
- **Zero Breaking Changes**: Completely backward compatible, pool auto-initialized in factory functions

**Performance Impact (Expected):**
- ⚡ **Throughput**: +15-20% on DirectIO with RangeEngine
- 🔧 **CPU Usage**: -10-15% (less memcpy/malloc/free)
- 📉 **Page Faults**: -30-50% reduction
- ♻️ **Allocator Calls**: -90% (buffer reuse vs per-operation allocation)

**API Usage (No Code Changes Required):**
```rust
// Factory functions automatically initialize pool
let store = direct_io_store_for_uri("file:///data/")?;  // ✅ Pool auto-initialized

// Constructors automatically initialize pool  
let config = FileSystemConfig::direct_io();  // ✅ Pool auto-initialized (32 × 64MB)
let config = FileSystemConfig::high_performance();  // ✅ Pool auto-initialized

// Default remains compatible with v0.9.8
let config = FileSystemConfig::default();  // ✅ No pool (backward compatible)
```

**Why Only DirectIO?**
- Network storage (S3/Azure/GCS): Latency (5-50ms) >> allocation (<0.1ms), pool provides <1% benefit
- Regular file I/O: Kernel page cache already handles efficiency
- DirectIO: Aligned allocations are expensive + frequent operations = pool is critical

**Technical Details:**
- Pool capacity: 32 buffers
- Buffer size: 64 MB per buffer
- Alignment: System page size (4096 bytes typical)
- Async-safe: Uses tokio channels for thread-safe operations
- Graceful fallback: Allocates new buffer if pool exhausted

📖 [Full Changelog v0.9.9](docs/Changelog.md#version-099) | [Testing Summary](docs/testing/v0.9.9-phase1-testing-summary.md)

---

## 📚 Version History

For detailed release notes and migration guides, see the [Complete Changelog](docs/Changelog.md).

**Recent versions:**
- **v0.9.10** (19, October 2024) - Pre-stat size cache for benchmarking (2.5x faster multi-object downloads)
- **v0.9.9** (18, October 2025) - Buffer pool optimization for DirectIO (15-20% throughput improvement)
- **v0.9.8** (17, October 2025) - Dual GCS backend options, configurable page cache hints
- **v0.9.6** (10, October 2025) - RangeEngine disabled by default (performance fix)
- **v0.9.5** (9, October 2025) - Adaptive concurrency for deletes (10-70x faster)
- **v0.9.3** (8, October 2025) - RangeEngine for Azure & GCS
- **v0.9.2** (8, October 2025) - Graceful shutdown & configuration hierarchy
- **v0.9.1** (8, October 2025) - Zero-copy Python API with BytesView
- **v0.9.0** (7, October 2025) - bytes::Bytes migration (BREAKING)
- **v0.8.x** (2024-2025) - Production features (universal commands, OpLog, TFRecord indexing)

---

## Storage Backend Support

### Universal Backend Architecture
s3dlio provides unified storage operations across all backends with consistent URI patterns:

- **🗄️ Amazon S3**: `s3://bucket/prefix/` - High-performance S3 operations (5+ GB/s reads, 2.5+ GB/s writes)
- **☁️ Azure Blob Storage**: `az://container/prefix/` - Complete Azure integration with **RangeEngine** (30-50% faster for large blobs)
- **🌐 Google Cloud Storage**: `gs://bucket/prefix/` or `gcs://bucket/prefix/` - Production ready with **RangeEngine** and full ObjectStore integration
- **📁 Local File System**: `file:///path/to/directory/` - High-speed local file operations with **RangeEngine** support
- **⚡ DirectIO**: `direct:///path/to/directory/` - Bypass OS cache for maximum I/O performance with **RangeEngine**

### RangeEngine Performance Features (v0.9.3+, Updated v0.9.6)
Concurrent range downloads hide network latency by parallelizing HTTP range requests.

**⚠️ IMPORTANT (v0.9.6+):** RangeEngine is **disabled by default** across all backends due to stat overhead causing up to 50% slowdown on typical workloads. Must be explicitly enabled for large-file operations.

**Backends with RangeEngine Support:**
- ✅ **Azure Blob Storage**: 30-50% faster for large files (must enable explicitly)
- ✅ **Google Cloud Storage**: 30-50% faster for large files (must enable explicitly)
- ✅ **Local File System**: Rarely beneficial due to seek overhead (disabled by default)
- ✅ **DirectIO**: Rarely beneficial due to O_DIRECT overhead (disabled by default)
- 🔄 **S3**: Coming soon

**Default Configuration (v0.9.6+):**
- **Status**: Disabled by default (was: enabled in v0.9.5)
- **Reason**: Extra HEAD request on every GET causes 50% slowdown for typical workloads
- **Threshold**: 16MB when enabled
- **Chunk size**: 64MB default
- **Max concurrent**: 32 ranges (network) or 16 ranges (local)

**How to Enable for Large-File Workloads:**
```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig};

let config = AzureConfig {
    enable_range_engine: true,  // Explicitly enable for large files
    ..Default::default()
};
let store = AzureObjectStore::with_config(config);
```

**When to Enable:**
- ✅ Large-file workloads (average size >= 64 MiB)
- ✅ High-bandwidth, high-latency networks
- ❌ Mixed or small-object workloads
- ❌ Local file systems

### S3 Backend Options
s3dlio supports two S3 backend implementations. **Native AWS SDK is the default and recommended** for production use:

```bash
# Default: Native AWS SDK backend (RECOMMENDED for production)
cargo build --release
# or explicitly:
cargo build --no-default-features --features native-backends

# Experimental: Apache Arrow object_store backend (optional, for testing)
cargo build --no-default-features --features arrow-backend
```

**Why native-backends is default:**
- Proven performance in production workloads
- Optimized for high-throughput S3 operations (5+ GB/s reads, 2.5+ GB/s writes)
- Well-tested with MinIO, Vast, and AWS S3

**About arrow-backend:**
- Experimental alternative implementation
- No proven performance advantage over native backend
- Useful for comparison testing and development
- Not recommended for production use

### GCS Backend Options (v0.9.7+)

s3dlio supports **two mutually exclusive GCS backend implementations** that can be selected at compile time. **Community backend (`gcs-community`) is the default and recommended** for production use:

```bash
# Default: Community backend (RECOMMENDED for production)
cargo build --release
# or explicitly:
cargo build --release --features gcs-community

# Experimental: Official Google backend (for testing only)
cargo build --release --no-default-features --features native-backends,s3,gcs-official
```

**Why gcs-community is default:**
- ✅ Production-ready and stable (10/10 tests pass consistently)
- ✅ Uses community-maintained `gcloud-storage` v1.1 crate
- ✅ Full ADC (Application Default Credentials) support
- ✅ All operations work reliably: GET, PUT, DELETE, LIST, STAT, range reads

**About gcs-official:**
- ⚠️ **Experimental only** - Known transport flakes in test suites
- Uses official Google `google-cloud-storage` v1.1 crate
- Individual operations work correctly (100% pass when tested alone)
- Full test suite experiences intermittent "transport error" failures (7/10 tests fail)
- **Root cause**: Upstream HTTP/2 connection pool flake in google-cloud-rust library
  - **Bug Report**: https://github.com/googleapis/google-cloud-rust/issues/3574
  - **Related Issue**: https://github.com/googleapis/google-cloud-rust/issues/3412
- Not recommended for production until upstream issue is resolved

**For more details:** See [GCS Backend Selection Guide](docs/GCS-BACKEND-SELECTION.md)

## Quick Start

### Installation

**Rust CLI:**
```bash
git clone https://github.com/russfellows/s3dlio.git
cd s3dlio
cargo build --release
```

**Python Library:**
```bash
pip install s3dlio
# or build from source:
./build_pyo3.sh && ./install_pyo3_wheel.sh
```

### Documentation

- **[CLI Guide](docs/CLI_GUIDE.md)** - Complete command-line interface reference with examples
- **[Python API Guide](docs/PYTHON_API_GUIDE.md)** - Complete Python library reference with examples
- **[Multi-Endpoint Guide](docs/MULTI_ENDPOINT_GUIDE.md)** - Load balancing across multiple storage endpoints (v0.9.14+)
- **[Rust API Guide v0.9.0](docs/api/rust-api-v0.9.0.md)** - Complete Rust library reference with migration guide
- **[Changelog](docs/Changelog.md)** - Version history and release notes
- **[Adaptive Tuning Guide](docs/ADAPTIVE-TUNING.md)** - Optional performance auto-tuning
- **[Testing Guide](docs/TESTING-GUIDE.md)** - Test suite documentation
- **[v0.9.2 Test Summary](docs/v0.9.2_Test_Summary.md)** - ✅ 122/130 tests passing (93.8%)

## Core Capabilities

### 🚀 Universal Copy Operations

s3dlio treats upload and download as enhanced versions of the Unix `cp` command, working across all storage backends:

**CLI Usage:**
```bash
# Upload to any backend with real-time progress
s3-cli upload /local/data/*.log s3://mybucket/logs/
s3-cli upload /local/files/* az://container/data/  
s3-cli upload /local/models/* gs://ml-bucket/models/
s3-cli upload /local/backup/* file:///remote-mount/backup/
s3-cli upload /local/cache/* direct:///nvme-storage/cache/

# Download from any backend  
s3-cli download s3://bucket/data/ ./local-data/
s3-cli download az://container/logs/ ./logs/
s3-cli download gs://ml-bucket/datasets/ ./datasets/
s3-cli download file:///network-storage/data/ ./data/

# Cross-backend copying workflow
s3-cli download s3://source-bucket/data/ ./temp/
s3-cli upload ./temp/* gs://dest-bucket/data/
```

**Advanced Pattern Matching:**
```bash
# Glob patterns for file selection (upload)
s3-cli upload "/data/*.log" s3://bucket/logs/
s3-cli upload "/files/data_*.csv" az://container/data/

# Regex patterns for listing (use single quotes to prevent shell expansion)
s3-cli ls -r s3://bucket/ -p '.*\.txt$'           # Only .txt files
s3-cli ls -r gs://bucket/ -p '.*\.(csv|json)$'    # CSV or JSON files
s3-cli ls -r az://acct/cont/ -p '.*/data_.*'      # Files with "data_" in path

# Count objects matching pattern (with progress indicator)
s3-cli ls -rc gs://bucket/data/ -p '.*\.npz$'
# Output: ⠙ [00:00:05] 71,305 objects (14,261 obj/s)
#         Total objects: 142,610 (10.0s, rate: 14,261 objects/s)

# Delete only matching files
s3-cli delete -r s3://bucket/logs/ -p '.*\.log$'
```

See **[CLI Guide](docs/CLI_GUIDE.md)** for complete command reference and pattern syntax.

### 🐍 Python Integration

**High-Performance Data Operations:**
```python
import s3dlio

# Universal upload/download across all backends
s3dlio.upload(['/local/data.csv'], 's3://bucket/data/')
s3dlio.upload(['/local/logs/*.log'], 'az://container/logs/')  
s3dlio.upload(['/local/models/*.pt'], 'gs://ml-bucket/models/')
s3dlio.download('s3://bucket/data/', './local-data/')
s3dlio.download('gs://ml-bucket/datasets/', './datasets/')

# High-level AI/ML operations
dataset = s3dlio.create_dataset("s3://bucket/training-data/")
loader = s3dlio.create_async_loader("gs://ml-bucket/data/", {"batch_size": 32})

# PyTorch integration
from s3dlio.torch import S3IterableDataset
from torch.utils.data import DataLoader

dataset = S3IterableDataset("gs://bucket/data/", loader_opts={})
dataloader = DataLoader(dataset, batch_size=16)
```

**Streaming & Compression:**
```python
# High-performance streaming with compression
options = s3dlio.PyWriterOptions()
options.compression = "zstd"
options.compression_level = 3

writer = s3dlio.create_s3_writer('s3://bucket/data.zst', options)
writer.write_chunk(large_data_bytes)
stats = writer.finalize()  # Returns (bytes_written, compressed_bytes)

# Data generation with configurable modes
s3dlio.put("s3://bucket/test-data-{}.bin", num=1000, size=4194304, 
          data_gen_mode="streaming")  # 2.6-3.5x faster for most cases
```

**Multi-Endpoint Load Balancing (v0.9.14+):**
```python
# Distribute I/O across multiple storage endpoints
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://bucket-1/data",
        "s3://bucket-2/data", 
        "s3://bucket-3/data",
    ],
    strategy="least_connections"  # or "round_t robin"
)

# Zero-copy data access (memoryview compatible)
data = store.get("s3://bucket-1/file.bin")
array = np.frombuffer(memoryview(data), dtype=np.float32)

# Monitor load distribution
stats = store.get_endpoint_stats()
for i, s in enumerate(stats):
    print(f"Endpoint {i}: {s['requests']} requests, {s['bytes_transferred']} bytes")
```
📖 **[Complete Multi-Endpoint Guide](docs/MULTI_ENDPOINT_GUIDE.md)** - Load balancing, configuration, use cases

## Performance

### Benchmark Results
s3dlio delivers world-class performance across all operations:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| **S3 PUT** | Up to 3.089 GB/s | Exceeds steady-state baseline by 17.8% |
| **S3 GET** | Up to 4.826 GB/s | Near line-speed performance |
| **Multi-Process** | 2-3x faster | Improvement over single process |
| **Streaming Mode** | 2.6-3.5x faster | For 1-8MB objects vs single-pass |

### Optimization Features
- **HTTP/2 Support**: Modern multiplexing for enhanced throughput (with Apache Arrow backend only)
- **Intelligent Defaults**: Streaming mode automatically selected based on benchmarks
- **Multi-Process Architecture**: Massive parallelism for maximum performance
- **Zero-Copy Streaming**: Memory-efficient operations for large datasets
- **Configurable Chunk Sizes**: Fine-tune performance for your workload

# Checkpoint system for model states
store = s3dlio.PyCheckpointStore('file:///tmp/checkpoints/')
store.save('model_state', your_model_data)
loaded_data = store.load('model_state')
```

**Ready for Production**: All core functionality validated, comprehensive test suite, and honest documentation matching actual capabilities.

## Configuration & Tuning

### Environment Variables
s3dlio supports comprehensive configuration through environment variables:

- **HTTP Client Optimization**: `S3DLIO_USE_OPTIMIZED_HTTP=true` - Enhanced connection pooling
- **Runtime Scaling**: `S3DLIO_RT_THREADS=32` - Tokio worker threads  
- **Connection Pool**: `S3DLIO_MAX_HTTP_CONNECTIONS=400` - Max connections per host
- **Range GET**: `S3DLIO_RANGE_CONCURRENCY=64` - Large object optimization
- **Operation Logging**: `S3DLIO_OPLOG_LEVEL=2` - S3 operation tracking

📖 [Environment Variables Reference](docs/api/Environment_Variables.md)

### Operation Logging (Op-Log)
Universal operation trace logging across all backends with zstd-compressed TSV format, warp-replay compatible.

```python
import s3dlio
s3dlio.init_op_log("operations.tsv.zst")
# All operations automatically logged
s3dlio.finalize_op_log()
```

See [S3DLIO OpLog Implementation](docs/S3DLIO_OPLOG_IMPLEMENTATION_SUMMARY.md) for detailed usage.

## Building from Source

### Prerequisites
- **Rust**: [Install Rust toolchain](https://www.rust-lang.org/tools/install)
- **Python 3.12+**: For Python library development
- **UV** (recommended): [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **HDF5**: Required for HDF5 support (`libhdf5-dev` on Ubuntu, `brew install hdf5` on macOS)

### Build Steps
```bash
# Python environment
uv venv && source .venv/bin/activate

# Rust CLI
cargo build --release

# Python library
./build_pyo3.sh && ./install_pyo3_wheel.sh
```

## Configuration

### Environment Setup
```bash
# Required for S3 operations
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_ENDPOINT_URL=https://your-s3-endpoint
AWS_REGION=us-east-1
```
Enable comprehensive S3 operation logging compatible with MinIO warp format:


## Advanced Features

### CPU Profiling & Analysis
```bash
cargo build --release --features profiling
cargo run --example simple_flamegraph_test --features profiling
```

### Compression & Streaming
```python
import s3dlio
options = s3dlio.PyWriterOptions()
options.compression = "zstd"
writer = s3dlio.create_s3_writer('s3://bucket/data.zst', options)
writer.write_chunk(large_data)
stats = writer.finalize()
```

## Container Deployment

```bash
# Use pre-built container
podman pull quay.io/russfellows-sig65/s3dlio
podman run --net=host --rm -it quay.io/russfellows-sig65/s3dlio

# Or build locally
podman build -t s3dlio .
```

**Note**: Always use `--net=host` for storage backend connectivity.

## Documentation & Support

- **🖥️ CLI Guide**: [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) - Complete command-line reference
- **🐍 Python API**: [docs/PYTHON_API_GUIDE.md](docs/PYTHON_API_GUIDE.md) - Python library reference
- **📚 API Documentation**: [docs/api/](docs/api/)
- **📝 Changelog**: [docs/Changelog.md](docs/Changelog.md)
- **🧪 Testing Guide**: [docs/TESTING-GUIDE.md](docs/TESTING-GUIDE.md)
- **🚀 Performance**: [docs/performance/](docs/performance/)

## 🔗 Related Projects

- **[sai3-bench](https://github.com/russfellows/sai3-bench)** - Multi-protocol I/O benchmarking suite built on s3dlio
- **[polarWarp](https://github.com/russfellows/polarWarp)** - Op-log analysis tool for parsing and visualizing s3dlio operation logs

## License

Licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file.

---

**🚀 Ready to get started?** Check out the [Quick Start](#quick-start) section above or explore our [example scripts](examples/) for common use cases!
