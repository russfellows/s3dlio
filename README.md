# s3dlio - Universal Storage I/O Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/russfellows/s3dlio)
[![Tests](https://img.shields.io/badge/tests-178%20passing-brightgreen)](docs/Changelog.md)
[![Rust Tests](https://img.shields.io/badge/rust%20tests-178%2F178-brightgreen)](docs/Changelog.md)
[![Version](https://img.shields.io/badge/version-0.9.37-blue)](https://github.com/russfellows/s3dlio/releases)
[![PyPI](https://img.shields.io/pypi/v/s3dlio)](https://pypi.org/project/s3dlio/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)

High-performance, multi-protocol storage library for AI/ML workloads with universal copy operations across S3, Azure, GCS, local file systems, and DirectIO.

## 📦 Installation

```bash
pip install s3dlio
```

## ✨ Key Features

- **5+ GB/s Performance**: High-throughput S3 reads, 2.5+ GB/s writes
- **Zero-Copy Architecture**: `bytes::Bytes` throughout for minimal memory overhead
- **Multi-Protocol**: S3, Azure Blob, GCS, file://, direct:// (O_DIRECT)
- **Python & Rust**: Native Rust library with zero-copy Python bindings (PyO3)
- **Multi-Endpoint Load Balancing**: RoundRobin/LeastConnections across storage endpoints
- **AI/ML Ready**: PyTorch DataLoader integration, TFRecord/NPZ format support
- **High-Speed Data Generation**: 50+ GB/s test data with configurable compression/dedup

## 🌟 Latest Release

**v0.9.37** (January 2026) - Test suite modernization, zero build warnings.

**Recent highlights:**
- **v0.9.36** - **BREAKING**: `ObjectStore::put()` now takes `Bytes` instead of `&[u8]` for true zero-copy
- **v0.9.35** - Hardware detection module, 50+ GB/s data generation
- **v0.9.30** - Zero-copy refactor, PyO3 0.27 migration
- **v0.9.27** - First PyPI release, DLIO Benchmark integration

📖 **[Complete Changelog](docs/Changelog.md)** - Full version history, migration guides, API details

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
