# s3dlio - Universal Storage I/O Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/russfellows/s3dlio)
[![Tests](https://img.shields.io/badge/tests-130%20passing-brightgreen)](docs/Changelog.md)
[![Rust Tests](https://img.shields.io/badge/rust%20tests-118%2F119-brightgreen)](docs/Changelog.md)
[![Python Tests](https://img.shields.io/badge/python%20tests-12%2F16-yellow)](docs/Changelog.md)
[![Version](https://img.shields.io/badge/version-0.9.7-blue)](https://github.com/russfellows/s3dlio/releases)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)

High-performance, multi-protocol storage library for AI/ML workloads with universal copy operations across S3, Azure, GCS, local file systems, and DirectIO.

## 🌟 Latest Release

### v0.9.6 - RangeEngine Disabled by Default (October 2025)

**⚠️ BREAKING CHANGE**: RangeEngine now disabled by default across all backends due to performance testing revealing up to 50% slowdown on typical workloads.

**🔧 Key Changes:**
- **RangeEngine Opt-In**: Must explicitly enable for large-file workloads (>= 64 MiB average)
- **Performance Fix**: Eliminates extra HEAD/STAT request overhead (2x requests → 1x request)
- **Default Threshold**: 16 MiB minimum split size when explicitly enabled
- **Universal Impact**: All backends (S3, Azure, GCS, file://, direct://) now disabled by default

**Migration Guide:**
```rust
// Enable for large-file workloads only
let config = AzureConfig {
    enable_range_engine: true,  // Opt-in
    ..Default::default()
};
```

**When to Enable:**
- ✅ Large files (>= 64 MiB average)
- ✅ High-bandwidth, high-latency networks
- ❌ Mixed workloads (keep disabled)
- ❌ Small objects (< 16 MiB)

📖 [Full Details](docs/v0.9.6_RangeEngine_Disabled_By_Default.md) | [Changelog v0.9.6](docs/Changelog.md#version-096)

## 📚 Recent Releases

### v0.9.5 - Performance Fixes & RangeEngine Tuning (October 2025)

**Critical performance improvements** fixing regressions and delivering 10-70x faster delete operations:

**🚀 Major Improvements:**
- **Adaptive Concurrency for Deletes**: 10-70x faster (500 objects: 70x, 7K objects: 12-25x, 93K objects: 10x+)
- **RangeEngine Threshold Fix**: Increased to 16 MiB to eliminate 10% regression on small files
- **Universal Backend Support**: Optimizations work across all 5 backends (S3, Azure, GCS, file://, direct://)

**Delete Performance Examples:**
- 500 objects: ~0.7s (was ~50s) - **70x faster**
- 7,000 objects: ~5.5s (was ~70-140s) - **12-25x faster**
- 93,000 objects: ~90s (was 15+ minutes) - **10x+ faster**

**Technical Details:**
- Adaptive concurrency: Scales from 10 to 1,000 concurrent deletions based on workload
- Progress tracking: Batched updates (every 50 operations) reduce overhead by 98%
- RangeEngine: 16 MiB threshold balances small-file efficiency with large-file performance

📖 [Full Changelog v0.9.5](docs/Changelog.md#version-095) | [Performance Analysis](docs/v0.9.5-PERFORMANCE-REGRESSION-ANALYSIS.md)

## 📚 Recent Releases

### v0.9.3 - RangeEngine for Azure & GCS (October 2025)

**Concurrent range downloads** for Azure and GCS with 30-50% throughput improvements on large files.

📖 [Changelog v0.9.3](docs/Changelog.md#version-093)

### v0.9.2 - Graceful Shutdown & Configuration (October 2025)

Production-ready enhancements with zero breaking changes:
- **CancellationToken Infrastructure**: Graceful shutdown for all DataLoader components
- **Configuration Hierarchy**: PyTorch-aligned three-level design with comprehensive documentation

📖 [Changelog](docs/Changelog.md) | [Rust API v0.9.2](docs/api/rust-api-v0.9.2.md) | [Python API v0.9.2](docs/api/python-api-v0.9.2.md)

### v0.9.x - Zero-Copy & Performance

- **v0.9.1**: True zero-copy Python API with `BytesView`, universal `get_many()`
- **v0.9.0** (BREAKING): `bytes::Bytes` migration (10-15% memory reduction), adaptive tuning, 3-8x faster batch loading

📖 Migration: [Rust v0.9.0](docs/api/rust-api-v0.9.0.md) | [Python v0.9.0](docs/api/python-api-v0.9.0.md)

### v0.8.x - Production Features (2024-2025)

Universal commands, operation logging, performance monitoring, and AI/ML training enhancements:
- Universal GET/PUT/ls/stat/rm with progress bars
- Op-Log system with warp-replay compatibility
- TFRecord indexing (~1200x faster random access)
- Page cache optimization and tracing framework

📖 [Complete Changelog](docs/Changelog.md)

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

- **[Rust API Guide v0.9.0](docs/api/rust-api-v0.9.0.md)** - Complete Rust library reference with migration guide
- **[Python API Guide v0.9.0](docs/api/python-api-v0.9.0.md)** - Complete Python library reference with migration guide
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
# Glob patterns for file selection
s3-cli upload "/data/*.log" s3://bucket/logs/
s3-cli upload "/files/data_*.csv" az://container/data/

# Regex patterns for powerful matching  
s3-cli upload "/logs/.*\.log$" s3://bucket/logs/
s3-cli upload "/data/file_[0-9]+\.txt" direct:///storage/data/
```

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

- **📚 API Documentation**: [docs/api/](docs/api/)
- **📝 Changelog**: [docs/Changelog.md](docs/Changelog.md)
- **🧪 Testing Guide**: [docs/TESTING-GUIDE.md](docs/TESTING-GUIDE.md)
- **🚀 Performance**: [docs/performance/](docs/performance/)

## License

Licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file.

---

**🚀 Ready to get started?** Check out the [Quick Start](#quick-start) section above or explore our [example scripts](examples/) for common use cases!
