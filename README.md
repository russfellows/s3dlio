# s3dlio - Universal 
## 📊 Performance Monitoring (v0.8.7)

Advanced HDR histogram-based performance monitoring for AI/ML workloads, providing precise tail latency analysis (P99, P99.9, P99.99+) and comprehensive throughput tracking. Built-in presets for training, inference, and distributed scenarios with thread-safe global metrics collection.

## 📇 TFRecord Index Generation (v0.8.10)

**NEW**: Generate NVIDIA DALI-compatible index files for efficient random access to TFRecord datasets. Enables shuffled data loading, distributed training, and O(1) record seeking with minimal overhead (0.4% file size). Pure Rust implementation with comprehensive Python API.

```python
import s3dlio

# Generate index for efficient random access
s3dlio.create_tfrecord_index("train.tfrecord")

# Read index and access any record directly
index = s3dlio.read_tfrecord_index("train.tfrecord.idx")
offset, size = index[42]  # O(1) access, ~1200x faster than sequential scan
```

**Use Cases**: Random shuffling, distributed training sharding, batch loading, DALI pipeline integration. Compatible with NVIDIA DALI and TensorFlow tooling. See [Quick Reference](docs/TFRECORD-INDEX-QUICKREF.md) for details.

## 🧠 AI/ML Training Enhancement (v0.8.6)

Added comprehensive LoaderOptions Realism Knobs for production AI/ML workloads, providing fine-grained control over data loading behavior, performance optimization, and training pipeline configuration with PyTorch/TensorFlow best practices.

## Storage Backend Support

### Universal Backend Architecture
s3dlio provides a unified interface for all storage operations, treating upload/download as enhanced copy commands that work across any backend:
nce Storage Library

## Overview

**s3dlio** is a high-performance, multi-protocol storage library designed for AI/ML workloads and data-intensive applications. Built in Rust with Python bindings, it provides zero-copy streaming, comprehensive checkpointing, and universal copy operations across multiple storage backends.

**Key Features:**
- 🚀 **Universal Copy Operations**: Upload/download work like Unix `cp` across tested storage backends
- ⚡ **High Performance**: Close to 4.8 line-speed throughput with intelligent optimization
- 📊 **Real-Time Progress Tracking**: CLI has progress bars with transfer rates and ETA
- 🔍 **Advanced Pattern Matching**: Support for glob patterns (`*.log`) and regex (`.*\.log$`)
- 🐍 **Python Integration**: Full PyTorch/TensorFlow/JAX compatibility with async support
- 🏗️ **Multi-Backend Architecture**: Support I/O across S3, Azure, file systems, and DirectIO

## �️ Page Cache Optimization (v0.8.8)

Added intelligent Linux/Unix page cache hints via `posix_fadvise()` with automatic mode selection based on file size. Optimizes kernel read-ahead behavior for both sequential large files (≥64MB) and random small file access patterns. Integrated with file:// and direct:// backends.

## 📝 Tracing Framework (v0.8.8)

Migrated from `log` crate to `tracing` ecosystem for enhanced observability and compatibility with dl-driver and s3-bench projects. Supports standard verbosity levels (`-v` for INFO, `-vv` for DEBUG) with preserved operation trace logging functionality.

## �📊 Performance Monitoring (v0.8.7)

Advanced HDR histogram-based performance monitoring for AI/ML workloads, providing precise tail latency analysis (P99, P99.9, P99.99+) and comprehensive throughput tracking. Built-in presets for training, inference, and distributed scenarios with thread-safe global metrics collection.

## 🧠 AI/ML Training Enhancement (v0.8.6)

Added comprehensive LoaderOptions Realism Knobs for production AI/ML workloads, providing fine-grained control over data loading behavior, performance optimization, and training pipeline configuration with PyTorch/TensorFlow best practices.

## Storage Backend Support

### Universal Backend Architecture
s3dlio provides a unified interface for all storage operations, treating upload/download as enhanced copy commands that work across any backend:

- **🗄️ Amazon S3**: `s3://bucket/prefix/` - High-performance S3 operations with multiple backend choices
- **☁️ Azure Blob Storage**: `az://container/prefix/` - Complete Azure integration with hot/cool tier support  
- **📁 Local File System**: `file:///path/to/directory/` - High-speed local file operations
- **⚡ DirectIO**: `direct:///path/to/directory/` - Bypass OS cache for maximum I/O performance

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

## Core Capabilities

### 🚀 Universal Copy Operations

s3dlio treats upload and download as enhanced versions of the Unix `cp` command, working across all storage backends:

**CLI Usage:**
```bash
# Upload to any backend with real-time progress
s3-cli upload /local/data/*.log s3://mybucket/logs/
s3-cli upload /local/files/* az://container/data/  
s3-cli upload /local/backup/* file:///remote-mount/backup/
s3-cli upload /local/cache/* direct:///nvme-storage/cache/

# Download from any backend  
s3-cli download s3://bucket/data/ ./local-data/
s3-cli download az://container/logs/ ./logs/
s3-cli download file:///network-storage/data/ ./data/

# Cross-backend copying workflow
s3-cli download s3://source-bucket/data/ ./temp/
s3-cli upload ./temp/* az://dest-container/data/
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
s3dlio.download('s3://bucket/data/', './local-data/')

# High-level AI/ML operations
dataset = s3dlio.create_dataset("s3://bucket/training-data/")
loader = s3dlio.create_async_loader("file:///data/", {"batch_size": 32})

# PyTorch integration
from s3dlio.torch import S3IterableDataset
from torch.utils.data import DataLoader

dataset = S3IterableDataset("s3://bucket/data/", loader_opts={})
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

### Version 0.8.6 - LoaderOptions Realism Knobs for AI/ML Training (Latest)
Comprehensive AI/ML training enhancement with 10 new LoaderOptions configuration knobs for production workloads. Features pin_memory GPU optimization, persistent_workers for epoch efficiency, configurable timeouts, multiprocessing context control, sampling strategies, memory format optimization, and async GPU transfers. Includes convenience presets (gpu_optimized, distributed_optimized, debug_mode) and full Python integration with PyLoaderOptions fluent builder pattern. Production-ready defaults aligned with PyTorch/TensorFlow best practices.

### Version 0.8.5 - Direct I/O Support & Async Loader Fixes
Complete Direct I/O dataset support for `direct://` URIs with async loader improvements. Added DirectIOBytesDataset using O_DIRECT for maximum throughput, fixed async loaders to return individual items by default, and resolved Python API baseline compatibility issues. Enhanced dataset factory function and comprehensive test coverage.

### Version 0.7.8 - Rust API Cleanup & O_DIRECT Implementation
Complete Rust API redesign with clean, stable interfaces for external developers and working O_DIRECT implementation. Introduces new `s3dlio::api` module with factory functions `store_for_uri()` and `direct_io_store_for_uri()`, unified `ObjectStore` trait, and backward compatibility. Features functional O_DIRECT streaming writer with `DirectIOWriter`, hybrid I/O support (automatic switching between O_DIRECT for aligned data and buffered I/O for unaligned chunks), and proper error handling. Includes comprehensive documentation in `docs/api/` directory, usage examples, and documented path forward for completing O_DIRECT data persistence optimization. Provides enterprise-ready stable API for external Rust developers while maintaining all existing functionality.

### Version 0.7.7 - Phase 2 Streaming API & Complete Python Bindings
Complete Phase 2 streaming infrastructure with production-ready Python bindings and comprehensive test coverage. Introduces universal `ObjectWriter` streaming APIs with `write_chunk()`, `write_owned_bytes()`, and `finalize()` methods across all storage backends (S3, Azure, Filesystem, Direct I/O). Features zero-copy optimization through `write_owned_bytes()`, optional zstd compression, and robust Python integration with `PyWriterOptions` and `PyObjectWriter` classes. Includes comprehensive test suites (7 Rust + 8 Python tests), proper error handling, and post-finalization statistics access. Enables memory-efficient streaming of arbitrarily large files with minimal memory footprint.  Also fixed the docker container build to not copy all of the local .venv environment, making for a substantially smaller container image.

### Version 0.7.5 - HTTP Client Optimization & Performance Enhancement
Advanced HTTP client optimization through strategic AWS SDK fork integration. Successfully forked `aws-smithy-http-client` to expose connection pool configuration, achieving +2-3% performance improvement with full backward compatibility. Features environment variable control (`S3DLIO_USE_OPTIMIZED_HTTP=true`) for easy A/B testing between AWS SDK defaults and optimized configuration. Includes enhanced connection pooling (200 max connections), optimized timeouts (800ms idle), and HTTP/2 improvements. All 84 tests pass with comprehensive performance validation.

### Version 0.7.3 - Modular Python API Architecture
Major architectural refactoring transforming the monolithic Python API into a clean modular structure. Split 1883-line `python_api.rs` into organized modules for core storage, AI/ML functions, and advanced features. Eliminated all compiler warnings, fixed critical bugs, and added comprehensive regression test suite with 16 test cases covering all 49 public functions. Zero breaking changes - all existing code continues to work unchanged.

### Version 0.7.2 - Complete Python Compression Integration
Full end-to-end compression support with Python API integration. The `compression_level` parameter now provides seamless compression/decompression across all storage backends with automatic save/load cycles. Achieves 99.8% compression ratios with data integrity preservation through streaming zstd compression and automatic decompression on read.

### Version 0.6.2 - Enhanced Data Integrity
Complete checksum integration across all storage backends providing CRC32C-based data integrity validation. All checkpoint operations now include computed checksums with zero-copy streaming preserved.

### Version 0.6.1 - Zero-Copy Streaming Infrastructure  
Revolutionary memory efficiency enabling processing of models larger than available RAM. Peak memory usage reduced from full model size to single chunk size (99%+ memory reduction for large models).

### Version 0.6.0 - Comprehensive Checkpointing System
Production-ready distributed checkpointing modeled after AWS S3 PyTorch Connector, with multi-backend support and framework integration for PyTorch, JAX, and TensorFlow.

### Version 0.5.3 - Advanced Async DataLoader
Dynamic batch formation with out-of-order completion, eliminating head-of-line blocking while maintaining complete backward compatibility.

📖 **[Complete Version History](docs/Changelog.md)** - Detailed changelog with all enhancements and technical details

## Configuration & Tuning

### Environment Variables
s3dlio supports comprehensive configuration through environment variables for performance tuning and optimization:

- **HTTP Client Optimization**: `S3DLIO_USE_OPTIMIZED_HTTP=true` enables enhanced connection pooling
- **Runtime Scaling**: `S3DLIO_RT_THREADS=32` controls Tokio worker threads  
- **Connection Pool**: `S3DLIO_MAX_HTTP_CONNECTIONS=400` sets max connections per host
- **Range GET**: `S3DLIO_RANGE_CONCURRENCY=64` for large object optimization
- **Operation Logging**: `S3DLIO_OPLOG_LEVEL=2` for detailed S3 operation tracking

📖 **[Complete Environment Variables Reference](docs/api/Environment_Variables.md)** - Comprehensive configuration guide with performance tuning examples

## Performance Profiling & Analysis

### Advanced Profiling Infrastructure (NEW in v0.7.6)
s3dlio includes comprehensive performance profiling capabilities for analyzing and optimizing AI/ML workloads:

## Building from Source

### Prerequisites
- **Rust**: [Install Rust toolchain](https://www.rust-lang.org/tools/install)
- **Python 3.12+**: For Python library development
- **UV** (recommended): [Install UV](https://docs.astral.sh/uv/getting-started/installation/) for Python environment management
- **HDF5**: Required for HDF5 format support
  ```bash
  # Ubuntu/Debian
  sudo apt update && sudo apt install -y libhdf5-dev
  
  # macOS
  brew install hdf5
  
  # RHEL/CentOS/Fedora
  sudo dnf install hdf5-devel  # or yum install hdf5-devel
  ```

### Build Steps

**1. Set up Python Environment:**
```bash
uv venv && source .venv/bin/activate
uv python install 3.12
```

**2. Build CLI:**
```bash
cargo build --release
# Binary available at ./target/release/s3-cli
```

**3. Build Python Library:**
```bash
./build_pyo3.sh && ./install_pyo3_wheel.sh
```

## Configuration

### Environment Setup
Create a `.env` file or set environment variables:

```bash
# Required for S3 operations
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_ENDPOINT_URL=https://your-s3-endpoint
AWS_REGION=us-east-1

# Optional: Custom CA bundle for self-signed certificates
AWS_CA_BUNDLE_PATH=/path/to/ca-bundle.pem
```

### Operation Logging
Enable comprehensive S3 operation logging compatible with MinIO warp format:

**CLI:**
```bash
s3-cli put s3://bucket/data --op-log /tmp/operations.tsv.zst
```

**Python:**
```python
import s3dlio
s3dlio.init_op_log("/tmp/python_ops.tsv.zst")
# ... perform operations ...
s3dlio.finalize_op_log()
```

## Advanced Features

### 🔥 CPU Profiling & Analysis
Enable detailed performance profiling for optimization:
```bash
# Build with profiling support
cargo build --release --features profiling

# Run large-scale performance analysis
S3DLIO_TEST_SIZE_GB=5 cargo run --example large_scale_s3_test --features profiling

# Generate flamegraph analysis
cargo run --example simple_flamegraph_test --features profiling
```

### 📊 Compression & Streaming
High-performance compression with multiple algorithms:
```python
import s3dlio

# Configure compression options
options = s3dlio.PyWriterOptions()
options.compression = "zstd"  # or "lz4", "gzip"
options.compression_level = 3

# Create compressed streaming writer
writer = s3dlio.create_s3_writer('s3://bucket/data.zst', options)
writer.write_chunk(large_data)
stats = writer.finalize()  # Returns compression ratios
```

### 🎯 Data Generation Modes
Optimize data generation for your workload:
```bash
# Streaming mode (default, 2.6-3.5x faster for most cases)
s3-cli put s3://bucket/data-{}.bin --num 1000 --size 4MB --data-gen-mode streaming

# Single-pass mode for specific use cases
s3-cli put s3://bucket/data-{}.bin --num 100 --size 32MB --data-gen-mode single-pass
```

## Container Deployment

### Pre-built Container
```bash
podman pull quay.io/russfellows-sig65/s3dlio
podman run --net=host --rm -it quay.io/russfellows-sig65/s3dlio
```

### Build Container
```bash
podman build -t s3dlio .
```

**⚠️ Note**: Always use `--net=host` for network connectivity to storage backends.

## Documentation & Support

- **📚 Complete API Documentation**: [docs/api/](docs/api/)
- **🚀 Performance Analysis**: [docs/performance/](docs/performance/) 
- **📝 Changelog**: [docs/Changelog.md](docs/Changelog.md)
- **🧪 Testing Guide**: [docs/TESTING-GUIDE.md](docs/TESTING-GUIDE.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**🚀 Ready to get started?** Check out the [Quick Start](#quick-start) section above or explore our [example scripts](examples/) for common use cases!
