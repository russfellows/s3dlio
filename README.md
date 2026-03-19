# s3dlio - Universal Storage I/O Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/russfellows/s3dlio)
[![Rust Tests](https://img.shields.io/badge/rust%20tests-580%2F580-brightgreen)](docs/Changelog.md)
[![Version](https://img.shields.io/badge/version-0.9.80-blue)](https://github.com/russfellows/s3dlio/releases)
[![PyPI](https://img.shields.io/pypi/v/s3dlio)](https://pypi.org/project/s3dlio/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)

High-performance, multi-protocol storage library for AI/ML workloads with universal copy operations across S3, Azure, GCS, local file systems, and DirectIO.

## 📦 Installation

### Quick Install (Python)

```bash
# If using uv package manager + uv virtual environment:
uv pip install s3dlio

# If using pip without uv:
pip install s3dlio
```

### Python Backend Profiles (PyPI vs Full Build)

- If using `uv` package manager + `uv` virtual environment: `uv pip install s3dlio`.
- If using standard `pip` without `uv`: `pip install s3dlio`.
- The default published wheel is now S3-focused (Azure Blob and GCS are excluded).
- If you want full backends (S3 + Azure Blob + GCS), build from source with:

```bash
# uv workflow:
uv pip install s3dlio --no-binary s3dlio --config-settings "cargo-extra-args=--features extension-module,full-backends"

# pip-only workflow:
pip install s3dlio --no-binary s3dlio --config-settings "cargo-extra-args=--features extension-module,full-backends"
```

You can still add a separate package name (for example `s3dlio-full`) later if you want a dedicated prebuilt full wheel distribution.

> Maintainer note: for PyPI uploads, publish the default (`./build_pyo3.sh`) wheel unless intentionally releasing a separate distribution. `full-backends` is currently source-build only via the command above.

### Building from Source (Rust)

#### System Dependencies

s3dlio requires some system libraries to build. **Only OpenSSL and pkg-config are required by default.** HDF5 and hwloc are optional and improve functionality but are not needed for the core library:

**Ubuntu/Debian:**
```bash
# Quick install - run our helper script
./scripts/install-system-deps.sh

# Or manually (required only):
sudo apt-get install -y build-essential pkg-config libssl-dev

# Optional - for NUMA topology support (--features numa):
sudo apt-get install -y libhwloc-dev

# Optional - for HDF5 data format support (--features hdf5):
sudo apt-get install -y libhdf5-dev

# All optional libraries at once:
sudo apt-get install -y libhdf5-dev libhwloc-dev cmake
```

**RHEL/CentOS/Fedora/Rocky/AlmaLinux:**
```bash
# Quick install
./scripts/install-system-deps.sh

# Or manually (required only):
sudo dnf install -y gcc gcc-c++ make pkg-config openssl-devel

# Optional - for NUMA topology support:
sudo dnf install -y hwloc-devel

# Optional - for HDF5 data format support:
sudo dnf install -y hdf5-devel

# All optional libraries at once:
sudo dnf install -y hdf5-devel hwloc-devel cmake
```

**macOS:**
```bash
# Quick install
./scripts/install-system-deps.sh

# Or manually (required only):
brew install pkg-config openssl@3

# Optional - for NUMA/HDF5 support:
brew install hdf5 hwloc cmake

# Set environment variables (add to ~/.zshrc or ~/.bash_profile):
export PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig:$PKG_CONFIG_PATH"
export OPENSSL_DIR="$(brew --prefix openssl@3)"
```

**Arch Linux:**
```bash
# Quick install
./scripts/install-system-deps.sh

# Or manually (required only):
sudo pacman -S base-devel pkg-config openssl

# Optional - for NUMA/HDF5 support:
sudo pacman -S hdf5 hwloc cmake
```

**WSL (Windows Subsystem for Linux) / Minimal Environments:**

If you are building on WSL or any environment where `libhdf5` or `libhwloc` may not be available, s3dlio builds without them by default. No extra libraries are required:
```bash
# Just the basics - works on WSL, Docker, CI, and minimal installs:
sudo apt-get install -y build-essential pkg-config libssl-dev
cargo build --release
# install Python package (no system HDF5/hwloc needed):
# uv workflow:
uv pip install s3dlio
# pip-only workflow:
pip install s3dlio
```

#### Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### Build s3dlio

```bash
# Clone the repository
git clone https://github.com/russfellows/s3dlio.git
cd s3dlio

# Build with default features (no HDF5 or NUMA required)
cargo build --release

# Build s3-cli with all cloud backends enabled (AWS + Azure + GCS)
cargo build --release --bin s3-cli --features full-backends

# Build s3-cli with GCS enabled only (plus default backends)
cargo build --release --bin s3-cli --features backend-gcs

# Build with NUMA topology support (requires libhwloc-dev)
cargo build --release --features numa

# Build with HDF5 data format support (requires libhdf5-dev)
cargo build --release --features hdf5

# Build with all optional features
cargo build --release --features numa,hdf5

# Run tests
cargo test

# Build Python bindings (optional)
./build_pyo3.sh

# Build Python bindings with full backends (S3 + Azure + GCS)
./build_pyo3.sh full

# Named profile form is also supported:
./build_pyo3.sh --profile full
./build_pyo3.sh --profile default

# Show profile/help usage
./build_pyo3.sh --help
```

### Build Profile Quick Reference

Rust backend feature profiles:

- Default build (`cargo build --release`): S3-focused default backend set.
- GCS-enabled build (`--features backend-gcs`): enables GCS in addition to default set.
- Full cloud build (`--features full-backends`): enables AWS + Azure + GCS.

Python wheel build profiles via `build_pyo3.sh`:

- `default` or `slim`: AWS + file/direct; excludes Azure and GCS.
- `full`: AWS + Azure + GCS + file/direct.
- Positional and named forms are equivalent:
    - `./build_pyo3.sh full`
    - `./build_pyo3.sh -p full`
    - `./build_pyo3.sh --profile full`

Optional extra Rust features for wheel builds can still be passed with `EXTRA_FEATURES`.
Example: `EXTRA_FEATURES="numa,hdf5" ./build_pyo3.sh full`.

**Note:** NUMA support (`--features numa`) improves multi-socket performance but requires the `hwloc2` C library. HDF5 support (`--features hdf5`) enables HDF5 data format generation but requires `libhdf5`. Both are optional and s3dlio is fully functional without them.

**Platform support:** s3dlio builds natively on Linux (x86\_64, aarch64), macOS (x86\_64 and Apple Silicon arm64), and WSL. Making `numa` and `hdf5` optional was the key change for broad platform support — all remaining dependencies are pure Rust or use platform-independent system libraries (OpenSSL). To cross-compile Python wheels for Linux ARM64 from an x86\_64 host, see `build_pyo3.sh` for instructions using the `--zig` linker. For macOS universal2 (fat binary covering both architectures), see the commented section in `build_pyo3.sh`.

## ✨ Key Features

- **High Performance**: High-throughput multi GB/s reads and writes on platforms with sufficient network and storage capabilities
- **Zero-Copy Architecture**: `bytes::Bytes` throughout for minimal memory overhead
- **Multi-Protocol**: S3, Azure Blob, GCS, file://, direct:// (O_DIRECT)
- **Python & Rust**: Native Rust library with zero-copy Python bindings (PyO3), bytearray support for efficient memory management
- **Multi-Endpoint Load Balancing**: RoundRobin/LeastConnections across storage endpoints
- **AI/ML Ready**: PyTorch DataLoader integration, TFRecord/NPZ format support
- **High-Speed Data Generation**: 50+ GB/s test data with configurable compression/dedup

## 🌟 Latest Release

**v0.9.80** (March 2026) - Critical fix: Python `list()` / `list_keys()` hung indefinitely on non-AWS endpoints; tracing deadlock inside `tokio::spawn` eliminated by refactoring `list_objects_stream` to an inline `async_stream::stream!`; all S3 bucket/delete operations now use async-safe helpers.

**Recent highlights:**
- **v0.9.80** - Python list hang fix (IMDSv2 legacy call removed); tracing deadlock fix (`tokio::spawn` → inline stream); async S3 delete/bucket helpers; deprecated Python APIs cleaned up
- **v0.9.76** - GCS RAPID/zonal support (stat/get/put); `RUST_LOG=debug` hang fix (issue #105); debug logging on all five ObjectStore backends; `rm` alias; `list` as primary command name; Python LogTracer conflict fixed
- **v0.9.70** - Added Python wheel backend profiles (`default` and `full`), `build_pyo3.sh` profile CLI options (`full` / `default` / `--profile`), and documentation for full-backend source builds
- **v0.9.65** - Fixed GCS PUT RESOURCE_EXHAUSTED (chunk size exceeded server 4 MiB protobuf message limit); centralised all GCS/gRPC constants; zero-copy `put_object(Bytes)`; RAPID auto-detection; subchannel auto-tune via `--jobs`; 10 new zero-copy unit tests
- **v0.9.60** - All GCS operations now use gRPC (BidiReadObject/BidiWriteObject) instead of JSON API; GCS RAPID/Hyperdisk ML zonal bucket support via `S3DLIO_GCS_RAPID`; 64-way concurrent batch deletes; byte-range optimization enabled by default at 32 MB threshold; NUMA and HDF5 now optional features
- **v0.9.50** - Python multi-threaded runtime fix (io_uring-style submit), s3torchconnector zero-copy rewrite, S3 range download optimization (76% faster for large objects), multipart upload zero-copy chunking
- **v0.9.40** - Enhanced Python bytearray documentation with performance benchmarks (2.5-3x speedup)
- **v0.9.37** - Test suite modernization, zero build warnings
- **v0.9.36** - **BREAKING**: `ObjectStore::put()` now takes `Bytes` instead of `&[u8]` for true zero-copy
- **v0.9.30** - Zero-copy refactor, PyO3 0.27 migration

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

### GCS Backend Options (Current)

GCS is now **optional** at build time.

- Default build (`cargo build --release`) does **not** include GCS.
- To include GCS, enable `backend-gcs` (or `full-backends`).
- When enabled, s3dlio uses the **official Google crates** (`google-cloud-storage` + gax) from a patched fork maintained for s3dlio.

```bash
# Default build (S3-focused; no GCS)
cargo build --release

# Enable GCS explicitly
cargo build --release --features backend-gcs

# Enable all cloud backends (AWS + Azure + GCS)
cargo build --release --features full-backends
```

**Patched official GCS fork used by s3dlio:**
- Repository: https://github.com/russfellows/google-cloud-rust
- Integration in this repo is pinned in [Cargo.toml](Cargo.toml) (currently via release tag from that fork).

**Legacy note:** `gcs-community` remains as a legacy opt-in path, but the primary supported path is the official Google crates from the patched `russfellows/google-cloud-rust` fork.

## Quick Start

### Installation

**Rust CLI:**
```bash
git clone https://github.com/russfellows/s3dlio.git
cd s3dlio
cargo build --release

# Full cloud backend CLI build:
cargo build --release --bin s3-cli --features full-backends
```

**Python Library:**
```bash
# uv workflow:
uv pip install s3dlio

# pip-only workflow:
pip install s3dlio

# or build from source:
./build_pyo3.sh && ./install_pyo3_wheel.sh

# build from source with full cloud backends:
./build_pyo3.sh --profile full && ./install_pyo3_wheel.sh
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
- **OpenSSL**: Required (`libssl-dev` on Ubuntu)
- **HDF5** *(optional)*: Only needed with `--features hdf5` (`libhdf5-dev` on Ubuntu, `brew install hdf5` on macOS)
- **hwloc** *(optional)*: Only needed with `--features numa` (`libhwloc-dev` on Ubuntu)

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
- **[google-cloud-rust (s3dlio patched fork)](https://github.com/russfellows/google-cloud-rust)** - Official Google Cloud Rust client fork used by s3dlio for patched GCS support

## License

Licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file.

---

**🚀 Ready to get started?** Check out the [Quick Start](#quick-start) section above or explore our [example scripts](examples/) for common use cases!
