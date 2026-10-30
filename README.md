# s3dlio - Universal Storage I/O Library

[![CI](https://github.com/russfellows/s3dlio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/russfellows/s3dlio/actions/workflows/ci.yml)
[![Rust Tests](https://img.shields.io/badge/rust%20tests-1893-brightgreen)](docs/Changelog.md)
[![Version](https://img.shields.io/badge/version-0.10.0-blue)](https://github.com/russfellows/s3dlio/releases)
[![PyPI](https://img.shields.io/pypi/v/s3dlio)](https://pypi.org/project/s3dlio/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94.1%2B-orange)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)

High-performance, multi-protocol storage library for AI/ML workloads with universal copy operations across S3, Azure, GCS, local file systems, and DirectIO.

Full-backend builds support the stable Azure Blob Storage Rust SDK 1.x API (1.1).

## Recent Releases

- **v0.10.0** — Caller-buffer GET, typed HEAD; zero copy everywhere; official GCS only — [PR #293](https://github.com/russfellows/s3dlio/pull/293)
- **v0.9.128** — Zero-copy writes, opt-in `.env`, Azure credentials — [PR #262](https://github.com/russfellows/s3dlio/pull/262)
- **v0.9.126** — Exact URI selection; routing, upload fixes — [PR #237](https://github.com/russfellows/s3dlio/pull/237)
- **v0.9.124** (not on PyPI) — Exact delete contract, op-log safety — [PR #217](https://github.com/russfellows/s3dlio/pull/217)
- **v0.9.122** (not on PyPI) — One op-log row per call — [PR #209](https://github.com/russfellows/s3dlio/pull/209)
- **v0.9.120** (not on PyPI) — Routing, hang and crash fixes — [PR #206](https://github.com/russfellows/s3dlio/pull/206)
- **v0.9.114** (not on PyPI) — Azure SDK stable 1.x — [PR #164](https://github.com/russfellows/s3dlio/pull/164)
- **v0.9.112** — Runtime clamp, MPI-aware thread pools — [PR #163](https://github.com/russfellows/s3dlio/pull/163)
- **v0.9.110** — 39-bug multi-agent audit fixes — [PR #159](https://github.com/russfellows/s3dlio/pull/159)
- **v0.9.108** — Performance and concurrency audit — [PR #149](https://github.com/russfellows/s3dlio/pull/149)
- **v0.9.106** — Write verification now opt-in — [PR #147](https://github.com/russfellows/s3dlio/pull/147)
- **v0.9.104** — Write integrity verification and retry — [PR #145](https://github.com/russfellows/s3dlio/pull/145)
- **v0.9.102** — SDK error diagnostics, retry knobs — [PR #144](https://github.com/russfellows/s3dlio/pull/144)

Full history, behaviour changes and migration notes: **[docs/Changelog.md](docs/Changelog.md)**.


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
- The PyPI wheels include S3, Azure Blob, file and direct. GCS is not in them (it needs git dependencies).
- If you also want GCS (S3 + Azure Blob + GCS), build from source with `./build_pyo3.sh full`, or:

```bash
# uv workflow:
uv pip install s3dlio --no-binary s3dlio --config-settings "cargo-extra-args=--features extension-module,full-backends"

# pip-only workflow:
pip install s3dlio --no-binary s3dlio --config-settings "cargo-extra-args=--features extension-module,full-backends"
```

> Maintainer note: the PyPI workflow builds with `--features extension-module,backend-azure`; `./build_pyo3.sh` (default) builds the same backend set. maturin ignores the `features` in `pyproject.toml` whenever `--features` is given on the command line, so keep the two in step. `full-backends` (GCS) is source-build only.

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

- Default build (`cargo build --release`): S3-focused default backend set (Azure needs `--features backend-azure`; the PyPI wheels add it).
- GCS-enabled build (`--features backend-gcs`): enables GCS in addition to default set.
- Full cloud build (`--features full-backends`): enables AWS + Azure + GCS.

Python wheel build profiles via `build_pyo3.sh`:

- `default`: AWS + Azure + file/direct (the PyPI wheel's backend set); excludes GCS.
- `slim`: AWS + file/direct only; excludes Azure and GCS.
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
- **Zero-Copy Architecture**: no payload copies except [documented exceptions](docs/PYTHON_API_GUIDE.md#remaining-copies-rule-1); Python buffers are read and filled in place
- **Multi-Protocol**: S3, Azure Blob, GCS, file://, direct:// (O_DIRECT)
- **HTTP/2 Support (opt-in)**: **HTTP/1.1 is the default** — HTTP/2 is almost always slower for bulk storage workloads. Opt in with `S3DLIO_HTTPS_H2=1` (TLS ALPN, `https://`), `S3DLIO_H2C=1` (cleartext h2c, `http://`), or `S3DLIO_ENABLE_HTTP2=1` (both). See [docs/HTTP2_ALPN_INVESTIGATION.md](docs/HTTP2_ALPN_INVESTIGATION.md).
- **Python & Rust**: Native Rust library with zero-copy Python bindings (PyO3), bytearray support for efficient memory management
- **Multi-Endpoint Load Balancing**: RoundRobin/LeastConnections across storage endpoints; a per-endpoint URI (`s3://host:port/bucket/key`) runs every operation on that server
- **Precise Object Selection**: an `s3://` URI selects exactly the object it names plus its `name/` directory — never `name.bak` — with explicit `*` / `**` globs and `--pattern` regex filtering; `delete -r` follows the same rule
- **Operation Tracing (Op-Log)**: zstd-compressed TSV trace of every storage operation, one row per call, with a replay tool
- **AI/ML Ready**: PyTorch DataLoader integration, TFRecord/NPZ format support
- **Parquet DataLoader**: Per-row-group epoch-aware DataLoader for any s3dlio storage backend (S3, Azure Blob, GCS, file://, direct://). Epoch-2 zero re-fetch speedup (2.5×+), Raw + Arrow IPC decode modes, 8-worker concurrency with shared metadata caches — [see guide](docs/Parquet_Data-Loader.md)
- **High-Speed Data Generation**: 50+ GB/s test data with configurable compression/dedup

## Storage Backend Support

### Universal Backend Architecture
s3dlio provides unified storage operations across all backends with consistent URI patterns:

- **🗄️ Amazon S3**: `s3://bucket/prefix/` - High-performance S3 operations (5+ GB/s reads, 2.5+ GB/s writes) with built-in concurrent range GETs (on by default)
- **☁️ Azure Blob Storage**: `az://account/container/prefix/` - Complete Azure integration with **RangeEngine** (concurrent ranges for large blobs; speed-ups are unmeasured). Authentication: managed identity, workload identity, service principal (secret or certificate), SAS, or `az login`, selected with `S3DLIO_AZURE_CREDENTIAL` (default `auto`; see [Environment Variables](docs/Environment_Variables.md#azure-blob-storage-configuration)). Shared Key (account key) is not supported.
- **🌐 Google Cloud Storage**: `gs://bucket/prefix/` or `gcs://bucket/prefix/` - Production ready with **RangeEngine** and full ObjectStore integration
- **📁 Local File System**: `file:///path/to/directory/` - High-speed local file operations with **RangeEngine** support
- **⚡ DirectIO**: `direct:///path/to/directory/` - Bypass OS cache for maximum I/O performance with **RangeEngine**

### Concurrent Range GET Performance Features
Concurrent range downloads hide network latency by parallelizing HTTP range requests.

**All backends support concurrent range GETs — but via two different mechanisms:**

**Mechanism 1 — S3 built-in (on by default)**
- ✅ **Amazon S3**: Concurrent range splitting enabled by default via `S3DLIO_ENABLE_RANGE_OPTIMIZATION` (default: on). Uses `get_object_concurrent_range_async()` — fires parallel `GetObject(Range: bytes=N-M)` requests via the AWS SDK with lock-free chunk assembly. Controlled by `S3DLIO_RANGE_THRESHOLD_MB` (default: 32 MiB) and `S3DLIO_RANGE_CONCURRENCY` (default: auto-scaled). Disable with `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0`.

**Mechanism 2 — RangeEngine (per-store config flag, must enable explicitly)**
- ✅ **Azure Blob Storage**: concurrent range downloads for large files (`enable_range_engine: true` in `AzureConfig`); the speed-up has not been measured
- ✅ **Google Cloud Storage**: concurrent range downloads for large files (`enable_range_engine: true` in `GcsConfig`); the speed-up has not been measured
- ⚠️ **Local File System**: Rarely beneficial due to seek overhead (disabled by default)
- ⚠️ **DirectIO**: Rarely beneficial due to O_DIRECT overhead (disabled by default)

**RangeEngine config flag defaults:**
- **Status**: `enable_range_engine: false` by default in all per-store config structs
- **Reason**: Extra HEAD request on every GET causes ~50% slowdown for small-object workloads
- **Threshold**: 32 MiB default (tunable per-store via `RangeEngineConfig::min_split_size`)

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
- ✅ Large-file workloads (average size >= 32 MiB)
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

Since v0.10.0 this is the only GCS client (`gcs-community` is removed). `GCS_ENDPOINT_URL` / `STORAGE_EMULATOR_HOST` point it at
an emulator; see [Environment Variables](docs/Environment_Variables.md#gcs-custom-endpoint-examples).

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
- **[Parquet DataLoader Guide](docs/Parquet_Data-Loader.md)** - Epoch-aware per-row-group Parquet DataLoader
- **[Multi-Endpoint Guide](docs/MULTI_ENDPOINT_GUIDE.md)** - Load balancing across multiple storage endpoints
- **[Rust API Guide v0.9.0](docs/api/rust-api-v0.9.0.md)** - Complete Rust library reference with migration guide
- **[Changelog](docs/Changelog.md)** - Version history and release notes
- **[Adaptive Tuning Guide](docs/ADAPTIVE-TUNING.md)** - Optional performance auto-tuning
- **[Testing Guide](docs/TESTING-GUIDE.md)** - Test suite documentation

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
s3-cli ls -r az://acct/cont/ -p '(^|/)data_'      # Names starting with "data_" (pattern is matched relative to the listed prefix)

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

**Read into your own buffer (v0.10.0):**
```python
import asyncio, s3dlio

async def read(uri):
    meta = await s3dlio.head_async(uri)          # ObjectMetadata(size, etag, version_id) or None
    if meta is None:
        raise FileNotFoundError(uri)
    buf = bytearray(meta.size)                   # any writable byte buffer: numpy, mmap, ctypes, ...
    await s3dlio.get_into_async(uri, buf, expected_len=meta.size)  # written into buf as it arrives
    return buf

data = asyncio.run(read("s3://bucket/key"))
s3dlio.object_capabilities("s3://bucket/key")  # {"provider": "s3", "get_into": True, "head": True}
```
The same calls work on `file://` and `direct://` (#295): `file://` reads with `pread` straight into
`buf`; `direct://` reads with O_DIRECT straight into a page-aligned `buf` and copies once, through a
bounded bounce pool, into an unaligned one (a NumPy slice at an odd address, for example), with no
whole-object allocation either way. `head_async` returns the size from `stat` and `None` for a missing file.
Do not touch `buf` until the `await` returns; a size other than `expected_len` raises
`LengthMismatchError`, and a local file modified while it is read (same size, new timestamps) raises
`RuntimeError` ("version changed") instead of returning mixed data. A `bytearray` is only 16-byte aligned, so on
`direct://` it takes the bounce copy: use an `mmap` or a page-aligned NumPy view for the zero-copy path (see the guide).
Rust embedders that build their own multi-thread tokio runtime can opt in to the worker hand-off around each read chunk with
`.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread)` (s3dlio's own runtimes already do; without it `direct://`
and `file://` reads and writes, fsyncs and finalize all run inline on the polling worker and never panic, a single reader is
up to about 2x faster and neighbouring tasks can wait up to 16 MiB of read time; a marked `spawn_blocking` thread that
drives a `LocalSet` needs `inline_blocking_io()`; details in `s3dlio::runtime_support`). Tests: `S3DLIO_TEST_DIR` names a real-filesystem scratch directory,
`S3DLIO_REQUIRE_ODIRECT_TESTS=1` turns skips into failures (CI), and the opt-in
`tests/direct_get_into_no_odirect_fs_295.rs` needs a ramfs (`S3DLIO_TEST_NO_ODIRECT_DIR`); see
[docs/Environment_Variables.md](docs/Environment_Variables.md). See the [Python API guide](docs/PYTHON_API_GUIDE.md#get_into_async--head_async--read-into-your-own-buffer--new-in-v0100).

**Sync wrappers inside a tokio `LocalSet` (#300):** `run_on_global_rt` and the blocking wrappers built on it (for example
`s3_utils::stat_object_uri`) work from `rt.block_on(local.run_until(..))`, `LocalSet::block_on` and `spawn_local` tasks, with
`panic = "abort"` too; they used to panic. They hand the worker off (`block_in_place`) on s3dlio's own runtimes and on an
embedder's runtime whose workers have tokio's default thread name, and wait plainly elsewhere. If your runtime sets custom
thread names, add `.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread)` so a worker is never parked without
a hand-off; a `spawn_blocking` closure (or a thread hand-named like tokio's default worker) that drives a `LocalSet` with the call inside needs `inline_blocking_io()`, else it panics; a runtime with custom thread names and no marker gets no hand-off (a hang if the awaited work needs that same runtime). `MultipartUploadSink`'s blocking
methods are a separate bug (#301), not covered. Details in `s3dlio::runtime_support` and `docs/Changelog.md`.

**Writes borrow your buffer (v0.10.0):** async writes read your buffer in place until the `await` finishes; pass
`borrow=False` for a copy.

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

**XorStream — dedup-safe data generation:**

`XorStream` generates unique, incompressible data at ~15 GB/s per core without Rayon
thread management overhead. Every `fill()` / `generate()` call is guaranteed to produce a
different 512-byte-block-level fingerprint. Ideal for PUT-heavy benchmarks where many
worker threads share a single generator.

```python
import s3dlio
import numpy as np

stream = s3dlio.XorStream()

# --- Fastest path: in-place fill into pre-allocated bytearray ---
buf = bytearray(8 * 1024 * 1024)   # 8 MiB working buffer
stream.fill(buf)                    # fill once — unique payload
stream.fill(buf)                    # fill again — different payload, guaranteed

print(stream.objects_generated)     # == 2

# --- Convenience path: allocate + fill in one call ---
data = stream.generate(8 * 1024 * 1024)  # returns BytesView
view = memoryview(data)                   # zero-copy Python view
arr  = np.frombuffer(view, dtype=np.uint8)  # zero-copy numpy array

# --- PUT with XorStream data (benchmark pattern) ---
import threading

def worker(stream, uri_template, n):
    buf = bytearray(8 * 1024 * 1024)
    for i in range(n):
        stream.fill(buf)       # reuse buffer, unique data each time
        s3dlio.put_bytes(uri_template.format(i), buf)

threads = [threading.Thread(target=worker, args=(stream, "s3://bucket/obj-{}.bin", 100))
           for _ in range(32)]
for t in threads: t.start()
for t in threads: t.join()
```

| Scenario | Best choice |
|---|---|
| High concurrency PUT (≥ 32 workers) | **`XorStream`** — no Rayon scheduling |
| Medium objects 1–32 MiB | **`XorStream`** — no per-call allocation |
| Controllable compress/dedup ratios | `generate_data()` / `Generator` |
| Very large objects (≥ 256 MiB) | `Generator.fill_chunk()` |


**Multi-Endpoint Load Balancing:**
```python
# Distribute I/O across multiple storage endpoints
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://bucket-1/data",
        "s3://bucket-2/data", 
        "s3://bucket-3/data",
    ],
    strategy="least_connections"  # or "round_robin"
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

### 📦 Parquet DataLoader — Epoch-Aware Training

The Parquet DataLoader provides per-row-group streaming for Parquet files on **any
s3dlio-accessible storage** — S3, Azure Blob, GCS, `file://`, and `direct://` (O_DIRECT;
tested and working). Only the URI prefix changes; no code changes needed to switch backends.
Two decode modes: `Raw` (Python decodes) and `ArrowIpc` (Rust decodes to Arrow IPC bytes).
**Zero footer re-fetches on epoch 2+** (row-group byte ranges cached in a process-global
DashMap after epoch 1; backend-agnostic).

```python
import s3dlio

# Works with any s3dlio URI — just change the prefix:
#   "s3://bucket/train/"           Amazon S3 / MinIO / Ceph
#   "az://account/container/train/"  Azure Blob Storage
#   "gs://bucket/train/"           Google Cloud Storage
#   "file:///mnt/data/train/"      Local filesystem
#   "direct:///mnt/nvme/train/"    Local O_DIRECT (bypass page cache)

# Raw mode — Python decodes with PyArrow (default)
loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "prefetch": 32}
)
for item in loader:
    # item["data"]: bytes, item["uri"]: str, item["rg_idx"]: int
    table = pyarrow.parquet.read_table(io.BytesIO(item["data"]))

# Arrow IPC mode — Rust decodes, Python gets ready-to-use RecordBatch bytes
loader = s3dlio.create_async_loader(
    "direct:///mnt/nvme/train/",   # same API, different backend
    {"format": "parquet", "decode": "arrow", "prefetch": 32}
)
for item in loader:
    batch = pa.ipc.open_stream(pa.py_buffer(item["data"])).read_next_batch()
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `"format"` | `str` | — | Must be `"parquet"` to activate Parquet mode |
| `"decode"` | `str` | `"raw"` | `"raw"` or `"arrow"` (Rust-side decode) |
| `"columns"` | `list[int]` | `None` | Column subset; `None` = all columns |
| `"footer_cap"` | `int` | 4 MiB | Bytes from file tail for footer parsing |
| `"prefetch"` | `int` | `32` | Concurrent in-flight row-group GETs |

**Epoch-2+ speedup** — measured against live MinIO:

| Epoch | Construction time | Notes |
|-------|-----------------|-------|
| 1 | 20.4 ms | `list_objects` + footer GETs |
| 2+ | 8.3 ms | `list_objects` only — **2.5× faster** |

**Memory**: only metadata in RAM; 8 concurrent workers share process-global caches (no 8× duplication).

📖 **[Complete Parquet DataLoader Guide](docs/Parquet_Data-Loader.md)**

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
- **HTTP/2 Support (opt-in)**: HTTP/2 is supported but **not the default** — HTTP/1.1 is used unless you set `S3DLIO_H2C=1`. HTTP/2 is available for scenarios that benefit from multiplexing, but is typically slower for bulk storage workloads.
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

- **NVIDIA AIStore**: `S3DLIO_FOLLOW_REDIRECTS=1` - Enable HTTP 307 redirect following for AIStore (opt-in, disabled by default); `S3DLIO_REDIRECT_MAX=5` - Maximum redirect hops per request
- **HTTP/2 mode**: `S3DLIO_H2C=1` - Force h2c (HTTP/2 cleartext) on http:// endpoints; `S3DLIO_H2C=0` - Force HTTP/1.1; unset = auto-probe (default)
- **Runtime Scaling**: `S3DLIO_RT_THREADS=32` - Tokio worker threads
- **Connection Pool**: `S3DLIO_POOL_MAX_IDLE_PER_HOST=32` - Max idle connections per host (default: 32)
- **S3 Range GET**: `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` - Disable concurrent range splitting (enabled by default); `S3DLIO_RANGE_THRESHOLD_MB=64` - Size threshold in MiB (default: 32); `S3DLIO_RANGE_CONCURRENCY=64` - Max concurrent range requests
- **Operation Logging**: `S3DLIO_OPLOG_LEVEL=2` - S3 operation tracking
- **`.env` files**: `S3DLIO_LOAD_DOTENV=1|0|<path>` - read a `.env` file (on for `s3-cli`, off for library use; see [`.env` files](#env-files))

📖 [Environment Variables Reference](docs/Environment_Variables.md)

### Operation Logging (Op-Log)
Universal operation trace logging across all backends with zstd-compressed TSV format, warp-replay compatible.

```python
import s3dlio
s3dlio.init_op_log("operations.tsv.zst")
# All operations automatically logged
s3dlio.finalize_op_log()
# A new session can start in the same process (repeated tests, notebooks): #294
s3dlio.init_op_log("operations2.tsv.zst")
s3dlio.finalize_op_log()
```

`init_op_log()` after `finalize_op_log()` starts a fresh session with its own file and stats (it waits for the previous session to finish draining); calling it while a session is active is a no-op for the same file and a `RuntimeError` (naming the file being written and the requested one, asking you to finalize first) for any other file. After `os.fork()` a child does not share the parent's session: it sees the op-log as not initialised and may start its own, but never on a file an ancestor process is still writing (a `RuntimeError`; a log the ancestor had already finalized is not protected). This is about the op-log only: a forked child must not use s3dlio I/O once the parent has (the parent's runtime threads are not copied by `fork`, so the child's first call can hang); use the `spawn` or `forkserver` start method, or fork before the first s3dlio call. A session whose file was renamed or deleted behind its back says so in the error (it keeps writing the open file); see [Operation Logging](docs/supplemental/OPERATION_LOGGING.md). Run the op-log Python tests with `uv run -q --no-project --isolated --python 3.12 --with <wheel> --with pytest python -m pytest tests/test_oplog_lifecycle_permutations_294.py` (needs the `zstd` command-line tool).

See [S3DLIO OpLog Implementation](docs/S3DLIO_OPLOG_IMPLEMENTATION_SUMMARY.md) for detailed usage.

## Building from Source

### Prerequisites
- **Rust 1.94.1+**: [Install Rust toolchain](https://www.rust-lang.org/tools/install)
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

### `.env` files

These settings can be exported in the shell or put in a `.env` file. Whether s3dlio reads a `.env`
file is decided by one switch, `S3DLIO_LOAD_DOTENV`, and the default depends on how s3dlio is used:

| Use | Default | Why |
|-----|---------|-----|
| `s3-cli` | **on**: `.env` from the working directory | command-line convenience, as always |
| Library (Rust and Python) | **off** | a `.env` in whatever directory a host process happened to start from must not silently change its credentials, endpoints and `S3DLIO_*` / `AWS_*` settings |

| `S3DLIO_LOAD_DOTENV` | Effect |
|----------------------|--------|
| `1` / `true` / `yes` / `on` (any case) | load `.env` from the working directory (the nearest parent directory that has one, as before) |
| `0` / `false` / `no` / `off` (any case) | never load a `.env` file |
| a file path | load exactly that file |
| unset | the default above |

- Any environment variable can come from the file (`AWS_*`, `S3DLIO_*`, `AZURE_*`, `GOOGLE_*`).
- **Precedence: the real environment always wins.** A variable that is already set is never overridden
  by the file; the file only fills in what is missing.
- The switch is read from the real environment (not from a `.env`) once per process, and the file is
  loaded once, at the first of: `s3-cli` start-up; Python `import s3dlio`; for Rust, the first use of the
  global runtime, `store_for_uri*`, an S3 / Azure / GCS client, an S3 dataset, `Logger::new` or a bucket/
  container listing. That is before s3dlio reads those settings, so `S3DLIO_RT_THREADS`,
  `S3DLIO_OPLOG_*`, the range-GET settings and the endpoint/region see the file's values. Limits: a
  `torch` (or anything) that your application imported before `import s3dlio` has already run, and
  s3dlio cannot know whether its host already has threads, so for a race-free environment load the file
  (or set the real environment) at the very start of your process. Only `s3-cli` loads it first thing on
  its main thread, before its Tokio runtime exists.
- To have the file's values even earlier (before your own code calls anything in s3dlio), load it
  yourself: Rust `s3dlio::dotenv::load_env_file("path/to/.env")?`, or in Python `python-dotenv`'s
  `load_dotenv()` **before** `import s3dlio`.
- When a `.env` exists but is not loaded because of the library default, s3dlio logs one INFO line saying
  how to enable it, and the "Missing AWS_ACCESS_KEY_ID ..." error carries the same hint (so a Python user
  who sees only the exception still learns that the file is being ignored).
- `s3-cli` prints a warning on stderr when `S3DLIO_LOAD_DOTENV` names a file it cannot read.

**Behaviour change for library users (since 0.9.128):** earlier releases loaded `.env` implicitly when the
first client was built. Set `S3DLIO_LOAD_DOTENV=1` (or a file path) to get that back. Details:
[Environment Variables Reference](docs/Environment_Variables.md#env-files-s3dlio_load_dotenv).

### Operation Logging (MinIO warp format)
S3 operation logging compatible with the MinIO warp format is described under
[Operation Logging (Op-Log)](#operation-logging-op-log) above.


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
- **📦 Parquet DataLoader**: [docs/Parquet_Data-Loader.md](docs/Parquet_Data-Loader.md) - Epoch-aware Parquet DataLoader guide
- **�📚 API Documentation**: [docs/api/](docs/api/)
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
