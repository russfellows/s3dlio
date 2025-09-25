# s3dlio - Multi-Protocol AI/ML Storage Library

## Overview
s3dlio is a high-performance, multi-protocol storage library designed specifically for AI/ML workloads. Built in Rust with Python bindings, it provides zero-copy streaming, comprehensive checkpointing, and advanced data loading capabilities across multiple storage backends including S3, Azure Blob Storage, local filesystems, and high-performance DirectIO.

This project supports both low-level storage operations (get, put, list, stat, delete) and high-level AI/ML operations including PyTorch/TensorFlow/JAX-compatible data loaders and distributed checkpointing systems. The library is designed to handle arbitrarily large models and datasets while maintaining memory efficiency through zero-copy streaming architecture.

## Key Components
1. **Rust Core Library**: High-performance storage operations with zero-copy streaming
2. **Python Library**: ML framework integration with seamless PyTorch/TensorFlow/JAX compatibility  
3. **CLI Tool**: Unified command-line interface supporting all storage backends

## Multi-Backend Support

### S3 Protocol Backends  
- **Apache Arrow Backend** ⭐: **World-class performance** (2.990 GB/s PUT, 4.826 GB/s GET) with enhanced HTTP/2
- **Native AWS SDK Backend**: **Exceptional performance** (3.089 GB/s PUT, 4.579 GB/s GET) exceeding hardware baseline by 17.8%

### Other Storage Backends
- **Azure Blob Storage**: Complete Azure integration with hot/cool tier support
- **Local Filesystem**: High-performance file operations with optional O_DIRECT support  
- **DirectIO**: Bypass OS page cache for maximum I/O performance

### Backend Selection
Choose your S3 backend at compile time:
```bash
# High-performance Arrow backend (recommended)
cargo build --no-default-features --features arrow-backend

# Native AWS SDK backend  
cargo build --no-default-features --features native-backends
```

## Recent Highlights

### 🚀 Version 0.8.2 - Configurable Data Generation Modes (September 25, 2025)

**CONFIGURABLE DATA GENERATION MODES**: Introducing **configurable data generation modes** with **streaming as the default** for optimal performance. Based on comprehensive benchmarking, **streaming mode provides 2.6-3.5x performance improvement** for 1-8MB objects and wins in **64% of real-world scenarios**.

**Key Features:**
- ✅ **DataGenMode Configuration**: Choose between `streaming` (default) and `single-pass` modes
- ✅ **CLI Integration**: `--data-gen-mode` and `--chunk-size` options for performance tuning
- ✅ **Python API Enhancement**: `data_gen_mode` and `chunk_size` parameters in `put()` function  
- ✅ **Intelligent Defaults**: Streaming mode automatically selected based on benchmark data
- ✅ **Real S3 Testing**: Verified with actual S3 backend achieving 80+ MiB/s throughput
- ✅ **Backward Compatibility**: All existing code benefits from new streaming defaults

**CLI Usage:**
```bash
# Default streaming mode (optimal performance)
s3-cli put s3://bucket/object-{}.bin --num 10 --size 4MB

# Explicit mode selection  
s3-cli put s3://bucket/object-{}.bin --num 10 --size 4MB --data-gen-mode streaming
s3-cli put s3://bucket/object-{}.bin --num 10 --size 16MB --data-gen-mode single-pass
```

**Python API:**
```python
import s3dlio

# Default streaming mode (2.6-3.5x faster for most cases)
s3dlio.put("s3://bucket/object-{}.bin", num=10, template="obj-{}-of-{}", size=4194304)

# Explicit configuration
s3dlio.put("s3://bucket/object-{}.bin", num=10, template="obj-{}-of-{}", 
          size=4194304, data_gen_mode="streaming", chunk_size=65536)
```

**Performance Benchmarks:**
- **1-8MB objects**: Streaming mode provides 2.6-3.5x performance improvement
- **Overall scenarios**: Streaming wins in 64% of real-world test cases
- **Memory efficiency**: Streaming uses less memory through on-demand generation  
- **16-32MB range**: Single-pass competitive for specific object sizes

### 🎯 Version 0.8.1 - Enhanced API & PyTorch Integration (September 24, 2025)

**PRODUCTION-READY ENHANCED API**: Complete **Enhanced API** with **100% working PyTorch integration** and **34/34 tests passing**. The new simplified API provides modern interfaces while maintaining full backward compatibility.

**Key Achievements:**
- ✅ **Enhanced API Functions**: `create_dataset()` and `create_async_loader()` with options support
- ✅ **PyTorch Integration Fixed**: `S3IterableDataset` working perfectly with DataLoader support  
- ✅ **100% Test Coverage**: All 34 tests passing with comprehensive functionality validation
- ✅ **Organized Documentation**: Complete API docs in `docs/api/` with version control
- ✅ **Critical Testing Guide**: Documented Python import gotchas and best practices

**Enhanced API Usage:**
```python
import s3dlio

# Enhanced API - Simple and powerful
dataset = s3dlio.create_dataset("file:///path/to/data")
loader = s3dlio.create_async_loader("s3://bucket/path", {"batch_size": 32})

# PyTorch Integration - Now working!  
from s3dlio.torch import S3IterableDataset
dataset = S3IterableDataset("s3://bucket/data", loader_opts={})
dataloader = DataLoader(dataset, batch_size=16)

# Legacy API - Still supported
s3dlio.get("s3://bucket/file", "/local/path")  # Still works
```

**Testing Results:**
| Category | Status | Details |
|----------|---------|---------|
| **Installation** | ✅ 9/9 PASSED | All Rust functions available |
| **Enhanced API** | ✅ 8/8 PASSED | Dataset/loader creation working |  
| **PyTorch** | ✅ 5/5 PASSED | S3IterableDataset functional |
| **Legacy API** | ✅ 7/7 PASSED | Backward compatibility maintained |
| **Error Handling** | ✅ 3/3 PASSED | Proper error messages |
| **Async Operations** | ✅ 2/2 PASSED | Streaming working |

**Total: 34/34 PASSED (100.0%)**

### 🚀 Version 0.8.0 - Multi-Process Performance Engine & Python Bindings (September 20, 2025)

**WARP-LEVEL S3 PERFORMANCE**: New multi-process architecture with **Python bindings** delivering **2,308 MB/s** (8 processes) vs **1,150 MB/s** (single process) = **2x performance improvement**. Python bindings achieve **100%+ CLI performance parity** at 1,712 MB/s.

### ⚡ Version 0.7.11 - Enhanced Performance Features & Progress Bars (September 20, 2025)

**WORLD-CLASS UPLOAD PERFORMANCE**: Enhanced s3dlio with **HTTP/2** and **warp-style progress bars**, achieving **3.089 GB/s uploads that exceed hardware baseline by 17.8%** (vs 2.623 GB/s warp baseline). This demonstrates world-class upload performance.

**Enhanced Performance vs Hardware Baseline:**
| Operation | Warp Baseline | s3dlio Enhanced | Performance vs Baseline | 
|-----------|---------------|-----------------|------------------------|
| **PUT** | 2.623 GB/s | **3.089 GB/s** | **+17.8% FASTER** ⚡ |  
| **GET** | 11.537 GB/s | **4.826 GB/s** | 41.8% of potential |

**New Features:**
- ⚡ **HTTP/2 Client Support**: Modern multiplexing protocol with significant performance gains  
-  **Warp-Style Progress Bars**: Real-time progress with throughput, ETA, and completion stats
- 🚀 **Multi-Process Operations**: Warp-level performance with massive parallelism
- 📈 **Comprehensive Performance Analysis**: Complete backend comparison with baseline validation

**Enhanced Build & Usage:**
```bash
# Build with all enhanced features
cargo build --release --features enhanced-http,io-uring

# CLI with beautiful progress bars
./target/release/s3-cli put s3://bucket/prefix/ -n 1000 -s 10485760
./target/release/s3-cli get s3://bucket/prefix/ -j 48

# Performance testing suite
./scripts/long_duration_performance_test.sh
```

**📊 Complete Performance Documentation**: See [`docs/performance/`](docs/performance/) for detailed analysis and comparison reports.

---

### 🚀 Version 0.7.10 - Apache Arrow Backend & Superior Performance (September 19, 2025)

**PERFORMANCE BREAKTHROUGH**: Introduced complete **Apache Arrow `object_store` backend** that **outperforms the native AWS SDK by 15%** for PUT operations and 6% for GET operations, proving modern object storage abstractions can exceed vendor-specific performance.

**Performance Highlights:**
| Backend | PUT Performance | GET Performance | 
|---------|----------------|-----------------|
| **Apache Arrow** | **349.86 MB/s** (+15%) | **2442.47 MB/s** (+6%) |  
| Native AWS SDK | 304.54 MB/s | 2299.26 MB/s |

**Key Features:**
- ✅ **Superior Performance**: Arrow backend consistently outperforms native AWS SDK
- ✅ **Drop-in Compatibility**: Complete API compatibility with existing code  
- ✅ **Feature Flag Selection**: Compile-time backend choice with `--features arrow-backend`
- ✅ **Production Ready**: Handles real-world S3 workloads with excellent performance

**Usage:**
```bash
# Build with high-performance Arrow backend
cargo build --no-default-features --features arrow-backend

# Run performance comparison tests
./scripts/run_backend_comparison.sh
```

**Why Apache Arrow**: Better ecosystem integration, active upstream development, and now **proven superior performance** make Arrow the preferred backend for new deployments.

### ✅ Version 0.7.9 - PRODUCTION READY Python API (September 2, 2025)

**BREAKTHROUGH RELEASE**: Successfully resolved all Python async/sync integration issues. The Python API is now **fully functional and production-ready** for AI/ML workloads.

**What's Fixed:**
- ✅ **Async Integration**: No more "no running event loop" errors - all functions work from regular Python code
- ✅ **Multi-Backend Streaming**: File, Azure, and Direct I/O streaming all validated and working
- ✅ **Compression System**: High-performance zstd compression achieving 400x+ ratios
- ✅ **Checkpoint System**: Complete save/load cycle with optional compression
- ✅ **Test Coverage**: 10/10 production validation tests passing

**Production Usage:**
```python
import s3dlio

# Create streaming writer for any backend
options = s3dlio.PyWriterOptions()
writer = s3dlio.create_filesystem_writer('file:///tmp/data.txt', options)
writer.write_chunk(b'Your data here')
stats = writer.finalize()  # Returns (bytes_written, compressed_bytes)

# Checkpoint system for model states
store = s3dlio.PyCheckpointStore('file:///tmp/checkpoints/')
store.save('model_state', your_model_data)
loaded_data = store.load('model_state')
```

**Ready for Production**: All core functionality validated, comprehensive test suite, and honest documentation matching actual capabilities.

### Version 0.7.8 - Rust API Cleanup & O_DIRECT Implementation (Latest)
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

- **🔥 CPU Profiling**: High-frequency sampling (100Hz) with flamegraph generation
- **📊 Benchmarking Suite**: Criterion microbenchmarks validating 25+ GB/s buffer operations  
- **⚡ Large-Scale Testing**: 5GB+ multi-object workloads achieving **1.88 GB/s upload** and **5.34 GB/s download**
- **🎯 Zero-Overhead Design**: Feature-gated profiling with no performance impact when disabled

### Running Performance Tests
```bash
# Set up environment for real S3 testing
cp .env.example .env
# Edit .env with your AWS credentials

# Large-scale profiling test (5GB dataset)
S3DLIO_TEST_SIZE_GB=5 S3DLIO_MIN_OBJECT_MB=2 S3DLIO_MAX_OBJECT_MB=8 \
cargo run --example large_scale_s3_test --features profiling --release

# Microbenchmark suite
cargo bench --features profiling

# Simple flamegraph validation
cargo run --example simple_flamegraph_test --features profiling --release
```

### Profiling Output
All profiling results are saved to the `profiles/` directory:
- **Upload Analysis**: `large_scale_upload_profile.svg` - CPU distribution across concurrent uploads
- **Download Analysis**: `large_scale_download_profile.svg` - Memory management optimization  
- **Validation**: `simple_test_profile.svg` - Basic profiling verification

📊 **[Complete Profiling Results](docs/performance/Profiling_Results_Summary.md)** - Detailed performance analysis with flamegraph insights

### Logging
This is a new feature, that logs all S3 operations.  The file format is designed to be compatible with the MinIO warp tool.  Thus, these s3 op-log files can be used with our warp-replay project.

#### Rust CLI Usage
To enable via the cli, use the ```--op-log <FILE>``` syntax.

#### Python Library Usage
To enable s3 operation logging in Python, you must first start the logging:
```
# Start op logging (warp-replay compatible TSV.ZST)
s3.init_op_log("/tmp/python_s3_ops_test.tsv.zst")
```

Then, prior to exiting, shut down the logger to finalize the file properly.
```
# Flush and close (safe to call multiple times)
s3.finalize_op_log()
```


#### Op logging Environment variables
There are a number of tuning parameters that may be set, they are as follows:

For normal situations, optimizes I/O throughput and works on best-effort with minimal I/O or CPU impact. 
```
bash
export S3DLIO_OPLOG_BUF=2048        # entries
export S3DLIO_OPLOG_WBUFCAP=262144  # 256 KiB
export S3DLIO_OPLOG_LEVEL=1
export S3DLIO_OPLOG_LOSSLESS=false
```

For high replay fidelity, i.e. lossless logging, and high burst tolerance, use the following:
```
bash
export S3DLIO_OPLOG_BUF=8192        # entries
export S3DLIO_OPLOG_WBUFCAP=1048576 # 1 MiB
export S3DLIO_OPLOG_LEVEL=1
export S3DLIO_OPLOG_LOSSLESS=true
```

## To Do and Enhancements
What is left?  Several items are already identified as future enhancements.  These are noted both in discussions for this repo, and in the "Issues".  Note that currently all outstanding issues are tagged as enhancements, as there are no known bugs.  There are ALWAYS things that could work better, or differently, but currently, there are no outstanding issues.

### Enhancements
These are listed in no particular order:
  * (✅ Completed!) Add AI/ML data loader functionality (issue #14 - closed 9, Aug. 2025)
  * (✅ Completed!) Add AI/ML check point writer feature (issue #18 - closed 26, Aug. 2025)
  * (✅ Completed!) Implement S3 operation logging (issue #17 - closed 27, July 2025)
  * Improve Get performance (issue #11)
  * Improve Put performance and add multi-part upload (issue #12)
  * Improve python binding performance, w/ zero-copy views (issue #13)
  * (✅ Completed!) Implement regex for operations (issue #23 - closed 10, Aug. 2025)

# How to Guide
## Multi-Protocol Storage Library and CLI
This guide shows how to use the s3dlio library, which supports both a compiled command line interface executable called ```s3-cli``` and Python library, compiled into a wheel file. Using this wheel, the Python library may be installed, imported via  `import s3dlio as s3` and used by Python programs.

### Purpose
The purpose of this library is to provide a unified interface for AI/ML storage operations across multiple backends including AWS S3, Azure Blob Storage, local filesystems, and high-performance DirectIO. The library is designed to enable testing and production use of storage systems via both CLI and Python interfaces. The intention is to provide low-level storage operations (get, put, list, stat, delete) along with higher-level AI/ML compatible operations including data-loader and checkpoint-writer functionality, similar to PyTorch and TensorFlow equivalents.

### Multi-Backend Support
The library automatically detects and routes to the appropriate backend based on URI schemes:
- **s3://**: AWS S3 storage operations
- **az://**: Azure Blob Storage operations  
- **file://**: Local filesystem operations
- **direct://**: High-performance DirectIO operations

### Core Operations
The following primary operations are supported across all backends:
 - get
 - put
 - list
 - stat
 - delete
 - create-bucket (where applicable)
 - delete-bucket (where applicable)

### Enhanced Operations
Additionally, the CLI supports commands to upload and download data:
 - upload
 - download

These operations will respectively upload/put a local file as an object, or download/get an object as a local file.

**Note** that bucket creation can occur explicitly, or as part of a Put operation if the create-bucket flag is passed.  

### Container
For ease of testing and deployment, both the CLI and library may be built into a container. The Rust source code is removed from this container to reduce its size.

### Plan for Publishing
The Python library `s3dlio` will be published on PyPi for distribution. The compiled Rust executable and Python wheel files are already published on GitHub and are available under the release section of this project.  

# How to Build
In order to build this project, you can build the code and libraries and use them directly. 

### Prerequisites 
1. Rust, see [https://www.rust-lang.org/tools/install] How to Install Rust
2. The `maturin` command, which may be installed with pip, uv or some other tools
3. The `patchelf` executable, best installed with system packages, such as `sudo apt-get install patchelf` or similar 

## Building Executables and Libraries
### Rust Environment Pre-Req
Given this is a Rust project, you must install the Rust toolchain, including cargo, etc.
The following installs the Rust toolchain:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Python Environment Pre-Req
A Python environment is required.  The uv package manager and virtual environment works well.  
The 'uv' tool may be installed via: 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, you will need a virtual environment, along with Python
Here are the steps:
1. ```uv venv```
2. ```source .venv/bin/activate```
3. ```uv python install 3.12```

### Building the CLI
To build the executable cli code:
 * At the top of the project issue: `cargo build --release`
 * Upon success the executable called `s3-cli` will exist in the ./target/release directory

### Building the Python Library
This step requires the Python Environment pre-reqs listed above.
 * First, make certain you have activated your virtual environment
 * If using uv, the command `source .venv/bin/activate` will do so
 * Next, to build the python library: `maturin build --release --features extension-module`
 * Upon success a python wheel will exist in the ./target/wheels directory

# Container
A container image may be built from the source code in this repository.  Also, examining the Dockerfile may help users better understand how to build the code outside of a container by examining the steps in the Dockerfile.  
The container will contain the executable, along with a python library.  This allows you to run either the Rust cli, or a python program that uses the Python library.
### Building / Pulling
Either docker or podman may be used to create the image, typically with a `podman build -t xxx .` where xxx is the name you want to use for your container image. For those that want to simply pull and run a pre-built container, a relatively recent image may be found on quay.io.  However, the version on quay may be older than that available by building from the source code in this repository.  

***Note:** A NEW container is here:  https://quay.io/repository/russfellows-sig65/s3dlio*

In order to pull the container using podman (or docker), you may execute the following:

```
podman pull quay.io/russfellows-sig65/s3dlio
```
***Note 2:** The above container may be pulled with the a specific tag name if it exists*

### Running Container
Here is an example of starting a container using podman, and listing the contents of the containers /app directory.  
***Note:** It is VERY important to use the "--net=host" or similar command when starting, since using S3 necessitates network connectivity to S3 storage.*

```
eval@loki-node:~ $ podman run --net=host --rm -it s3dlio:0.4.2
root@loki-node:/app# ls
LICENSE  README.md  aws-env  configs  docs  local-env  python  target  tests  uv.lock  wheels
root@loki-node:/app# which python
/app/.venv/bin/python
root@loki-node:/app# python --version
Python 3.12.11

root@loki-node:/app# python ./python/tests/test_regex.py 
--- Creating S3 bucket russ-test5... ---
--- Setting up test files in S3... ---
Setup complete.

--- Testing s3dlio.list() ---
Non-recursive list found 2 items: ['test-recursive-py/data/', 'test-recursive-py/dummy1.txt']
Recursive list found 3 items: ['test-recursive-py/data/dummy2.txt', 'test-recursive-py/data/logs/dummy3.txt', 'test-recursive-py/dummy1.txt']
✅ List tests passed!

--- Testing s3dlio.download() ---
Non-recursive download got: ['dummy1.txt']
Recursive download got: ['dummy3.txt', 'dummy1.txt', 'dummy2.txt']
✅ Download tests passed!

--- Cleaning up test files... ---
--- Deleting Empty S3 bucket russ-test5... ---
✅ Cleanup successful!
root@loki-node:/app# exit 
```

# S3 Access via CLI or Python Library
This library and CLI currently support accessing S3 via http without TLS, and S3 with TLS via https for both official, and self signed certificates.  There is a method to direct the library to use a PEM file bundle of self-signed certificates, which is detailed below.

## Lack of Insecure TLS Option
There is NO option to connect via TLS and ignore certificate errors.  There are several reasons for this, the two biggest ones being:
1) It is VERY difficult to do this with the current Rust crates and AWS S3 SDK
2) It is VERY dangerous, and strongly discouraged

## Important S3 Environment Info
Setting up the environment variables and information necessary to access S3 can be somewhat tricky.  This code supports the use of variables read in from a ```.env``` file, OR environment variables.  Note that the examples here have sample .env files.  The values included from a private test environment, thus the ACCESS_KEY and SECRET_KEY values are not a concern.  Again, these may be set as environment variables if desired, but probably easier to set them in the .env file, which is read in by default.

***Note:** The following values are required:*
```
AWS_ACCESS_KEY_ID=some-id
AWS_SECRET_ACCESS_KEY=my-secret
AWS_ENDPOINT_URL=https://dns-name-or-ipaddr:port-number
AWS_REGION=region-if-needed
```
***Note:** The following value is OPTIONAL, but **REQUIRED** if using a self signed certificate with TLS access to S3:*
```
AWS_CA_BUNDLE_PATH=/a/path/to-myfile.pem
```

Here are the contents of the sample .env file:
```
eval@loki-node3:~/real-dealio2$ cat .env
AWS_ACCESS_KEY_ID=BG0XPVXISBP41DCXOQR8
AWS_SECRET_ACCESS_KEY=kGSmlMHBl0ohc/nYRtGbBx4KCfpdPN1/fLbtjUyX
AWS_ENDPOINT_URL=http://10.9.0.21
AWS_REGION=us-east-1
S3_BUCKET=my-bucket2
```

# Python
The intention of this entire library is to provide a Python library for interacting with S3.  The Rust CLI provides a way to check the underlying functions without worrying about constructing a Python program, or syntax errors.  Because it is pure Rust, it also more accurately displays the potential performance that may be attained.  However, using the Python library is the intention.  In the "docs" subdirectory, there is a quick overview of the Python API, written by a chat agent.  For those who prefer examples, there is a sample Python script to test the APIs, shown below.

## Compression Example
The Python API now supports seamless compression with the `compression_level` parameter, providing automatic compression during save and decompression during load:

```python
import s3dlio as s3

# Create a checkpoint store with compression (level 1-21, default 3)
store = s3.CheckpointStore("s3://my-bucket/checkpoints/", compression_level=10)

# Save data with compression - automatically adds .zst extension
import numpy as np
data = np.random.rand(1000, 1000).astype(np.float32)  # 4MB of random data
store.put_shard("model_weights", 0, data.tobytes())

# Load data with automatic decompression - seamlessly handles .zst files  
loaded_data = store.get_shard("model_weights", 0)
loaded_array = np.frombuffer(loaded_data, dtype=np.float32).reshape(1000, 1000)

# Compression works across all backends: S3, Azure, FileSystem, DirectIO
# Achieves 99.8% compression ratios for typical ML data
```

## Running a Python Test
Note: The following example shows running the `test_new_dlio_s3.py` Python script, after changing the number of objects to 500, by editing the file.  The default number of objects to use for testing is only 5 objects.  In order to display more accurate performance values, the following example shows running the test with `NUM_OBJECTS = 500` setting.

```
root@loki-node3:/app# uv run test_new_dlio_s3.py 
=== Sync LIST ===
Found 999 objects under s3://my-bucket2/my-data2/ in 0.02s

=== Sync PUT ===
Uploaded 500 objects (10000.00 MiB) in 3.34s -> 149.66 ops/s, 2993.14 MiB/s

=== Sync STAT ===
Stat for s3://my-bucket2/test_rust_api/object_0_of_5.dat:
  size: 20971520
  last_modified: 2025-07-16T22:53:33Z
  etag: "05a7bc9e4bab6974ebbb712ad54c18de"
  content_type: application/octet-stream
  content_language: None
  storage_class: None
  metadata: {}
  version_id: None
  replication_status: None
  server_side_encryption: None
  ssekms_key_id: None

=== Sync GET Many ===
Fetched 500 objects (10000.00 MiB) in 7.50s -> 66.67 ops/s, 1333.49 MiB/s

=== Sync GET Many Stats ===
Stats: 500 objects, 10485760000 bytes (10000.00 MiB) in 2.21s -> 226.25 ops/s, 4524.91 MiB/s

=== Sync DELETE ===
Deleted test objects in 0.37s

=== Async LIST ===
Found 999 objects under s3://my-bucket2/my-data2/ in 0.02s

=== Async PUT ===
Uploaded 500 objects (10000.00 MiB) in 3.10s -> 161.18 ops/s, 3223.68 MiB/s

=== Async GET Many ===
Fetched 500 objects (10000.00 MiB) in 7.64s -> 65.41 ops/s, 1308.29 MiB/s

=== Async GET Many Stats ===
Stats: 500 objects, 10485760000 bytes (10000.00 MiB) in 2.41s -> 207.79 ops/s, 4155.79 MiB/s

=== Async DELETE ===
Deleted test objects in 0.36s
root@loki-node3:/app# 
```

# Using the Rust CLI
If you built this project locally, outside a container, you will have to install the executable, or provide the path to its location, which by default is in the "./target/release" directory.
If running in the container, the Rust executable is in your path.  A  ```which s3-cli``` shows its location:

```
root@loki-node3:/app# which s3-cli
/usr/local/bin/s3-cli
```

Running the command without any arguments shows the usage:

```
root@loki-node3:/app# s3-cli                           
Usage: s3-cli <COMMAND>

Commands:
  create-bucket  Create a new S3 bucket
  delete-bucket  Delete an S3 bucket. The bucket must be empty
  list           List objects in an S3 path, the final part of path is treated as a regex. Example: s3-cli list s3://my-bucket/data/.*\\.csv
  stat           Stat object, show size & last modify date of a single object
  delete         Delete one object or every object that matches the prefix
  get            Download one or many objects concurrently
  put            Upload one or more objects concurrently, uses ObjectType format filled with random data
  upload         Upload local files (supports glob patterns) to S3, concurrently to jobs
  download       Download object(s) to named directory (uses globbing pattern match)
  help           Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...     Increase log verbosity: -v = Info, -vv = Debug
      --op-log <FILE>  Write warp‑replay compatible op‑log (.tsv.zst). Disabled if not provided
  -h, --help           Print help
  -V, --version        Print version

root@loki-node3:/app# 
```

### Example --help for subcommand

First is an example of getting help for the `get` subcommand:

```
root@loki-node3:/app# s3-cli get --help 
Download one or many objects

Usage: s3-cli get [OPTIONS] <URI>

Arguments:
  <URI>  S3 URI – can be a full key or a prefix ending with `/`

Options:
  -j, --jobs <JOBS>  Maximum concurrent GET requests [default: 64]
  -h, --help         Print help


```
Next is an example of help for the `put` subcommand:

```
$ s3-cli put --help
Upload one or more objects concurrently, uses ObjectType format filled with random data

Usage: s3-cli put [OPTIONS] <URI_PREFIX>

Arguments:
  <URI_PREFIX>
          S3 URI prefix (e.g. s3://bucket/prefix)

Options:
  -c, --create-bucket
          Optionally create the bucket if it does not exist

  -d, --dedup <DEDUP_F>
          Deduplication factor (integer ≥1). 1 => fully unique
          [default: 1]

  -j, --jobs <JOBS>
          Maximum concurrent uploads (jobs), but is modified to be min(jobs, num)
          [default: 32]

  -n, --num <NUM>
          Number of objects to create and upload
          [default: 1]

  -o, --object-type <OBJECT_TYPE>
          Specify Type of object to generate:
          [default: RAW]
          [possible values: NPZ, TFRECORD, HDF5, RAW]

  -s, --size <SIZE>
          Object size in bytes (default 20 MB)
          [default: 20971520]

  -t, --template <TEMPLATE>
          Template for names. Use '{}' for replacement, first '{}' is object number, 2nd is total count
          [default: object_{}_of_{}.dat]

  -x, --compress <COMPRESS_F>
          Compression factor (integer ≥1). 1 => fully random
          [default: 1]

      --data-gen-mode <DATA_GEN_MODE>          ⭐ NEW in v0.8.2
          Data generation mode for performance optimization

          Possible values:
          - streaming:   Streaming generation with configurable chunk size (default, faster for most workloads)
          - single-pass: Single-pass generation (legacy, faster for 16-32MB objects)
          [default: streaming]

      --chunk-size <CHUNK_SIZE>                ⭐ NEW in v0.8.2  
          Chunk size for streaming generation mode (bytes)
          [default: 262144]

  -h, --help
          Print help (see a summary with '-h')
```



### Additional CLI Examples
Here are several examples of running the ```s3-cli``` command:

```
root@loki-node3:/app# s3-cli list s3://my-bucket4/
...
/russ-test3-989.obj
/russ-test3-99.obj
/russ-test3-990.obj
/russ-test3-991.obj
/russ-test3-992.obj
/russ-test3-993.obj
/russ-test3-994.obj
/russ-test3-995.obj
/russ-test3-996.obj
/russ-test3-997.obj
/russ-test3-998.obj
/russ-test3-999.obj

Total objects: 1200
```
#### Example Delete command and List
```
root@loki-node3:/app# s3-cli delete s3://my-bucket4/
Deleting 1200 objects…
Done.
root@loki-node3:/app# s3-cli list s3://my-bucket4/

Total objects: 0
```
#### Example put and then delete to cleanup
Note: This use of put only uses a single set of braces `{}` , and will thus each object will have only the object number, and not the total number in the name:
```
root@loki-node3:/app# s3-cli put -c -n 1200 -t russ-test3-{}.obj s3://my-bucket4/
Uploaded 1200 objects (total 25165824000 bytes) in 7.607254643s (157.74 objects/s, 3154.88 MB/s)

root@loki-node3:/app# 
root@loki-node3:/app# s3-cli get -j 32 s3://my-bucket4/
Fetching 1200 objects with 32 jobs…
downloaded 24000.00 MB in 4.345053775s (5523.52 MB/s)

root@loki-node3:/app# s3-cli delete s3://my-bucket4/

Deleting 1200 objects…
Done.
root@loki-node3:/app#
```
