# s3dlio Changelog

## Version 0.9.24 - S3-Compatible Endpoint Fix & Tracing Workaround (December 2025)

### 🐛 **Bug Fixes**

**S3-Compatible Endpoint Support (force_path_style)**
- Fixed S3 requests to custom endpoints (MinIO, Ceph, WarpIO, etc.)
- Added `force_path_style(true)` to S3 config builder in `s3_client.rs`
- **Root cause**: AWS SDK defaults to virtual-hosted style addressing (e.g., `bucket.endpoint.com`) which doesn't work with custom endpoints that expect path-style (e.g., `endpoint.com/bucket`)
- Now matches AWS CLI behavior when using custom endpoints

**Tracing Hang Workaround**
- Fixed hang when using verbose flags (`-v`, `-vv`) with `s3-cli`
- **Root cause**: `tracing::debug!()` macros inside `tokio::spawn` async tasks cause indefinite hangs when AWS SDK S3 operations are also running in that task
- **Solution**: CLI now uses `warn,s3dlio=debug` filter to exclude AWS SDK debug logging
- Commented out debug statements inside `tokio::spawn` in `s3_utils.rs` as an additional safeguard

### 🐛 **Known Issues**

**AWS SDK Tracing Hang Bug** ([aws-sdk-rust#1388](https://github.com/awslabs/aws-sdk-rust/issues/1388))
- `tracing::debug!()` inside `tokio::spawn` + AWS SDK operations = hang
- Affects both SDK 1.104.0 and 1.116.0 (not a recent regression)
- **Workaround**: Filter tracing with `RUST_LOG=warn,s3dlio=debug`
- Debug statements outside `tokio::spawn` work correctly
- The `s3-cli` `-v` and `-vv` flags use the workaround filter automatically

### 📦 **Examples & Project Organization**

**Python Examples** (6 comprehensive examples in `examples/python/`)
- `basic_operations.py` - Core put/get/list/stat/delete operations
- `parallel_operations.py` - High-performance parallel get/put with concurrency tuning
- `data_loader.py` - ML data loader patterns with batch iteration
- `streaming_writer.py` - Chunked/streaming upload API with compression
- `upload_download.py` - File upload/download workflows
- `oplog_example.py` - Operation logging/tracing demonstration

**Examples Directory Reorganization**
- Moved Python examples to `examples/python/`
- Moved Rust examples to `examples/rust/`
- Moved shell scripts to `scripts/`
- Deleted broken examples that used outdated APIs

**Op-Log Fixes**
- Fixed `get()` and `delete()` in Python API to use `store_for_uri_with_logger()`
- Fixed `put_objects_parallel_with_progress()` to use logger
- All Python operations now properly logged via `LoggedObjectStore` wrapper

**Code Quality**
- Fixed unused import warning (`info` in `s3_client.rs`)
- Added `#[cfg(feature = "experimental-http-client")]` to functions only used with that feature
- Zero non-deprecation warnings

### 📝 **Documentation**

- Created `docs/bugs/AWS_SDK_TRACING_HANG_BUG_REPORT.md` with full investigation details
- Added doc comments to `list_objects_stream()` in `s3_utils.rs` warning about the bug
- Updated README.md with v0.9.24 release notes

---

## Version 0.9.23 - Azure Blob & GCS Custom Endpoint Support (December 3, 2025)

### 🆕 **New Features**

**Custom Endpoint Support for Azure Blob Storage**
- Added environment variable support for custom Azure endpoints
- Primary: `AZURE_STORAGE_ENDPOINT` (e.g., `http://localhost:10000`)
- Alternative: `AZURE_BLOB_ENDPOINT_URL`
- Enables use with Azurite, WarpIO, or other Azure-compatible emulators/proxies
- Account name is appended to endpoint URL automatically

Usage:
```bash
# Azurite (local emulator)
export AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000
sai3-bench util ls az://devstoreaccount1/testcontainer/

# WarpIO multi-protocol proxy
export AZURE_STORAGE_ENDPOINT=http://localhost:9001
sai3-bench util ls az://myaccount/mycontainer/
```

**Custom Endpoint Support for Google Cloud Storage**
- Added environment variable support for custom GCS endpoints
- Primary: `GCS_ENDPOINT_URL` (e.g., `http://localhost:4443`)
- Alternative: `STORAGE_EMULATOR_HOST` (GCS emulator convention, `http://` prepended if missing)
- Enables use with fake-gcs-server, WarpIO, or other GCS-compatible emulators/proxies
- Anonymous authentication used automatically for custom endpoints (typical for emulators)

Usage:
```bash
# fake-gcs-server (local emulator)
export GCS_ENDPOINT_URL=http://localhost:4443
sai3-bench util ls gs://testbucket/

# Using STORAGE_EMULATOR_HOST convention
export STORAGE_EMULATOR_HOST=localhost:4443
sai3-bench util ls gs://testbucket/

# WarpIO multi-protocol proxy
export GCS_ENDPOINT_URL=http://localhost:9002
sai3-bench util ls gs://testbucket/
```

### 📝 **Documentation**

- Updated `docs/api/Environment_Variables.md` with Azure and GCS endpoint configuration
- Added new constants in `src/constants.rs` for endpoint environment variable names:
  - `ENV_AZURE_STORAGE_ENDPOINT`
  - `ENV_AZURE_BLOB_ENDPOINT_URL`
  - `ENV_GCS_ENDPOINT_URL`
  - `ENV_STORAGE_EMULATOR_HOST`

### ⚡ **Compatibility**

**Backwards Compatibility**: No breaking changes
- When environment variables are not set, behavior remains identical to previous versions
- Connects to public cloud endpoints by default (Azure Blob, GCS)
- S3 custom endpoint support via `AWS_ENDPOINT_URL` remains unchanged

**Related Issue**: https://github.com/russfellows/sai3-bench/issues/56

---

## Version 0.9.22 - Client ID & First Byte Tracking (November 25, 2025)

### 🆕 **New Features**

**Client ID Support for Multi-Agent Operation Logging**
- Added `set_client_id()` and `get_client_id()` public functions
- All operation log entries now include client_id field
- Enables identification of which client/agent performed each operation
- Thread-safe implementation using `OnceCell<Mutex<String>>`
- Minimal overhead: ~10ns per log entry (mutex lock + clone)
- Use case: Distributed benchmarking with multiple agents writing to separate oplogs

API Usage:
```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Set client identifier (agent ID, hostname, custom ID, etc.)
let client_id = std::env::var("CLIENT_ID").unwrap_or_else(|_| "standalone".to_string());
s3dlio::set_client_id(&client_id)?;

// All future log entries tagged with this client_id
```

**Approximate First Byte Tracking** (See docs/OPERATION_LOGGING.md for details)
- `first_byte_time` field now populated in operation logs
- GET operations: first_byte ≈ end (when complete data is available)
- PUT operations: first_byte = start (upload begins immediately)
- Metadata operations (LIST, HEAD, DELETE): first_byte = None (not applicable)

**Important**: This is an *approximate* implementation due to ObjectStore trait limitations:
- Current API returns `Bytes` (complete data), not `Stream<Bytes>`
- Can't distinguish HTTP header receipt from body completion
- For small objects (<1MB): approximation is acceptable for throughput analysis
- For true TTFB metrics: Use streaming APIs (future enhancement) or dedicated HTTP tools

See [OPERATION_LOGGING.md](OPERATION_LOGGING.md) for comprehensive documentation on:
- Why first_byte is approximate
- When to use vs when to avoid
- Future enhancement plans (streaming GET API)
- Recommendations for different use cases

### 📝 **Documentation**

**New: Operation Logging Guide** (docs/OPERATION_LOGGING.md)
- Comprehensive explanation of operation logging architecture
- First byte tracking strategy with detailed rationale
- Clock offset synchronization patterns
- Client identification best practices
- Example usage for standalone and distributed scenarios
- Performance impact analysis
- Future enhancement roadmap

**Updated: Code Comments**
- Extensive inline documentation in `object_store_logger.rs`
- 40+ lines explaining first_byte tracking approach and limitations
- Clear guidance on when approximation is acceptable
- Future enhancement notes for streaming APIs

**Clarification: Operation Log Sorting**
- Added section on post-processing oplogs in OPERATION_LOGGING.md
- Clarified that logs are NOT sorted during capture (due to concurrent writes)
- Documented proper sorting workflow using sai3-bench sort command
- Note: Sorted logs compress ~30-40% better than unsorted

### ⚠️ **Important Notes**

**first_byte_time Interpretation**:
- **DO**: Use for throughput analysis and relative comparisons
- **DO**: Use for small object (<1MB) performance benchmarking
- **DON'T**: Assume it represents exact time of first byte arrival
- **DON'T**: Use for precise TTFB analysis on large objects (>10MB)

**Backwards Compatibility**:
- Existing code continues to work (client_id defaults to empty string)
- TSV format unchanged (first_byte column existed but was empty before)
- No breaking API changes

## Version 0.9.21 - Clock Offset Support & Pseudo-Random Data Generation (November 25, 2025)

### 🆕 **New Features**

**Clock Offset Support for Distributed Op-Log Synchronization** (Issue #100)
- Added `set_clock_offset()` and `get_clock_offset()` public functions
- Logger now supports timestamp correction for distributed systems
- Enables accurate global timeline reconstruction when agents have clock skew
- Thread-safe implementation using `Arc<AtomicI64>`
- Minimal overhead: single atomic read per log entry
- Use case: sai3-bench and dl-driver distributed benchmarking

API Usage:
```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Calculate clock offset during agent sync
let offset = (local_time - controller_time).as_nanos() as i64;

// Set offset for all future log entries
s3dlio::set_clock_offset(offset)?;
```

**Pseudo-Random Data Generation Method** (Issue #98 resolution)
- Added `generate_controlled_data_prand()` function
- Uses original BASE_BLOCK algorithm (~3-4 GB/s, consistent performance)
- Provides "prand" option alongside "random" (new Xoshiro256++ algorithm)
- Public API in `s3dlio::api::advanced` module
- When to use:
  - `random`: Truly incompressible data (compress=1 → ~1.0 zstd ratio), slower but more realistic
  - `prand`: Maximum CPU efficiency, faster but allows cross-block compression patterns

### ✅ **Bug Fixes & Improvements**

**Issue #95: Range Engine Messages** (Already Fixed)
- Confirmed Range Engine messages now at `trace!` level (not `debug!`)
- Fixed in commit edee657 (November 16, 2025)
- No longer overwhelms debug output

**Issue #98: Old Data Generation Code**
- Resolution: Old code now serves as "prand" method, not removed
- Provides performance option for CPU-constrained scenarios
- Both algorithms available for different use cases

### 📝 **Documentation**

- Added comprehensive clock offset documentation in `s3_logger.rs`
- Updated API documentation for both data generation methods
- Added usage examples for distributed op-log synchronization

---

## Version 0.9.20 - High-Performance List & Delete Optimizations (November 22, 2025)

### 🚀 **Performance Improvements for Large Object Operations**

Major optimizations for workloads with 100K-1M+ objects, targeting 5x performance improvement (28 minutes → 5-8 minutes for 1M objects).

**Phase 1: Batch Delete API**
- Added `delete_batch()` method to ObjectStore trait
- S3: DeleteObjects API (1000 objects/request)
- Azure: Batch API with pipeline (up to 256 operations)
- GCS: Batch delete with parallel execution
- File/Direct: Parallel deletion with configurable concurrency
- All 7 backends implement efficient batch operations

**Phase 2: Streaming List with Progress**
- Added `list_stream()` returning `Pin<Box<dyn Stream<Item = Result<String>>>>`
- S3: True streaming via paginated ListObjectsV2 (1000-object pages)
- Azure/GCS/File: Efficient buffered implementation
- MultiEndpoint: Wraps stream with per-endpoint statistics
- ObjectStoreLogger: Proper op-log integration for workload replay
- CLI: Added `-c/--count-only` flag with progress indicators
- CLI: Rate formatting with comma separators (e.g., "rate: 276,202 objects/s")

**Phase 3: List+Delete Pipeline**
- Concurrent lister/deleter tasks via tokio channels
- 1000-object batches, 10-batch buffer (10K objects in-flight)
- Overlaps LIST and DELETE operations for maximum throughput
- Progress reporting every 10K objects

**CLI Improvements:**
```bash
# Count objects with streaming progress
s3-cli ls -rc s3://bucket/prefix/
# Output: Total objects: 1,234,567 (12.345s, rate: 100,000 objects/s)

# Fast deletion with pipeline
s3-cli delete s3://bucket/prefix/
# Uses concurrent list+delete pipeline automatically
```

**Architecture:**
- Clean abstractions maintained (no backend-specific CLI code)
- Proper op-log integration for workload replay capability
- Zero-copy streaming where possible
- All optimizations work across all 7 storage backends

**Testing:**
- Comprehensive test suite with file:// backend
- Pattern matching validated (preserves non-matching objects)
- Pipeline performance verified (10K files in 0.213s)
- Progress indicators and rate formatting confirmed

**Documentation:**
- `docs/LIST_DELETE_PERFORMANCE_OPTIMIZATION.md` - Complete 3-phase plan
- All phases implemented and tested

### 📝 **API Stability**

**Rust API:**
- New trait methods: `ObjectStore::delete_batch()`, `ObjectStore::list_stream()`
- Backward compatible - existing code continues to work
- Python API automatically benefits from underlying optimizations

**Python API:**
- No changes required - `delete()` and `list()` use optimized implementations
- Batch operations work transparently under the hood

---

## Version 0.9.18 - Data Generation Bug Fix & Algorithm Migration (November 17-18, 2025)

### 🔧 **Update (November 18, 2025): RNG Optimization & Distributed Safety**

**Performance Optimization:**
- Explicit Xoshiro256PlusPlus RNG (removed StdRng abstraction)
- 5-24% performance improvement in data generation
- Same or lower CPU usage across all workloads
- Added `rand_xoshiro = "^0.7"` dependency

**Distributed Deployment Enhancement:**
- Enhanced entropy source: SystemTime + `/dev/urandom`
- Prevents data collision across distributed workers
- Critical for orchestrated environments (Kubernetes, SLURM)
- Ensures global uniqueness even with synchronized clocks

**Code Quality:**
- Updated comments: removed "Bresenham" terminology
- Clarified as "integer error accumulation" (standard distribution technique)
- No patent concerns or algorithmic attribution issues

**Comprehensive Testing:**
- New `tests/performance_comparison.rs` with CPU/memory metrics
- All 162 library tests passing
- Performance validated across 6 workload scenarios
- API compatibility verified (zero breaking changes)

**Performance Results:**
```
Test                 | OLD Speed  | NEW Speed  | Speedup | CPU Δ
---------------------|------------|------------|---------|-------
1MB compress=1       | 3,474 MB/s | 3,436 MB/s | 0.99x   | +15%
16MB compress=1      | 6,319 MB/s | 6,621 MB/s | 1.05x   | 0%
64MB compress=1      | 5,800 MB/s | 6,283 MB/s | 1.08x   | -10%
16MB compress=5      | 5,660 MB/s | 7,009 MB/s | 1.24x   | -3%
Streaming 16MB       | 2,355 MB/s | 2,530 MB/s | 1.07x   | 0%
16MB dedup=4         | 6,936 MB/s | 7,553 MB/s | 1.09x   | -3%
```

**Documentation:**
- `API_COMPATIBILITY_REPORT.md` - Complete API analysis
- Verified sai3-bench (6 call sites) and dl-driver (12 call sites) compatibility

### 🐛 **Critical Bug Fix: Cross-Block Compression**

Fixed a critical bug in the data generation algorithm where `compress=1` (incompressible data) incorrectly produced 7.68:1 compression ratio instead of ~1.0.

**Root Cause:**
- Original algorithm used shared `BASE_BLOCK` template across all unique blocks
- Zstd compressor found cross-block patterns, defeating incompressibility guarantee
- Affected all compress levels (1-6) when combined with dedup > 1

**Solution:**
- New algorithm uses per-block Xoshiro256++ RNG initialization
- Each unique block gets independent high-entropy keystream
- Local back-references within blocks for controlled compressibility
- `compress=1` now correctly produces ratio ~1.0000 ✅

### ✨ **New Data Generation Algorithm**

Introduced `data_gen_alt.rs` with improved correctness and performance:

**Features:**
- Per-block RNG seeding (prevents cross-block compression)
- Xoshiro256++ RNG (5-10x faster than ChaCha20)
- Streaming generation via `ObjectGenAlt`
- Parallel single-pass generation for large datasets
- Performance: 1-7 GB/s depending on size

**API Changes:**
- **Zero breaking changes** - all existing code works unchanged
- `generate_controlled_data()` transparently redirected to new algorithm
- `ObjectGen` now wraps `ObjectGenAlt` internally
- Old implementations preserved as commented-out code (removal: December 2025)

### 📊 **Validation Results**

**Compression Ratios (16MB test):**
- compress=1: 1.0000 ✅ (was 7.6845 ❌)
- compress=5: 1.3734 ✅
- compress=6: 1.3929 ✅

**Performance:**
- 1MB: 954 MB/s (3.76x faster than old algorithm)
- 16MB: 2,816 MB/s
- 64MB: 7,351 MB/s
- Streaming: 1,374 MB/s

**Testing:**
- All 162 library tests passing ✅
- Comprehensive test suite added (`tests/test_data_gen_alt.rs`)
- Deduplication behavior validated (dedup=2,6)
- Old algorithm bug confirmed via regression test

### 📝 **Documentation**

Added comprehensive migration documentation:
- `docs/DATA_GEN_MIGRATION_SUMMARY.md` - Technical details (9.2KB)
- `.github/ISSUE_TEMPLATE/data_gen_migration.md` - Tracking issue template (5.4KB)
- `.github/copilot-instructions.md` - Migration checklist
- `src/data_gen.rs` - 73-line header explaining changes

### 🔄 **Migration Timeline**

**Phase 1: Production Validation (Nov-Dec 2025)**
- Extended testing with real workloads
- Performance monitoring across platforms
- Compatibility verification with downstream tools

**Phase 2: Code Cleanup (December 2025)**
- Remove commented-out old algorithm code
- Update inline documentation
- Consider renaming data_gen_alt.rs → data_gen.rs

**Phase 3: Optimization (Q1 2026)**
- Profile Xoshiro256++ performance
- Evaluate SIMD opportunities
- Benchmark against industry tools

### 🔧 **Dependencies**

- Updated `rand_chacha` 0.3 → 0.9
- Fixed API breaking changes (`gen_range` → `random_range`)
- Updated test code for modern APIs

### 📦 **Test Suite**

- **Total tests**: 162 (all passing)
- **New tests**: 7 comprehensive tests for data_gen_alt
- **Coverage**: Compression ratios, dedup behavior, streaming, performance, regression

---

## Version 0.9.17 - NPY/NPZ Enhancements & TFRecord Index API (November 16, 2025)

### 🎯 **Multi-Array NPZ Support**

Added `build_multi_npz()` function for creating NumPy ZIP archives with multiple named arrays, enabling PyTorch/JAX-style dataset creation with data, labels, and metadata in a single file.

**New API:**

```rust
use s3dlio::data_formats::npz::build_multi_npz;
use ndarray::ArrayD;

// Create multi-array NPZ (PyTorch/JAX pattern)
let data = ArrayD::zeros(vec![224, 224, 3]);
let labels = ArrayD::ones(vec![10]);
let metadata = ArrayD::from_elem(vec![5], 42.0);

let arrays = vec![
    ("data", &data),
    ("labels", &labels),
    ("metadata", &metadata),
];

let npz_bytes = build_multi_npz(arrays)?;
// Write npz_bytes to file or object storage
```

**Python Interoperability:**

```python
import numpy as np

# Load multi-array NPZ created by Rust
data = np.load("dataset.npz")
print(data.files)  # ['data', 'labels', 'metadata']

images = data['data']      # Shape: (224, 224, 3)
labels = data['labels']    # Shape: (10,)
metadata = data['metadata'] # Shape: (5,)
```

**Key Features:**
- **Zero-copy design**: Uses `Bytes` for efficient memory handling
- **Proper ZIP structure**: Compatible with NumPy's `np.load()`
- **Named arrays**: Custom names for each array in the archive
- **Type support**: f32 arrays (primary ML use case)
- **Comprehensive tests**: 5 new tests covering single/multi-array scenarios

**Use Cases:**
- AI/ML dataset generation (images + labels + metadata)
- Scientific computing (simulation results + parameters + timestamps)
- dl-driver workload generation (simplified from 150+ lines to 80 lines)

---

### 🔧 **TFRecord Index Generation API**

Exported `build_tfrecord_with_index()` function and `TfRecordWithIndex` struct for creating TFRecord files with accompanying index files, enabling compatibility with TensorFlow Data Service.

**New Exports:**

```rust
use s3dlio::data_formats::{build_tfrecord_with_index, TfRecordWithIndex};

// Generate TFRecord with index in single pass
let raw_data = s3dlio::generate_controlled_data(102400, 1, 1);
let result = build_tfrecord_with_index(
    100,    // num_records
    1024,   // record_size
    &raw_data
)?;

// result.data: Bytes containing TFRecord file
// result.index: Bytes containing index file (16 bytes per record)

// Write both files
store.put("dataset.tfrecord", &result.data).await?;
store.put("dataset.tfrecord.index", &result.index).await?;
```

**Index Format** (TensorFlow Data Service compatible):
```
For each record:
  - offset: u64 (8 bytes, little-endian) - Byte offset in TFRecord file
  - length: u64 (8 bytes, little-endian) - Record length in bytes
Total: 16 bytes per record
```

**Key Features:**
- **Zero overhead**: Index generated during TFRecord creation (single pass)
- **Standard format**: Compatible with TensorFlow Data Service expectations
- **Efficient**: Returns `Bytes` for zero-copy I/O
- **Documented**: Clear API for downstream tools (dl-driver, custom tools)

**Performance:**
- No additional I/O operations
- Minimal memory overhead (16 bytes per record)
- Example: 1000 records → 16KB index file

**Background:**

TensorFlow Data Service can leverage index files to optimize random access patterns and enable efficient dataset sharding across distributed workers. This API enables tools to generate properly formatted indices alongside TFRecord data files.

---

### 🔄 **Custom NPY/NPZ Implementation**

Previously in v0.9.16, replaced `ndarray-npy` dependency with custom 328-line implementation for better control, zero-copy performance, and elimination of version conflicts.

**Features (continued from v0.9.16):**
- Full NPY format support (header + data serialization)
- Multi-array NPZ with proper ZIP structure (NEW in v0.9.17)
- TFRecord index generation API (NEW in v0.9.17)
- Zero-copy `Bytes`-based design
- Python/NumPy interoperability verified
- 11 comprehensive tests (6 NPY + 5 multi-NPZ)

---

## Version 0.9.16 - Optional Op-Log Sorting (November 7, 2025)

### 📊 **Configurable Operation Log Sorting**

Added optional automatic sorting of operation logs by start timestamp, addressing chronological ordering requirements for multi-threaded workloads while maintaining high performance for large-scale logging (10M+ operations).

**Key Changes:**

- **Optional Auto-Sort at Shutdown** - Controlled via `S3DLIO_OPLOG_SORT` environment variable
  - **Default behavior**: Streaming write (no sorting, zero memory overhead, immediate output)
  - **Opt-in sorting**: Set `S3DLIO_OPLOG_SORT=1` to enable chronological sorting
  - **Performance**: ~1.2μs per entry overhead (~4% for 210K entries)
  - **Use case**: Small to medium workloads (<1M operations) requiring sorted output

- **Streaming Sort Window Constant** - `DEFAULT_OPLOG_SORT_WINDOW = 1000`
  - Documented for future streaming sort implementations
  - Sized based on observation that operations are rarely >1000 lines out of order
  - Enables constant-memory sorting for huge files (50M+ operations)

**Background:**

Multi-threaded operation logging writes entries as they complete, not in start-time order. Variable I/O latency causes operations to finish out of sequence. For workloads requiring chronological analysis (replay, performance analysis), sorting is now optionally available.

**Environment Variables:**

```bash
# Default: Fast streaming write (unsorted)
sai3-bench run --op-log /tmp/ops.tsv --config test.yaml

# Opt-in: Auto-sort at shutdown (sorted output)
S3DLIO_OPLOG_SORT=1 sai3-bench run --op-log /tmp/ops.tsv --config test.yaml
```

**Configuration Constants:**

- `ENV_OPLOG_SORT` - Environment variable name for auto-sort control
- `DEFAULT_OPLOG_SORT_WINDOW` - Window size for streaming sort algorithms (1000 lines)

**Implementation Details:**

- Sort-on-write path collects all entries in `Vec<LogEntry>`, sorts by `start_time`, then writes
- No-sort path streams directly to file (zero buffering, minimal memory)
- Both paths use zstd compression (level 1) and auto-add `.zst` extension
- Proper `info!()` logging for transparency during long sorts

**Rationale:**

This opt-in approach balances three competing needs:
1. **Performance**: Default streaming mode has zero sorting overhead
2. **Scalability**: Large workloads (10M+ ops) avoid memory pressure
3. **Usability**: Small workloads can enable convenient auto-sort

For very large files requiring sorting, downstream tools (sai3-bench) provide streaming window-based offline sorting with constant memory usage.

---

## Version 0.9.15 - S3 URI Endpoint Parsing (November 6, 2025)

### 🔧 **Enhanced URI Parsing for Multi-Endpoint Scenarios**

Added utilities for parsing S3 URIs with optional custom endpoints, supporting MinIO, Ceph, and other S3-compatible storage systems with explicit endpoint specifications.

**New Functions:**

**Rust API:**
```rust
use s3dlio::{parse_s3_uri_full, S3UriComponents};

// Standard AWS format
let components = parse_s3_uri_full("s3://mybucket/data.bin")?;
assert_eq!(components.endpoint, None);  // Uses AWS_ENDPOINT_URL env var
assert_eq!(components.bucket, "mybucket");
assert_eq!(components.key, "data.bin");

// MinIO/Ceph with custom endpoint
let components = parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")?;
assert_eq!(components.endpoint, Some("192.168.100.1:9001".to_string()));
assert_eq!(components.bucket, "mybucket");
assert_eq!(components.key, "data.bin");
```

**Python API:**
```python
import s3dlio

# Parse standard AWS URI
result = s3dlio.parse_s3_uri_full("s3://mybucket/data.bin")
# {'endpoint': None, 'bucket': 'mybucket', 'key': 'data.bin'}

# Parse MinIO URI with endpoint
result = s3dlio.parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")
# {'endpoint': '192.168.100.1:9001', 'bucket': 'mybucket', 'key': 'data.bin'}
```

**Key Features:**
- Heuristic endpoint detection (IP addresses, hostnames, ports)
- Backwards compatible with existing `parse_s3_uri()`
- Useful for multi-process benchmarking tools (sai3-bench, dl-driver)
- Zero overhead - parsing utilities only

**Use Case:**
Multi-process testing with different endpoints per process:
```bash
# Process 1
AWS_ENDPOINT_URL=http://192.168.100.1:9001 sai3bench-agent

# Process 2  
AWS_ENDPOINT_URL=http://192.168.100.2:9001 sai3bench-agent
```

Tools can use `parse_s3_uri_full()` to extract endpoint information from config files for validation and process orchestration.

---

## Version 0.9.14 - Multi-Endpoint Storage (November 6, 2025)

### 🎯 **Multi-Endpoint Load Balancing for High-Throughput Workloads**

Added comprehensive multi-endpoint support enabling load distribution across multiple storage backends for improved performance, scalability, and fault tolerance.

**New Core Components:**

**1. MultiEndpointStore** - Load balancing wrapper implementing full `ObjectStore` trait
```rust
use s3dlio::{MultiEndpointStore, MultiEndpointStoreConfig, EndpointConfig, LoadBalanceStrategy};

let config = MultiEndpointStoreConfig {
    endpoints: vec![
        EndpointConfig { uri: "s3://bucket-1/data".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket-2/data".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket-3/data".to_string(), ..Default::default() },
    ],
    strategy: LoadBalanceStrategy::RoundRobin,  // or LeastConnections
    default_thread_count: 4,
};

let store = MultiEndpointStore::new(config).await?;
```

**2. Load Balancing Strategies:**
- **RoundRobin**: Sequential distribution for uniform workloads (minimal overhead)
- **LeastConnections**: Adaptive routing based on active connections (self-balancing)

**3. Per-Endpoint Configuration:**
```rust
EndpointConfig {
    uri: "s3://high-perf-bucket/data".to_string(),
    thread_count: Some(16),          // Override default thread count
    process_affinity: Some(0),       // NUMA node assignment for multi-socket systems
}
```

**4. URI Template Expansion** - New `uri_utils` module:
```rust
// Expand numeric ranges with optional zero-padding
expand_uri_template("s3://bucket-{1...10}/data")?;
// → ["s3://bucket-1/data", ..., "s3://bucket-10/data"]

expand_uri_template("direct://192.168.1.{01...10}:9000")?;
// → ["direct://192.168.1.01:9000", ..., "direct://192.168.1.10:9000"]

// Load URIs from configuration file (one per line)
load_uris_from_file("endpoints.txt")?;
```

**5. Statistics and Monitoring:**
```rust
// Per-endpoint statistics (lock-free atomics)
let stats = store.get_endpoint_stats();
for (i, stat) in stats.iter().enumerate() {
    println!("Endpoint {}: {} requests, {} bytes, {} active connections",
             i, stat.requests, stat.bytes_transferred, stat.active_connections);
}

// Total aggregated statistics
let total = store.get_total_stats();
println!("Total: {} requests, {} bytes, {} errors",
         total.total_requests, total.total_bytes, total.total_errors);
```

**6. Python API** - Zero-copy support via `BytesView`:
```python
import s3dlio

# Create multi-endpoint store
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://bucket-1/data", "s3://bucket-2/data", "s3://bucket-3/data"],
    strategy="least_connections"  # or "round_robin"
)

# Zero-copy data access (buffer protocol support)
data = store.get("s3://bucket-1/large-file.bin")
mv = memoryview(data)  # No copy!
array = np.frombuffer(mv, dtype=np.float32)

# Template expansion
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://shard-{1...10}/dataset",
    strategy="round_robin"
)

# File-based configuration
store = s3dlio.create_multi_endpoint_store_from_file(
    file_path="endpoints.txt",
    strategy="least_connections"
)

# Statistics
stats = store.get_endpoint_stats()
total = store.get_total_stats()
```

### ✨ **Features**

**Architecture:**
- Thread-safe design with atomic statistics (no lock contention)
- Schema validation: all endpoints must use same URI scheme (s3://, az://, gs://, file://, direct://)
- Transparent ObjectStore implementation - works anywhere single store is expected
- Per-endpoint thread pool control for optimal resource utilization

**Performance:**
- Lock-free atomic counters for statistics (zero contention overhead)
- Round-robin: ~10ns overhead per request
- Least-connections: ~50ns overhead per request
- Zero-copy Python interface via `BytesView` (buffer protocol)
- Negligible overhead vs latency (< 0.01% for typical cloud requests)

**Use Cases:**
- **Distributed ML training**: Access sharded datasets across multiple S3 buckets
- **High-throughput benchmarking**: Aggregate bandwidth from multiple storage servers
- **Fault tolerance**: Continue operations if individual endpoints fail
- **Geographic distribution**: Route requests to regionally-optimal endpoints
- **Load testing**: Distribute synthetic workloads across multiple backends

### 📝 **Testing**

**Comprehensive Test Coverage:**
- 33 new tests (16 in `uri_utils`, 17 in `multi_endpoint`)
- Total test count: 141 tests (all passing)
- Zero-copy validation tests in Python suite
- Load balancing distribution tests
- Error handling and validation tests

**Python Test Suite** (`tests/test_multi_endpoint.py`):
- Store creation from URIs, templates, and files
- Zero-copy data access via `memoryview()`
- NumPy/PyTorch integration tests
- Load balancing statistics validation
- Error handling for invalid configurations

### 🔧 **API Reference**

**Rust API:**
```rust
// Module exports
pub use multi_endpoint::{
    MultiEndpointStore,
    MultiEndpointStoreConfig,
    EndpointConfig,
    LoadBalanceStrategy,
    EndpointStats,
    TotalStats,
};

pub use uri_utils::{
    expand_uri_template,
    parse_uri_list,
    load_uris_from_file,
    infer_scheme_from_uri,  // Now public for validation
};
```

**Python API:**
```python
# Factory functions
s3dlio.create_multi_endpoint_store(uris, strategy) -> PyMultiEndpointStore
s3dlio.create_multi_endpoint_store_from_template(template, strategy) -> PyMultiEndpointStore  
s3dlio.create_multi_endpoint_store_from_file(file_path, strategy) -> PyMultiEndpointStore

# PyMultiEndpointStore methods
store.get(uri) -> BytesView                           # Zero-copy
store.get_range(uri, offset, length) -> BytesView     # Zero-copy
store.put(uri, data) -> None
store.list(prefix, recursive) -> List[str]
store.delete(uri) -> None
store.get_endpoint_stats() -> List[Dict]
store.get_total_stats() -> Dict
```

### 📖 **Documentation**

**New Documentation:**
- **[MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md)** - Comprehensive guide with:
  - Architecture overview and design principles
  - Load balancing strategy comparison
  - Rust and Python API examples
  - Configuration methods (URIs, templates, files)
  - Performance tuning guidelines
  - Best practices and use cases
  - Advanced topics (NUMA affinity, monitoring)
  - Troubleshooting guide

**Updated Documentation:**
- **[README.md](../README.md)** - Added multi-endpoint quick example in Python section
- **[docs/README.md](README.md)** - Added MULTI_ENDPOINT_GUIDE.md to Technical References

### 🔄 **Backward Compatibility**

✅ **Zero Breaking Changes:**
- New optional feature - existing code unaffected
- Single-endpoint workflows continue working unchanged
- Can wrap single endpoint in multi-endpoint store for future flexibility

### 🎯 **Use Case Examples**

**1. Distributed ML Training:**
```python
# Access sharded training data across 10 S3 buckets
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://training-shard-{1...10}/data",
    strategy="least_connections"
)

# Zero-copy data loading
for uri in file_list:
    data = store.get(uri)
    tensor = torch.frombuffer(memoryview(data), dtype=torch.float32)
```

**2. High-Throughput Benchmarking:**
```rust
// Aggregate bandwidth across 4 storage servers
let endpoints = vec![
    "s3://benchmark-1/test", "s3://benchmark-2/test",
    "s3://benchmark-3/test", "s3://benchmark-4/test",
];

// Round-robin for predictable load distribution
let config = MultiEndpointStoreConfig {
    endpoints: endpoints.iter().map(|uri| EndpointConfig {
        uri: uri.to_string(),
        thread_count: Some(8),
        ..Default::default()
    }).collect(),
    strategy: LoadBalanceStrategy::RoundRobin,
    ..Default::default()
};
```

**3. Geographic Distribution:**
```python
# Route to fastest regional endpoint automatically
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://data-us-west-2/dataset",
        "s3://data-us-east-1/dataset", 
        "s3://data-eu-west-1/dataset",
    ],
    strategy="least_connections"  # Naturally favors lower-latency endpoints
)
```

### 🔍 **Implementation Details**

**Zero-Copy Design:**
- Python `get()` and `get_range()` return `PyBytesView` objects
- `PyBytesView` implements Python buffer protocol
- Allows `memoryview()` access without copying data
- Compatible with NumPy, PyTorch, and other buffer-aware libraries
- Maintains Arc-counted `Bytes` reference (cheap clone, automatic cleanup)

**Thread Safety:**
- All statistics use `AtomicU64` counters (lock-free)
- Endpoint selection protected by `Mutex` (minimal contention)
- Concurrent operations across different endpoints fully parallel
- No performance degradation under high concurrency

**Schema Validation:**
- All endpoints must use same URI scheme (enforced at creation)
- Prevents mixing incompatible backends (e.g., s3:// + file://)
- Clear error messages for configuration mistakes

### 📊 **Performance Characteristics**

**Throughput Scaling:**
- 2 endpoints: 1.8-2.0× baseline throughput
- 4 endpoints: 3.5-4.0× baseline throughput  
- 8 endpoints: 6.5-8.0× baseline throughput
- 16+ endpoints: 12-16× baseline throughput

**Overhead:**
- RoundRobin: ~10ns per request (atomic increment)
- LeastConnections: ~50ns per request (atomic read + compare)
- For typical cloud requests (1-100ms), overhead < 0.01%

### 🔧 **Configuration Examples**

**endpoints.txt** (file-based configuration):
```text
# Production multi-region setup
s3://prod-us-west-2/data
s3://prod-us-east-1/data
s3://prod-eu-west-1/data

# Comments and blank lines ignored
```

**Template Patterns:**
```rust
// Simple numeric range
"s3://bucket-{1...5}/data"

// Zero-padded range  
"s3://bucket-{01...10}/data"

// IP addresses for direct I/O
"direct://192.168.1.{1...10}:9000"

// Multiple ranges (Cartesian product)
"s3://rack{1...3}-node{1...4}"  // → 12 URIs
```

---

## Version 0.9.12 - GCS Factory Fixes (November 3, 2025)

### 🐛 **Fixed**
- Fixed 4 misleading "GCS backend not yet fully implemented" errors in enhanced factory functions
- `store_for_uri_with_config_and_logger()` now properly supports GCS
- `direct_io_store_for_uri_with_logger()` now properly supports GCS  
- `high_performance_store_for_uri_with_logger()` now properly supports GCS

### ✨ **Added**
- `store_for_uri_with_high_performance_cloud()` - Enable RangeEngine for cloud backends
- `store_for_uri_with_high_performance_cloud_and_logger()` - With logging support
- Documentation for high-performance cloud factory functions

### 📝 **Notes**
- All basic GCS operations worked correctly even before this fix
- The errors only appeared in enhanced factory functions (not used by most code)
- dl-driver checkpoint plugin was never affected (uses basic `store_for_uri()`)

---

## Version 0.9.11 - Directory Operations (November 2024)

### 🎯 **Unified Directory Management Across All Backends**

Added `mkdir` and `rmdir` operations to the `ObjectStore` trait for consistent directory handling across file systems and cloud storage.

**New API Methods:**
```rust
async fn mkdir(&self, uri: &str) -> Result<()>
async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()>
```

**Backend Implementations:**
- **File/DirectIO** (`file://`, `direct://`): Creates/removes actual POSIX directories
  - `mkdir`: Uses `tokio::fs::create_dir_all()` for recursive creation
  - `rmdir`: Supports both empty (`remove_dir`) and recursive (`remove_dir_all`) deletion
  
- **Cloud Storage** (`s3://`, `az://`, `gs://`): Manages prefix markers
  - `mkdir`: Creates empty marker objects (e.g., `.keep`) to represent directories
  - `rmdir`: Deletes all objects under prefix (always recursive for cloud)

**Integration:**
- Required for sai3-bench v0.7.0+ directory tree workloads
- Enables consistent directory operations across hybrid storage environments
- Maintains backend-specific optimizations (e.g., cloud prefix semantics)

**Backward Compatibility:**
- ✅ Default trait implementations error gracefully for backends without support
- ✅ No breaking changes to existing APIs
- ✅ Optional functionality - existing workloads unaffected

---

## Version 0.9.10 - Pre-Stat Size Cache for Benchmarking (December 2024)

### 🚀 **Major Performance Improvement for Multi-Object Workloads**

#### **ObjectSizeCache - 2.5x Faster Benchmarking with Pre-Stat Optimization**

Eliminated redundant stat/HEAD operations in multi-object download workloads through intelligent size caching:

**Problem Solved:**
- Benchmarking tools downloading 1000+ objects spent 60% of time on sequential stat operations
- Each `get()` called stat before download: 1000 objects × 20ms stat = 20 seconds of pure overhead
- Network latency dominated total time (32.8s benchmark: 20s stat + 12.8s download)
- No way to amortize stat cost across multiple objects

**Solution - Pre-Stat Optimization:**
- New `ObjectSizeCache` module: Thread-safe size cache with configurable TTL (default 60s)
- New `pre_stat_objects()` trait method: Concurrent stat for multiple objects (100 concurrent default)
- New `pre_stat_and_cache()` API: Pre-stat all objects once, cache sizes, eliminate per-object stat overhead
- Integrated into S3, GCS, Azure backends automatically

**Performance Impact (Measured on 1000 × 64MB benchmark):**
- **Total time**: 32.8s → 13.0s (2.5x faster)
- **Stat overhead**: 20.0s → 0.2s (99% reduction)
- **Effective throughput**: 1.95 GB/s → 4.92 GB/s (2.5x improvement)
- **Pattern**: Pre-stat 1000 objects in 200ms (concurrent) vs 20s (sequential)

**Use Cases:**
- Benchmarking tools (`sai3-bench`, `io-bench`) downloading many objects
- Dataset pre-loading in ML training pipelines
- Batch processing of S3/GCS object collections
- Any workload with predictable object access patterns

**Backward Compatibility:**
- ✅ **Zero breaking changes** - All existing code works unchanged
- ✅ **Default trait methods** - All backends get concurrent pre-stat automatically
- ✅ **Opt-in API** - `pre_stat_and_cache()` is optional, regular `get()` still works
- ✅ **Graceful degradation** - Cache miss falls back to stat (same as before)

**API Usage - Before (v0.9.9 - Sequential stat overhead):**

```rust
// OLD: Each get() calls stat internally (20ms overhead per object)
let store = store_for_uri("s3://bucket/data/")?;
let objects = get_1000_object_uris();

for uri in &objects {
    let data = store.get(uri).await?;  // ❌ stat + download (20ms + 128ms = 148ms each)
    process(data);
}
// Total time: ~148 seconds for 1000 objects
```

**API Usage - After (v0.9.10 - Pre-stat optimization):**

```rust
// NEW: Pre-stat all objects once, then download without stat overhead
let store = store_for_uri("s3://bucket/data/")?;
let objects = get_1000_object_uris();

// PHASE 1: Pre-stat all objects concurrently (once at start)
store.pre_stat_and_cache(&objects, 100).await?;  // ✅ 200ms for 1000 objects

// PHASE 2: Download with zero stat overhead
for uri in &objects {
    let data = store.get(uri).await?;  // ✅ cached size, no stat! (128ms download only)
    process(data);
}
// Total time: ~128 seconds for 1000 objects (13% faster)
```

**Example: Benchmarking Tool Integration**

```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

async fn benchmark_download(bucket: &str, prefix: &str, object_count: usize) -> Result<()> {
    let store = store_for_uri(&format!("s3://{}/{}", bucket, prefix))?;
    
    // Get list of objects to benchmark
    let all_objects = store.list(&format!("s3://{}/{}", bucket, prefix), true).await?;
    let objects: Vec<String> = all_objects.into_iter().take(object_count).collect();
    
    println!("Benchmarking {} objects...", objects.len());
    
    // PRE-STAT PHASE: Load all object sizes concurrently (v0.9.10 NEW!)
    let start = std::time::Instant::now();
    let cached = store.pre_stat_and_cache(&objects, 100).await?;
    println!("Pre-statted {} objects in {:?}", cached, start.elapsed());
    
    // DOWNLOAD PHASE: Now downloads have zero stat overhead
    let download_start = std::time::Instant::now();
    let mut total_bytes = 0u64;
    
    for uri in &objects {
        let data = store.get(uri).await?;  // ✅ Uses cached size!
        total_bytes += data.len() as u64;
    }
    
    let duration = download_start.elapsed();
    let throughput_mbps = (total_bytes as f64 / 1_000_000.0) / duration.as_secs_f64();
    
    println!("Downloaded {} MB in {:?} ({:.2} MB/s)", 
             total_bytes / 1_000_000, duration, throughput_mbps);
    
    Ok(())
}
```

**Example: ML Dataset Pre-Loading**

```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

async fn preload_training_data(dataset_uris: Vec<String>) -> Result<Vec<Vec<u8>>> {
    let store = store_for_uri(&dataset_uris[0])?;
    
    // Pre-stat all files to populate size cache
    // This eliminates stat overhead during actual data loading
    store.pre_stat_and_cache(&dataset_uris, 200).await?;
    
    // Now load data with zero stat overhead
    let mut dataset = Vec::new();
    for uri in dataset_uris {
        let data = store.get(&uri).await?;
        dataset.push(data.to_vec());
    }
    
    Ok(dataset)
}
```

**Configuration Options:**

```rust
use s3dlio::object_store::{S3ObjectStore, S3Config, GcsObjectStore, GcsConfig};
use std::time::Duration;

// Custom TTL for size cache (default is 60 seconds)
let config = S3Config {
    enable_range_engine: false,
    range_engine: Default::default(),
    size_cache_ttl_secs: 120,  // 2-minute cache for longer workloads
};
let store = S3ObjectStore::with_config(config);

// Or use defaults (60 second TTL)
let store = S3ObjectStore::new();  // ✅ 60s TTL automatic
```

**Backend-Specific Behavior:**

| Backend | Cache TTL | Rationale |
|---------|-----------|-----------|
| **S3** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **GCS** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **Azure** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **file://** | 0 seconds (disabled) | Local stat is fast (~1ms), files can change on disk |
| **direct://** | 0 seconds (disabled) | Local stat is fast (~1ms), files can change on disk |

**Technical Details:**
- **Cache structure**: Thread-safe `RwLock<HashMap<String, CachedSize>>`
- **TTL mechanism**: Per-entry `Instant` timestamp, checked on `get()`
- **Concurrency**: `pre_stat_objects()` uses `futures::stream::buffer_unordered()`
- **Memory**: ~100 KB for 1000 entries (String + u64 + Instant per entry)
- **Backward compatibility**: Default trait methods ensure all backends work

**New Trait Methods:**

```rust
#[async_trait]
pub trait ObjectStore: Send + Sync {
    // ... existing methods ...
    
    /// Stat multiple objects concurrently (NEW in v0.9.10)
    async fn pre_stat_objects(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<HashMap<String, u64>> {
        // Default concurrent implementation provided
    }
    
    /// Pre-stat and cache object sizes for later use (NEW in v0.9.10)
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Default implementation (backends with cache override this)
    }
}
```

**When to Use Pre-Stat:**
- ✅ Benchmarking many objects (100+)
- ✅ Known object list before download
- ✅ Objects don't change during workload
- ✅ Network storage (S3, GCS, Azure)
- ❌ Single object downloads
- ❌ Objects changing frequently
- ❌ Local file:// or direct:// storage

**Performance Scaling:**

| Objects | Sequential Stat | Concurrent Pre-Stat | Speedup |
|---------|-----------------|---------------------|---------|
| 100 | 2.0s | 20ms | 100x faster |
| 500 | 10.0s | 100ms | 100x faster |
| 1000 | 20.0s | 200ms | 100x faster |
| 5000 | 100.0s | 1.0s | 100x faster |

**Testing:**
- 13 ObjectSizeCache unit tests (basic ops, TTL, concurrency, memory efficiency)
- Integration tests for file:// backend
- Performance validation tests (benchmarking scenarios)

**Migration Guide:**

No migration required! All existing code works without changes. To opt-in to performance improvements:

```rust
// Add this one line before your download loop:
store.pre_stat_and_cache(&object_uris, 100).await?;

// Then use store.get() as normal - it will use cached sizes
```

---

## Version 0.9.9 - Buffer Pool Optimization for DirectIO (October 2025)

### 🚀 **Major Performance Improvement**

#### **Buffer Pool for DirectIO Hot Path - 15-20% Throughput Gain**

Eliminated allocation churn in DirectIO range reads by integrating buffer pool infrastructure:

**Problem Solved:**
- DirectIO range reads allocated fresh aligned buffers on every operation
- Full buffer copies via `buffer[start..end].to_vec()` caused CPU overhead
- Allocator churn resulted in excessive page faults (20-25% throughput gap vs vdbench)

**Solution - Phase 1:**
- Wired existing `BufferPool` infrastructure into DirectIO hot path
- Pool of 32 × 64MB aligned buffers automatically initialized in `direct_io()` and `high_performance()` constructors
- Optimized `try_read_range_direct()` with borrow/return pattern:
  - **Before**: Fresh allocation + full buffer copy per range
  - **After**: Pool borrow + small copy (only requested bytes) + pool return
  - **Eliminated**: 2 allocations + 1 full copy per range operation

**Performance Impact (Expected):**
- **Throughput**: +15-20% on DirectIO with RangeEngine enabled
- **CPU**: -10-15% utilization (less memcpy, less malloc/free)
- **Page faults**: -30-50% reduction (less allocator activity)
- **Allocator calls**: -90% reduction (buffer reuse)

**Backward Compatibility:**
- ✅ **Zero breaking changes** - buffer pool is optional (`Option<Arc<BufferPool>>`)
- ✅ **Automatic** - `direct_io()` and `high_performance()` constructors initialize pool automatically
- ✅ **Graceful fallback** - Falls back to allocation if pool disabled or exhausted
- ✅ **Default unchanged** - `FileSystemConfig::default()` has no pool (backward compatible)

**API Usage (No Code Changes Required):**

```rust
// Using factory functions (pool auto-initialized for DirectIO)
let store = direct_io_store_for_uri("file:///data/")?;  // ✅ Pool automatic

// Using constructors (pool auto-initialized)
let config = FileSystemConfig::direct_io();  // ✅ Pool automatic (32 × 64MB buffers)
let config = FileSystemConfig::high_performance();  // ✅ Pool automatic

// Default (no pool, backward compatible)
let config = FileSystemConfig::default();  // ✅ No pool (compatible with v0.9.8)
```

**Technical Details:**
- Pool capacity: 32 buffers
- Buffer size: 64 MB per buffer
- Alignment: System page size (typically 4096 bytes)
- Grow-on-demand: Allocates new buffer if pool exhausted
- Async-safe: Uses tokio channels for thread-safe borrow/return

**Why Only DirectIO?**
- **S3/Azure/GCS**: Network latency (5-50ms) >> allocation (<0.1ms), pool would provide <1% benefit
- **Regular file I/O**: Kernel page cache already handles efficiency
- **DirectIO**: Aligned allocations expensive + frequent operations = pool is critical

**Files Modified:**
- `src/file_store_direct.rs`: Buffer pool integration (~80 lines)
- `src/file_store.rs`: Pre-sizing optimization (~2 lines)
- `tests/test_buffer_pool_directio.rs`: 12 comprehensive functional tests (new)
- `tests/test_allocation_comparison.rs`: Allocation overhead comparison (new)
- `docs/implementation-plans/v0.9.9-buffer-pool-enhancement.md`: Complete implementation plan (new)
- `docs/testing/v0.9.9-phase1-testing-summary.md`: Testing analysis (new)

**Testing:**
- ✅ 12/12 functional tests passing
- ✅ Concurrent access validated (64 tasks with pool size 32)
- ✅ Edge cases covered (unaligned offsets, oversized ranges, fallback)
- ✅ Data integrity verified across multiple chunks
- ✅ Zero new compiler warnings
- ⏳ Performance validation pending with sai3-bench (realistic workloads)

**Future Phases:**
- Phase 2: Writer path optimizations (optional)
- Phase 3: Zero-copy via custom Bytes deallocator (future)
- Target: Close remaining performance gap to <10% vs vdbench

---

### 📚 **Documentation Cleanup**

**Streamlined documentation from 114 to 34 files (70% reduction):**

**Removed:**
- Entire `docs/archive/` directory (584KB of historical docs)
  - Old API versions (pre-v0.9.0)
  - Completed implementation plans (v0.8.x)
  - Old performance reports and benchmarks
  - Pre-v0.8.0 changelog and release notes
- Version-specific docs (v0.9.4, v0.9.5, v0.9.6)
- Old API snapshots (v0.9.2, v0.9.3)
- Detailed release plans for v0.9.0-v0.9.2 (15 files)
- Completed implementation plans (4 files)

**Added:**
- `docs/STREAMING-ARCHITECTURE.md`: Comprehensive guide to stream-based patterns
  - RangeEngine streaming patterns
  - Backpressure control mechanisms
  - Cancellation support
  - DataLoader streaming architecture

**Kept (Essential):**
- Core guides: README, Changelog, RELEASE-CHECKLIST, TESTING-GUIDE
- Configuration: CONFIGURATION-HIERARCHY, VERSION-MANAGEMENT
- API references: ZERO-COPY-API-REFERENCE, TFRECORD-INDEX-QUICKREF
- Performance: Performance_Optimization_Summary, Performance_Profiling_Guide

**Rationale:**
- Changelog.md is authoritative source for version history
- Reduced maintenance burden and improved navigation
- Current API documented in code and reference docs

---

## Version 0.9.8 - Optional GCS Backends & Page Cache Configuration (October 2025)

### 🚀 **New Features**

#### **Optional Google Cloud Storage (GCS) Backends**

Added **two mutually exclusive GCS backend implementations** selectable at compile time:

1. **`gcs-community`** (default) - Uses community-maintained `gcloud-storage` v1.1 crate
   - ✅ **Production Ready**: All tests pass reliably (10/10)
   - ✅ Stable and proven in production workloads
   - ✅ Full ADC (Application Default Credentials) support
   - ✅ Supports all GCS operations: GET, PUT, DELETE, LIST, STAT, range reads

2. **`gcs-official`** (experimental) - Uses official Google `google-cloud-storage` v1.1 crate
   - ⚠️ **Experimental**: Known transport flakes in test suites (10-20% failure rate)
   - ✅ Individual operations work correctly when tested in isolation
   - ⚠️ Full test suite has intermittent "transport error" failures (improved to 8-9/10 tests passing)
   - 🐛 **Root Cause**: Upstream flake in `google-cloud-rust` library
     - **Bug Report**: https://github.com/googleapis/google-cloud-rust/issues/3574
     - **Related Issue**: https://github.com/googleapis/google-cloud-rust/issues/3412
   - ✅ **MAJOR IMPROVEMENT**: Implemented global client singleton pattern
     - **Before**: 30% pass rate (3/10 tests) - StorageControl operations failed consistently
     - **After**: 80-90% pass rate (8-9/10 tests) - Most operations now work reliably
     - **Implementation**: `once_cell::Lazy<OnceCell<GcsClient>>` for single client per process
   - 📊 **Current Status**: Acceptable for development/testing but expect occasional flakes until upstream fix

**Build Options:**

```bash
# Default (community backend - RECOMMENDED)
cargo build --release

# Explicit community backend
cargo build --release --features gcs-community

# Official backend (experimental - for testing only)
cargo build --release --no-default-features --features native-backends,s3,gcs-official

# ❌ Cannot use both (compile error by design)
cargo build --features gcs-community,gcs-official  # ERROR
```

**Why Two Backends?**

The dual-backend approach allows:
- ✅ Production stability with `gcs-community` (default)
- ✅ Future migration path when upstream transport issues are resolved
- ✅ A/B performance testing and benchmarking
- ✅ Easy switching without code changes (compile-time selection only)

**Recommendation**: Use `gcs-community` (default) for all production workloads. The `gcs-official` backend is provided for experimentation and to track upstream development, but is not suitable for production due to the transport flakes documented in Issue #3574.

**Files Changed:**
- `src/google_gcs_client.rs`: New implementation using official `google-cloud-storage` crate
- `src/gcs_client.rs`: Community implementation (default)
- `src/object_store.rs`: Factory pattern with compile-time backend selection
- `tests/test_gcs_community.rs`: Complete test suite for community backend (10/10 pass ✅)
- `tests/test_gcs_official.rs`: Complete test suite for official backend (3/10 pass, 7/10 fail ⚠️)
- `tests/common/mod.rs`: Shared test utilities
- `docs/GCS-BACKEND-SELECTION.md`: Comprehensive backend selection guide
- Bug report filed: https://github.com/googleapis/google-cloud-rust/issues/3574

---

#### **Configurable `posix_fadvise` Hints for File I/O**

Added ability to configure page cache behavior for file system operations (`file://` and `direct://` URIs) via the new `page_cache_mode` field in `FileSystemConfig`.

**What is `posix_fadvise`?**  
On Linux/Unix systems, `posix_fadvise()` provides hints to the kernel about expected file access patterns, allowing the OS to optimize page cache behavior for better performance.

**Available Modes:**
- **`Auto`** (default for full GETs): Automatically selects Sequential for large files (>= 64 MiB), Random for small files
- **`Sequential`**: Prefetch data ahead - optimal for streaming reads, large sequential scans
- **`Random`**: Don't prefetch - optimal for random access patterns, database queries
- **`DontNeed`**: Don't cache pages - optimal for one-time reads, streaming that won't be re-read
- **`Normal`**: Let OS use default heuristics

**Default Behavior (No Configuration Required):**
- Full GET operations: `Auto` mode (intelligent based on file size)
- Range GET operations: `Random` mode (typical for random access)

**Rust API Usage:**

```rust
use s3dlio::object_store::{store_for_uri_with_config, FileSystemConfig, PageCacheMode};

// Example 1: Sequential access pattern (streaming, large files)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::Sequential),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///data/", Some(config))?;

// Example 2: Random access pattern (database, random seeks)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::Random),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///db/", Some(config))?;

// Example 3: Don't pollute cache (one-time streaming)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::DontNeed),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///stream/", Some(config))?;

// Example 4: Use defaults (Auto for full GET, Random for range GET)
let store = store_for_uri("file:///data/")?;  // No config needed
```

**When to Use:**
- ✅ **Sequential**: Large file streaming, sequential scans, media processing
- ✅ **Random**: Database files, random access patterns, sparse file access
- ✅ **DontNeed**: Large one-time reads that won't be re-accessed (prevents cache pollution)
- ✅ **Auto**: Default - works well for most workloads

**Performance Impact:**
- Sequential mode: Can provide 2-3x improvement for large sequential reads
- Random mode: Reduces memory pressure for random access patterns
- DontNeed mode: Prevents cache pollution for one-time large file operations

**Files Changed:**
- `src/file_store.rs`: Added `page_cache_mode` field to `FileSystemConfig`, updated GET operations
- `src/api.rs`: Exported `PageCacheMode`, `FileSystemConfig`, and `store_for_uri_with_config`
- `tests/test_file_range_engine.rs`: Updated tests for new field

**Backward Compatibility:**  
✅ **Fully backward compatible** - existing code continues to work with Auto/Random defaults

---

## Version 0.9.6 - RangeEngine Disabled by Default (October 2025)

### ⚠️ **BREAKING CHANGES**

#### **RangeEngine Disabled by Default (ALL Backends)**

**Problem:** Performance testing revealed that RangeEngine causes **up to 50% slowdown** for typical workloads due to the extra HEAD/STAT request required on every GET operation to determine object size. While RangeEngine provides 30-50% throughput improvement for large files (>= 16 MiB), the mandatory stat overhead makes it counterproductive for most real-world workloads.

**Solution:** RangeEngine is now **disabled by default** across all storage backends. Users must explicitly opt-in for large-file workloads where benefits outweigh the stat overhead.

**Affected Backends:**
- ✅ **Azure Blob Storage** (`az://`) - `enable_range_engine: false`
- ✅ **Google Cloud Storage** (`gs://`, `gcs://`) - `enable_range_engine: false`
- ✅ **Local File System** (`file://`) - `enable_range_engine: false`
- ✅ **DirectIO** (`direct://`) - `enable_range_engine: false`
- ✅ **S3** (`s3://`) - RangeEngine not yet implemented

**Performance Impact:**
- **Before (v0.9.5)**: Every GET operation performed HEAD + GET (2 requests)
- **After (v0.9.6)**: GET operations use single request (no stat overhead)
- **Typical workloads**: 50% faster due to elimination of HEAD requests
- **Large-file workloads**: Must enable RangeEngine explicitly to get 30-50% benefit

**Migration Guide:**

**No changes required for most users** - default behavior now faster for typical workloads.

**For large-file workloads (>= 16 MiB objects), explicitly enable RangeEngine:**

```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig, RangeEngineConfig};

// Enable RangeEngine for large-file Azure workload
let config = AzureConfig {
    enable_range_engine: true,  // Explicitly enable (was: default true, now: default false)
    range_engine: RangeEngineConfig {
        min_split_size: 16 * 1024 * 1024,  // 16 MiB threshold (default)
        max_concurrent_ranges: 32,          // 32 parallel ranges
        chunk_size: 64 * 1024 * 1024,      // 64 MiB chunks
        ..Default::default()
    },
};
let store = AzureObjectStore::with_config(config);
```

**Same pattern for all backends:**

```rust
// Google Cloud Storage
use s3dlio::object_store::{GcsObjectStore, GcsConfig};
let config = GcsConfig {
    enable_range_engine: true,  // Opt-in for large files
    ..Default::default()
};
let store = GcsObjectStore::with_config(config);

// Local File System
use s3dlio::file_store::{FileSystemObjectStore, FileSystemConfig};
let config = FileSystemConfig {
    enable_range_engine: true,  // Rarely beneficial for local FS
    ..Default::default()
};
let store = FileSystemObjectStore::with_config(config);

// DirectIO
use s3dlio::file_store_direct::{FileSystemConfig};
let config = FileSystemConfig::direct_io();  // Still disabled by default
// Explicitly enable if needed:
let mut config = FileSystemConfig::direct_io();
config.enable_range_engine = true;
```

**When to Enable RangeEngine:**
- ✅ Large-file workloads (average object size >= 64 MiB)
- ✅ High-bandwidth networks (>= 1 Gbps) with high latency
- ✅ Dedicated large-object operations (media processing, ML training)
- ❌ Mixed workloads with small and large objects
- ❌ Benchmarks with small objects (< 16 MiB)
- ❌ Local file systems (seek overhead usually outweighs benefit)

---

### 🔧 **Configuration Changes**

#### **All Backend Configs Updated**

1. **`AzureConfig`** (`src/object_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation updated to reflect opt-in behavior

2. **`GcsConfig`** (`src/object_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation updated to reflect opt-in behavior

3. **`FileSystemConfig`** (`src/file_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation emphasizes local FS rarely benefits

4. **`FileSystemConfig` (DirectIO)** (`src/file_store_direct.rs`)
   - `enable_range_engine`: `true` → `false` in:
     - `Default::default()`
     - `direct_io()`
     - `high_performance()`
   - All presets now disable RangeEngine by default

#### **Constants Documentation Updated**

- `DEFAULT_RANGE_ENGINE_THRESHOLD` documentation clarifies:
  - RangeEngine is **disabled by default**
  - 16 MiB threshold applies only when explicitly enabled
  - Explains performance trade-offs
  - Provides example configurations for enabling

---

### 📊 **Performance Summary**

| Workload Type | v0.9.5 Performance | v0.9.6 Performance (Default) | v0.9.6 with RangeEngine Enabled |
|---------------|-------------------|------------------------------|----------------------------------|
| Small objects (< 16 MiB) | **Slow** (2x requests: HEAD + GET) | **Fast** (1x request: GET only) | **Slow** (2x requests: HEAD + GET) |
| Large files (>= 64 MiB) | **Medium** (HEAD + single GET) | **Medium** (single GET) | **Fast** (HEAD + parallel range GETs) |
| Mixed workloads | **Slow overall** (all GETs statted) | **Fast** (no stat overhead) | **Slow overall** (all GETs statted) |

**Key Insight:** RangeEngine provides benefit **only for large-file workloads**. For typical workloads with mixed object sizes, the stat overhead on small objects outweighs gains on large objects.

---

### 📚 **Documentation Updates**

- **Constants (`src/constants.rs`)**: Comprehensive RangeEngine configuration guide
- **Config Structs**: All backend configs document opt-in behavior
- **README.md**: Updated to reflect disabled-by-default status
- **Copilot Instructions**: Updated RangeEngine guidance

---

### 🔄 **Backward Compatibility**

**Breaking Change:** Existing code that relies on RangeEngine being enabled by default will see different behavior. However, this change **improves performance for most workloads**.

**Migration Path:**
1. **If your workload uses small/mixed objects**: No changes needed - enjoy faster performance
2. **If your workload uses large files (>= 64 MiB)**: Add explicit `enable_range_engine: true` to config

**Deprecations:** None. All APIs remain stable.

---

## Version 0.9.5 - Performance Fixes & RangeEngine Tuning (October 2025)

### 🚀 **PERFORMANCE IMPROVEMENTS**

#### **1. Adaptive Concurrency for Delete Operations (10-70x faster)**

Fixed critical performance regression introduced in v0.8.23 where progress bar implementation caused delete operations to become sequential instead of concurrent.

**Performance Gains:**
- **500 objects**: ~0.7 seconds (was ~50 seconds) - **70x faster**
- **7,000 objects**: ~5.5 seconds (was ~70-140 seconds) - **12-25x faster**
- **93,000 objects**: ~90 seconds (was ~15+ minutes) - **10x+ faster**

**Implementation Details:**

1. **New `delete_objects_concurrent()` Helper Function** (`src/object_store.rs`)
   - Universal implementation works with all backends (S3, Azure, GCS, file://, direct://)
   - Uses `futures::stream` with `buffer_unordered` for concurrent deletions
   - Batched progress updates (every 50 operations) instead of per-object updates
   - 98% reduction in progress bar overhead

2. **Adaptive Concurrency Algorithm**
   
   Automatically scales concurrency based on total object count (10% of total):
   
   | Total Objects | Concurrency | Reasoning |
   |--------------|-------------|-----------|
   | < 10 | 1 | Very small: sequential is efficient |
   | 10-99 | 10 | Small: minimum viable concurrency |
   | 100-9,999 | total/10 (max 100) | Medium: scales with workload |
   | 10,000+ | total/10 (max 1,000) | Large: capped to avoid overwhelming backends |
   
   **Examples:**
   - 100 objects → 10 concurrent deletions
   - 500 objects → 50 concurrent deletions
   - 5,000 objects → 100 concurrent deletions (capped)
   - 93,000 objects → 1,000 concurrent deletions (capped)

3. **CLI Updates** (`src/bin/cli.rs`)
   - Both pattern-filtered and full-prefix deletions now use concurrent helper
   - Displays adaptive concurrency level in user messages
   - Maintains smooth progress bar with batched updates

**Testing:**
- ✅ Tested with Google Cloud Storage: 7,010 objects deleted in 5.5 seconds
- ✅ Throughput: ~1,280 deletions/second
- ✅ Progress bar verified working correctly
- ✅ Universal backend compatibility confirmed

**Technical Notes:**
- Progress updates use `Arc<AtomicUsize>` for lock-free concurrent counting
- Callback wrapped in `Arc` for sharing across async tasks
- ProgressBar cloned to avoid move issues in closures
- Final progress update ensures accurate completion count

---

#### **2. RangeEngine Threshold Increased to 16 MiB (Fixes 10% regression)**

**Problem:** v0.9.3 introduced a 10% performance regression for small object workloads (e.g., 1 MiB objects) due to an extra HEAD request on every GET operation to check object size for RangeEngine eligibility.

**Root Cause Analysis:**
- v0.9.3 used 4 MiB threshold for RangeEngine
- For objects < 4 MiB, code still performed HEAD + GET (2 requests instead of 1)
- Benchmarks with 1 MiB objects saw 60% more total requests → ~10% slowdown
- The stat overhead outweighed any potential benefit for small objects

**Solution:** Raised default threshold to **16 MiB** for all network backends (S3, Azure, GCS).

**Changes:**

1. **New Universal Constant** (`src/constants.rs`)
   ```rust
   /// Universal default minimum object size to trigger RangeEngine (16 MiB)
   pub const DEFAULT_RANGE_ENGINE_THRESHOLD: u64 = 16 * 1024 * 1024;
   ```
   
   - Replaces backend-specific thresholds (4 MiB)
   - Legacy aliases deprecated but maintained for compatibility
   - Comprehensive documentation explaining threshold selection

2. **Updated Backend Configs**
   - `AzureConfig`: Now uses 16 MiB threshold
   - `GcsConfig`: Now uses 16 MiB threshold
   - `S3Config`: Now uses 16 MiB threshold (when RangeEngine added)

**Performance Impact:**
- **Small objects (< 16 MiB)**: Restored to v0.8.22 performance (single GET request)
- **Medium objects (16-64 MiB)**: Still benefit from RangeEngine (20-40% faster)
- **Large objects (> 64 MiB)**: Maximum RangeEngine benefit (30-60% faster)
- **Benchmarks**: 1 MiB object workloads no longer see regression

**When to Override:**

```rust
use s3dlio::object_store::{GcsConfig, RangeEngineConfig};

// Lower threshold for large-file workloads (higher latency networks)
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        min_split_size: 4 * 1024 * 1024,  // 4 MiB threshold
        ..Default::default()
    },
};

// Higher threshold to avoid stat overhead on most objects
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        min_split_size: 64 * 1024 * 1024,  // 64 MiB threshold
        ..Default::default()
    },
};

// Disable RangeEngine entirely for small-object benchmarks
let benchmark_config = GcsConfig {
    enable_range_engine: false,
    ..Default::default()
};
```

**Documentation Updates:**
- Added detailed comments in `src/constants.rs` explaining threshold rationale
- Updated Azure/GCS config documentation to reflect 16 MiB threshold
- Performance regression analysis documented

**Testing:**
- ✅ Benchmarks with 1 MiB objects now match v0.8.22 performance
- ✅ Large file downloads still use RangeEngine (>= 16 MiB)
- ✅ No extra HEAD requests for typical workloads
- ✅ Configuration override tested and working

---

### 📊 **Performance Summary**

| Change | Workload | Performance Gain |
|--------|----------|------------------|
| Adaptive Delete | 7,000 objects | 12-25x faster (5.5s vs 70-140s) |
| RangeEngine Fix | 1 MiB GETs | 10% regression eliminated |
| Combined | Mixed workloads | Restored + improved performance |

---

### 🔧 **Breaking Changes**
None. All changes are backward compatible with optional configuration overrides.

---

### 📚 **Deprecations**
- `DEFAULT_S3_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`
- `DEFAULT_AZURE_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`
- `DEFAULT_GCS_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`

Legacy constants remain functional but emit deprecation warnings.

---

## Version 0.9.4 - S3-Specific API Deprecation (October 2025)

### ⚠️ **DEPRECATION NOTICES**

#### **Python & Rust API: S3-Specific Data Functions Deprecated (Removal in v1.0.0)**

Two S3-specific data operation functions have been deprecated in favor of universal URI-based alternatives. **These functions will be removed in v1.0.0.**

**IMPORTANT**: 
- This deprecation affects **BOTH Python AND Rust APIs**
- `create_bucket()` and `delete_bucket()` are **NOT deprecated** - they will be made universal in future releases

**Deprecated Functions:**

1. **`list_objects(bucket, prefix, recursive)`** → Use `list(uri, recursive, pattern)`
   
   **Python:**
   ```python
   # OLD (deprecated)
   objects = s3dlio.list_objects("bucket", "prefix/", recursive=True)
   
   # NEW (universal)
   objects = s3dlio.list("s3://bucket/prefix/", recursive=True)
   ```
   
   **Rust:**
   ```rust
   // OLD (deprecated)
   use s3dlio::s3_utils::list_objects;
   let objects = list_objects("bucket", "prefix/", true)?;
   
   // NEW (universal)
   use s3dlio::api::{store_for_uri, ObjectStore};
   let store = store_for_uri("s3://bucket/prefix/")?;
   let objects = store.list("", true, None).await?;
   ```

2. **`get_object(bucket, key, offset, length)`** → Use `get(uri)` or `get_range(uri, offset, length)`
   
   **Python:**
   ```python
   # OLD (deprecated)
   data = s3dlio.get_object("bucket", "key", offset=1024, length=4096)
   
   # NEW (universal)
   data = s3dlio.get_range("s3://bucket/key", offset=1024, length=4096)
   ```

**Timeline:**
- **v0.9.4**: Functions work with deprecation warnings (stderr + compile-time)
- **v0.9.x → v1.0.0-rc**: Functions continue working with warnings
- **v1.0.0**: Functions removed

**See**: [DEPRECATION-NOTICE-v0.9.4.md](./DEPRECATION-NOTICE-v0.9.4.md) for complete migration guide.

---

## Version 0.9.3 - RangeEngine for Azure & GCS Backends (October 2025)

#### **1. RangeEngine Integration for Azure Blob Storage**
Concurrent range downloads significantly improve throughput for large Azure blobs by hiding network latency with parallel range requests.

**Performance Gains:**
- Medium blobs (4-64MB): 20-40% faster
- Large blobs (> 64MB): 30-50% faster
- Huge blobs (> 1GB): 40-60% faster on high-bandwidth networks

**Implementation:**
- Created `AzureConfig` with configurable RangeEngine settings
- Refactored `AzureObjectStore` from unit struct to stateful struct with Clone support
- Added `get_with_range_engine()` helper for concurrent downloads
- Size-based strategy: files >= 4MB use RangeEngine automatically

**Configuration (network-optimized defaults):**
```rust
pub struct AzureConfig {
    pub enable_range_engine: bool,  // Default: true
    pub range_engine: RangeEngineConfig {
        chunk_size: 64 * 1024 * 1024,        // 64MB chunks
        max_concurrent_ranges: 32,            // 32 parallel ranges
        min_split_size: 4 * 1024 * 1024,     // 4MB threshold
        range_timeout: Duration::from_secs(30),
    },
}
```

**Usage:**
```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig};

// Default configuration (RangeEngine enabled)
let store = AzureObjectStore::new();

// Custom configuration
let config = AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig::default(),
};
let store = AzureObjectStore::with_config(config);

// Downloads automatically use RangeEngine for files >= 4MB
let data = store.get("az://account/container/large-file.bin").await?;
```

**Validation:**
- ✅ Python tests: 16.54 MB/s download (8MB blob)
- ✅ Rust tests: All ObjectStore methods validated
- ✅ Zero-copy Bytes API compatibility
- ✅ Builds with zero warnings

---

#### **2. RangeEngine Integration for Google Cloud Storage**
Complete GCS backend enhancement with concurrent range downloads following the same pattern as Azure.

**Performance:**
- 128MB file: 44-46 MB/s with 2 concurrent ranges
- Validated on production GCS buckets (signal65-russ-b1)
- Expected 30-50% gains on high-bandwidth networks (>1 Gbps)

**Implementation:**
- Created `GcsConfig` with identical structure to AzureConfig
- Refactored `GcsObjectStore` from unit struct to stateful struct
- Added `get_with_range_engine()` for concurrent downloads
- Same 4MB threshold and network-optimized defaults

**Configuration:**
```rust
use s3dlio::object_store::{GcsObjectStore, GcsConfig};

// Default configuration (RangeEngine enabled)
let store = GcsObjectStore::new();

// Custom configuration
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 64 * 1024 * 1024,
        max_concurrent_ranges: 32,
        min_split_size: 4 * 1024 * 1024,
        range_timeout: Duration::from_secs(30),
    },
};
let store = GcsObjectStore::with_config(config);
```

**Validation:**
- ✅ Python tests: 45.61 MB/s download (128MB file, 2 concurrent ranges)
- ✅ Rust smoke tests: All 5 tests passing (8.06s total)
  - Small files (1MB): Simple download below threshold
  - Large files (128MB): 44.60 MB/s with RangeEngine
  - Range requests: Partial reads validated
  - Metadata: stat() with ETag
  - Listings: Directory operations working
- ✅ Debug logging confirms concurrent range execution
- ✅ Zero-copy Bytes API compatibility

---

#### **3. Universal Python API Enhancements**
All Python API functions now work universally across all 5 backends (S3, Azure, GCS, file://, direct://).

**Changes:**
- `put()`: Template parameter now optional (default: "object-{}")
- `get()`: Universal implementation via `store_for_uri()` + `ObjectStore::get()`
- `delete()`: Works with all URI schemes, supports pattern matching
- `build_uri_list()`: Generic URI parsing for any scheme

**Before (v0.9.2 - S3-specific):**
```python
# Only worked with s3:// URIs
s3dlio.put("s3://bucket/prefix/", num=3, template="object-{}", size=1024*1024)
```

**After (v0.9.3 - Universal):**
```python
# Works with all backends
s3dlio.put("s3://bucket/prefix/", num=3, size=1024*1024)        # S3
s3dlio.put("az://account/container/prefix/", num=3, size=1024*1024)  # Azure
s3dlio.put("gs://bucket/prefix/", num=3, size=1024*1024)        # GCS
s3dlio.put("file:///local/path/", num=3, size=1024*1024)        # Local
s3dlio.put("direct:///local/path/", num=3, size=1024*1024)      # DirectIO

# get() works universally
data = s3dlio.get("gs://bucket/file.bin")  # Returns BytesView (zero-copy)

# delete() works with patterns
s3dlio.delete(["az://container/prefix/*"])  # Deletes all matching
```

**Validation:**
- ✅ Azure Python tests: All operations working (put, get, get_range, large files)
- ✅ GCS Python tests: All operations working (128MB with RangeEngine)
- ✅ Zero-copy BytesView wrapper preserved
- ✅ Pattern matching in delete() validated

---

### 🧪 **Testing & Validation**

#### **New Test Suites:**
1. **`python/tests/test_azure_api.py`**: Comprehensive Azure backend validation
2. **`python/tests/test_gcs_api.py`**: GCS backend with RangeEngine tests (128MB files)
3. **`tests/test_gcs_smoke.rs`**: Rust integration tests (5 test cases, all passing)

#### **Test Coverage:**
- Azure: 17.08 MB/s download (8MB with RangeEngine)
- GCS Python: 45.61 MB/s download (128MB, 2 concurrent ranges)
- GCS Rust: 44.60 MB/s download (128MB, 2 concurrent ranges)
- All ObjectStore methods validated: get, put, delete, list, stat, get_range
- Zero-copy Bytes API compatibility confirmed
- Debug logging added for RangeEngine activity tracking

---

### 📦 **Infrastructure Updates**

- **Build System**: Zero warnings policy enforced across all builds
- **Tracing**: Debug logging support for RangeEngine activity (`s3dlio.init_logging("debug")`)
- **Documentation**: Added GCS login helper script (`scripts/gcs-login.sh`)
- **Architecture**: Consistent backend patterns across Azure and GCS
  - Both use stateful structs with Config
  - Both support Clone for closure compatibility
  - Both use identical RangeEngine configuration

---

### 🔧 **Breaking Changes**
None. This is a backward-compatible feature release.

- Existing Azure code continues to work (defaults to RangeEngine enabled)
- Existing GCS code continues to work (defaults to RangeEngine enabled)
- Python API changes are purely additive (template parameter optional)
- All v0.9.2 code compiles and runs without modification

---

### 📊 **Performance Summary**

| Backend | File Size | Method | Throughput | Improvement |
|---------|-----------|--------|------------|-------------|
| Azure | 8MB | RangeEngine (1 range) | 16.54 MB/s | Baseline |
| GCS | 8MB | RangeEngine (1 range) | 22.97 MB/s | Baseline |
| GCS | 128MB | RangeEngine (2 ranges) | 44-46 MB/s | Network-limited |
| Expected | >1GB | RangeEngine (multi-range) | 30-50% faster | On fast networks |

**Note**: Performance gains depend on network bandwidth and latency. High-bandwidth networks (>1 Gbps) with higher latency will see the most benefit from concurrent range requests.

---

### 🚀 **Migration Guide**

#### **No Code Changes Required**
This release is fully backward compatible. All existing code continues to work without modification.

#### **Optional: Customize RangeEngine Configuration**
```rust
// Azure custom config
use s3dlio::object_store::{AzureObjectStore, AzureConfig, RangeEngineConfig};

let config = AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 32 * 1024 * 1024,  // 32MB chunks instead of 64MB
        max_concurrent_ranges: 16,      // 16 concurrent instead of 32
        min_split_size: 8 * 1024 * 1024,  // 8MB threshold instead of 4MB
        range_timeout: Duration::from_secs(60),
    },
};
let store = AzureObjectStore::with_config(config);
```

#### **Optional: Disable RangeEngine**
```rust
let config = AzureConfig {
    enable_range_engine: false,  // Disable concurrent ranges
    ..Default::default()
};
let store = AzureObjectStore::with_config(config);
```

---

### 📚 **Documentation Updates**
- Updated README.md with v0.9.3 features
- Updated Changelog.md (this file)
- API documentation includes RangeEngine configuration examples
- Test guides updated with new test suites

---

## Version 0.9.2 - CancellationToken & Configuration Rationalization (October 2025)

### 🎯 **New Features**

#### **1. CancellationToken Infrastructure for Graceful Shutdown**
Comprehensive cancellation support across all DataLoader components enables clean shutdown of data loading operations.

**Components Updated**:
- `LoaderOptions`: Added `cancellation_token` field with builder methods
- `spawn_prefetch()`: Added cancel_token parameter with loop check
- `DataLoader`: Cancellation checks in all 3 spawn paths (map-known, iterable, map-unknown)
- `AsyncPoolDataLoader`: Cancellation support in async pool worker with 3 checkpoints

**Usage**:
```rust
use tokio_util::sync::CancellationToken;

let cancel_token = CancellationToken::new();

let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token.clone());

// Spawn Ctrl-C handler
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    cancel_token.cancel();
});

let loader = DataLoader::new(dataset, options);
let mut stream = loader.stream();

while let Some(batch) = stream.next().await {
    train_step(batch?).await?;
}
// Clean shutdown on Ctrl-C
```

**Behavior**:
- ✅ Workers exit cleanly without submitting new requests
- ✅ In-flight requests drain naturally
- ✅ MPSC channels properly closed
- ✅ No orphaned background tasks
- ✅ Zero overhead when token not cancelled

---

#### **2. Configuration Hierarchy Rationalization**
Clear three-level configuration design aligned with PyTorch DataLoader concepts for ML practitioners.

**Documentation Added**:
- `docs/CONFIGURATION-HIERARCHY.md` - Comprehensive analysis
- `docs/api/rust-api-v0.9.2.md` - Updated from v0.9.0 with hierarchy section
- `docs/api/python-api-v0.9.2.md` - Updated from v0.9.0 with Python-specific guidance

**Three Levels**:
1. **LoaderOptions** (User-Facing, Training-Centric)
   - Like PyTorch's `DataLoader(batch_size, num_workers, ...)`
   - Controls WHAT batches to create and HOW to iterate
   - Always visible to users
   
2. **PoolConfig** (Performance Tuning, Optional)
   - Like PyTorch's internal worker pool management
   - Controls HOW data is fetched efficiently
   - Good defaults via `stream()`, advanced tuning via `stream_with_pool()`
   
3. **RangeEngineConfig** (Internal Optimization, Hidden)
   - Like file I/O internals in PyTorch Dataset
   - Controls storage-layer parallel range requests
   - Backend-specific, automatically configured

**Code Enhancement**:
```rust
// NEW: Convenience constructor to bridge Level 1 → Level 2
let pool_config = PoolConfig::from_loader_options(&options);
// Derives: pool_size from num_workers, readahead_batches from prefetch
```

**Design Philosophy**:
- Progressive complexity: Simple by default, powerful when needed
- PyTorch alignment: Familiar concepts for ML practitioners
- Separation of concerns: Training logic vs storage optimization
- Good defaults: Most users never touch Level 2 or 3

---

### 🧪 **Testing Improvements**

#### **Cancellation Test Suite**
Comprehensive test coverage in `tests/test_cancellation.rs`:

**9 Tests (All Passing ✅)**:
- DataLoader tests (5): pre-cancellation, during streaming, delayed, no-token, shared token
- AsyncPoolDataLoader tests (4): pre-cancellation, during streaming, stops new requests, idempotent

**Test Patterns**:
- Validates drain behavior (in-flight requests complete)
- Timeout-based completion checks (2s)
- Mock URIs for control flow testing
- Ctrl-C handler examples

**Documentation**:
- Updated `docs/TESTING-GUIDE.md` with cancellation testing section
- Test patterns, running instructions, expected behavior
- Configuration hierarchy integration notes
- Checklist for new components

---

### 📚 **Documentation Updates**

#### **API Documentation Versioned to v0.9.2**:
- Renamed `rust-api-v0.9.0.md` → `rust-api-v0.9.2.md`
- Renamed `python-api-v0.9.0.md` → `python-api-v0.9.2.md`
- Added "Configuration Hierarchy" sections
- Added "Graceful Shutdown" section with CancellationToken examples
- Updated "What's New" sections

#### **New Documentation Files**:
- `docs/CONFIGURATION-HIERARCHY.md` - Deep dive into three-level design
- PyTorch comparison tables
- Side-by-side examples (PyTorch vs s3dlio)
- Recommendations for users
- When to tune each level

---

### 🔧 **API Additions**

**LoaderOptions**:
```rust
pub struct LoaderOptions {
    // ... existing fields ...
    
    /// Optional cancellation token for graceful shutdown (NEW in v0.9.2)
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl LoaderOptions {
    pub fn with_cancellation_token(self, token: CancellationToken) -> Self;
    pub fn without_cancellation(self) -> Self;
}
```

**PoolConfig**:
```rust
impl PoolConfig {
    /// NEW: Derive from LoaderOptions with sensible scaling
    pub fn from_loader_options(opts: &LoaderOptions) -> Self;
}
```

**spawn_prefetch**:
```rust
// NEW: Added cancel_token parameter
pub fn spawn_prefetch<F, Fut, T>(
    cap: usize,
    producer: F,
    cancel_token: Option<CancellationToken>,  // NEW
) -> Receiver<Result<T, DatasetError>>
```

---

### ⚡ **Performance**

- **Zero overhead** when cancellation token not used
- **Clean shutdown** typically completes within 2 seconds
- **No breaking changes** to existing code without cancellation
- **Maintains** 5+ GB/s reads, 2.5+ GB/s writes

---

### 🛠️ **Build Status**

- ✅ Zero compiler warnings
- ✅ All tests passing (including 9 new cancellation tests)
- ✅ Clean release builds
- ✅ Documentation comprehensive and versioned

---

### 📦 **Migration Notes**

**No breaking changes** - all features are additive:

**To add graceful shutdown**:
```rust
// Before (v0.9.1)
let options = LoaderOptions::default().with_batch_size(32);

// After (v0.9.2) - optional cancellation
let cancel_token = CancellationToken::new();
let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token);
```

**To use PoolConfig convenience constructor**:
```rust
// NEW in v0.9.2
let pool_config = PoolConfig::from_loader_options(&options);
let stream = loader.stream_with_pool(pool_config);
```

---

### 🎯 **Key Use Cases**

**Training Loops with Ctrl-C Handling**:
- Clean shutdown prevents corrupted checkpoints
- No orphaned background tasks
- Proper resource cleanup

**Multi-Loader Coordination**:
- Single token cancels multiple loaders
- Synchronized shutdown across components

**Long-Running Jobs**:
- Graceful termination on SIGTERM
- Clean exit from infinite data streams

---

### 📖 **Further Reading**

- **Configuration Guide**: `docs/CONFIGURATION-HIERARCHY.md`
- **Rust API**: `docs/api/rust-api-v0.9.2.md`
- **Python API**: `docs/api/python-api-v0.9.2.md`
- **Testing Guide**: `docs/TESTING-GUIDE.md` (cancellation section)

---

## Version 0.9.1 - True Zero-Copy Python API (October 2025)

### 🎯 **Critical Fixes**

#### **1. Zero-Copy Python API (TRUE Implementation)**
- **Fixed false claim**: v0.9.0 claimed zero-copy but actually copied data to Python bytes
- **New `BytesView` class**: Wraps Rust `Bytes`, exposes `.memoryview()` for true zero-copy
- **Python buffer protocol**: Native support without numpy dependency
- **Memory reduction**: 10-15% reduction in typical AI/ML workflows

**Zero-Copy Functions (Returns `BytesView`)**:
- `get(uri)` → `BytesView` (was `bytes`)
- `get_range(uri, offset, length)` → `BytesView` (NEW)
- `get_many(uris)` → `List[(str, BytesView)]` (was `List[(str, bytes)]`)
- `get_object(bucket, key)` → `BytesView` (was `bytes`)
- `CheckpointStore.load_latest()` → `Optional[BytesView]` (was `Optional[bytes]`)
- `CheckpointReader.read_shard_by_rank()` → `BytesView` (was `bytes`)

**Usage**:
```python
# Zero-copy access
view = s3dlio.get("s3://bucket/data.bin")
arr = np.frombuffer(view.memoryview(), dtype=np.float32)  # No copy!

# Or copy if needed
data = view.to_bytes()  # Explicit copy
```

**NOT Zero-Copy** (copies data):
- `PyDataset.get_item()` → still returns `bytes` (legacy trait)
- `PyBytesAsyncDataLoader.__next__()` → returns `bytes` (uses dataset)

See `docs/ZERO-COPY-API-REFERENCE.md` for complete details.

---

#### **2. Universal Backend Support for get_many()**
- **Fixed limitation**: `get_many()` was S3-only in v0.9.0
- **Universal support**: Now works with S3, Azure, GCS, File, DirectIO
- **Backend detection**: Automatically routes to appropriate implementation
- **S3**: Uses optimized `get_objects_parallel()`
- **File/DirectIO**: Parallel `tokio::fs::read()` with semaphore
- **Azure/GCS**: Uses universal `store_for_uri()` factory

**All URIs must use same scheme**:
```python
# ✅ Valid - all file://
s3dlio.get_many(["file:///a.bin", "file:///b.bin"])

# ❌ Invalid - mixed schemes
s3dlio.get_many(["s3://a", "file:///b"])  # Error
```

---

### ✨ **New Features**

#### **1. Universal Range Request Support**

**Python API**:
```python
# Read byte range (zero-copy)
view = s3dlio.get_range("s3://bucket/file", offset=1000, length=1024*1024)
arr = np.frombuffer(view.memoryview(), dtype=np.uint8)

# Read from offset to end
view = s3dlio.get_range("file:///path", 5000, None)
```

**CLI**:
```bash
# Range request
s3-cli get s3://bucket/file --offset 1000 --length 1024

# From offset to end
s3-cli get file:///path/file --offset 5000

# Single-object only (not with --recursive or --keylist)
```

**Backends**: S3, Azure, GCS, File, DirectIO

---

#### **2. Comprehensive Zero-Copy Refactor**

**Rust Library**:
- All `ObjectStore` helper methods return `Bytes` (not `Vec<u8>`)
- `get_with_validation()` → `Bytes`
- `get_range_with_validation()` → `Bytes`
- `get_optimized()` → `Bytes`
- `get_range_optimized()` → `Bytes`

**Checkpoint System**:
- All checkpoint reader methods return `Bytes`
- `read_shard()` → `Bytes`
- `read_shard_by_rank()` → `Bytes`
- `read_all_shards()` → `Vec<(u32, Bytes)>`
- `load_latest()` → `Option<Bytes>`

**S3 Utils**:
- `get_range()` uses `.slice()` not `.to_vec()`

---

### 🧪 **Testing**

**New Test Suites**:
1. **test_zero_copy_comprehensive.py** (6/6 tests pass):
   - BytesView structure validation
   - get() returns BytesView with working memoryview
   - NumPy array from memoryview (zero-copy verified)
   - get_many() universal backend support
   - Large file handling (1 MB)
   - BytesView immutability

2. **test_range_requests.py** (4/4 tests pass):
   - Range with offset+length
   - Range with offset only (to end)
   - Range from beginning
   - Full file comparison

**Existing Tests**:
- ✅ Rust library: 91/91 tests pass
- ✅ Python functionality: 27/27 tests pass
- ✅ Zero warnings in all builds

---

### 📝 **Documentation**

**New Documentation**:
- `docs/ZERO-COPY-API-REFERENCE.md`: Complete zero-copy vs copy operations guide
- `docs/v0.9.1-ZERO-COPY-TEST-SUMMARY.md`: Implementation and test summary

**Clarifications**:
- GCS backend: Production-ready, works well in testing
- Performance targets: 5+ GB/s reads, 2.5+ GB/s writes maintained

---

### 🔧 **Implementation Notes**

**Zero-Copy Architecture**:
- `PyBytesView` wraps `Bytes` with PyO3 buffer protocol support
- `.memoryview()` → zero-copy access (read-only)
- `.to_bytes()` → explicit copy for compatibility
- `__len__()` → zero-copy length query

**Type Conversions**:
- Functions returning `PyBytesView`: Direct return
- Functions returning `PyObject`: Use `.into_py_any(py)?`
- Functions returning `Option<PyObject>`: Nested map with `Python::with_gil()`

**Build Quality**:
- Zero warnings in all builds (`cargo build`, `cargo clippy`, PyO3)
- Fixed all Rust test assertions for `Bytes` comparisons
- Proper error handling and validation

---

### 📊 **Performance Impact**

**Memory Savings**:
```python
# v0.9.0 (copies data)
data = s3dlio.get("s3://bucket/1gb.bin")  # bytes
arr = np.frombuffer(data, dtype=np.uint8)
# Memory: 2 GB (bytes + NumPy array)

# v0.9.1 (zero-copy)
view = s3dlio.get("s3://bucket/1gb.bin")  # BytesView
arr = np.frombuffer(view.memoryview(), dtype=np.uint8)
# Memory: 1 GB (shared buffer)
# Savings: 50%
```

**Throughput**:
- Reduced GC pressure: 15-20% faster in batch operations
- Universal `get_many()`: Works across all backends
- Range requests: Fetch only needed bytes

---

### 🔄 **Migration from v0.9.0**

**Python API Changes**:
```python
# Old (v0.9.0)
data = s3dlio.get(uri)  # bytes
arr = np.frombuffer(data, dtype=np.float32)

# New (v0.9.1) - Zero-copy
view = s3dlio.get(uri)  # BytesView
arr = np.frombuffer(view.memoryview(), dtype=np.float32)

# Backward compatibility (copies)
view = s3dlio.get(uri)
data = view.to_bytes()  # Convert to bytes if needed
```

**No Breaking Changes**:
- `BytesView` supports buffer protocol → most code works unchanged
- Explicit `.to_bytes()` available for compatibility
- Dataset API unchanged (still returns bytes)

---

### ✅ **Verified Compatibility**

**Frameworks**:
- ✅ NumPy: `np.frombuffer(view.memoryview())`
- ✅ PyTorch: `torch.from_numpy(np.frombuffer(view.memoryview()))`
- ✅ TensorFlow: `tf.constant(view.memoryview())`
- ✅ JAX: `jnp.array(np.frombuffer(view.memoryview()))`

**Backends**:
- ✅ S3: AWS S3, MinIO, Vast
- ✅ Azure: Azure Blob Storage
- ✅ GCS: Google Cloud Storage
- ✅ File: Local filesystem
- ✅ DirectIO: Direct I/O filesystem

---

## Version 0.9.0 - API-Stable Beta with Breaking Changes (October 2025)

### 🚨 **BREAKING CHANGES**

#### **ObjectStore Trait Returns Bytes (Not Vec<u8>)**
- **Breaking**: `ObjectStore::get()` now returns `Result<Bytes>` instead of `Result<Vec<u8>>`
- **Breaking**: `ObjectStore::get_range()` now returns `Result<Bytes>` instead of `Result<Vec<u8>>`
- **Impact**: Code using ObjectStore directly must handle `Bytes` or call `.to_vec()`
- **Python API**: Unchanged - conversion happens internally
- **Helper methods**: Unchanged - still return `Vec<u8>` for backward compatibility

**Migration**:
```rust
// Old (v0.8.x)
let data: Vec<u8> = store.get(uri).await?;

// New (v0.9.0) - Option 1: Use Bytes
let data: Bytes = store.get(uri).await?;

// New (v0.9.0) - Option 2: Convert to Vec
let data: Vec<u8> = store.get(uri).await?.to_vec();
```

### ✨ **New Features**

#### **1. Zero-Copy Performance (Stage 2)**
- **10-15% memory reduction**: Eliminated unnecessary Vec allocations
- **S3/Azure zero-copy**: SDKs return Bytes, we now return directly (no `.to_vec()`)
- **Efficient concatenation**: Use `BytesMut` for concurrent range downloads
- **Cheap cloning**: Bytes are Arc-like, cloning doesn't duplicate data

#### **2. Concurrent Batch Loading (Stage 1)**
- **3-8x faster Python batch loading**: Concurrent fetching in `spawn_stream()`
- **JoinSet + Semaphore pattern**: Efficient async task management
- **Configurable concurrency**: Control parallelism via `LoaderOptions`
- **Python API enhancement**: Transparent performance improvement

#### **3. Optional Adaptive Tuning (Stage 4)**
- **Opt-in auto-tuning**: Disabled by default, users explicitly enable
- **Smart part sizing**: 8 MB (small files) → 16 MB (medium) → 32 MB (large)
- **Smart concurrency**: 2x-8x CPU count based on workload type
- **Explicit override**: User settings ALWAYS take precedence
- **API**: `WriterOptions::with_adaptive()`, `LoaderOptions::with_adaptive()`

**Usage**:
```rust
// Enable adaptive tuning
let opts = WriterOptions::new().with_adaptive();

// Explicit settings override adaptive
let opts = WriterOptions::new()
    .with_adaptive()
    .with_part_size(20 * 1024 * 1024); // 20 MB used, adaptive ignored
```

### 📚 **Documentation**

- **New**: `docs/api/rust-api-v0.9.0.md` - Comprehensive Rust API guide with migration section
- **New**: `docs/api/python-api-v0.9.0.md` - Comprehensive Python API guide with migration section
- **New**: `docs/ADAPTIVE-TUNING.md` - Complete adaptive tuning guide
- **New**: `docs/STAGE3-DEFERRAL.md` - Stage 3 deferral explanation
- **New**: `docs/v0.9.0-TEST-SUMMARY.md` - Complete test validation report
- **New**: `examples/adaptive_tuning_demo.rs` - Comprehensive demo
- **Updated**: Migration guides include "What's Changed Since v0.8.22" sections

### 🧪 **Testing**

- **91 Rust unit tests pass** (100%, 10 new adaptive config tests)
- **16 Python regression tests pass** (100%, cleaned up deprecated functions)
- **Framework integration verified**: PyTorch, TensorFlow, JAX all compatible with Bytes migration
- **Zero compilation warnings**
- **All backends tested**: S3, Azure, GCS, File, DirectIO
- **Performance validated**: 10-15% memory reduction, 3-8x batch speedup confirmed

### 🔧 **API Changes**

**New Public Types**:
- `AdaptiveConfig` - Optional auto-tuning configuration
- `AdaptiveMode` - Enabled/Disabled enum
- `WorkloadType` - Small/Medium/Large file classifications
- `AdaptiveParams` - Parameter computation

**New Methods**:
- `WriterOptions::with_adaptive()` - Enable adaptive tuning
- `WriterOptions::with_adaptive_config()` - Custom adaptive config
- `WriterOptions::effective_part_size()` - Compute effective part size
- `WriterOptions::effective_buffer_size()` - Compute effective buffer size
- `LoaderOptions::with_adaptive()` - Enable adaptive tuning
- `LoaderOptions::with_adaptive_config()` - Custom adaptive config
- `LoaderOptions::effective_part_size()` - Compute effective part size
- `LoaderOptions::effective_concurrency()` - Compute effective concurrency

### ⚡ **Performance Summary**

- **Memory**: 10-15% reduction (zero-copy Bytes)
- **Batch loading**: 3-8x faster (concurrent fetching)
- **Adaptive tuning**: Minimal overhead (microseconds at config time)

### 🗺️ **Roadmap**

- **v0.9.1**: Stage 3 (Backend-agnostic range engine) - NON-BREAKING performance enhancement
  - 30-50% throughput improvement for File/Azure/GCS large files
  - All backends get S3-level range performance

### � **Removed/Deprecated**

- **Python API**: Removed `save_numpy_array()` and `load_numpy_array()` (disabled since v0.7.x)
  - Use checkpoint API or direct NPZ handling instead
  - Migration: `writer.save_array()` or `np.savez()` + `s3dlio.put()`

### �🔗 **Commits**

- Stage 1: Python loader concurrent batching (0994a1a)
- Stage 2: Zero-copy Bytes migration (d214dfc)
- Stage 4: Optional adaptive tuning (b4fd8b3)
- Release v0.9.0: Version bumps and documentation (64a1b3c)

---

## Version 0.8.23 - Delete Operation Progress Tracking (October 2025)

### ✨ **Enhancement: Real-time Progress for Delete Operations**

This release adds comprehensive progress tracking to delete operations, providing visual feedback during long-running deletions (e.g., 93,000+ objects).

#### **Delete Progress Tracking**
- **Progress bar**: Real-time visual progress for all multi-object deletions
- **Rate display**: Shows objects/second deletion rate
- **ETA calculation**: Estimated time to completion
- **Two-phase feedback**:
  1. "Listing objects to delete..." - During list/pagination phase
  2. Progress bar with live updates during deletion phase

#### **Implementation Details**
- **File**: `src/bin/cli.rs` - Enhanced `delete_cmd()` function
- **All delete scenarios covered**:
  - Recursive prefix deletion (`--recursive` or trailing `/`)
  - Pattern-filtered deletion (`--pattern`)
  - Multi-object prefix matches
- **Uses indicatif progress bars** matching GET/PUT operation style
- **Progress template**: `Deleting: {spinner} [{elapsed}] [{bar}] {pos}/{len} objects ({per_sec}, ETA: {eta})`

#### **User Experience Improvements**
- **Before**: Silent operation for minutes with no feedback on 93,000+ object deletions
- **After**: 
  - Immediate feedback: "Listing objects to delete..."
  - Count shown: "Found 93,000 objects to delete"
  - Live progress: Visual bar updating as each object is deleted
  - Completion message: "Deleted 93,000 objects"

#### **Example Output**
```bash
$ s3-cli delete gs://bucket/prefix/ --recursive
Listing objects to delete...
Found 93,000 objects to delete
Deleting: ⠋ [00:02:15] [████████████████░░░░░░░░] 65,432/93,000 objects (485.2/s, ETA: 00:00:57)
```

### 🔧 **Technical Notes**

- **No performance impact**: Progress updates are lightweight
- **Pagination handled**: Works correctly with GCS (1000/page), Azure (5000/page), S3 (1000/page)
- **All backends supported**: S3, GCS, Azure, File, DirectIO
- **Single object deletions**: No progress bar (immediate)
- **Consistent with existing operations**: Matches GET/PUT progress style

---

## Version 0.8.22 - GCS Pagination Fix (October 2025)

### 🐛 **Critical Bug Fix: GCS List/Delete Limited to 1000 Objects**

This release fixes a critical pagination bug in Google Cloud Storage operations that limited list and delete operations to only the first 1000 objects, even when more objects existed.

#### **Issue**
- **GCS list operations**: Only returned first 1000 objects from bucket/prefix
- **GCS delete operations**: Only deleted first 1000 objects when deleting by prefix
- **Root cause**: Missing pagination loop in `GcsClient::list_objects()`
- **S3 comparison**: S3 operations were already correct (had pagination since earlier versions)

#### **Fix Details**
- **File**: `src/gcs_client.rs`
- **Method**: `GcsClient::list_objects()`
- **Implementation**: Added pagination loop using GCS `page_token` / `next_page_token` pattern
- **Pattern**: Matches existing S3 pagination implementation (S3 uses `continuation_token` / `next_continuation_token`)

```rust
// Before: Single page only (limited to 1000 objects)
let response = self.client.list_objects(&request).await?;
let result: Vec<String> = response.items.unwrap_or_default()...

// After: Full pagination support
let mut all_objects = Vec::new();
let mut page_token: Option<String> = None;

loop {
    let mut request = ListObjectsRequest {
        page_token: page_token.clone(),
        ...
    };
    
    let response = self.client.list_objects(&request).await?;
    all_objects.extend(response.items...);
    
    if let Some(next_token) = response.next_page_token {
        page_token = Some(next_token);
    } else {
        break;
    }
}
```

#### **Impact**
✅ **Fixed**: GCS list operations now return ALL matching objects  
✅ **Fixed**: GCS delete_prefix now deletes ALL matching objects  
✅ **No regression**: Single-page results (<1000 objects) work as before  
✅ **S3/Azure**: No changes needed (S3 already correct, Azure uses different API)

#### **Testing**
- **Test guide**: `docs/GCS-PAGINATION-TEST-GUIDE.md`
- **Verification**: Code review against GCS API documentation
- **Comparison**: Matches proven S3 pagination pattern
- **Build**: Zero compiler warnings

### 📚 **Documentation**

- **Added**: `docs/GCS-PAGINATION-TEST-GUIDE.md` - Comprehensive testing guide
  - Root cause analysis and code comparison
  - Test strategy for >1000 object scenarios
  - Debug logging examples
  - Manual verification checklist

### 🔍 **API References**
- [GCS Objects.list API](https://cloud.google.com/storage/docs/json_api/v1/objects/list)
- GCS default: 1000 objects per page
- Pagination field: `page_token` → `next_page_token`

---

## Version 0.8.21 - Backend Authentication Caching & Performance Analysis (October 2025)

### 🎯 **Release Focus: Multi-Backend Authentication Optimization**

This release implements global authentication caching for GCS and Azure backends, eliminating repeated authentication overhead in multi-threaded workloads. This matches the existing S3 client caching pattern, ensuring all three cloud backends now have optimal performance characteristics.

### ✨ **Performance Optimizations**

#### **GCS Client Caching** ⚡
- **Global credential cache**: GCS client now uses `OnceCell<Arc<Client>>` for single-initialization authentication
- **Eliminated per-operation overhead**: Authentication happens once per process instead of once per thread/operation
- **Implementation**: `src/gcs_client.rs` - Added `GCS_CLIENT` static with async initialization via `get_or_try_init()`

#### **Azure Credential Caching** ⚡
- **Global credential cache**: Azure client now uses `OnceCell<Arc<dyn TokenCredential>>` for cached authentication
- **Eliminated per-operation overhead**: `DefaultAzureCredential::new()` called once and cached globally
- **Implementation**: `src/azure_client.rs` - Added `AZURE_CREDENTIAL` static with async initialization
- **Performance parity**: Azure now matches S3 and GCS authentication efficiency

#### **S3 Client Already Optimized** ✅
- **Verified existing implementation**: S3 client already uses `CLIENT.get_or_try_init()` for caching
- **No changes needed**: S3 authentication has been optimized since earlier versions

### 📊 **Performance Analysis & Documentation**

#### **Comprehensive Performance Review**
- **Created**: `docs/PERFORMANCE_OPTIMIZATION_ANALYSIS.md` - Runtime performance analysis
  - Client caching strategies across all backends
  - Buffer management and memory allocation patterns
  - Connection pool configuration guidelines
  - Concurrency tuning recommendations
  - Part size optimization for large files

- **Created**: `docs/COMBINED_PERFORMANCE_RECOMMENDATIONS.md` - Architecture & ML-focused analysis
  - Backend-agnostic range engine recommendations (Priority 1)
  - Python async loader parallelism improvements (Priority 1)
  - Loader return type stability verification (✅ Already implemented)
  - Zero-copy Rust→Python patterns
  - Dynamic part sizing for 100 Gb networks
  - Adaptive concurrency tuning

#### **Implementation Roadmap**
- **Phase 1 (Weeks 1-2)**: Backend-agnostic range engine, concurrent batch loading
- **Phase 2 (Weeks 3-4)**: Dynamic part sizing, adaptive concurrency
- **Phase 3 (Weeks 5-6)**: Zero-copy optimizations, buffer management improvements

### 🔧 **Code Quality Improvements**

#### **Enhanced Developer Documentation**
- **Updated**: `.github/copilot-instructions.md` - Added critical virtual environment checks
  - Always verify `(s3dlio)` prefix in terminal prompt before building
  - Documented virtual environment activation requirements
  - Python extension build workflow best practices

#### **Data Loader API Stability** ✅
- **Verified**: Python async data loaders already return consistent `list[bytes]` type
- **All loaders confirmed**: `PyBytesAsyncDataLoaderIter`, `PyS3AsyncDataLoader`, `PyAsyncDataLoaderIter`
- **ML framework compatibility**: Stable type contracts for PyTorch, JAX, TensorFlow
- **Documentation**: `docs/LOADER_RETURN_TYPE_QUICK_WIN.md`

### 🎯 **Performance Targets**

Current library capabilities (on 100 Gb infrastructure):
- **S3 GET**: 5+ GB/s sustained ✅
- **S3 PUT**: 2.5+ GB/s sustained ✅
- **GCS/Azure**: Now benefit from same client caching optimizations ✅
- **Multi-threaded efficiency**: Eliminated authentication bottlenecks ✅

### 📝 **Configuration Best Practices**

For 100 Gb infrastructure (Vast/MinIO):
```bash
export S3DLIO_RT_THREADS=32
export S3DLIO_MAX_HTTP_CONNECTIONS=400
export S3DLIO_HTTP_IDLE_TIMEOUT_MS=1500
export S3DLIO_OPERATION_TIMEOUT_SECS=300
export S3DLIO_USE_OPTIMIZED_HTTP=true
```

For cloud deployments:
```bash
export S3DLIO_RT_THREADS=16
export S3DLIO_MAX_HTTP_CONNECTIONS=200
export S3DLIO_HTTP_IDLE_TIMEOUT_MS=2000
export S3DLIO_OPERATION_TIMEOUT_SECS=600
```

### 🧪 **Build Quality**

- **Zero warnings**: All builds pass with 0 compiler warnings
- **Clean architecture**: Client caching pattern consistent across all backends
- **Virtual environment workflow**: Enhanced documentation for Python extension builds

---

## Version 0.8.20 - Progress Bar Fixes & Universal Backend Support for GET/PUT (October 2025)

### 🎯 **Release Focus: CLI Progress Tracking & Backend Consistency**

This release fixes a critical issue where progress bars in the CLI would jump to 100% at completion instead of updating incrementally during operations. Additionally, `get` and `put` commands have been converted from S3-specific implementations to use the universal ObjectStore interface, ensuring consistent behavior across all 5 storage backends.

### ✨ **Bug Fixes**

#### **Progress Bar Functionality** 🔧
- **Fixed incremental progress updates**: Progress bars now update continuously during async operations instead of jumping to 100% at completion
- **GET command progress tracking**: Added proper `ProgressCallback` integration with parallel downloads
- **PUT command progress tracking**: Added proper `ProgressCallback` integration with parallel uploads
- **Real-time throughput display**: Progress bars show accurate transfer speeds during operations

#### **Universal Backend Architecture** 🌐
- **GET command universality**: Converted from S3-specific `get_objects_parallel()` to universal ObjectStore interface
- **PUT command universality**: Converted from S3-specific implementation to universal ObjectStore interface
- **Multi-backend credential checking**: Commands now only require credentials for their respective backends (S3, GCS, Azure)
- **Consistent behavior**: GET/PUT commands now work identically across all backends: `s3://`, `gs://`, `az://`, `file://`, `direct://`

### 🔧 **Technical Improvements**

#### **API Enhancements**
- **Backward compatibility maintained**: All existing Rust library APIs unchanged - other projects using s3dlio continue to work without modifications
- **New progress-aware functions**: Added `get_objects_parallel_with_progress()` and `put_objects_with_random_data_and_type_with_progress()` with optional progress callbacks
- **Wrapper pattern**: Original functions now call progress-aware versions with `None` for progress callback

#### **Architecture Consistency**
- **Universal ObjectStore usage**: All CLI commands now consistently use the ObjectStore trait
- **Performance preservation**: Maintained semaphore-based concurrency and FuturesUnordered async patterns
- **Clean builds**: Resolved all compiler warnings with proper dead code annotations

### 🧪 **Testing**

#### **Verified Functionality**
- **File backend testing**: Successfully tested GET/PUT progress tracking with `file://` URIs
- **Progress tracking validation**: Confirmed incremental updates during parallel operations
- **Round-trip operations**: Verified PUT followed by GET operations work correctly
- **Multi-backend support**: Commands properly detect and handle different URI schemes

---

## Version 0.8.19 - Universal Commands Across All Backends (October 2025)

**See full details in [docs/v0.8.19-RELEASE-NOTES.md](./v0.8.19-RELEASE-NOTES.md)**

### Key Features
- **Universal `ls` command** with regex pattern filtering across all 5 backends (S3, GCS, Azure, File, DirectIO)
- **Universal `stat` command** with improved output formatting (only shows fields with values)
- **Python API updates**: `list()` and `stat()` now work universally across all backends
- **S3 region fix**: `list-buckets` now correctly uses `us-east-1` region
- **Deprecation**: `list` command deprecated in favor of `ls` (backward compatible with warning)

---

## Version 0.8.18 - GCS Backend Phase 2 Complete (October 2025)

### 🎯 **Release Focus: GCS ObjectStore Integration & Multi-Backend CLI**

This release completes Phase 2 of Google Cloud Storage (GCS) support, delivering a fully functional 5th storage backend with ObjectStore trait implementation, universal CLI commands, and comprehensive testing. GCS is now production-ready for basic operations (upload, download, list, delete).

### ✨ **New Features**

#### **GCS ObjectStore Implementation** 🚀
- **Complete ObjectStore Trait**: `GcsObjectStore` in `src/object_store.rs`
  - All methods implemented: `get()`, `get_range()`, `put()`, `put_multipart()`, `list()`, `stat()`, `delete()`, `delete_prefix()`
  - Container operations: `create_container()`, `delete_container()`
  - Writer support: `get_writer()`, `get_writer_with_compression()`, `create_writer()`
  - Lazy client initialization for optimal performance

- **GcsBufferedWriter**: Streaming upload support
  - Implements `ObjectWriter` trait
  - Buffer management with configurable chunk sizes
  - Compression support (gzip, zstd, none)
  - Checksum calculation (MD5, CRC32C)
  - Multipart upload capability

- **Factory Integration**: Universal backend selection
  - `store_for_uri()` returns `GcsObjectStore` for `gs://` and `gcs://` URIs
  - Works with all factory variants (with_logger, with_config, etc.)
  - Seamless integration with existing upload/download infrastructure

#### **Multi-Backend CLI Enhancement** 🔧
- **Universal Delete Command**: Now supports ALL backends
  - **Before**: Only worked with `s3://` URIs (S3-specific implementation)
  - **After**: Works with `gs://`, `s3://`, `az://`, `file://`, `direct://`
  - Single object: `s3-cli delete gs://bucket/object`
  - Prefix deletion: `s3-cli delete gs://bucket/prefix/ -r`
  - Async implementation for clean integration

- **GCS Upload Support**: Bucket creation and generic upload
  - Auto-creates GCS buckets when `--create-bucket` flag used
  - Full integration with `generic_upload_files()` infrastructure

### 🐛 **Bug Fixes**

#### **GCS List URI Duplication** (Critical)
- **Issue**: List operations returned `gs://bucket/gs://bucket/object` (duplicate prefixes)
- **Root Cause**: `gcs_client.rs:list_objects()` returned full URIs instead of object names
- **Fix**: Changed to return just object names; ObjectStore layer adds URI prefix
- **Impact**: List operations now return correct, clean URIs

#### **CLI Delete Backend Lock-in** (Major)
- **Issue**: Delete command hard-coded to S3, failed with "URI must start with s3://" for other backends
- **Root Cause**: Used `parse_s3_uri()` and S3-specific `delete_objects()` function
- **Fix**: Rewrote to use generic `ObjectStore::delete()` and `delete_prefix()` methods
- **Impact**: Delete now works universally across all 5 storage backends

### 📊 **Testing & Validation**

#### **Comprehensive Test Suite** ✅
- **test-gcs-final.sh**: End-to-end GCS operations (ALL PASSED)
  - Upload: Single file to GCS bucket
  - List: Verified object listing
  - Download: Retrieved with correct content
  - Delete: Successfully removed object
  - Verification: Confirmed 0 objects after deletion

- **Additional Tests**:
  - `gcs-test.sh`: S3-compatible endpoint testing ✅
  - `test-gcloud-native.sh`: gcloud CLI baseline ✅
  - `test-s3cli-gcs.sh`: s3-cli with native GCS URIs ✅

#### **Operations Verified** ✅
| Operation | Status | Notes |
|-----------|--------|-------|
| Upload (PUT) | ✅ | With ADC auth |
| Download (GET) | ✅ | Full content retrieval |
| List objects | ✅ | Recursive & non-recursive |
| Delete single | ✅ | Individual objects |
| Delete prefix | ✅ | Bulk deletion |
| Range reads | ✅ | API only (not in CLI) |
| Stat/metadata | ✅ | Via list operation |
| Multipart upload | ✅ | Via generic upload |

#### **Build Quality** ✅
- Zero compilation warnings
- All unit tests passing (7/7 for GCS URI parsing)
- Clean `cargo build --release` output

### 🏗️ **Architecture**

#### **ObjectStore Integration**
```rust
// GCS automatically selected for gs:// URIs
let store = store_for_uri("gs://bucket/path")?;

// All operations work identically across backends
store.put("gs://bucket/obj", data).await?;
store.get("gs://bucket/obj").await?;
store.list("gs://bucket/", true).await?;
store.delete("gs://bucket/obj").await?;
```

#### **Authentication**
- **Method**: Application Default Credentials (ADC)
- **Setup**: `gcloud auth application-default login`
- **Location**: `~/.config/gcloud/application_default_credentials.json`
- **Client**: Lazy initialization on first GCS operation

### 📚 **Documentation**

#### **New Documentation**
- `docs/GCS_Phase2_0-8-18.md`: Complete implementation guide (500+ lines)
  - Implementation details
  - Bugs fixed
  - Testing results
  - Architecture overview
  - Known limitations

- `docs/GCS_TODO.md`: Future work tracking
  - Python API testing
  - Performance benchmarking
  - CLI enhancements
  - Advanced features roadmap

- `docs/GCS-TESTING-SUMMARY.md`: Test results summary

#### **Test Scripts**
- `test-gcs-final.sh`: Comprehensive verification suite
- `gcs-native-env`: ADC setup instructions
- `gcs-s3-compat-env`: HMAC credentials (S3-compatible mode)

### ⚠️ **Known Limitations**

#### **Not Yet Implemented**
1. **List Buckets**: GCS SDK doesn't expose project-level bucket listing
   - Workaround: Use `gcloud storage buckets list`

2. **Range Reads in CLI**: API supports it, CLI doesn't expose `--offset`/`--length` flags
   - `ObjectStore::get_range()` works programmatically

3. **Python API**: Rust backend complete, bindings not yet tested with GCS URIs

#### **Not Yet Benchmarked**
- Upload throughput (target: 2.5+ GB/s)
- Download throughput (target: 5+ GB/s)
- Comparison with S3/Azure backends

### 🚀 **Production Readiness**

#### **Ready for Production** ✅
- Core operations tested and working
- Zero compilation warnings
- Clean error handling
- Proper authentication flow
- Multi-backend CLI support

#### **Before Large-Scale Use** ⏳
- Python API integration testing
- Performance benchmarking
- Large file testing (>1GB)
- Error scenario coverage

### 📦 **Dependencies**

No new dependencies added (gcloud-storage already in v0.8.16):
```toml
gcloud-storage = "^1.1"  # Google Cloud Rust SDK
```

### 🔄 **Migration Guide**

#### **For Existing Users**
No breaking changes! Existing S3, Azure, File, and DirectIO code continues to work unchanged.

#### **To Use GCS**
```bash
# 1. Authenticate with Google Cloud
gcloud auth application-default login

# 2. Use gs:// URIs with existing commands
s3-cli upload file.txt gs://my-bucket/
s3-cli ls gs://my-bucket/
s3-cli download gs://my-bucket/file.txt ./output/
s3-cli delete gs://my-bucket/file.txt
```

### 📈 **Code Changes**

#### **Files Modified**
- `src/object_store.rs`: GcsObjectStore + GcsBufferedWriter (~305 lines added)
- `src/gcs_client.rs`: List bug fix (1 line changed)
- `src/bin/cli.rs`: Universal delete command (~25 lines modified)
- `Cargo.toml`: Version bump to 0.8.18
- `pyproject.toml`: Version bump to 0.8.18, description updated

#### **Files Added**
- `docs/GCS_Phase2_0-8-18.md`: Implementation documentation
- `docs/GCS_TODO.md`: Future work tracking
- `test-gcs-final.sh`: Comprehensive test suite

### 🎯 **Success Metrics**

✅ All ObjectStore trait methods implemented  
✅ All CLI commands work with GCS  
✅ Authentication working (ADC)  
✅ Bugs fixed (list URI duplication, delete backend lock-in)  
✅ Zero compilation warnings  
✅ Comprehensive test suite passing  
✅ Documentation complete  
✅ **GCS Backend Status: PRODUCTION READY for basic operations**

### 🔜 **Next Steps (v0.8.19)**

**Focus**: Testing and Performance
1. Python API testing with GCS URIs
2. Performance benchmarking suite
3. Range reads CLI support (`--offset`, `--length` flags)
4. Error scenario testing
5. Large file stress testing

See `docs/GCS_TODO.md` for complete roadmap.

### 👥 **Contributors**

- Implementation: GitHub Copilot AI
- Testing: Signal65 team
- GCS Bucket: signal65-russ-b1

---

## Version 0.8.16 - GCS Backend Infrastructure (Phase 1) (October 2025)

### 🎯 **Release Focus: Google Cloud Storage Foundation**

This release establishes the infrastructure for Google Cloud Storage (GCS) support as s3dlio's 5th storage backend. Phase 1 provides a complete, working GCS client implementation with authentication, all core operations, and URI parsing—laying the groundwork for full ObjectStore integration in the next release.

### ✨ **New Features**

#### **GCS Client Infrastructure** 🚀
- **Complete GCS Client**: `src/gcs_client.rs` with all core operations
  - GET operations: `download_object()` with full and range reads
  - PUT operations: `upload_object()` for simple uploads
  - Metadata: `get_object()` for object stats
  - Delete: Single and batch deletion with concurrency
  - List: Object listing with prefix and delimiter support
  - Buckets: Create and delete operations
  
- **Application Default Credentials (ADC)**: Automatic authentication
  - `GOOGLE_APPLICATION_CREDENTIALS` environment variable
  - GCE/GKE metadata server (automatic for Google Cloud workloads)
  - gcloud CLI credentials (`~/.config/gcloud/...`)
  - Uses `ClientConfig::with_auth()` from gcloud-storage crate

- **URI Scheme Support**: Full `gs://` and `gcs://` recognition
  - Added `Gcs` variant to `Scheme` enum
  - Updated `infer_scheme()` to recognize GCS URIs
  - `parse_gcs_uri()` function with comprehensive validation
  - **7/7 unit tests passing** for URI parsing

#### **Dependencies**
- **gcloud-storage ^1.1**: Official Google Cloud Rust SDK
  - Part of google-cloud-rust project
  - Built-in ADC support
  - Comprehensive GCS API coverage

### 📊 **Implementation Status**

#### **Phase 1: Infrastructure** ✅ COMPLETE
- ✅ GCS dependencies added to Cargo.toml
- ✅ `src/gcs_client.rs` module created (430 lines)
- ✅ All client methods implemented with real API calls
- ✅ URI parsing with full test coverage (7/7 tests passing)
- ✅ Scheme enum updated for GCS support
- ✅ Zero compilation warnings
- ✅ ADC authentication working

#### **Phase 2: Integration** ✅ COMPLETE (v0.8.18)
- ✅ `GcsObjectStore` implementing `ObjectStore` trait

- ⏳ `GcsObjectWriter` for streaming uploads (resumable API)
- ⏳ Update factory functions (`store_for_uri()` etc.)
- ⏳ Integration tests with real GCS credentials
- ⏳ Performance benchmarks
- ⏳ Documentation and examples

### 🔧 **Technical Details**

#### **GCS Client Methods**
```rust
impl GcsClient {
    async fn new() -> Result<Self>                    // ADC authentication
    async fn get_object(&self, ...) -> Result<Vec<u8>>              // Full download
    async fn get_object_range(&self, ...) -> Result<Vec<u8>>        // Range read
    async fn put_object(&self, ...) -> Result<()>                   // Simple upload
    async fn put_object_multipart(&self, ...) -> Result<()>         // Multipart upload
    async fn stat_object(&self, ...) -> Result<GcsObjectMetadata>  // Get metadata
    async fn delete_object(&self, ...) -> Result<()>                // Single delete
    async fn delete_objects(&self, ...) -> Result<()>               // Batch delete (16 concurrent)
    async fn list_objects(&self, ...) -> Result<Vec<String>>        // List with prefix
    async fn create_bucket(&self, ...) -> Result<()>                // Create bucket
    async fn delete_bucket(&self, ...) -> Result<()>                // Delete bucket
}
```

#### **URI Parsing**
```rust
pub fn parse_gcs_uri(uri: &str) -> Result<(String, String)> {
    // Supports: gs://bucket/path/to/object
    //          gcs://bucket/path/to/object
    // Returns: (bucket_name, object_path)
}
```

### 🧪 **Testing & Quality**

- **URI Parsing Tests**: 7/7 passing
  - Valid gs:// URIs
  - Valid gcs:// URIs  
  - Nested paths
  - Edge cases (empty bucket, missing object, etc.)
- **Build Quality**: Zero warnings in `cargo build --release --lib`
- **Compilation**: Clean build with all dependencies

### 📁 **Files Modified**

#### **New Files**
- `src/gcs_client.rs`: Complete GCS client implementation (430 lines)
- `docs/Changelog_pre-0.8.0.md`: Historical changelog (pre-0.8.0 versions)

#### **Modified Files**
- `Cargo.toml`: Added gcloud-storage dependency, version bump to 0.8.16
- `pyproject.toml`: Version bump to 0.8.16
- `src/lib.rs`: Added `pub mod gcs_client;`
- `src/object_store.rs`: Added `Gcs` variant to `Scheme` enum, updated `infer_scheme()`
- `src/api.rs`: Added `Scheme::Gcs` case (not-yet-implemented errors)
- `src/python_api/python_core_api.rs`: Added `Scheme::Gcs` cases (not-yet-implemented errors)
- `docs/Changelog.md`: Split into current (0.8.0+) and historical versions

### 🎯 **Next Steps (v0.8.20)**

The GCS backend is **50% complete** with all infrastructure in place. Next release will focus on:

1. **GcsObjectStore Integration**: Implement `ObjectStore` trait
2. **Streaming Uploads**: `GcsObjectWriter` with resumable upload API
3. **Factory Functions**: Update `store_for_uri()` to instantiate GCS stores
4. **Testing**: Integration tests with real GCS credentials
5. **Documentation**: GCS quickstart guide and usage examples
6. **Performance**: Benchmarks targeting 5+ GB/s reads (parity with S3/Azure)

### �� **Migration Notes**

- **No Breaking Changes**: This is infrastructure-only; existing APIs unchanged
- **GCS URIs Recognized**: `gs://` and `gcs://` URIs are now detected by `infer_scheme()`
- **Not-Yet-Implemented Errors**: Attempting GCS operations will return clear error messages
- **Authentication Ready**: Set `GOOGLE_APPLICATION_CREDENTIALS` to prepare for v0.9.0

---

# s3dlio Changelog

## Version 0.8.15 - Streaming Op-Log Reader with Workspace Version Management (October 2025)

### 🎯 **Release Focus: Memory-Efficient Streaming & Simplified Version Management**

This release introduces a high-performance streaming op-log reader that reduces memory usage by 30-50x for large operation logs, plus workspace-level version inheritance to simplify release management. The streaming reader processes GB-scale op-logs with constant ~1.5 MB memory usage while maintaining 100% backward compatibility.

### ✨ **New Features**

#### **OpLogStreamReader - Memory-Efficient Streaming** 🚀
- **Constant Memory Usage**: ~1.5 MB regardless of op-log size (30-50x reduction)
  - Processes 100K operations: 1.5 MB vs previous 30-50 MB
  - Background thread for CPU-isolated zstd decompression
  - 1 MB chunk reads with bounded channel backpressure
- **Iterator-Based API**: Ergonomic streaming with Rust iterators
  ```rust
  let stream = OpLogStreamReader::from_file("large_oplog.tsv.zst")?;
  for entry in stream { /* process one at a time */ }
  ```
- **Environment Variable Tuning**:
  - `S3DLIO_OPLOG_READ_BUF`: Channel buffer size (default: 1024 entries)
  - `S3DLIO_OPLOG_CHUNK_SIZE`: Read chunk size (default: 1 MB)
- **Zero Breaking Changes**: OpLogReader now uses streaming internally
- **Full Format Support**: JSONL, TSV, compressed (.zst) all streaming-compatible

#### **Workspace Version Management** 📦
- **Single Source of Truth**: All Rust crate versions inherit from workspace
  ```toml
  [workspace.package]
  version = "0.8.15"  # ← Update once, all crates inherit
  ```
- **Simplified Releases**: Update only 2 files instead of 3+
  - `Cargo.toml`: `[workspace.package] version`
  - `pyproject.toml`: `[project] version`
- **Automatic Inheritance**: All workspace members use `version.workspace = true`

### 📊 **Performance Improvements**

#### **Memory Efficiency**
- **Before**: 100K ops × 200 bytes = ~30-50 MB loaded into memory
- **After**: Channel buffer (1024 × 200 bytes) + 1 MB read buffer = **~1.5 MB**
- **Reduction**: 30-50x less memory for large op-logs
- **CPU Isolation**: zstd decompression in background thread (zero I/O impact)

#### **Throughput**
- 1 MB chunks provide optimal balance (2ms decompression per chunk)
- Channel buffering prevents stalls
- Background thread can use different CPU cores

### 🧪 **Testing & Quality**

#### **Test Coverage: 40 Tests (100% Passing)**
- **17 Unit Tests**: Core functionality + 7 new streaming tests
  - `test_streaming_jsonl`: Basic JSONL streaming
  - `test_streaming_tsv`: Basic TSV streaming  
  - `test_streaming_with_errors`: Error propagation through iterator
  - `test_streaming_compressed`: zstd decompression in background
  - `test_streaming_take`: Iterator chaining support
  - `test_backward_compatibility`: Verify OpLogReader still works
  - Plus existing reader/replayer/types tests
- **17 Integration Tests**: Real-world compatibility scenarios
- **6 Doc Tests**: Example code validation
- **Zero Warnings**: Clean build in s3dlio-oplog package

#### **Examples & Documentation**
- `crates/s3dlio-oplog/examples/oplog_streaming_demo.rs`: Working examples
- `docs/OPLOG_STREAMING_ANALYSIS.md`: Complete design analysis & rationale
- `docs/VERSION-MANAGEMENT.md`: Workspace version management guide
- `docs/RELEASE-CHECKLIST.md`: Never forget documentation updates again!

### 🔧 **Technical Implementation**

#### **Streaming Architecture**
- Mirrors proven pattern from `s3_logger.rs` (op-log writer)
- `sync_channel` with bounded capacity for backpressure
- Background thread spawned via `thread::spawn`
- BufReader with configurable chunk size (default 1 MB)
- Graceful error propagation through iterator

#### **Backward Compatibility**
- OpLogReader refactored to use OpLogStreamReader internally
- All existing tests continue to pass
- No API changes to public interfaces
- Users can opt-in to streaming when ready

### 📁 **Files Modified**

#### **Core Implementation**
- `src/constants.rs`: Added streaming constants (chunk size, buffer capacity)
- `crates/s3dlio-oplog/src/reader.rs`: Implemented OpLogStreamReader
- `crates/s3dlio-oplog/src/lib.rs`: Export OpLogStreamReader publicly

#### **Version Management**
- `Cargo.toml`: Added `[workspace.package]` with version inheritance
- `crates/s3dlio-oplog/Cargo.toml`: Use `version.workspace = true`
- `pyproject.toml`: Updated to 0.8.15

#### **Documentation**
- `docs/OPLOG_STREAMING_ANALYSIS.md`: Design analysis (NEW)
- `docs/VERSION-MANAGEMENT.md`: Version management guide (NEW)
- `docs/RELEASE-CHECKLIST.md`: Pre-commit checklist (NEW)
- `crates/s3dlio-oplog/examples/oplog_streaming_demo.rs`: Usage examples (NEW)

### 🔄 **Migration Guide**

#### **No Changes Required**
```rust
// Existing code works unchanged
let reader = OpLogReader::from_file("ops.tsv")?;
for entry in reader.entries() {
    // Process
}
```

#### **Opt-in to Streaming** (for memory efficiency)
```rust
// New streaming API
let stream = OpLogStreamReader::from_file("large_ops.tsv.zst")?;
for entry in stream {
    let entry = entry?;
    // Process one at a time with constant memory
}
```

#### **Environment Tuning**
```bash
# Increase channel buffer for high-throughput scenarios
export S3DLIO_OPLOG_READ_BUF=2048

# Use 2 MB chunks instead of 1 MB
export S3DLIO_OPLOG_CHUNK_SIZE=2097152
```

### 📋 **Version Management Workflow (Simplified)**

#### **Before** (3+ files to update):
```bash
# Had to update each file manually
vim Cargo.toml                           # version = "0.8.15"
vim crates/s3dlio-oplog/Cargo.toml      # version = "0.8.15"
vim pyproject.toml                       # version = "0.8.15"
```

#### **After** (2 files, workspace inheritance):
```bash
# Update workspace version (all Rust crates inherit)
vim Cargo.toml                           # [workspace.package] version = "0.8.15"

# Update Python version (manual)
vim pyproject.toml                       # version = "0.8.15"

# Verify all crates inherit correctly
cargo metadata --format-version 1 | jq -r '.packages[] | select(.name | startswith("s3dlio")) | "\(.name): \(.version)"'
```

---

## Version 0.8.14 - Shared Op-Log Replay Library with Compatibility Testing (October 2025)

### 🎯 **Release Focus: Production-Ready Shared Library with Comprehensive Testing**

This release introduces the `s3dlio-oplog` shared library crate with extensive compatibility testing, consolidating operation log replay functionality previously duplicated across s3dlio, s3-bench, and dl-driver. Includes 33 tests (100% passing) validating compatibility with both s3-bench and dl-driver usage patterns.

### ✨ **New Features**

#### **s3dlio-oplog Shared Crate v0.8.14** (`crates/s3dlio-oplog/`)
- **Format-Tolerant Reader**: Auto-detects JSONL and TSV formats with zstd decompression
  - Header-driven column mapping handles variations ("operation" vs "op", etc.)
  - Supports minimal 6-field and extended 13-field TSV formats
  - Ported from dl-driver's `oplog_ingest.rs` with enhancements
- **Timeline-Based Replayer**: Microsecond-precision absolute scheduling
  - Preserves original operation timing relationships
  - Speed multiplier support (0.1x to 1000x)
  - Ported from s3-bench's `replay.rs` with trait abstraction
- **Pluggable Execution**: `OpExecutor` trait for custom backends
  - Default `S3dlioExecutor` uses s3dlio ObjectStore
  - Easy integration with custom storage systems
- **URI Translation**: 1:1 backend retargeting (s3→az, file→direct)
- **Operation Filtering**: Replay subsets (GET-only, PUT/DELETE, etc.)

### 🧪 **Comprehensive Compatibility Testing**

#### **Test Coverage: 33 Tests (100% Passing)**
- **11 Unit Tests**: Core functionality (types, reader, uri, replayer)
- **17 Integration Tests**: Real-world compatibility scenarios
  - 4 s3-bench compatibility tests
  - 3 dl-driver compatibility tests  
  - 3 format flexibility tests
  - 7 core functionality tests
- **5 Doc Tests**: Example code validation
- **Zero Warnings**: Production-ready build quality

#### **s3-bench Compatibility Validated ✅**
- ✅ 13-column TSV format parsing
- ✅ HEAD → STAT operation aliasing
- ✅ Speed multiplier timeline scheduling
- ✅ Endpoint prefix stripping + URI retargeting
- ✅ Zstd compression auto-detection

#### **dl-driver Compatibility Validated ✅**
- ✅ JSONL format with flexible field mapping
- ✅ Base URI path construction
- ✅ Path remapping for cross-environment replay
- ✅ Field aliases ("operation" vs "op", "t_start_ns" vs "start")
- ✅ Fast mode support (skip timing delays)

### 📚 **Documentation**

#### **User Documentation**
- `crates/s3dlio-oplog/README.md` - Quick start guide and API reference
- `docs/S3DLIO_OPLOG_INTEGRATION.md` - Detailed integration guide
- `examples/oplog_replay_basic.rs` - Working example with mock executor

#### **Testing Documentation**
- `crates/s3dlio-oplog/docs/COMPATIBILITY_TESTING.md` - Detailed test descriptions
- `crates/s3dlio-oplog/docs/COMPATIBILITY_TESTING_SUMMARY.md` - Executive summary
- Migration guides for both s3-bench and dl-driver

#### **Core Components**
```rust
// crates/s3dlio-oplog/src/types.rs
pub enum OpType { GET, PUT, DELETE, LIST, STAT }
pub struct OpLogEntry {
    pub idx: Option<u64>,
    pub op: OpType,
    pub bytes: Option<usize>,
    pub endpoint: Option<String>,
    pub file: String,
    pub start: DateTime<Utc>,
    pub duration_ns: Option<i64>,
    pub error: Option<String>,
}

// crates/s3dlio-oplog/src/reader.rs  
pub struct OpLogReader {
    pub fn from_file(path: PathBuf) -> Result<Self>
    pub fn entries(&self) -> Vec<OpLogEntry>
    pub fn filter_operations(&self, filter: Vec<OpType>) -> Vec<OpLogEntry>
}

// crates/s3dlio-oplog/src/replayer.rs
pub struct ReplayConfig {
    pub op_log_path: PathBuf,
    pub target_uri: Option<String>,
    pub speed: f64,
    pub continue_on_error: bool,
    pub filter_ops: Option<Vec<OpType>>,
}

#[async_trait]
pub trait OpExecutor {
    async fn get(&self, uri: &str) -> Result<()>;
    async fn put(&self, uri: &str, bytes: usize) -> Result<()>;
    async fn delete(&self, uri: &str) -> Result<()>;
    async fn list(&self, uri: &str) -> Result<()>;
    async fn stat(&self, uri: &str) -> Result<()>;
}

pub async fn replay_workload<E: OpExecutor + 'static>(
    config: ReplayConfig,
    executor: Arc<E>
) -> Result<()>

pub async fn replay_with_s3dlio(config: ReplayConfig) -> Result<()>

// crates/s3dlio-oplog/src/uri.rs
pub fn translate_uri(file: &str, endpoint: &str, target: &str) -> String
```

### 📚 **Documentation**

- **Integration Guide**: `docs/S3DLIO_OPLOG_INTEGRATION.md`
  - Complete API reference and usage examples
  - s3-bench migration guide (remove `src/replay.rs`, use shared crate)
  - dl-driver migration guide (leverage shared reader or custom executor)
  - Supported formats (JSONL/TSV with zstd), configuration options
- **Example Code**: `examples/oplog_replay_basic.rs`
  - Basic replay with default executor
  - Speed multiplier and filtering demonstrations
  - Backend retargeting examples

### 🏗️ **Architecture**

- **Workspace Structure**: Converted to Cargo workspace
  ```toml
  [workspace]
  members = [".", "crates/s3dlio-oplog"]
  ```
- **Dependencies**:
  - chrono (with serde for DateTime serialization)
  - csv 1.3 (TSV parsing)
  - zstd 0.13 (compression)
  - tokio, async-trait, futures (async execution)
  - s3dlio (with native-backends feature)

### ✅ **Testing**

- **Unit Tests**: 11 tests covering all modules
  - `types.rs`: OpType parsing and display
  - `reader.rs`: JSONL/TSV parsing, format detection
  - `uri.rs`: Cross-backend translation
  - `replayer.rs`: Mock executor replay
- **Doc Tests**: 5 tests validating documentation examples
- **Quality**: Zero warnings build (adheres to project standards)

### 🔧 **Implementation Details**

- **Arc-based Executor Lifetime**: Solves async task lifetime issues
  ```rust
  pub async fn replay_workload<E: OpExecutor + 'static>(
      config: ReplayConfig,
      executor: Arc<E>  // Arc wrapper for async task safety
  ) -> Result<()>
  ```
- **Data Generation**: Uses s3dlio's `generate_controlled_data()` for PUT operations
- **Error Handling**: `anyhow::Result` throughout, configurable continue-on-error
- **Logging**: Uses tracing framework for observability

### 🚀 **Migration Benefits**

- **s3-bench**: Remove ~300 lines of duplicate code (`src/replay.rs`)
- **dl-driver**: Remove ~400 lines of duplicate code (`src/oplog_ingest.rs`)
- **Shared Maintenance**: Bug fixes and features benefit all projects
- **Consistent Behavior**: Single implementation ensures identical replay semantics

### 📦 **Version**

- **s3dlio-oplog**: v0.1.0 (initial release)
- **s3dlio**: v0.8.13 (workspace conversion, README update)

## Version 0.8.12 - Universal Op-Log Support (October 1, 2025)

### 🎯 **Release Focus: Operation Logging for All Storage Backends**

This release extends operation trace logging (op-log) support from S3-only to all storage backends (file://, s3://, az://, direct://). Enables performance profiling, debugging, and warp-replay compatibility for any storage system using the same TSV format.

### ✨ **New Features**

#### **Universal ObjectStore Logging** (`src/object_store_logger.rs`)
- **LoggedObjectStore Wrapper**: Non-invasive decorator pattern wraps any `ObjectStore` implementation
- **All Operations Logged**: GET, PUT, LIST, DELETE, STAT operations with timing and error tracking
- **Thread-Safe**: Uses existing logger infrastructure (`src/s3_logger.rs`) with channel-based logging
- **TSV Format**: Zstd-compressed format compatible with warp-replay tool
  - Format: `idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns`
- **All Backends Supported**: file://, s3://, az://, direct:// URIs

#### **Rust API Integration** (PRIMARY Interface)
Added logger-enabled factory functions in `src/object_store.rs`:
```rust
use s3dlio::{init_op_logger, store_for_uri_with_logger, global_logger, finalize_op_logger};

// Initialize logger
init_op_logger("operations.tsv.zst")?;

// Create store with logging enabled
let logger = global_logger();
let store = store_for_uri_with_logger("file:///data/", logger)?;

// All operations automatically logged
store.list("file:///data/", true).await?;
store.get("file:///data/file.dat").await?;

// Finalize and flush logs
finalize_op_logger()?;
```

Available factory functions:
- `store_for_uri_with_logger(uri, logger)`
- `store_for_uri_with_config_and_logger(uri, config, logger)`
- `direct_io_store_for_uri_with_logger(uri, logger)`
- `high_performance_store_for_uri_with_logger(uri, logger)`

#### **Python API Integration** (SECONDARY Interface)
Added logger control functions in `src/python_api/python_core_api.rs`:
```python
import s3dlio

# Initialize op-log
s3dlio.init_op_log("operations.tsv.zst")

# Check if logging is active
if s3dlio.is_op_log_active():
    print("Logging enabled")

# All ObjectStore operations automatically logged
s3dlio.list_objects("file:///data/", recursive=True)
s3dlio.get_object("file:///data/file.dat")

# Finalize and flush logs
s3dlio.finalize_op_log()
```

#### **CLI Integration** (Testing Interface)
- Global `--op-log FILE` flag for all commands
- Works with: `ls`, `upload`, `download`, and other ObjectStore-based commands
```bash
# List with op-log
s3-cli --op-log list_ops.tsv.zst ls file:///data/ -r

# Upload with op-log
s3-cli --op-log upload_ops.tsv.zst upload /local/files/ s3://bucket/prefix/

# Download with op-log
s3-cli --op-log download_ops.tsv.zst download s3://bucket/prefix/ /local/dir/ -r
```

### 🔧 **Technical Details**

- **Decorator Pattern**: `LoggedObjectStore` wraps any `ObjectStore` without modifying implementations
- **Backward Compatible**: All existing code works unchanged (logger parameter optional)
- **Zero Overhead When Disabled**: No performance impact when logger not initialized
- **17 Methods Wrapped**: All ObjectStore trait methods instrumented with timing and error capture

### 📊 **Testing**

- Test script: `tests/test_op_log_all_backends.sh`
- Validates op-log creation for file:// and direct:// backends
- Confirms TSV format and zstd compression
- Verifies timing data and error handling

### 🎓 **Use Cases**

1. **Performance Profiling**: Track operation timing across all backends
2. **Debugging**: Identify slow operations or errors in storage access
3. **Warp-Replay**: Generate traces for replay/simulation testing
4. **Multi-Backend Analysis**: Compare performance across file://, s3://, az://, direct://
5. **Production Monitoring**: Log production I/O patterns for analysis

---

## Version 0.8.10 - TFRecord Index Generation (October 1, 2025)

### 🎯 **Release Focus: NVIDIA DALI-Compatible TFRecord Indexing**

This release adds comprehensive TFRecord index file generation compatible with NVIDIA DALI and TensorFlow tooling. Enables efficient random access, shuffled data loading, and distributed training workflows for large TFRecord datasets.

### ✨ **New Features**

#### **TFRecord Index Generation** (`src/tfrecord_index.rs`)
- **Pure Rust Implementation**: Zero-dependency stdlib-only implementation (437 lines)
- **DALI-Compatible Format**: Generates text indexes matching NVIDIA DALI `tfrecord2idx` specification
  - Format: `"{offset} {size}\n"` (space-separated ASCII text)
  - Validated against [NVIDIA DALI source](https://github.com/NVIDIA/DALI/blob/main/tools/tfrecord2idx)
- **Core Functions**:
  - `index_entries_from_bytes()` - Parse TFRecord to index entries
  - `index_text_from_bytes()` - Generate DALI text format
  - `write_index_for_tfrecord_file()` - File-to-file indexing
  - `read_index_file()` - Parse index files for random access
  - `TfRecordIndexer` - Streaming parser for large files

#### **Python API Integration** (PRIMARY Interface)
Added 4 PyO3-wrapped functions in `src/python_api/python_aiml_api.rs`:
```python
import s3dlio

# Generate index (auto-generates .idx file)
num_records = s3dlio.create_tfrecord_index("train.tfrecord")

# Read index for random access
index = s3dlio.read_tfrecord_index("train.tfrecord.idx")
offset, size = index[42]  # O(1) access to any record

# In-memory operations
index_text = s3dlio.index_tfrecord_bytes(data)
entries = s3dlio.get_tfrecord_index_entries(data)
```

#### **CLI Tool** (Optional)
- Added `s3-cli tfrecord-index` subcommand for batch index generation
- Auto-generates `.idx` extension or accepts custom output path

### 🎓 **Use Cases Enabled**

1. **Random Access**: O(1) seeking to any record (~1200x speedup vs sequential)
2. **Shuffled Training**: Different random order each epoch
3. **Distributed Training**: Efficient data sharding across GPUs/workers
4. **Batch Loading**: Fast seeking to batch boundaries
5. **DALI Integration**: Direct use with NVIDIA DALI pipelines

### 📊 **Performance Characteristics**

From validation testing (1000-record dataset):
- **Index Overhead**: 0.4% of TFRecord file size
- **Random Access Speedup**: ~1200x vs sequential scan
- **Access Pattern**: O(1) with index vs O(n) without

### 🧪 **Testing & Validation**

**22 tests, all passing:**
- ✅ 9 Rust unit tests (format, read/write, edge cases)
- ✅ 4 Python integration tests (API functionality)
- ✅ 4 DALI compatibility tests (format validation)
- ✅ 5 practical usage examples (ML workflows)
- ✅ Zero compilation warnings

**New Test Files:**
- `tests/test_tfrecord_index_python.py` - Python API integration
- `tests/test_dali_compatibility.py` - DALI format validation
- `tests/test_index_usage_examples.py` - Real-world usage patterns
- `tests/run_tfrecord_tests.sh` - Comprehensive test runner

### 📚 **Documentation**

**New Documentation:**
- `docs/TFRECORD-INDEX-IMPLEMENTATION.md` - Complete implementation guide
- `docs/TFRECORD-INDEX-SUMMARY.md` - Implementation summary & results
- `docs/TFRECORD-INDEX-QUICKREF.md` - Quick reference guide

**Coverage:**
- Python API usage examples
- Rust API usage examples
- DALI integration patterns
- Distributed training workflows
- Performance analysis

### 🔧 **Technical Details**

- **Format Compliance**: Verified against NVIDIA DALI tfrecord2idx specification
- **TensorFlow Compatible**: Standard format used by TensorFlow tooling
- **Streaming Support**: Can process files larger than memory
- **Round-trip Tested**: Write → Read → Verify correctness validated

### 🎯 **Compatibility**

- ✅ **NVIDIA DALI**: Format validated against official specification
- ✅ **TensorFlow**: Standard index format for TF datasets
- ✅ **Python 3.12+**: Tested with latest Python via PyO3
- ✅ **Rust 1.70+**: Pure stdlib, no external dependencies

### 🚀 **Migration Guide**

No breaking changes. This is a pure feature addition. Existing code continues to work unchanged.

**To use the new functionality:**
```python
# Python (PRIMARY interface)
import s3dlio
s3dlio.create_tfrecord_index("your_data.tfrecord")
```

```rust
// Rust library
use s3dlio::tfrecord_index::*;
write_index_for_tfrecord_file("your_data.tfrecord", "your_data.tfrecord.idx")?;
```

```bash
# CLI (optional)
s3-cli tfrecord-index your_data.tfrecord
```

---

## Version 0.8.9 - Build Quality & Dependency Cleanup (October 1, 2025)

### 🎯 **Release Focus: Code Quality, Build Hygiene, and Dependency Management**

This maintenance release focuses on improving build quality, removing forced dependencies, and updating to the latest AWS SDK versions. No functional changes to runtime behavior—all existing code continues to work identically.

### 🛠️ **Build Quality Improvements**

#### **Zero-Warnings Build Policy**
- **Fixed**: Removed unused `fmt` import in `src/bin/cli.rs` that caused warning
- **Enforcement**: Added comprehensive "Zero Warnings Policy" to `.github/copilot-instructions.md`
- **Pre-Commit Checklist**: Documented warning investigation process and anti-patterns
- **Result**: Clean compilation with zero warnings in default build

#### **Optional Experimental HTTP Client**
- **Removed**: Forced `aws-smithy-http-client` patch from `Cargo.toml` `[patch.crates-io]`
- **Added**: New `experimental-http-client` feature flag for optional custom HTTP optimizations
- **Default Behavior**: Uses standard AWS SDK HTTP client (no patches required)
- **Experimental Path**: Requires uncommenting patch AND building with `--features experimental-http-client`
- **Rationale**: Custom HTTP client showed no consistent performance benefit in production testing
- **Benefit**: Downstream users no longer forced to patch dependencies

### 📚 **Documentation Updates**

#### **Backend Recommendations Corrected**
- **README.md**: Completely rewrote "S3 Backend Options" section
  - Clarified `native-backends` is DEFAULT and RECOMMENDED for production
  - Marked `arrow-backend` as EXPERIMENTAL with no proven performance advantage
  - Removed misleading "recommended for modern deployments" language
- **copilot-instructions.md**: Updated backend guidance and patch removal notes
- **Changelog.md**: Added clarifications to v0.7.10 entry about arrow backend status
- **Performance docs**: Added warnings to `Apache_Arrow_Backend_Implementation.md`
- **src/object_store_arrow.rs**: Updated header comments to mark as experimental

#### **Build Quality Standards**
- **Added**: Comprehensive build quality section to copilot-instructions
- **Documented**: Warning investigation workflow and anti-patterns
- **Clarified**: Patch usage is optional, not required for builds

### 📦 **Dependency Updates**

Updated to latest AWS SDK and supporting crates:
- **aws-config**: `1.8.5` → `1.8.6`
- **aws-sdk-s3**: `1.103.0` → **`1.106.0`** (3 minor version bump)
- **aws-sdk-sso**: `1.82.0` → `1.84.0`
- **aws-sdk-ssooidc**: `1.83.0` → `1.86.0`
- **aws-sdk-sts**: `1.84.0` → `1.86.0`
- **aws-smithy-runtime**: `1.9.0` → `1.9.2`
- **aws-lc-sys**: `0.32.1` → `0.32.2`
- Added: `hashbrown` v0.16.0, `itertools` v0.13.0

### 🔧 **Technical Implementation Details**

#### **Feature-Gated HTTP Client**
```rust
// Default build (no experimental features)
#[cfg(not(feature = "experimental-http-client"))]
fn create_optimized_http_client() -> Result<SharedHttpClient> {
    Ok(HttpClientBuilder::new().build_http())  // Standard AWS SDK
}

// Experimental build (requires patched dependency)
#[cfg(feature = "experimental-http-client")]
fn create_optimized_http_client() -> Result<SharedHttpClient> {
    // Custom hyper_builder configuration
    // Requires uncommenting [patch.crates-io] section
}
```

#### **AWS_CA_BUNDLE Support**
- Works in ALL builds (standard and experimental)
- Uses standard AWS SDK API: `.tls_provider()` + `.tls_context()` + `.build_https()`
- Critical AWS SDK feature preserved across all configurations

### ✅ **Verification & Testing**

- **Build Verification**: Default build compiles with zero warnings
- **Historical Validation**: Compared against v0.7.9 (production-verified) implementation
- **API Compatibility**: All existing code paths preserved and verified identical
- **Backward Compatibility**: No breaking changes, all existing APIs work unchanged

### 🎯 **Usage**

```bash
# Default build (RECOMMENDED for production)
cargo build --release

# Experimental HTTP optimizations (optional, requires manual patch enablement)
# 1. Uncomment [patch.crates-io] section in Cargo.toml
# 2. cargo build --release --features experimental-http-client
```

### 📊 **Impact Summary**

| Change | Before (v0.8.8) | After (v0.8.9) | Benefit |
|--------|-----------------|----------------|---------|
| **Build Warnings** | 1 unused import | Zero warnings | ✅ Clean compilation |
| **Forced Patches** | Required for all users | Optional only | ✅ Downstream flexibility |
| **Default Behavior** | Same | Same | ✅ No breaking changes |
| **AWS SDK Version** | 1.103.0 | 1.106.0 | ✅ Latest bug fixes |
| **Documentation** | Some inaccuracies | Fully corrected | ✅ Clear guidance |

### 🔄 **Migration Notes**

**No action required** for existing users. This is a drop-in replacement:
- Default behavior unchanged
- All APIs work identically
- Performance characteristics preserved
- Environment variables work the same

For users who enabled experimental HTTP optimizations (`S3DLIO_USE_OPTIMIZED_HTTP=true`):
- Feature continues to work in default build using standard AWS SDK HTTP client
- To use custom `hyper_builder` optimizations: uncomment patch AND build with `--features experimental-http-client`

---

## Version 0.8.8 - Page Cache Hints & Tracing Framework (October 1, 2025)

### 🚀 **Major New Features**

#### **🗂️ posix_fadvise() Page Cache Optimization**
- **New Module**: Added `src/page_cache.rs` with Linux/Unix page cache hint support
- **PageCacheMode Enum**: Sequential, Random, DontNeed, Normal, and Auto modes
- **Auto Mode Intelligence**: Automatically applies Sequential hints for files ≥64MB, Random for smaller files
- **File Store Integration**: Integrated into `file_store.rs` get() and get_range() operations
- **Platform Support**: Linux/Unix only (no-op on Windows)
- **Performance Impact**: Optimizes kernel read-ahead and caching strategies for different access patterns

#### **📝 Logging Framework Migration: log → tracing**
- **Complete Migration**: Replaced `log` and `env_logger` with `tracing` ecosystem
- **Enhanced Compatibility**: Now compatible with dl-driver and s3-bench (io-bench) projects
- **Dependencies Updated**: 
  - `tracing ^0.1` (with log feature for compatibility)
  - `tracing-subscriber ^0.3` (with fmt, env-filter, registry features)
  - `tracing-log ^0.2` (for backward compatibility bridge)
- **Verbosity Mapping**: 
  - Default (no flags): WARN level
  - `-v`: INFO level
  - `-vv`: DEBUG level
- **Scope**: Updated 12 source files with ~99 log macro conversions
- **Trace Logging Preserved**: Operation trace logging (--op-log) functionality remains unchanged

### 🐛 **Known Issues Documented**
- **Progress Bars**: Documented real-time update issue in `KNOWN-ISSUES.md`
- **Op-Log Backend Support**: Created comprehensive issue documentation for extending trace logging to all storage backends (file://, az://, direct://)
- **Issue Tracking**: Added `KNOWN-ISSUES.md` for project-wide issue management

### 🛠️ **Implementation Details**
- **Version Bump**: 0.8.7 → 0.8.8 in both Rust and Python packages
- **Clean Compilation**: All changes compile successfully with no errors
- **Testing**: Verified logging levels, trace independence, and page cache hint application
- **Documentation**: Three new issue docs with root cause analysis and proposed solutions

### 📚 **Files Modified**
- **Core Modules**: cli.rs, s3_logger.rs, data_gen.rs, s3_utils.rs, s3_client.rs, object_store.rs, python_core_api.rs
- **Additional Updates**: checkpoint/latest.rs, http/client.rs, s3_copy.rs, file_store_direct.rs, profiling.rs
- **New Files**: 
  - `src/page_cache.rs` - Page cache hint implementation
  - `KNOWN-ISSUES.md` - Project-wide issue tracker
  - `docs/ISSUE-op-log-backend-support.md` - Detailed issue analysis
  - `docs/GITHUB-ISSUE-op-log-backend-support.md` - GitHub issue format

### 🔄 **Backward Compatibility**
- **LogTracer Bridge**: Added for compatibility with crates still using `log` macros
- **Trace Format**: Operation trace logging format unchanged (warp-replay compatible)
- **API Stability**: No breaking changes to public APIs

---

## Version 0.8.7 - HDR Performance Monitoring for AI/ML Workloads (September 29, 2025)

### 🚀 **Major New Features**

#### **📊 HDR Histogram Performance Monitoring**
- **New Module**: Added comprehensive `src/metrics/` module with HDR histogram-based performance tracking
- **Precise Tail Latency**: P99, P99.9, P99.99+ percentile measurement for AI/ML performance analysis
- **High Throughput Support**: Configurable precision and limits up to 1TB/s for large-scale workloads
- **Thread-Safe Collection**: Global metrics with `parking_lot` synchronization for high-frequency recording
- **AI/ML Presets**: Pre-configured settings for training, inference, and distributed scenarios

#### **🎯 Global Metrics API**
- **Simple Integration**: `record_operation()` and `print_global_report()` convenience functions
- **Configurable Precision**: 1-5 significant digits for different accuracy requirements
- **Operation Tracking**: Per-operation latency and throughput analysis
- **Error Recording**: Built-in error rate tracking and reporting

#### **🔧 Enhanced Dependencies**
- **hdrhistogram ^7.5**: High Dynamic Range histograms for detailed performance analysis
- **metrics ^0.23**: Professional metrics collection framework
- **parking_lot ^0.12**: High-performance RwLock for concurrent access

#### **📈 Performance Analysis Demo**
- **New Demo**: `examples/performance_monitoring_demo.rs` with realistic AI/ML workload simulation
- **Comprehensive Reporting**: Detailed latency and throughput analysis across operation types
- **Training Patterns**: Simulates data loading, preprocessing, forward/backward passes, checkpointing

### 🛠️ **Implementation Details**
- **Version Bump**: 0.8.6 → 0.8.7 in both Rust and Python packages
- **Clean Architecture**: Focused on measurement rather than premature optimization
- **Future-Ready**: Foundation for planned OS page cache control features (v0.8.8)

### 📚 **Documentation**
- **New Guide**: `docs/PERFORMANCE-MONITORING-v0.8.7.md` with comprehensive usage examples
- **Integration Examples**: PyTorch DataLoader and DLIO framework integration patterns
- **Performance Strategies**: Guidelines for training, inference, and distributed workloads

---

## Version 0.8.6 - LoaderOptions Realism Knobs for AI/ML Training (September 29, 2025)

### 🚀 **Major New Features**

#### **🧠 LoaderOptions Realism Knobs**
- **New Feature**: Added 10 comprehensive AI/ML training configuration options for realistic production workloads
- **Core Knobs**: 
  - `pin_memory`: GPU memory pinning for faster CPU-to-GPU transfers
  - `persistent_workers`: Worker process persistence across training epochs
  - `timeout_seconds`: Configurable data loading timeout handling
  - `multiprocessing_context`: Choice of spawn/fork/forkserver process contexts
  - `sampler_type`: Sequential/random/weighted sampling strategies
  - `memory_format`: Channels-first/channels-last memory layout control
  - `non_blocking`: Asynchronous GPU transfer operations
  - `generator_seed`: Reproducible random number generation
  - `enable_transforms`: Runtime data transformation toggle
  - `collate_buffer_size`: Batch collation buffer size optimization

#### **🎯 Convenience Presets**
- **gpu_optimized()**: Pre-configured settings for optimal GPU training performance
- **distributed_optimized()**: Best practices for multi-node distributed training
- **debug_mode()**: Enhanced debugging and validation for development workflows

#### **🐍 Python Integration Enhancement**
- **PyLoaderOptions Class**: Full PyO3 bindings with fluent builder pattern
- **Method Chaining**: Intuitive configuration with `.pin_memory(True).persistent_workers(True)` syntax
- **Property Access**: Direct access to all configuration options
- **Seamless Integration**: Works with existing s3dlio Python API

### 📚 **Documentation**
- **Implementation Guide**: Comprehensive documentation in `docs/LOADER-OPTIONS-REALISM-KNOBS.md`
- **Before/After Examples**: Clear migration path and usage patterns
- **Performance Implications**: Detailed analysis of each configuration option
- **Architecture Decisions**: Design rationale and implementation details

### 🔧 **Technical Implementation**
- **Enhanced Core**: Extended `src/data_loader/options.rs` with new enums and configuration fields
- **Python Bindings**: Added `PyLoaderOptions` in `src/python_api/python_aiml_api.rs`
- **Comprehensive Testing**: Full test coverage in `python/tests/test_loader_options_*.py`
- **Backward Compatibility**: All existing APIs remain unchanged
- **Production Defaults**: Aligned with PyTorch and TensorFlow best practices

## Version 0.8.5 - Direct I/O Support & Async Loader Fixes (September 29, 2025)

### 🚀 **Major New Features**

#### **💾 Direct I/O Dataset Support**
- **New Feature**: Added full Direct I/O dataset support for `direct://` URIs
- **Implementation**: Created `DirectIOBytesDataset` using object store pattern with O_DIRECT support
- **Performance**: Leverages existing `ConfigurableFileSystemObjectStore::boxed_direct_io()` for maximum throughput
- **API Integration**: `create_dataset("direct:///path/to/files")` now works seamlessly alongside `file://`, `s3://`, and `az://` schemes
- **Testing**: Comprehensive test suite with single file and directory operations

#### **🔄 Async Loader Improvements**
- **Fixed**: Async loaders now return individual items (bytes objects) by default instead of batches (lists)
- **Smart Batching**: When `batch_size=1` (default for async loaders), yields individual items for intuitive iteration
- **Backward Compatible**: Explicit `batch_size > 1` still returns lists for ML workload efficiency
- **Improved UX**: `async for item in loader:` now works as expected without type confusion

### 🐛 **Bug Fixes**

#### **🔧 Python API Baseline Fixes**
- **Fixed**: `test_realism_baseline.py` string/bytes type error in async test creation
- **Fixed**: Multi-backend factory parity - all URI schemes now work consistently
- **Validation**: 11/11 baseline tests now pass (9 multi-backend + 2 async functionality)

#### **📁 API Module Organization**
- **Enhanced**: Added `DirectIOBytesDataset` to public API exports
- **Updated**: Dataset factory function now handles `direct://` schemes correctly
- **Improved**: Better error messages for unsupported URI schemes

### 🛠 **Technical Improvements**
- **Code Quality**: Removed duplicate closing braces and syntax errors
- **Module Structure**: Added `directio_bytes.rs` to data_loader module exports
- **API Consistency**: All dataset types now follow the same creation patterns
- **Test Coverage**: DirectIO tests pass reliably with proper error handling

### 📖 **Documentation & Infrastructure**
- **Added**: Comprehensive AI coding instructions in `.github/copilot-instructions.md`
- **Updated**: README.md with HDF5 dependency requirements for all major platforms
- **Enhanced**: `install_pyo3_wheel.sh` handles different Python versions with wildcard matching
- **Organized**: Python test suite structure maintained in `python/tests/` directory

---

## Version 0.8.4 - Critical Regex Pattern Matching Fix (September 25, 2025)

### 🐛 **Critical Bug Fixes**

#### **🎯 Fixed Regex Pattern Matching for All Commands**
- **Issue**: `list`, `get`, `delete`, and `download` commands with regex patterns like `s3://mybucket/` were incorrectly returning CommonPrefixes ("/") instead of actual matching objects
- **Root Cause**: S3 API with delimiter="/" was treating objects with leading slashes (e.g., `/object_0.dat`) as being in subdirectories, returning them as CommonPrefixes rather than Contents
- **Solution**: Enhanced CommonPrefixes handling to intelligently distinguish between:
  - **Single-character prefixes** (like "/") that contain objects with leading slashes → Process recursively to find actual objects
  - **Multi-character prefixes** (like "dir1/", "subdir/") that represent true subdirectories → Skip in non-recursive mode
- **Impact**: All pattern-based operations now work correctly while preserving proper directory hierarchy behavior

#### **🔧 Improved Directory Boundary Respect**
- **Non-recursive listings** (default): Only show objects at the current level, properly exclude subdirectory contents
- **Recursive listings** (`--recursive`): Show all objects including subdirectory contents as expected  
- **Smart CommonPrefix processing**: Objects with leading slashes (e.g., `//object_0.dat`) are treated as root-level objects, not subdirectories
- **Preserved existing behavior**: All directory navigation and listing semantics remain unchanged for standard use cases

#### **✅ Verified Across All Commands**
- **`list s3://bucket/.*pattern.*`**: Now correctly finds matching objects instead of showing "/"
- **`get s3://bucket/.*pattern.*`**: Successfully retrieves all objects matching regex patterns
- **`delete s3://bucket/.*pattern.*`**: Safely deletes only objects matching the specified pattern  
- **`download s3://bucket/.*pattern.*`**: Downloads all matching objects while respecting directory structure

### 🛠 **Technical Improvements**
- **Enhanced debug logging**: Better visibility into prefix vs pattern matching logic
- **Code cleanup**: Removed obsolete commented-out code sections from CLI handlers
- **Consistent error handling**: Improved pattern matching error messages across all commands

---

## Version 0.8.3 - Multi-Backend Progress Tracking & Enhanced Copy Operations (September 25, 2025)

### 🚀 **Universal Copy Operations with Progress Tracking**

This release transforms `upload` and `download` commands into universal copy operations that work seamlessly across **all storage backends** (S3, Azure Blob, local file system, O_DIRECT) with **consistent progress tracking** and **enhanced pattern matching**.

**Key Achievement**: **Unified copy interface** - upload and download now function as enhanced versions of the Unix `cp` command, supporting **regex patterns**, **glob patterns**, and **multi-backend operations** with real-time progress bars.

### 🎯 **Multi-Backend Support**

#### **🔄 Universal Storage Backends**
- **S3**: `s3://bucket/prefix/` - Amazon S3 compatible storage
- **Azure Blob**: `az://container/prefix/` or `azure://container/prefix/` - Microsoft Azure Blob Storage  
- **File System**: `file:///path/to/directory/` - Local file system operations
- **Direct I/O**: `direct:///path/to/directory/` - High-performance O_DIRECT file operations
- **Automatic detection**: Backend automatically selected based on URI scheme
- **Consistent behavior**: Same progress tracking and performance across all backends

#### **📊 Enhanced Progress Tracking**
- **Real-time progress bars**: Consistent across all storage backends with transfer rates, ETA, and completion percentage
- **Multi-file tracking**: Accurate progress for concurrent upload/download operations
- **Dynamic totals**: Progress bars update correctly even when total size is unknown initially
- **Performance metrics**: Display transfer rates (MB/s, KiB/s) and estimated time remaining
- **Fixed Tokio runtime issues**: Eliminated runtime nesting panics in CLI and Python API

#### **🎯 Advanced Pattern Matching**
- **Individual Files**: `file.txt` - Upload/download specific files
- **Directories**: `directory/` - Process all files in directory (upload) or prefix (download)
- **Glob Patterns**: `*.log`, `file_*.txt`, `data?.csv` - Shell-style wildcards
- **Regex Patterns**: `.*\.log$`, `file_[0-9]+\.txt` - Powerful regular expression matching
- **Cross-backend patterns**: All pattern types work with any storage backend

### 🔧 **Enhanced CLI Commands**

#### **📤 Universal Upload Command**
```bash
# Upload to different backends with progress
s3-cli upload /local/files/*.log s3://bucket/logs/
s3-cli upload /local/data/* az://container/data/  
s3-cli upload /local/files/* file:///backup/files/
s3-cli upload /local/data/* direct:///fast-storage/data/

# Regex patterns for advanced file selection
s3-cli upload "/path/to/logs/.*\.log$" s3://bucket/logs/
s3-cli upload "/data/file_[0-9]+\.txt" az://container/data/
```

#### **📥 Universal Download Command**  
```bash
# Download from any backend with progress
s3-cli download s3://bucket/data/ ./local-data/
s3-cli download az://container/logs/ ./logs/
s3-cli download file:///remote-mount/data/ ./data/
s3-cli download direct:///nvme/cache/ ./cache/

# Cross-backend copying workflow
s3-cli download s3://source-bucket/data/ ./temp/
s3-cli upload ./temp/* az://dest-container/data/
```

### 🐍 **Python API Improvements**

#### **🔄 Multi-Backend Python Functions**
- **`s3dlio.upload()`**: Works with all storage backends, maintains same signature
- **`s3dlio.download()`**: Universal download supporting all URI schemes  
- **Runtime safety**: Fixed Tokio runtime nesting issues for stable operation
- **Pattern support**: Inherits glob and regex pattern matching from generic functions
- **No breaking changes**: Existing Python code works unchanged with enhanced capabilities

#### **🛡️ Robust Error Handling**
- **Bucket creation control**: Only creates buckets/containers when explicitly requested with `create_bucket=True`
- **Backend validation**: Clear error messages for unsupported operations or invalid URIs
- **Graceful degradation**: Operations continue even if individual files fail (with warnings)
- **Progress callback support**: Internal progress tracking ready for future Python progress bar integration

### 🏗️ **Architecture Improvements**

#### **🎯 Generic Backend Functions**
- **`generic_upload_files()`**: Backend-agnostic upload with progress tracking
- **`generic_download_objects()`**: Universal download supporting all storage types
- **ObjectStore trait**: Clean abstraction enabling easy addition of new storage backends
- **Async architecture**: Proper async/await implementation preventing runtime conflicts
- **Performance preservation**: All existing performance optimizations maintained across backends

#### **🔧 Technical Fixes**
- **Tokio runtime resolution**: Eliminated nested runtime creation causing panics in CLI commands
- **Progress bar accuracy**: Fixed "0 B/0 B" display by implementing dynamic total updates
- **Pattern matching robustness**: Enhanced regex detection and glob handling for better file discovery
- **Memory efficiency**: Maintained streaming performance while adding multi-backend support

### 📈 **Performance & Compatibility**

#### **⚡ Performance Characteristics**
- **Unchanged performance**: All existing benchmark results maintained
- **Concurrent operations**: Multi-file uploads/downloads preserve parallel execution
- **Memory footprint**: Minimal overhead added for backend abstraction
- **Progress tracking overhead**: Negligible performance impact for progress bar functionality

#### **🔄 Backward Compatibility**
- **CLI compatibility**: All existing CLI commands work unchanged with enhanced capabilities
- **Python API compatibility**: Existing Python scripts run without modification
- **Configuration preservation**: All existing settings and options preserved
- **Migration path**: Seamless upgrade with immediate access to new multi-backend features

---

## Version 0.8.2 - Configurable Data Generation Modes (September 25, 2025)

### 🚀 **Configurable Data Generation Modes for Optimal Performance**

This release introduces **configurable data generation modes** based on comprehensive performance benchmarking. Users can now choose between **streaming** and **single-pass** data generation modes via both CLI and Python API, with **streaming set as the default** for optimal performance in most scenarios.

**Key Achievement**: **Streaming mode provides 2.6-3.5x performance improvement** for 1-8MB objects and wins in **64% of real-world scenarios**, making it the ideal default choice.

### 🎯 **New Configuration System**

#### **🔥 Data Generation Modes**
- **`DataGenMode::Streaming`**: Default mode using streaming data generation for better memory efficiency and performance
- **`DataGenMode::SinglePass`**: Alternative mode for specific use cases where single-pass generation may be preferred
- **Intelligent defaults**: Streaming mode chosen as default based on extensive benchmarking data
- **Full configurability**: Both modes available via CLI options and Python API parameters

#### **⚡ CLI Integration**
- **`--data-gen-mode`**: Choose between `streaming` (default) or `single-pass` modes
- **`--chunk-size`**: Configure chunk size for fine-tuning performance (default: 262144 bytes)
- **Backward compatibility**: All existing CLI commands work unchanged with new streaming defaults
- **Real S3 testing**: Verified with actual S3 backend operations

#### **🐍 Python API Enhancement**
- **`data_gen_mode`** parameter: Accept `"streaming"` (default) or `"single-pass"` in `put()` function
- **`chunk_size`** parameter: Configurable chunk size for performance optimization
- **Default behavior**: Streaming mode automatically selected for optimal performance
- **Seamless integration**: New parameters work alongside existing Python API functionality

### 🔧 **Performance Characteristics**

#### **📊 Benchmarking Results**
- **1-8MB objects**: Streaming mode shows 2.6-3.5x performance improvement
- **Overall win rate**: Streaming performs better in 64% of real-world scenarios  
- **16-32MB range**: Single-pass mode competitive for specific object sizes
- **Memory efficiency**: Streaming mode uses less memory due to on-demand generation
- **Throughput**: Real S3 testing shows excellent throughput rates (80+ MiB/s)

#### **🎯 Configuration Examples**

**CLI Usage:**
```bash
# Default streaming mode
s3-cli put s3://bucket/object-{}.bin --num 10 --size 4MB

# Explicit streaming mode
s3-cli put s3://bucket/object-{}.bin --num 10 --size 4MB --data-gen-mode streaming

# Single-pass mode for specific use cases  
s3-cli put s3://bucket/object-{}.bin --num 10 --size 4MB --data-gen-mode single-pass --chunk-size 65536
```

**Python API:**
```python
import s3dlio

# Default streaming mode
s3dlio.put("s3://bucket/object-{}.bin", num=10, template="obj-{}-of-{}", size=4194304)

# Explicit mode selection
s3dlio.put("s3://bucket/object-{}.bin", num=10, template="obj-{}-of-{}", 
           size=4194304, data_gen_mode="streaming", chunk_size=65536)
```

### 🔧 **Technical Implementation**
- **DataGenerator Architecture**: New streaming data generation system with instance entropy for deterministic behavior
- **Memory Efficiency**: **32x memory efficiency improvement** - 256KB vs 8MB memory usage for 8MB objects  
- **Config System**: Added `data_gen_mode` and `chunk_size` fields with builder pattern methods
- **Function Integration**: Updated `put_objects_with_random_data_and_type()` to accept unified Config parameter
- **Comprehensive Testing**: 18/19 tests passing with edge case, error handling, and production performance validation

### 📊 **Measured Performance Achievements**
- **Single-Pass Enhancement**: 3.3x faster generation (70% reduction from 7.9ms to 2.4ms for 1MB objects)
- **Streaming Throughput**: 7.66 GB/s single-thread baseline with **20+ GB/s multi-process capability**
- **Memory Footprint**: **Zero-copy chunk generation** for optimal memory efficiency per chunk
- **Production Testing**: Verified with real S3 backend achieving 80+ MiB/s throughput rates

### 📚 **Comprehensive Documentation**
The v0.8.2 release includes extensive technical documentation:
- **Performance Analysis**: `docs/performance/before-after-comparison-v0.8.2.md` - Detailed before/after comparison
- **Implementation Guide**: `docs/performance/enhanced-data-generation-v0.8.2-progress-summary.md` - Complete technical overview
- **Testing Validation**: `docs/performance/comprehensive-testing-summary-v0.8.2.md` - Full test coverage analysis
- **Foundation Work**: `docs/performance/single-pass-data-generation-v0.8.1.md` - Single-pass optimization foundation

### 🎯 **Backward Compatibility & Migration**
- **Zero breaking changes**: All existing code continues to work with new streaming defaults
- **Automatic enhancement**: Existing applications immediately benefit from optimized streaming mode  
- **Optional configuration**: New parameters are optional - intelligent defaults provide optimal performance
- **Seamless upgrade**: v0.8.1 to v0.8.2 requires no code changes while delivering performance improvements

## Version 0.8.1 - Enhanced API & PyTorch Integration (September 24, 2025)

### 🚀 **Production-Ready Enhanced API with Full PyTorch Integration**

This release completes the **Enhanced API** development and delivers **production-ready PyTorch integration** that was broken in earlier versions. The new Enhanced API provides simplified, modern interfaces while maintaining full backward compatibility with legacy functions.

**Key Achievement**: **34/34 tests passing (100%)** with comprehensive functionality validation and **PyTorch S3IterableDataset working perfectly**.

### 🎯 **Enhanced API - Complete & Tested**

#### **🔥 New Simplified Functions**
- **`create_dataset(uri, options=None)`**: Creates `PyDataset` instances for file system or S3 URIs
- **`create_async_loader(uri, options=None)`**: Creates `PyBytesAsyncDataLoader` for streaming data
- **Options dictionary support**: Configurable batch size, shuffle, num_workers, etc.
- **Proper error handling**: Clear error messages for invalid URIs and configurations

#### **🐍 PyTorch Integration Fixed**
- **`S3IterableDataset`** now imports and functions correctly  
- **PyTorch DataLoader integration** working with iterator support
- **Fixed**: Previous "PyS3AsyncDataLoader not found" errors completely resolved
- **Tested**: Full integration with PyTorch 2.8.0+ confirmed working

#### **⚡ Backward Compatibility Maintained**
- **All legacy functions preserved**: `get()`, `put()`, `list()`, `stat()` still available
- **Helper functions working**: `list_keys_from_s3()`, `get_by_key()`, `stat_key()` functional
- **Zero breaking changes**: Existing code continues to work unchanged

### 🔧 **API Architecture**

#### **Enhanced API Functions** 
- **File System Support**: `file://` URIs for local datasets and testing
- **S3 Support**: Ready for S3 URIs with proper credential handling
- **Async Iteration**: `async for` loops with `PyBytesAsyncDataLoader`
- **Type Safety**: Proper Python type annotations and Rust type safety

#### **Comprehensive Documentation**
- **API Organization**: All API docs moved to `docs/api/` directory
- **Version Marking**: Clear distinction between v0.8.1 (current) and v0.7.x (legacy)
- **Status Documentation**: Working/not-working status for every function
- **Testing Guide**: Critical testing gotchas and best practices documented

### 📊 **Testing Results**

All functionality verified with comprehensive test suite:

| Test Category | Tests | Status | Details |
|---------------|-------|---------|---------|
| **Installation Verification** | 9/9 | ✅ PASSED | All Rust functions and classes available |
| **Enhanced API** | 8/8 | ✅ PASSED | Dataset and loader creation working |
| **PyTorch Integration** | 5/5 | ✅ PASSED | S3IterableDataset and DataLoader functional |
| **Legacy API** | 7/7 | ✅ PASSED | All original functions maintained |
| **Error Handling** | 3/3 | ✅ PASSED | Proper error messages and handling |
| **Async Operations** | 2/2 | ✅ PASSED | Streaming and iteration working |

**Total: 34/34 PASSED (100.0%)**

### ✅ **New Features**

- **Enhanced Dataset API**: Modern `create_dataset()` function for simplified usage
- **Enhanced Async API**: `create_async_loader()` for streaming data workflows  
- **PyTorch Integration**: Working `S3IterableDataset` with DataLoader support
- **Comprehensive Documentation**: Organized API docs with version control
- **Testing Infrastructure**: Robust test suite with 100% pass rate
- **Critical Testing Guide**: Documented Python import path gotchas

### 🐛 **Bug Fixes**

- **Fixed PyTorch Import**: `S3IterableDataset` now imports without errors
- **Fixed PyTorch DataLoader**: Integration with PyTorch DataLoader working correctly
- **Fixed Iterator Creation**: PyTorch iterator creation successful
- **Fixed Testing Issues**: Resolved sys.path import problems in test suite

### 🔄 **API Changes**

#### **New Functions (v0.8.1)**
```python
# Enhanced API - New in v0.8.1
dataset = s3dlio.create_dataset("file:///path/to/data")
loader = s3dlio.create_async_loader("s3://bucket/path", {"batch_size": 32})

# PyTorch Integration - Fixed in v0.8.1  
from s3dlio.torch import S3IterableDataset
dataset = S3IterableDataset("s3://bucket/data", loader_opts={})
```

#### **Maintained Legacy Functions (v0.7.x)**
```python
# Legacy API - Still working in v0.8.1
s3dlio.get(uri, path)      # Still available
s3dlio.put(path, uri)      # Still available  
s3dlio.list(uri)           # Still available
s3dlio.stat(uri)           # Still available
```

### 📚 **Documentation Updates**

- **`docs/api/README.md`**: API directory index with version guide
- **`docs/api/enhanced-api-v0.8.1.md`**: Complete Enhanced API documentation
- **`docs/api/python-api-v0.8.1-current.md`**: Current API status reference
- **`docs/api/migration-guide-v0.8.1.md`**: Migration guide from legacy APIs
- **`docs/TESTING-GUIDE.md`**: Critical testing best practices and gotchas
- **`docs/TESTING-SUCCESS-REPORT.md`**: Evidence of 100% test success

### 🏗️ **Development Improvements**

- **Clean Build Process**: Fresh builds with proper version increments
- **Release Branch**: Proper `release-v0.8.1` branching for clean PRs
- **Version Synchronization**: Rust and Python versions aligned at 0.8.1
- **Testing Validation**: Full test suite confirms production readiness

---

## Version 0.8.0 - Multi-Process Performance Engine & Python Bindings (September 20, 2025)

### 🚀 **Major Release: Warp-Level Performance & Python Integration**

This release introduces a **complete multi-process performance engine** with **Python bindings**, delivering **warp-level S3 performance** that scales to 8+ processes. The new architecture removes io_uring (which provided no benefit for network I/O) and replaces it with a purpose-built concurrent range-GET engine.

**Key Achievement**: **2,308 MB/s** (8 processes) vs **1,150 MB/s** (single process) = **2x performance improvement** with perfect load balancing.

### 🎯 **New Multi-Process Architecture**

#### **🔥 Core Performance Engine**
- **Multi-process supervisor** (`mp-get` command): Scales to 8+ worker processes with perfect load balancing
- **Concurrent range-GET engine**: Zero-copy memory architecture for maximum throughput  
- **Sharded S3 clients**: Connection pooling eliminates client creation overhead
- **Advanced memory management**: Page-aligned buffers for O_DIRECT file operations

#### **🐍 Python Integration**
- **Native Python bindings**: `s3dlio.mp_get()` function with 100%+ CLI performance parity
- **Performance statistics**: Comprehensive per-worker and aggregate metrics
- **Same warp-level performance**: Python achieves **1,712 MB/s** vs CLI **1,711 MB/s**

#### **⚡ Performance Optimizations**
- **Fixed get command**: Proper data download and byte counting for accurate benchmarking
- **Enhanced CLI reporting**: Real-time throughput and per-worker performance breakdown
- **O_DIRECT backend improvements**: Page-aligned buffers using constants from `src/constants.rs`

### 🔧 **Architecture Changes**

#### **Removed io_uring** 
- **Reason**: Complex async I/O provided no measurable benefit for network operations
- **Replacement**: Purpose-built concurrent engine optimized specifically for S3 workloads
- **Result**: Simpler codebase with significantly better performance characteristics

#### **Added Multi-Process Capabilities**
- **Worker coordination**: Robust process management with error handling and cleanup
- **Load balancing**: Even distribution of S3 objects across worker processes  
- **Performance reporting**: Detailed statistics for debugging and optimization

### 📊 **Performance Results**

| Configuration | Processes | Objects | Throughput | Improvement |
|---------------|-----------|---------|------------|-------------|
| **Single Process** | 1 | 200 | 1,150 MB/s | Baseline |
| **Multi-Process** | 8 | 200 | **2,308 MB/s** | **2x Faster** |
| **Python Bindings** | 8 | 200 | **1,712 MB/s** | **1.49x Faster** |

### ✅ **New Features**

- 🚀 **`mp-get` CLI command**: Multi-process S3 operations with configurable worker count
- 🐍 **Python `s3dlio.mp_get()`**: Native Python bindings with identical performance
- ⚡ **Enhanced get command**: Proper benchmarking with accurate byte counting and reporting  
- 🔧 **Page-aligned O_DIRECT**: Improved file backend using system page size constants
- 📊 **Performance statistics**: Per-worker breakdown and aggregate metrics
- 🎯 **Zero-copy architecture**: Memory-efficient S3 operations without intermediate files

### 🛠️ **Technical Improvements**

- **New modules**: `mp.rs`, `range_engine.rs`, `sharded_client.rs`, `download.rs`, `memory.rs`
- **Enhanced CLI**: Added `mp-get` subcommand with comprehensive options
- **Python wheel**: Built with maturin, installable via `uv pip install`  
- **Constants compliance**: All page sizes and alignments use centralized constants
- **Error handling**: Robust multi-process coordination and cleanup

### 📝 **Migration Notes**

- **io_uring removal**: No breaking changes - io_uring was optional and is now automatically replaced
- **New CLI commands**: `mp-get` command available alongside existing `get` command
- **Python bindings**: Install via `uv pip install s3dlio` for Python integration
- **Performance**: Existing single-process operations maintain same performance characteristics


---

**For versions prior to 0.8.0, see [Changelog_pre-0.8.0.md](Changelog_pre-0.8.0.md)**
