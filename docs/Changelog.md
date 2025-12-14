# s3dlio Changelog

## Version 0.9.26 - DLIO Benchmark Integration (December 2025)

### 🆕 **New Features**

**DLIO Benchmark Integration**
- Added comprehensive integration support for [Argonne DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark)
- Two installation options:
  - **Option 1 (Recommended):** New `storage_type: s3dlio` with explicit configuration
  - **Option 2:** Drop-in replacement for existing S3 configurations
- Enables DLIO to use all s3dlio backends: S3, Azure, GCS, file://, direct://

**Zero-Copy Write Functions (Rust + Python)**
- `put_bytes(uri, data)` - Write bytes to any backend using zero-copy from Python memory
- `put_bytes_async(uri, data)` - Async version
- `mkdir(uri)` - Create directories/prefixes via ObjectStore trait
- `mkdir_async(uri)` - Async version
- Uses `PyBytes.as_bytes()` for zero-copy data transfer, avoiding allocation overhead

### 📦 **New Files**

```
python/s3dlio/integrations/dlio/
├── __init__.py           # Helper functions for installation
├── s3dlio_storage.py     # Option 1: S3dlioStorage class
└── s3_torch_storage.py   # Option 2: Drop-in replacement

docs/integration/
└── DLIO_BENCHMARK_INTEGRATION.md  # Comprehensive guide (477 lines)
```

### 🔧 **Other Changes**

**Azure SDK Update**
- Updated from Azure SDK 0.4.0 to 0.7.0 API
- Maintains backward compatibility with existing functionality

### 📝 **Documentation**

- Added `docs/integration/DLIO_BENCHMARK_INTEGRATION.md` - comprehensive guide with:
  - Architecture diagram
  - Step-by-step instructions for both integration options
  - Environment variable configuration (S3/Azure/GCS)
  - API mapping table (DLIO → s3dlio)
  - Troubleshooting section
  - Multi-protocol examples
  - Advanced features (multi-endpoint load balancing, performance tuning)
  - Feature compatibility table
- Updated `docs/PYTHON_API_GUIDE.md` with new functions

---

## Version 0.9.24 - S3-Compatible Endpoint Fix & Tracing Workaround (December 2025)

### 🐛 **Bug Fixes**

**S3-Compatible Endpoint Support (force_path_style)**
- Fixed S3 requests to custom endpoints (MinIO, Ceph, etc.)
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
- Enables use with Azurite or other Azure-compatible emulators/proxies
- Account name is appended to endpoint URL automatically

Usage:
```bash
# Azurite (local emulator)
export AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000
sai3-bench util ls az://devstoreaccount1/testcontainer/

# Multi-protocol proxy
export AZURE_STORAGE_ENDPOINT=http://localhost:9001
sai3-bench util ls az://myaccount/mycontainer/
```

**Custom Endpoint Support for Google Cloud Storage**
- Added environment variable support for custom GCS endpoints
- Primary: `GCS_ENDPOINT_URL` (e.g., `http://localhost:4443`)
- Alternative: `STORAGE_EMULATOR_HOST` (GCS emulator convention, `http://` prepended if missing)
- Enables use with fake-gcs-server or other GCS-compatible emulators/proxies
- Anonymous authentication used automatically for custom endpoints (typical for emulators)

Usage:
```bash
# fake-gcs-server (local emulator)
export GCS_ENDPOINT_URL=http://localhost:4443
sai3-bench util ls gs://testbucket/

# Using STORAGE_EMULATOR_HOST convention
export STORAGE_EMULATOR_HOST=localhost:4443
sai3-bench util ls gs://testbucket/

# Multi-protocol proxy
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


---

**For v0.9.0 and earlier releases, see [archive/Changelog_pre_v090.md](archive/Changelog_pre_v090.md)**
