# s3dlio Changelog (v0.9.0 and Earlier)

> **Archived**: December 2025  
> **Covers**: v0.9.0 through v0.8.0  
> **Current changelog**: [../Changelog.md](../Changelog.md) (v0.9.1+)

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
