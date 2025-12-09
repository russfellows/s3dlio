# GCS Backend Selection Guide

## Overview

s3dlio supports **two Google Cloud Storage (GCS) implementations** that can be selected at compile time via Cargo features:

1. **`gcs-community`** (default) - Community-maintained `gcloud-storage` crate
2. **`gcs-official`** (experimental) - Official Google `google-cloud-storage` crate

Both implementations expose **identical APIs**, allowing A/B testing and benchmarking without code changes.

## Current Status

| Backend | Status | Recommendation |
|---------|--------|---------------|
| `gcs-community` | ✅ **Production Ready** | **Use this** (default) |
| `gcs-official` | ⚠️ **Experimental** | Has known transport flakes |

### Known Issues

**gcs-official Backend:**
- ✅ Individual operations work correctly when tested in isolation (100% pass rate)
- ⚠️ Full test suite has intermittent "transport error" failures (10-20% failure rate)
- 🐛 **Root cause**: Upstream HTTP/2 connection pool flake in `google-cloud-rust` library
  - **Our Bug Report**: https://github.com/googleapis/google-cloud-rust/issues/3574
  - **Related Upstream Issue**: https://github.com/googleapis/google-cloud-rust/issues/3412
- ✅ **MAJOR IMPROVEMENT**: Implemented global client singleton pattern
  - **Before**: 30% test pass rate (3/10 tests)
  - **After**: 80-90% test pass rate (8-9/10 tests)
  - **Implementation**: Used `once_cell::Lazy<OnceCell<GcsClient>>` for single client per process
  - **Proof**: Improvement confirms issue is client/connection pool lifecycle
- 🔧 **Best Practices Implemented**:
  - Global client singleton via `Lazy<OnceCell>` (recommended pattern from Google docs)
  - Correct bucket format (`projects/_/buckets/{bucket}`)
  - Sequential test execution (`--test-threads=1`)
- 📊 **Current Failure Pattern**: 
  - Most operations now work reliably (80-90% success rate)
  - Remaining failures are intermittent and transport-layer related
  - Storage client operations (read_object, write_object): ✅ Always work
  - StorageControl operations: ✅ Mostly work now (was: consistently failed in suites)
  - Error message: "the transport reports an error: transport error" or "cannot serialize request"

### What Works in gcs-official (When Not Hitting Transport Errors)

✅ **All operations function correctly** when transport layer succeeds:
- **Authentication**: Application Default Credentials (ADC) ✅ 100% reliable
- **Object Operations** (via Storage client - always works):
  - PUT object (write_object): ✅ 100% reliable
  - GET object (read_object): ✅ 100% reliable  
  - GET range (read_object with range): ✅ 100% reliable
- **Metadata/Control Operations** (via StorageControl client - 80-90% reliable):
  - STAT object (get_object metadata): ✅ Works when transport succeeds
  - LIST objects recursive: ✅ Works when transport succeeds
  - LIST objects non-recursive with delimiter: ✅ Works when transport succeeds (v0.9.8+)
  - DELETE single object: ✅ Works when transport succeeds
  - DELETE multiple objects (batch): ✅ Works when transport succeeds
- **Non-Recursive Listing** (v0.9.8+):
  - ✅ Returns files + subdirectory prefixes (ending with "/")
  - ✅ Uses `.set_delimiter("/")` and `.by_page()` to access full response
  - ✅ Matches S3 and gcs-community behavior exactly

### What Does NOT Work in gcs-official

❌ **Transport layer flakes** (intermittent, 10-20% of operations):
- Error: "transport reports an error: transport error"
- Error: "cannot serialize request"
- **NOT operation-specific**: All operations can fail with transport errors
- **NOT reproducible**: Same operation succeeds when retried
- **Root cause**: Upstream `google-cloud-rust` HTTP/2 connection pool issue
- **Workaround**: Retry logic (not implemented yet), or use gcs-community backend

**Recommendation**: Use `gcs-community` for production (100% reliable); `gcs-official` acceptable for development/testing but expect occasional flakes (10-20%) until upstream issue resolved

## Build Options

### Default Build (Community Backend)
```bash
cargo build --release
# or explicitly:
cargo build --release --features gcs-community
```

### Official Backend (Experimental)
```bash
cargo build --release --no-default-features \
  --features native-backends,s3,gcs-official
```

### ❌ Cannot Use Both
```bash
# This will fail with compile error:
cargo build --features gcs-community,gcs-official
# Error: "Enable only one of: gcs-community or gcs-official"
```

## Testing Against Real GCS

### Prerequisites

1. **GCS Test Bucket**: Create or use an existing bucket for testing
2. **Authentication**: Set up one of:
   - Service account: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`
   - gcloud CLI: `gcloud auth application-default login`
   - GCE/GKE: Automatic via metadata server

### Running Tests

#### Quick Test (Community Backend Only)
```bash
export GCS_TEST_BUCKET=your-test-bucket
cargo test --test test_gcs_functional --release -- --nocapture
```

#### Comprehensive Test (Both Backends)
```bash
export GCS_TEST_BUCKET=your-test-bucket
./scripts/test_gcs_backends.sh
```

This script will:
1. ✅ Build and test `gcs-community` (should pass 10/10 tests)
2. ⚠️ Build and test `gcs-official` (will fail 7/10 tests due to upstream transport flake)
3. Show comparison summary

### What Gets Tested

The functional tests cover all GCS operations:
- ✅ Authentication (Application Default Credentials)
- ✅ PUT object (single upload)
- ✅ GET object (full download)
- ✅ GET object range (partial download)
- ✅ STAT object (metadata/HEAD)
- ✅ LIST objects (with/without prefix, recursive/non-recursive)
- ✅ DELETE object (single)
- ✅ DELETE objects (batch with adaptive concurrency)
- ✅ Multipart upload (large files)
- ✅ URI parsing (gs:// and gcs:// schemes)

### Test Output Example

```
╔══════════════════════════════════════════════════════════════╗
║  Comprehensive GCS Functional Test Suite                    ║
║  Backend: gcs-community (gcloud-storage)                     ║
╚══════════════════════════════════════════════════════════════╝

=== Testing GCS Authentication ===
✓ GCS client created successfully with Application Default Credentials

=== Testing GCS PUT and GET Operations ===
Uploading object: gs://my-bucket/s3dlio-test/test-put-get.txt
✓ PUT successful: 35 bytes
Downloading object: gs://my-bucket/s3dlio-test/test-put-get.txt
✓ GET successful: 35 bytes
✓ Data integrity verified
✓ Cleanup successful

[... more tests ...]

╔══════════════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED                                          ║
╚══════════════════════════════════════════════════════════════╝
```

## Authentication Setup

### Option 1: Service Account (Recommended for CI/CD)
```bash
# Download service account JSON from Google Cloud Console
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Option 2: gcloud CLI (Recommended for Development)
```bash
# Login once
gcloud auth application-default login

# Tests will automatically use these credentials
```

### Option 3: GCE/GKE Metadata Server
If running on Google Cloud infrastructure, authentication is automatic.

## Official Backend Implementation Status

The `gcs-official` backend is **fully implemented** in `src/google_gcs_client.rs` but has known reliability issues.

**Implementation Details:**
- ✅ Uses official `google-cloud-storage` v1.1.0 and `google-cloud-gax` v1.1.0 crates
- ✅ Storage client operations: `read_object()`, `write_object()` (fully working)
- ⚠️ StorageControl operations: `delete_object()`, `get_object()`, `list_objects()` (flaky in test suites)
- ✅ Proper ADC (Application Default Credentials) support
- ✅ Global client caching via `OnceCell` pattern (recommended best practice)
- ✅ Correct API usage per official documentation

**Why Not Use It?**

While fully implemented and technically correct:
- ❌ **Reliability Issue**: Transport errors in test suites (see Known Issues above)
- ❌ **Upstream Bug**: Waiting for google-cloud-rust team to fix connection pool flake
- ✅ **Alternative Available**: Community backend (`gcloud-storage`) is stable and production-ready
- 🔮 **Future**: Will become default once upstream Issue #3574 is resolved

**When to Consider Using It:**
- You need to test compatibility with official Google SDK
- You want to help debug upstream issues
- You're evaluating future migration once transport flake is fixed
- You're running single operations in isolation (individual ops work reliably)

## Performance Comparison (Future Work)

Once `gcs-official` is implemented, you can benchmark both backends:

```bash
# Benchmark community backend
cargo bench --features gcs-community

# Benchmark official backend  
cargo bench --no-default-features --features native-backends,s3,gcs-official
```

## Architecture Details

### How Feature Gating Works

#### Cargo.toml
```toml
[features]
default = ["native-backends", "gcs-community"]
gcs-community = ["dep:gcloud-storage"]
gcs-official = ["dep:google-cloud-storage"]

[dependencies]
gcloud-storage = { version = "^1.1", optional = true }
google-cloud-storage = { version = "^1.1", optional = true }
```

#### src/lib.rs
```rust
#[cfg(feature = "gcs-community")]
pub mod gcs_client;

#[cfg(feature = "gcs-official")]
pub mod google_gcs_client;

// Compile-time check prevents using both
#[cfg(all(feature = "gcs-community", feature = "gcs-official"))]
compile_error!("Enable only one of: gcs-community or gcs-official");
```

#### src/object_store.rs
```rust
#[cfg(feature = "gcs-community")]
use crate::gcs_client::{GcsClient, GcsObjectMetadata, parse_gcs_uri};

#[cfg(feature = "gcs-official")]
use crate::google_gcs_client::{GcsClient, GcsObjectMetadata, parse_gcs_uri};

// Rest of code uses GcsClient without knowing which backend
```

### Key Design Principles

1. **Identical API**: Both implementations expose the same `GcsClient` struct and methods
2. **Zero Code Changes**: Higher-level code doesn't know which backend is used
3. **Compile-Time Selection**: No runtime overhead from feature flags
4. **Mutual Exclusion**: Cannot accidentally enable both backends
5. **Backward Compatible**: Existing `gcs_client.rs` is completely untouched

## Troubleshooting

### Authentication Errors
```
Error: Failed to initialize GCS authentication
```

**Solution**: Set up authentication (see above)

### Bucket Not Found
```
Error: 404 Not Found
```

**Solution**: Check `GCS_TEST_BUCKET` is set correctly and you have access

### Permission Denied
```
Error: 403 Forbidden
```

**Solution**: Ensure your service account/user has these IAM roles:
- `roles/storage.objectAdmin` (for full CRUD operations)
- `roles/storage.bucketReader` (for listing)

### Stub Implementation Errors (Official Backend)
```
Error: google_gcs_client::get_object not yet implemented
```

**Expected**: The official backend is stub-only. Use `gcs-community` instead.

## Recommendation

**For production use, stick with `gcs-community` (default).** It's:
- ✅ Well-tested and stable
- ✅ Actively maintained
- ✅ Simpler API
- ✅ Already integrated and working

Only implement `gcs-official` if you encounter specific issues or need features that only the official SDK provides.
