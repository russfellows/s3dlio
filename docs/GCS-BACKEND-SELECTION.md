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
| `gcs-official` | ⚠️ **Stub Only** | Needs implementation |

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
1. ✅ Build and test `gcs-community` (should pass)
2. ⚠️ Build and test `gcs-official` (will fail - stub only)
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

## Implementing the Official Backend

The `gcs-official` backend is currently a **stub implementation** in `src/google_gcs_client.rs`. To implement it:

1. **Review the stub** - All methods return `bail!("not yet implemented")`
2. **Implement using official crate**:
   ```rust
   use google_cloud_storage::client::Storage;
   use google_cloud_storage::model::Object;
   // etc.
   ```
3. **Match the existing API** - Signatures must match `gcs_client.rs` exactly
4. **Run tests** - Use the same `test_gcs_functional.rs` tests

### Why Not Implement It Yet?

The community backend (`gcloud-storage`) works well and has proven stable. The official backend should only be implemented if:
- Community backend becomes unmaintained
- You need bleeding-edge GCS features (managed folders, etc.)
- Official backend shows measurable performance benefits

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
