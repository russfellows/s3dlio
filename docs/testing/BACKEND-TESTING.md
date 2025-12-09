# Backend Testing Guide

**Last Updated:** October 9, 2025  
**Version:** v0.9.4  
**Status:** All 5 backends production-ready

---

## Overview

s3dlio supports **5 storage backends** with comprehensive testing for correctness, performance, and edge cases:
- **S3** (AWS S3, MinIO, Vast)
- **Azure Blob Storage**
- **Google Cloud Storage (GCS)**
- **Local Filesystem** (file://)
- **DirectIO** (direct://)

This guide consolidates testing approaches, common issues, and validation strategies across all backends.

---

## Pagination Testing

### Summary
All backends correctly handle pagination for lists exceeding default page limits (typically 1000-5000 objects).

### Backend-Specific Implementations

#### ✅ S3 - Manual Pagination Loop
**File**: `src/s3_utils.rs:244`  
**Implementation**: Manual continuation token loop

```rust
loop {
    if let Some(token) = &cont {
        req_builder = req_builder.continuation_token(token);
    }
    
    let resp = req_builder.send().await?;
    // collect objects...
    
    if let Some(next) = resp.next_continuation_token() {
        cont = Some(next.to_string());
    } else {
        break;
    }
}
```

**Details**:
- Default page size: ~1000 objects
- Pagination field: `continuation_token` → `next_continuation_token`
- Handles unlimited objects via loop

**Testing**:
```bash
# Create 2000+ objects
for i in {1..2500}; do
    echo "test" | aws s3 cp - s3://bucket/prefix/file_${i}.txt
done

# Verify all listed
s3-cli ls s3://bucket/prefix/ | wc -l  # Should show 2500
```

---

#### ✅ GCS - Manual Page Token Loop (Fixed v0.8.22)
**File**: `src/gcs_client.rs:252`  
**Status**: Fixed pagination bug in v0.8.22

**Bug History**: Prior to v0.8.22, GCS list operations were limited to 1000 objects (missing pagination loop).

**Current Implementation**:
```rust
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

**Details**:
- Default page size: 1000 objects (GCS API default)
- Pagination field: `page_token` → `next_page_token`
- Handles unlimited objects via loop
- **API Reference**: [GCS Objects.list](https://cloud.google.com/storage/docs/json_api/v1/objects/list)

**Testing**:
```bash
# Create 1500 objects
for i in {1..1500}; do
    echo "test" | gsutil cp - gs://bucket/prefix/file_${i}.txt
done

# Verify all listed
s3-cli ls gs://bucket/prefix/ | wc -l  # Should show 1500
```

---

#### ✅ Azure Blob Storage - SDK-Managed Pagination
**File**: `src/azure_client.rs:177`  
**Implementation**: Azure SDK's PageIterator (automatic)

```rust
let mut pager = container.list_blobs(Some(opts))?;
let mut out = Vec::new();

while let Some(next) = pager.next().await {
    let resp = next?;
    let body: ListBlobsFlatSegmentResponse = resp.into_body().await?;
    for it in body.segment.blob_items {
        if let Some(name) = it.name.and_then(|bn| bn.content) {
            out.push(name);
        }
    }
}
```

**Details**:
- Default page size: 5000 blobs (Azure API default)
- Pagination: Automatic via SDK PageIterator
- Azure SDK handles continuation tokens internally

**Testing**:
```bash
# Create 6000 blobs
for i in {1..6000}; do
    echo "test" | az storage blob upload --container bucket --name prefix/file_${i}.txt --data @-
done

# Verify all listed
s3-cli ls az://bucket/prefix/ | wc -l  # Should show 6000
```

---

#### ✅ Local Filesystem - No Pagination Needed
**File**: `src/file_store.rs`  
**Implementation**: Direct filesystem traversal

**Details**:
- Uses `WalkDir` crate for recursive directory traversal
- No pagination needed (OS handles filesystem iteration)
- Performance: O(n) with filesystem caching

**Testing**:
```bash
# Create 10000 files
mkdir -p /tmp/test/prefix
for i in {1..10000}; do
    echo "test" > /tmp/test/prefix/file_${i}.txt
done

# Verify all listed
s3-cli ls file:///tmp/test/prefix/ | wc -l  # Should show 10000
```

---

#### ✅ DirectIO - No Pagination Needed
**File**: `src/directio_store.rs`  
**Implementation**: Same as Local Filesystem (uses WalkDir)

**Details**:
- Identical to file:// for list operations
- Difference only in read/write paths (O_DIRECT flag)
- No pagination concerns

---

## Concurrent Range Download Testing

All backends support **concurrent range downloads** for files > 4MB, providing 30-50% performance improvement.

### Implementation Summary

| Backend | Implementation | File | Test Status |
|---------|---------------|------|-------------|
| S3 | Custom `concurrent_range_get_impl()` | `src/s3_utils.rs` | ✅ Tested |
| Azure | Generic `RangeEngine` | `src/range_engine_generic.rs` | ✅ Tested |
| GCS | Generic `RangeEngine` | `src/range_engine_generic.rs` | ✅ Tested |
| File | Generic `RangeEngine` | `src/range_engine_generic.rs` | ✅ Tested |
| DirectIO | Generic `RangeEngine` | `src/range_engine_generic.rs` | ✅ Tested |

**See Also**: `docs/v0.9.4_BackEnd_Range-Summary.md` for comprehensive analysis.

### Testing Range Downloads

```bash
# Create large test file (100MB)
dd if=/dev/urandom of=/tmp/test_100mb.bin bs=1M count=100

# Upload to each backend
s3-cli put /tmp/test_100mb.bin s3://bucket/test_100mb.bin
s3-cli put /tmp/test_100mb.bin az://container/test_100mb.bin
s3-cli put /tmp/test_100mb.bin gs://bucket/test_100mb.bin

# Download and verify (automatic range detection)
s3-cli get s3://bucket/test_100mb.bin /tmp/s3_download.bin
s3-cli get az://container/test_100mb.bin /tmp/azure_download.bin
s3-cli get gs://bucket/test_100mb.bin /tmp/gcs_download.bin

# Verify checksums match
sha256sum /tmp/test_100mb.bin /tmp/*_download.bin
```

---

## Python API Testing

### Testing All Backends

```python
import s3dlio

# Test S3
s3dlio.upload(['test.txt'], 's3://bucket/prefix/')
s3dlio.download('s3://bucket/prefix/*.txt', './s3_local/')

# Test Azure
s3dlio.upload(['test.txt'], 'az://container/prefix/')
s3dlio.download('az://container/prefix/*.txt', './azure_local/')

# Test GCS
s3dlio.upload(['test.txt'], 'gs://bucket/prefix/')
s3dlio.download('gs://bucket/prefix/*.txt', './gcs_local/')

# Test Local
s3dlio.upload(['test.txt'], 'file:///tmp/prefix/')
s3dlio.download('file:///tmp/prefix/*.txt', './local_copy/')

# Test DirectIO
s3dlio.upload(['test.txt'], 'direct:///tmp/directio/prefix/')
s3dlio.download('direct:///tmp/directio/prefix/*.txt', './directio_copy/')
```

### DataLoader Integration Testing

```python
from s3dlio import S3IterableDataset
from torch.utils.data import DataLoader

# Test each backend
datasets = [
    S3IterableDataset("s3://bucket/data/*.bin", batch_size=32),
    S3IterableDataset("az://container/data/*.bin", batch_size=32),
    S3IterableDataset("gs://bucket/data/*.bin", batch_size=32),
    S3IterableDataset("file:///data/*.bin", batch_size=32),
]

for dataset in datasets:
    loader = DataLoader(dataset, batch_size=None, num_workers=4)
    for batch in loader:
        print(f"Batch shape: {len(batch)}")
        break  # Test first batch only
```

---

## Common Testing Scenarios

### 1. Empty Prefix/Bucket
```bash
# Should return empty list, not error
s3-cli ls s3://bucket/nonexistent_prefix/
s3-cli ls az://container/nonexistent_prefix/
s3-cli ls gs://bucket/nonexistent_prefix/
```

### 2. Special Characters in Object Names
```bash
# Test with spaces, unicode, special chars
s3-cli put test.txt "s3://bucket/prefix/file with spaces.txt"
s3-cli put test.txt "s3://bucket/prefix/файл.txt"  # Cyrillic
s3-cli put test.txt "s3://bucket/prefix/文件.txt"  # Chinese
```

### 3. Large Object Counts (10k+ objects)
```bash
# Verify no pagination issues
for i in {1..15000}; do
    echo "$i" | s3-cli put - s3://bucket/stress/file_${i}.txt
done

# Should return 15000
s3-cli ls s3://bucket/stress/ | wc -l
```

### 4. Concurrent Operations
```bash
# Test concurrent uploads (all backends)
seq 1 100 | xargs -P 16 -I {} sh -c '
    echo "data" | s3-cli put - s3://bucket/concurrent/file_{}.txt
'

# Verify all 100 uploaded
s3-cli ls s3://bucket/concurrent/ | wc -l
```

### 5. Range Download Edge Cases
```bash
# Small files (< 4MB) - should NOT use range engine
dd if=/dev/urandom of=/tmp/small.bin bs=1M count=2
s3-cli put /tmp/small.bin s3://bucket/small.bin
s3-cli get s3://bucket/small.bin /tmp/small_download.bin

# Large files (> 4MB) - should use range engine
dd if=/dev/urandom of=/tmp/large.bin bs=1M count=50
s3-cli put /tmp/large.bin s3://bucket/large.bin
s3-cli get s3://bucket/large.bin /tmp/large_download.bin

# Verify both methods produce correct results
sha256sum /tmp/small.bin /tmp/small_download.bin
sha256sum /tmp/large.bin /tmp/large_download.bin
```

---

## Regression Testing

### Test Suite Execution
```bash
# Rust tests (all backends)
cargo test --release --lib

# Python tests
./build_pyo3.sh && ./install_pyo3_wheel.sh
python tests/test_functionality.py
python tests/test_modular_api_regression.py

# CLI tests
./scripts/test_all.sh
```

### Backend-Specific Test Commands
```bash
# S3 backend tests
AWS_ENDPOINT_URL=http://localhost:9000 cargo test s3_ --release

# Azure backend tests (requires credentials)
AZURE_STORAGE_ACCOUNT=myaccount cargo test azure_ --release

# GCS backend tests (requires credentials)
GOOGLE_APPLICATION_CREDENTIALS=key.json cargo test gcs_ --release

# Local filesystem tests (no credentials needed)
cargo test file_ --release
cargo test directio_ --release
```

---

## Performance Benchmarking

### Backend Comparison Script
```bash
# Build all variants
./scripts/build_performance_variants.sh

# Run comparative benchmarks
./scripts/run_backend_comparison.sh
```

### Manual Performance Testing
```bash
# Upload throughput (target: 2.5 GB/s)
dd if=/dev/zero bs=1M count=1000 | pv | s3-cli put - s3://bucket/1gb.bin

# Download throughput (target: 5 GB/s)
s3-cli get s3://bucket/1gb.bin - | pv > /dev/null

# List performance (10k objects)
time s3-cli ls s3://bucket/prefix/
```

---

## Troubleshooting

### GCS Pagination Issues (Pre-v0.8.22)
**Symptom**: Only 1000 objects returned from gs:// URIs  
**Solution**: Upgrade to v0.8.22+ (pagination bug fixed)

### Azure Timeout Errors
**Symptom**: Timeout errors during large operations  
**Solution**: Increase Azure SDK timeout in `src/azure_client.rs`

### S3 Connection Pool Exhaustion
**Symptom**: "Too many open connections" errors  
**Solution**: Tune `max_idle_connections` in `src/config.rs`

### DirectIO Alignment Errors
**Symptom**: "Invalid argument" errors with direct:// URIs  
**Solution**: Ensure files are 4KB-aligned (automatic in v0.9.0+)

---

## Test Coverage Status

| Feature | S3 | Azure | GCS | File | DirectIO |
|---------|-----|-------|-----|------|----------|
| List (< 1k objects) | ✅ | ✅ | ✅ | ✅ | ✅ |
| List (> 1k objects) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Range downloads | ✅ | ✅ | ✅ | ✅ | ✅ |
| Concurrent uploads | ✅ | ✅ | ✅ | ✅ | ✅ |
| Special characters | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python API | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| PyTorch DataLoader | ✅ | ✅ | ⚠️ | ✅ | ✅ |

**Legend**: ✅ Tested | ⚠️ Needs testing | ❌ Not supported

---

## See Also

- **TESTING-GUIDE.md** - General testing procedures
- **v0.9.4_BackEnd_Range-Summary.md** - RangeEngine implementation details
- **docs/api/python-api-v0.9.3-addendum.md** - Python API reference
- **docs/api/rust-api-v0.9.3-addendum.md** - Rust API reference
