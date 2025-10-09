# s3dlio Python API - v0.9.3 Addendum

**Version**: 0.9.3  
**Date**: October 2025  
**Status**: Non-breaking enhancements to v0.9.2

---

## Table of Contents

- [What's New in v0.9.3](#whats-new-in-v093)
- [Universal API Enhancements](#universal-api-enhancements)
- [RangeEngine Benefits](#rangeengine-benefits)
- [Debug Logging](#debug-logging)
- [Migration from v0.9.2](#migration-from-v092)

---

## What's New in v0.9.3

### Universal API Support

All Python API functions now work universally across **all 5 backends** (S3, Azure, GCS, file://, direct://):

**Updated Functions:**
- `put()`: Template parameter now optional (default: "object-{}")
- `get()`: Universal implementation via ObjectStore
- `delete()`: Works with all URI schemes, supports pattern matching
- All functions automatically detect backend from URI scheme

### RangeEngine for Azure & GCS

Azure Blob Storage and Google Cloud Storage downloads now use concurrent range requests for large files (≥4MB), delivering 30-50% throughput improvements with **zero code changes**.

---

## Universal API Enhancements

### put() - Universal Upload

**Before v0.9.3 (S3-only):**
```python
import s3dlio

# Only worked with s3:// URIs
s3dlio.put("s3://bucket/prefix/", num=3, template="object-{}", size=1024*1024)
```

**After v0.9.3 (Universal):**
```python
import s3dlio

# Works with all backends - template is now optional
s3dlio.put("s3://bucket/prefix/", num=3, size=1024*1024)           # S3
s3dlio.put("az://account/container/prefix/", num=3, size=1024*1024)  # Azure
s3dlio.put("gs://bucket/prefix/", num=3, size=1024*1024)           # GCS
s3dlio.put("file:///local/path/", num=3, size=1024*1024)           # Local
s3dlio.put("direct:///local/path/", num=3, size=1024*1024)         # DirectIO

# Template still works (optional)
s3dlio.put("gs://bucket/data/", num=5, template="train-{}.bin", size=2*1024*1024)
```

**Signature:**
```python
def put(
    prefix: str,              # URI prefix (any scheme)
    num: int,                 # Number of objects to create
    template: str = None,     # Optional: defaults to "object-{}"
    max_in_flight: int = 64,
    size: int = None,         # Required: size in bytes
    should_create_bucket: bool = False,
    object_type: str = "zeros",
    dedup_factor: int = 1,
    compress_factor: int = 1,
    data_gen_mode: str = "streaming",
    chunk_size: int = 262144
) -> None
```

### get() - Universal Download

**Before v0.9.3 (Limited backends):**
```python
# Only worked reliably with s3:// URIs
data = s3dlio.get("s3://bucket/file.bin")
```

**After v0.9.3 (Universal):**
```python
# Works with all backends
data = s3dlio.get("s3://bucket/file.bin")                    # S3
data = s3dlio.get("az://account/container/file.bin")         # Azure (with RangeEngine!)
data = s3dlio.get("gs://bucket/file.bin")                    # GCS (with RangeEngine!)
data = s3dlio.get("file:///local/path/file.bin")             # Local
data = s3dlio.get("direct:///local/path/file.bin")           # DirectIO

# Returns BytesView (zero-copy wrapper)
print(f"Downloaded {len(data)} bytes")

# Convert to bytes if needed
bytes_data = bytes(data)
```

**Performance:**
- Azure: 16.54 MB/s (8MB files with RangeEngine)
- GCS: 44-46 MB/s (128MB files, 2 concurrent ranges)
- Automatic RangeEngine for files ≥ 4MB

### delete() - Universal Delete

**Before v0.9.3 (S3-focused):**
```python
# Pattern matching was limited
s3dlio.delete(["s3://bucket/file1.bin", "s3://bucket/file2.bin"])
```

**After v0.9.3 (Universal with patterns):**
```python
# Works with all backends
s3dlio.delete(["s3://bucket/file.bin"])                      # S3
s3dlio.delete(["az://account/container/file.bin"])           # Azure
s3dlio.delete(["gs://bucket/file.bin"])                      # GCS
s3dlio.delete(["file:///local/path/file.bin"])               # Local
s3dlio.delete(["direct:///local/path/file.bin"])             # DirectIO

# Pattern matching works on all backends
s3dlio.delete(["gs://bucket/prefix/*"])                      # Delete all matching
s3dlio.delete(["az://container/data/*.tmp"])                 # Delete temp files

# Multiple URIs
s3dlio.delete([
    "gs://bucket/file1.bin",
    "gs://bucket/file2.bin",
    "gs://bucket/file3.bin"
])
```

### get_range() - Partial Reads

**Universal across all backends:**
```python
# Read bytes 1024-2047 (1KB starting at offset 1024)
data = s3dlio.get_range("gs://bucket/file.bin", offset=1024, length=1024)

# Works with all backends
data = s3dlio.get_range("az://container/file.bin", offset=0, length=1024*1024)
data = s3dlio.get_range("file:///path/file.bin", offset=5*1024*1024, length=10*1024*1024)
```

---

## RangeEngine Benefits

### Automatic Performance Improvements

v0.9.3 automatically uses RangeEngine for large files on Azure and GCS:

```python
import s3dlio

# Small file (< 4MB): Simple download
data = s3dlio.get("az://container/small-2mb.bin")
# Uses sequential download (fastest for small files)

# Large file (≥ 4MB): RangeEngine
data = s3dlio.get("az://container/large-128mb.bin")
# Automatically uses 2+ concurrent range requests
# 30-50% faster on high-bandwidth networks
```

**No code changes needed** - RangeEngine activates automatically based on file size.

### Performance Validation

**Azure Blob Storage:**
```python
import s3dlio
import time

# 8MB file test
start = time.time()
data = s3dlio.get("az://account/container/test-8mb.bin")
elapsed = time.time() - start
print(f"Downloaded {len(data)} bytes in {elapsed:.2f}s = {len(data)/1024/1024/elapsed:.2f} MB/s")
# Result: ~16-17 MB/s with RangeEngine
```

**Google Cloud Storage:**
```python
import s3dlio
import time

# 128MB file test (creates 2 concurrent ranges)
start = time.time()
data = s3dlio.get("gs://bucket/test-128mb.bin")
elapsed = time.time() - start
print(f"Downloaded {len(data)} bytes in {elapsed:.2f}s = {len(data)/1024/1024/elapsed:.2f} MB/s")
# Result: ~44-46 MB/s with RangeEngine
```

---

## Debug Logging

### Enable Debug Output

See RangeEngine activity and performance metrics:

```python
import s3dlio

# Enable debug logging at start of script
s3dlio.init_logging("debug")

# Download operations now show detailed logs
data = s3dlio.get("gs://bucket/large-file.bin")
```

**Example output:**
```
DEBUG GCS STAT: bucket=bucket, object=large-file.bin
DEBUG GCS STAT success: 134217728 bytes
DEBUG GCS object size 134217728 >= threshold 4194304, using RangeEngine
DEBUG Object size 134217728 >= threshold 4194304, using concurrent ranges
DEBUG Splitting 134217728 bytes into 2 ranges of ~67108864 bytes each
DEBUG GCS GET RANGE: bucket=bucket, object=large-file.bin, offset=0, length=Some(67108864)
DEBUG GCS GET RANGE: bucket=bucket, object=large-file.bin, offset=67108864, length=Some(67108864)
INFO Downloaded 134217728 bytes in 2 ranges: 46.27 MB/s (0.39 Gbps)
INFO RangeEngine (GCS) downloaded 134217728 bytes in 2 ranges: 46.27 MB/s (0.39 Gbps)
```

### Logging Levels

```python
s3dlio.init_logging("trace")   # Most verbose (all debug details)
s3dlio.init_logging("debug")   # Debug information (recommended for troubleshooting)
s3dlio.init_logging("info")    # Performance metrics only
s3dlio.init_logging("warn")    # Warnings only (default)
s3dlio.init_logging("error")   # Errors only
```

---

## Migration from v0.9.2

### No Code Changes Required

All v0.9.2 Python code works unchanged in v0.9.3:

```python
# v0.9.2 code (still works in v0.9.3)
import s3dlio

s3dlio.put("s3://bucket/prefix/", num=3, template="object-{}", size=1024*1024)
data = s3dlio.get("s3://bucket/file.bin")
s3dlio.delete(["s3://bucket/file.bin"])
```

### Optional: Leverage New Features

Take advantage of universal API and simplified parameters:

```python
# v0.9.3 simplified syntax
import s3dlio

# Template is now optional (defaults to "object-{}")
s3dlio.put("s3://bucket/prefix/", num=3, size=1024*1024)

# Works with all backends
s3dlio.put("gs://bucket/prefix/", num=5, size=2*1024*1024)
data = s3dlio.get("az://container/file.bin")  # Automatically uses RangeEngine!
s3dlio.delete(["gs://bucket/temp/*"])         # Pattern matching
```

### Use Debug Logging

Add at the start of your script to see RangeEngine activity:

```python
import s3dlio

# Enable debug logging
s3dlio.init_logging("debug")

# Your existing code unchanged
data = s3dlio.get("gs://bucket/large-file.bin")
# Debug logs show RangeEngine activity
```

---

## Complete Example

```python
#!/usr/bin/env python3
import s3dlio
import time

# Enable debug logging to see RangeEngine activity
s3dlio.init_logging("debug")

print("=== Testing Universal API with RangeEngine ===")

# Test 1: Upload to GCS (universal API)
print("\n1. Uploading 3 files to GCS...")
s3dlio.put(
    "gs://my-bucket/test-data/",
    num=3,
    size=5 * 1024 * 1024,  # 5MB each
    object_type="random"
)
print("✅ Upload complete")

# Test 2: Download with RangeEngine (automatic for files ≥ 4MB)
print("\n2. Downloading 5MB file (RangeEngine active)...")
start = time.time()
data = s3dlio.get("gs://my-bucket/test-data/object-0")
elapsed = time.time() - start
throughput = (len(data) / 1024 / 1024) / elapsed
print(f"✅ Downloaded {len(data)} bytes in {elapsed:.2f}s = {throughput:.2f} MB/s")

# Test 3: Range request (universal)
print("\n3. Partial read (first 1MB)...")
partial = s3dlio.get_range("gs://my-bucket/test-data/object-0", offset=0, length=1024*1024)
print(f"✅ Read {len(partial)} bytes")

# Test 4: Pattern delete (universal)
print("\n4. Cleaning up...")
s3dlio.delete(["gs://my-bucket/test-data/*"])
print("✅ Cleanup complete")

print("\n=== All tests passed! ===")
```

---

## Summary

v0.9.3 Python API enhancements:
- ✅ **Universal**: All functions work with all 5 backends
- ✅ **Simplified**: Template parameter optional in `put()`
- ✅ **Faster**: Automatic RangeEngine for Azure/GCS large files
- ✅ **Zero breaking changes**: All v0.9.2 code works unchanged
- ✅ **Observable**: Debug logging shows RangeEngine activity

**Recommendation**: Existing code works unchanged. Optionally add debug logging to observe RangeEngine performance.

---

**For complete v0.9.2 API reference, see**: [python-api-v0.9.2.md](python-api-v0.9.2.md)
