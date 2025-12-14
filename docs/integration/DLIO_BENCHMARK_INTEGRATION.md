# DLIO Benchmark Integration Guide

**Version:** 0.9.26+  
**Last Updated:** December 13, 2025

## Overview

This guide explains how to integrate s3dlio with the [Argonne DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark) as a storage backend. s3dlio provides high-performance multi-protocol storage supporting S3, Azure Blob, GCS, and local filesystems.

> **Note:** This integration requires s3dlio v0.9.26 or later for the `exists()`, `put_bytes()`, and `mkdir()` functions.

## Why Use s3dlio with DLIO?

| Feature | s3torchconnector | s3dlio |
|---------|------------------|--------|
| Protocols | S3 only | S3, Azure, GCS, file://, direct:// |
| Performance | Good | Optimized (5+ GB/s) |
| Range requests | Yes | Yes |
| Multi-endpoint | No | Yes (load balancing) |
| O_DIRECT support | No | Yes |
| Rust backend | C++ (CRT) | Rust (async) |
| Custom endpoints | Limited | Full (MinIO, Ceph, etc.) |

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DLIO Benchmark                                │
├─────────────────────────────────────────────────────────────────┤
│  Data Generator  │  Data Loader  │  Checkpoint  │  Profiling    │
├─────────────────────────────────────────────────────────────────┤
│                    StorageFactory                                │
├──────────────────┬──────────────────┬───────────────────────────┤
│   FileStorage    │ S3TorchConnector │    S3dlioStorage          │
│   (local fs)     │   (S3 only)      │    (multi-protocol)       │
├──────────────────┴──────────────────┴───────────────────────────┤
│                         s3dlio                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐ │
│  │   S3    │ │   GCS   │ │  Azure  │ │ file:// │ │ direct://  │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Two Integration Options

We provide **two ways** to integrate s3dlio with DLIO:

| Option | Approach | DLIO Changes | Best For |
|--------|----------|--------------|----------|
| **Option 1** | New storage type `s3dlio` | Small patch (2 files) | Production use, clear separation |
| **Option 2** | Drop-in replacement | None (overwrites file) | Quick testing, can't modify DLIO |

---

## Option 1: New Storage Type (Recommended)

This adds a new `storage_type: s3dlio` option to DLIO, keeping the original S3 storage intact.

### Step 1: Install s3dlio

```bash
pip install s3dlio
```

### Step 2: Copy Storage File

```bash
# Automatic (using Python helper)
python3 -c "
from s3dlio.integrations.dlio import install_s3dlio_storage
install_s3dlio_storage('/path/to/dlio_benchmark')
"

# Or manual copy
cp $(python3 -c "from s3dlio.integrations.dlio import get_s3dlio_storage_path; print(get_s3dlio_storage_path())") \
   /path/to/dlio_benchmark/storage/s3dlio_storage.py
```

### Step 3: Add Storage Type Enum

Edit `dlio_benchmark/common/enumerations.py`:

```python
class StorageType(Enum):
    """
    Different types of underlying storage
    """
    LOCAL_FS = 'local_fs'
    PARALLEL_FS = 'parallel_fs'
    S3 = 's3'
    S3DLIO = 's3dlio'  # <-- ADD THIS LINE

    def __str__(self):
        return self.value
```

### Step 4: Register in Factory

Edit `dlio_benchmark/storage/storage_factory.py`:

```python
@staticmethod
def get_storage(storage_type, namespace, framework=None):
    if storage_type == StorageType.LOCAL_FS:
        return FileStorage(namespace, framework)
    
    # ADD THIS BLOCK (before the S3 elif)
    elif storage_type == StorageType.S3DLIO:
        from dlio_benchmark.storage.s3dlio_storage import S3dlioStorage
        return S3dlioStorage(namespace, framework)
    
    elif storage_type == StorageType.S3:
        # ... existing S3 code ...
```

### Step 5: Update Your Config

```yaml
storage:
  storage_type: s3dlio              # Use s3dlio backend
  storage_root: s3://bucket/prefix  # S3
  # storage_root: az://container/prefix  # Azure
  # storage_root: gs://bucket/prefix     # GCS  
  # storage_root: file:///mnt/data       # Local filesystem
```

### Verification

```bash
# Run DLIO - it will use s3dlio for storage
dlio_benchmark workload=unet3d ++workload.storage.storage_type=s3dlio
```

---

## Option 2: Drop-in Replacement

This replaces DLIO's `s3_torch_storage.py` with an s3dlio-based implementation. The class is still named `S3PyTorchConnectorStorage` for compatibility.

### Step 1: Install s3dlio

```bash
pip install s3dlio
```

### Step 2: Install Drop-in Replacement

```bash
python3 -c "
from s3dlio.integrations.dlio import install_dropin_replacement
install_dropin_replacement('/path/to/dlio_benchmark')
"
```

This will:
- Backup the original to `s3_torch_storage.py.original.bak`
- Replace with s3dlio-based implementation

### Step 3: Use Existing S3 Config

No config changes needed - use `storage_type: s3` as before:

```yaml
storage:
  storage_type: s3
  storage_root: s3://bucket/prefix
```

### Reverting to Original

```bash
cd /path/to/dlio_benchmark/storage
mv s3_torch_storage.py.original.bak s3_torch_storage.py
```

---

## Environment Variables

### For S3 (including MinIO, Ceph)

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://minio:9000  # For custom endpoints
```

### For Azure Blob Storage

```bash
export AZURE_STORAGE_ACCOUNT_NAME=your_account
export AZURE_STORAGE_ACCOUNT_KEY=your_key
```

### For Google Cloud Storage

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## API Mapping

How DLIO storage methods map to s3dlio functions:

| DLIO Method | s3dlio Function | Description |
|-------------|-----------------|-------------|
| `put_data(id, data)` | `s3dlio.put_bytes(uri, data)` | Write bytes to storage |
| `get_data(id)` | `s3dlio.get(uri)` | Read full object |
| `get_data(id, offset, length)` | `s3dlio.get_range(uri, offset, length)` | Read byte range |
| `walk_node(id)` | `s3dlio.list(uri)` | List objects |
| `isfile(id)` | `s3dlio.exists(uri)` | Check if object exists |
| `create_node(id)` | `s3dlio.mkdir(uri)` | Create directory/prefix |
| `delete_node(id)` | `s3dlio.delete(uri)` | Delete object |
| `get_node(id)` | `s3dlio.stat(uri)` | Get object metadata |

---

## Programmatic Installation

### Python API

```python
from s3dlio.integrations import dlio

# Option 1: New storage type
dlio.install_s3dlio_storage('/path/to/dlio_benchmark')
# Prints manual steps for enum/factory changes

# Option 1: Generate patch file instead
dlio.generate_patch(output_file='dlio_s3dlio.patch')
# Then: patch -p1 < dlio_s3dlio.patch

# Option 2: Drop-in replacement
dlio.install_dropin_replacement('/path/to/dlio_benchmark')

# Get file paths (for manual copying)
print(dlio.get_s3dlio_storage_path())    # Option 1 file
print(dlio.get_storage_file_path())       # Option 2 file
```

---

## Troubleshooting

### "No module named 's3dlio'"

```bash
pip install s3dlio
# Verify
python3 -c "import s3dlio; print(s3dlio.version())"
```

### "Unknown storage type: s3dlio"

You haven't completed Step 3 (enum) or Step 4 (factory) for Option 1.

### "put_bytes not found"

You need s3dlio v0.9.26+. Upgrade:
```bash
pip install --upgrade s3dlio
```

### Azure/GCS authentication errors

Check environment variables are set correctly. For GCS, ensure the service account JSON file exists and is readable.

### Permission denied on S3

Verify AWS credentials have `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` permissions.

---

## Example Configurations

### S3 with MinIO

```yaml
storage:
  storage_type: s3dlio
  storage_root: s3://mybucket/dlio-data

# Set environment:
# AWS_ENDPOINT_URL=http://minio:9000
# AWS_ACCESS_KEY_ID=minioadmin
# AWS_SECRET_ACCESS_KEY=minioadmin
```

### Azure Blob Storage

```yaml
storage:
  storage_type: s3dlio
  storage_root: az://mycontainer/dlio-data

# Set environment:
# AZURE_STORAGE_ACCOUNT_NAME=myaccount
# AZURE_STORAGE_ACCOUNT_KEY=mykey
```

### Local Filesystem with O_DIRECT

```yaml
storage:
  storage_type: s3dlio
  storage_root: direct:///mnt/nvme/dlio-data
```

### Google Cloud Storage

```yaml
storage:
  storage_type: s3dlio
  storage_root: gs://mybucket/dlio-data

# Set environment:
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## How to Tell Which Backend is Active

### Option 1 (storage_type: s3dlio)
Check your config - it explicitly says `storage_type: s3dlio`

### Option 2 (drop-in replacement)
The class is named `S3PyTorchConnectorStorage` but uses s3dlio internally.
Look for `[s3dlio]` prefixes in error messages, or check:
```python
# In DLIO code or debug session
from dlio_benchmark.storage.s3_torch_storage import S3PyTorchConnectorStorage
import inspect
print(inspect.getfile(S3PyTorchConnectorStorage))
# If it mentions s3dlio, you're using the replacement
```

---

## Multi-Protocol Examples

s3dlio supports multiple storage backends with the same API:

```yaml
# AWS S3
dataset:
  data_folder: s3://my-bucket/data

# Google Cloud Storage  
dataset:
  data_folder: gs://my-bucket/data

# Azure Blob Storage
dataset:
  data_folder: az://mycontainer/data

# Local filesystem
dataset:
  data_folder: file:///mnt/data

# Direct I/O (O_DIRECT, bypasses page cache)
dataset:
  data_folder: direct:///mnt/nvme/data
```

---

## Advanced Features

### Multi-Endpoint Load Balancing

For high-performance setups with multiple S3 endpoints:

```python
# Configure multiple endpoints via environment
import os
os.environ['S3_ENDPOINT_URIS'] = 'http://node1:9000,http://node2:9000,http://node3:9000'
os.environ['S3_LOAD_BALANCE_STRATEGY'] = 'round_robin'  # or 'least_connections'
```

### Performance Tuning

```python
import s3dlio

# Configure logging level (reduce noise in production)
s3dlio.init_logging("warn")  # Options: trace, debug, info, warn, error

# For maximum throughput, use async APIs
import asyncio

async def read_batch(uris):
    return await s3dlio.get_many_async(uris)

# Run async code
data = asyncio.run(read_batch(file_uris))
```

---

## Testing the Integration

```python
# Quick test script
import s3dlio
import io
import numpy as np

# Test write
data = np.random.rand(100, 100)
buffer = io.BytesIO()
np.save(buffer, data)
s3dlio.put_bytes("s3://my-bucket/test.npy", buffer.getvalue())

# Test read
content = bytes(s3dlio.get("s3://my-bucket/test.npy"))
loaded = np.load(io.BytesIO(content))
print(f"Data shape: {loaded.shape}")

# Test list
files = s3dlio.list("s3://my-bucket/")
print(f"Files: {files}")

# Test exists
exists = s3dlio.exists("s3://my-bucket/test.npy")
print(f"Exists: {exists}")

# Test stat
meta = s3dlio.stat("s3://my-bucket/test.npy")
print(f"Size: {meta['size']} bytes")

# Cleanup
s3dlio.delete("s3://my-bucket/test.npy")
```

---

## Feature Compatibility

s3dlio v0.9.26+ includes all functions needed for DLIO:

| Feature | Status | Notes |
|---------|--------|-------|
| `exists()` | ✅ v0.9.26+ | Direct existence check |
| `put_bytes()` | ✅ v0.9.26+ | Zero-copy write from Python |
| `mkdir()` | ✅ v0.9.26+ | Create directories/prefixes |
| `stat()` | ✅ Available | Returns metadata dict |
| `get()` / `get_range()` | ✅ Available | Single object read |
| `list()` | ✅ Available | Returns full URIs |
| `delete()` | ✅ Available | Single object deletion |
| Multi-protocol | ✅ Available | S3, GCS, Azure, file://, direct:// |

---

## Contributing

To contribute improvements to the s3dlio-DLIO integration:

1. Fork the s3dlio repository
2. Create a feature branch
3. Submit a pull request with tests

---

## Support

- **s3dlio Issues:** https://github.com/russfellows/s3dlio/issues
- **DLIO Benchmark:** https://github.com/argonne-lcf/dlio_benchmark

## License

s3dlio is licensed under Apache 2.0, compatible with DLIO's Apache 2.0 license.
