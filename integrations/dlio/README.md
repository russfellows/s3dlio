# dlio-s3dlio-storage

High-performance multi-protocol storage backend for [Argonne DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark) using [s3dlio](https://github.com/russfellows/s3dlio).

## Features

- **Multi-protocol support**: S3, Azure Blob, Google Cloud Storage, local filesystem, O_DIRECT
- **High performance**: 5+ GB/s throughput with optimized Rust backend
- **Drop-in replacement**: Works with existing DLIO configurations
- **Load balancing**: Multi-endpoint support for distributed storage
- **Custom endpoints**: Works with MinIO, Ceph, and other S3-compatible systems

## Installation

```bash
# Install the storage adapter (also installs s3dlio)
pip install dlio-s3dlio-storage

# Or install from source
cd s3dlio/integrations/dlio
pip install -e .
```

## Quick Setup

### 1. Copy Storage Class to DLIO

```bash
# Copy the storage adapter to your DLIO installation
cp src/dlio_s3dlio_storage/s3dlio_storage.py \
   /path/to/dlio_benchmark/storage/
```

### 2. Register in StorageFactory

Edit `dlio_benchmark/storage/storage_factory.py`:

```python
# Add at the top with other imports
# (no import needed - we'll use lazy import)

# In get_storage() method, add this condition:
elif storage_type == 's3dlio':
    from dlio_benchmark.storage.s3dlio_storage import S3DLIOStorage
    return S3DLIOStorage(namespace, framework)
```

### 3. Update Workload Config

```yaml
storage:
  storage_type: s3dlio  # Use s3dlio backend
  storage_root: s3://my-bucket/data
```

## Supported Storage Backends

| Protocol | URI Format | Example |
|----------|------------|---------|
| Amazon S3 | `s3://bucket/key` | `s3://my-bucket/train/` |
| Google Cloud | `gs://bucket/key` | `gs://my-bucket/train/` |
| Azure Blob | `az://account/container/key` | `az://myaccount/data/train/` |
| Local FS | `file:///path` | `file:///mnt/data/train/` |
| Direct I/O | `direct:///path` | `direct:///nvme/train/` |

## Environment Variables

```bash
# AWS S3 / S3-compatible
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_ENDPOINT_URL="http://localhost:9000"  # For MinIO

# Google Cloud Storage
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Azure Blob Storage
export AZURE_STORAGE_ACCOUNT_NAME="myaccount"
export AZURE_STORAGE_ACCOUNT_KEY="your-key"
```

## Performance Comparison

| Metric | s3torchconnector | s3dlio |
|--------|------------------|--------|
| Protocols | S3 only | 5 backends |
| Throughput | ~2 GB/s | 5+ GB/s |
| O_DIRECT | No | Yes |
| Multi-endpoint | No | Yes |

## Documentation

- [Full Integration Guide](https://github.com/russfellows/s3dlio/blob/main/docs/integration/DLIO_BENCHMARK_INTEGRATION.md)
- [s3dlio Python API](https://github.com/russfellows/s3dlio/blob/main/docs/PYTHON_API_GUIDE.md)
- [DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark)

## License

Apache 2.0 - Compatible with DLIO Benchmark license.
