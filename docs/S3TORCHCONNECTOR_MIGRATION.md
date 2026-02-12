# Migrating from s3torchconnector to s3dlio

This guide helps you migrate from AWS's `s3torchconnector` to `s3dlio` with **zero code changes** except for imports.

## Quick Migration

### 1. Single-Line Change

**Before** (s3torchconnector):
```python
from s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint
```

**After** (s3dlio):
```python
from s3dlio.compat.s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint
```

That's it! **All other code remains identical.**

### 2. Update Dependencies

**Before** (`requirements.txt`):
```
s3torchconnector>=1.4.0
torch>=2.0.0
```

**After** (`requirements.txt`):
```
s3dlio>=0.9.39
torch>=2.0.0
```

Install:
```bash
pip uninstall s3torchconnector
pip install s3dlio
```

## Complete Example

### Original Code (s3torchconnector)

```python
from s3torchconnector import S3IterableDataset, S3Checkpoint
from torch.utils.data import DataLoader
import torch

# Create dataset
dataset = S3IterableDataset.from_prefix(
    "s3://my-bucket/train/",
    region="us-east-1",
    enable_sharding=True
)

# Use with DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Train loop
for batch in loader:
    for item in batch:
        bucket = item.bucket
        key = item.key
        data = item.read()  # Returns bytes
        # ... process data ...

# Checkpoint saving
checkpoint = S3Checkpoint(region="us-east-1")

with checkpoint.writer("s3://my-bucket/checkpoints/model.pt") as writer:
    torch.save(model.state_dict(), writer)

with checkpoint.reader("s3://my-bucket/checkpoints/model.pt") as reader:
    state_dict = torch.load(reader)
```

### Migrated Code (s3dlio)

```python
# ONLY LINE THAT CHANGED! ↓
from s3dlio.compat.s3torchconnector import S3IterableDataset, S3Checkpoint
from torch.utils.data import DataLoader
import torch

# Create dataset - IDENTICAL CODE!
dataset = S3IterableDataset.from_prefix(
    "s3://my-bucket/train/",
    region="us-east-1",
    enable_sharding=True
)

# Use with DataLoader - IDENTICAL CODE!
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Train loop - IDENTICAL CODE!
for batch in loader:
    for item in batch:
        bucket = item.bucket
        key = item.key
        data = item.read()  # Returns bytes
        # ... process data ...

# Checkpoint saving - IDENTICAL CODE!
checkpoint = S3Checkpoint(region="us-east-1")

with checkpoint.writer("s3://my-bucket/checkpoints/model.pt") as writer:
    torch.save(model.state_dict(), writer)

with checkpoint.reader("s3://my-bucket/checkpoints/model.pt") as reader:
    state_dict = torch.load(reader)
```

## Key Benefits of s3dlio

| Feature | s3torchconnector | s3dlio |
|---------|------------------|--------|
| **S3 Support** | ✅ | ✅ |
| **Azure Blob** | ❌ | ✅ |
| **Google Cloud Storage** | ❌ | ✅ |
| **Local Filesystem** | ❌ | ✅ (`file://`) |
| **Direct I/O** | ❌ | ✅ (`direct://`) |
| **Multi-Endpoint** | ❌ | ✅ Load balancing |
| **Performance** | Good | **5+ GB/s** |
| **Backend** | C++ (CRT) | **Rust (async)** |
| **Custom Endpoints** | Limited | Full (MinIO, Ceph, etc.) |

## Multi-Protocol Support

Unlike s3torchconnector, s3dlio works with **any** storage backend:

### S3-Compatible (AWS, MinIO, Ceph)
```python
dataset = S3IterableDataset.from_prefix(
    "s3://bucket/prefix/",
    region="us-east-1"
)

# With custom endpoint (MinIO, Ceph)
import os
os.environ["AWS_ENDPOINT_URL"] = "http://minio:9000"
```

### Azure Blob Storage
```python
import os
os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "myaccount"
os.environ["AZURE_STORAGE_ACCOUNT_KEY"] = "key"

dataset = S3IterableDataset.from_prefix("az://container/prefix/")
```

### Google Cloud Storage
```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/creds.json"

dataset = S3IterableDataset.from_prefix("gs://bucket/prefix/")
```

### Local Filesystem (Testing)
```python
# Perfect for local development/testing!
dataset = S3IterableDataset.from_prefix("file:///data/train/")
```

## Environment Variables

s3dlio uses the **same environment variables** as s3torchconnector:

### AWS S3
```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://endpoint:9000  # For MinIO/Ceph
```

### Azure
```bash
export AZURE_STORAGE_ACCOUNT_NAME=myaccount
export AZURE_STORAGE_ACCOUNT_KEY=your-key
```

### Google Cloud
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Performance Comparison

Run benchmarks to compare:

```python
import time
from s3torchconnector import S3IterableDataset as AWSDataset
from s3dlio.compat.s3torchconnector import S3IterableDataset as S3DLIODataset

def benchmark(dataset_class, name):
    dataset = dataset_class.from_prefix("s3://bucket/data/", region="us-east-1")
    start = time.time()
    total_bytes = 0
    count = 0
    
    for item in dataset:
        count += 1
        total_bytes += len(item.read())
        if count >= 1000:
            break
    
    elapsed = time.time() - start
    throughput = total_bytes / elapsed / 1024 / 1024  # MB/s
    print(f"{name}: {count} objects, {throughput:.1f} MB/s")

benchmark(AWSDataset, "s3torchconnector")
benchmark(S3DLIODataset, "s3dlio")
```

## Testing Both Implementations

You can conditionally use either library:

```python
USE_S3DLIO = True  # Toggle to switch backends

if USE_S3DLIO:
    from s3dlio.compat.s3torchconnector import S3IterableDataset
else:
    from s3torchconnector import S3IterableDataset

# Rest of code is identical!
dataset = S3IterableDataset.from_prefix(...)
```

Or via environment variable:

```python
import os

if os.getenv("USE_S3DLIO", "true").lower() == "true":
    from s3dlio.compat.s3torchconnector import S3IterableDataset
else:
    from s3torchconnector import S3IterableDataset
```

## API Compatibility Matrix

| Class/Method | s3torchconnector | s3dlio.compat | Notes |
|--------------|------------------|---------------|-------|
| `S3IterableDataset` | ✅ | ✅ | Fully compatible |
| `S3IterableDataset.from_prefix()` | ✅ | ✅ | Same signature |
| `enable_sharding` | ✅ | ✅ | Auto-sharding |
| `item.bucket` | ✅ | ✅ | Bucket name |
| `item.key` | ✅ | ✅ | Object key |
| `item.read()` | ✅ | ✅ | Returns bytes |
| `S3MapDataset` | ✅ | ✅ | Fully compatible |
| `S3MapDataset.from_prefix()` | ✅ | ✅ | Same signature |
| `dataset[index]` | ✅ | ✅ | Random access |
| `len(dataset)` | ✅ | ✅ | Dataset size |
| `S3Checkpoint` | ✅ | ✅ | Fully compatible |
| `checkpoint.writer()` | ✅ | ✅ | Context manager |
| `checkpoint.reader()` | ✅ | ✅ | Context manager |

## Known Limitations

1. **S3ReaderConstructor**: Advanced reader configuration not yet supported
   - Workaround: Use default reader (works for most cases)
   
2. **Distributed Checkpoints**: DCP integration not yet available
   - Use standard S3Checkpoint for now

3. **Lightning Integration**: Not yet available
   - Use standard PyTorch checkpointing

## Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 's3dlio'`

**Solution**: Install s3dlio
```bash
pip install s3dlio
```

### Compatibility Issues

**Error**: `AttributeError: 'S3Item' object has no attribute 'XYZ'`

**Solution**: File an issue - we aim for 100% API compatibility!
```bash
# Report at: https://github.com/russfellows/s3dlio/issues
```

### Performance Issues

**Slow iteration**: Check network connectivity and endpoint configuration

```python
import os
# Enable debug logging
os.environ["RUST_LOG"] = "s3dlio=debug"
```

## Migration Checklist

- [ ] Replace import statements
- [ ] Update dependencies (`requirements.txt`, `pyproject.toml`)
- [ ] Remove s3torchconnector installation
- [ ] Install s3dlio
- [ ] Test with existing code (should work unchanged!)
- [ ] Benchmark performance (should be faster!)
- [ ] Consider using additional s3dlio features (multi-protocol, multi-endpoint)

## Getting Help

- **s3dlio Issues**: https://github.com/russfellows/s3dlio/issues
- **Examples**: See `s3dlio/python/examples/compat_s3torchconnector_example.py`
- **Documentation**: See `s3dlio/docs/integration/DLIO_BENCHMARK_INTEGRATION.md`

## Contributing

Found an incompatibility? Please report it!

We aim for **100% API compatibility** with s3torchconnector to make migration effortless.
