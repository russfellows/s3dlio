# s3dlio Python Library API Guide - v0.9.0

**Version**: 0.9.0 (API-stable beta)  
**Date**: October 2025  
**Status**: Compatible with v0.8.22 - minimal breaking changes

---

## Table of Contents

- [Overview](#overview)
- [What's New in v0.9.0](#whats-new-in-v090)
- [Breaking Changes from v0.8.22](#breaking-changes-from-v0822)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Data Loaders](#data-loaders)
- [Framework Integration](#framework-integration)
- [Checkpoint Operations](#checkpoint-operations)
- [Advanced Features](#advanced-features)
- [Migration Guide](#migration-guide)

---

## Overview

s3dlio is a high-performance Python library for AI/ML data loading from cloud storage:

- **Fast**: 5+ GB/s reads, 2.5+ GB/s writes
- **Universal**: S3, Azure, local files, DirectIO
- **AI/ML optimized**: Built-in PyTorch, TensorFlow, JAX integration
- **Zero-copy**: Minimal memory allocations via Rust backend

**Key Features**:
- Async data loaders with prefetching
- Concurrent batch loading (3-8x faster)
- Checkpoint save/load
- Adaptive performance tuning (v0.9.0)
- NPZ, HDF5, TFRecord support

---

## What's New in v0.9.0

### User-Facing Improvements

1. **Concurrent Batch Loading** 🚀
   - 3-8x faster batch fetching
   - Automatic parallelization
   - No API changes required

2. **Adaptive Tuning** 🎯 (Optional)
   - Auto-optimization based on workload
   - Smart defaults for different file sizes
   - Opt-in via loader options

3. **Enhanced Data Integrity** ✅
   - Zero-copy performance (10-15% memory reduction)
   - Improved error handling
   - Better async iteration

4. **Framework Integration Verified** 🧪
   - PyTorch: Fully tested with DataLoaders
   - TensorFlow: TFRecord and Dataset support
   - JAX: NumPy array integration

### Internal Improvements

- **Bytes migration**: Underlying Rust API now uses `bytes::Bytes` (transparent to Python)
- **Better memory management**: Reference-counted bytes reduce copies
- **Improved async runtime**: More efficient task scheduling

---

## Breaking Changes from v0.8.22

### Minimal Breaking Changes

**Good news**: v0.9.0 is **largely compatible** with v0.8.22 Python code!

#### Removed Functions (Deprecated in v0.7.x)

- `save_numpy_array()` - Use checkpoint API or NPZ generation instead
- `load_numpy_array()` - Use `get()` + NumPy load instead

**Migration**:
```python
# Old (v0.8.22 and earlier)
# save_numpy_array() was disabled/deprecated

# New (v0.9.0) - Use standard patterns
import numpy as np
import io

# Save via checkpoint API
checkpoint_writer.save_array("weights", array)

# Or generate NPZ directly
s3dlio.put(
    prefix="s3://bucket/data/",
    num=1,
    template="weights.npz",
    object_type='npz',
    size=array.nbytes
)

# Load NPZ
data = s3dlio.get("s3://bucket/data/weights.npz")
npz = np.load(io.BytesIO(data))
array = npz['array_name']
```

---

## Installation

### From PyPI (when published)

```bash
pip install s3dlio
```

### From Source

```bash
# Clone repository
git clone https://github.com/russfellows/s3dlio.git
cd s3dlio

# Build and install
./build_pyo3.sh
./install_pyo3_wheel.sh

# Or with UV
uv pip install .
```

### Dependencies

**Required**:
- Python 3.8+
- NumPy

**Optional** (for framework integration):
- PyTorch ≥ 1.12
- TensorFlow ≥ 2.10
- JAX ≥ 0.4

---

## Quick Start

### Basic S3 Operations

```python
import s3dlio

# Read file from S3
data = s3dlio.get("s3://my-bucket/data.bin")
print(f"Read {len(data)} bytes")

# Write file to S3
s3dlio.put(
    prefix="s3://my-bucket/output/",
    num=1,
    template="result.bin",
    object_type='random',
    size=1024 * 1024  # 1 MB
)

# List files
files = s3dlio.list("s3://my-bucket/data/")
for f in files:
    print(f"{f['key']}: {f['size']} bytes")

# Delete file
s3dlio.delete("s3://my-bucket/old.bin")

# Get metadata
info = s3dlio.stat("s3://my-bucket/data.bin")
print(f"Size: {info['size']} bytes")
```

### Data Loading with Async

```python
import s3dlio
import asyncio

async def load_data():
    # Create async loader
    loader = s3dlio.create_async_loader(
        uri="s3://bucket/training/",
        opts={
            'batch_size': 32,
            'prefetch': 2,
            'shuffle': True
        }
    )
    
    # Iterate batches
    async for batch in loader:
        # batch is a list of bytes objects
        for item in batch:
            process_item(item)

# Run async code
asyncio.run(load_data())
```

---

## Core API

### File Operations

#### get() - Read File

```python
import s3dlio

# Read entire file
data = s3dlio.get("s3://bucket/file.bin")  # Returns bytes

# Works with any URI scheme
local_data = s3dlio.get("file:///tmp/data.bin")
azure_data = s3dlio.get("az://container/blob.bin")

# Use data
import numpy as np
import io

# Load NPZ
npz = np.load(io.BytesIO(data))
array = npz['data']
```

#### put() - Generate and Write Files

**Important**: `put()` is designed for **efficient data generation**, not arbitrary uploads.

```python
import s3dlio

# Generate multiple files
s3dlio.put(
    prefix="s3://bucket/training/",
    num=1000,                    # Generate 1000 files
    template="sample_{:06d}.npz",  # Naming pattern
    object_type='npz',           # NPZ format
    size=1024 * 1024,            # 1 MB each
    max_in_flight=16             # Parallel uploads
)

# Object types:
# 'zeros' - All-zero data (fast)
# 'random' - Random bytes
# 'npz' - NumPy compressed arrays
# 'hdf5' - HDF5 files
# 'tfrecord' - TensorFlow records
```

#### list() - List Files

```python
# List all files under prefix
files = s3dlio.list("s3://bucket/data/")

for f in files:
    print(f"Key: {f['key']}")
    print(f"Size: {f['size']} bytes")
    print(f"ETag: {f.get('etag', 'N/A')}")
    print()
```

#### delete() - Delete Files

```python
# Delete single file
s3dlio.delete("s3://bucket/old.bin")

# Delete multiple files (via list + delete)
files = s3dlio.list("s3://bucket/temp/")
for f in files:
    s3dlio.delete(f"s3://bucket/{f['key']}")
```

#### stat() - Get Metadata

```python
# Get file info without downloading
info = s3dlio.stat("s3://bucket/large.bin")

print(f"Size: {info['size']} bytes")
print(f"ETag: {info.get('etag')}")
print(f"Modified: {info.get('last_modified')}")
```

---

## Data Loaders

### Creating Loaders

#### Async Loader (Recommended)

```python
import s3dlio
import asyncio

async def train():
    # Create loader
    loader = s3dlio.create_async_loader(
        uri="s3://bucket/training/",
        opts={
            'batch_size': 64,
            'prefetch': 3,
            'max_concurrency': 16,
            'shuffle': True,
            'seed': 42
        }
    )
    
    # Iterate batches
    epoch = 0
    async for batch in loader:
        # batch: list of bytes objects
        print(f"Epoch {epoch}, batch size: {len(batch)}")
        
        for item in batch:
            # Process item (bytes object)
            process_training_sample(item)
        
        epoch += 1

asyncio.run(train())
```

#### Dataset (For PyTorch/TF Integration)

```python
import s3dlio

# Create dataset
dataset = s3dlio.create_dataset(
    uri="s3://bucket/data/",
    opts={
        'batch_size': 32,
        'shuffle': True
    }
)

# Use with async loader
loader = s3dlio.create_async_loader_from_dataset(dataset)
```

### Loader Options

```python
{
    'batch_size': 32,         # Items per batch (default: 32)
    'prefetch': 2,            # Batches to prefetch (default: 2)
    'max_concurrency': 16,    # Parallel downloads (default: 16)
    'shuffle': False,         # Shuffle items (default: False)
    'seed': None,             # Shuffle seed (default: None)
    'part_size': 8388608,     # 8 MB parts (default)
    
    # NEW in v0.9.0: Adaptive tuning
    'adaptive': {
        'mode': 'enabled',    # 'enabled' or 'disabled'
        'workload_type': 'auto',  # 'small', 'medium', 'large', 'batch', 'auto'
    }
}
```

### Adaptive Tuning (v0.9.0)

```python
# Enable adaptive tuning - auto-optimizes based on workload
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={
        'batch_size': 32,
        'adaptive': {
            'mode': 'enabled',
            'workload_type': 'auto'  # Auto-detect optimal settings
        }
    }
)

# Specify workload type
loader = s3dlio.create_async_loader(
    uri="s3://bucket/large-files/",
    opts={
        'adaptive': {
            'mode': 'enabled',
            'workload_type': 'large'  # Optimize for large files
        }
    }
)

# Workload types:
# 'small' - < 1 MB files: Higher concurrency
# 'medium' - 1-64 MB files: Balanced
# 'large' - > 64 MB files: Larger parts
# 'batch' - Multiple files: Maximum parallelism
# 'auto' - Detect automatically
```

---

## Framework Integration

### PyTorch Integration

```python
import s3dlio
import torch
import numpy as np
import io
import asyncio

async def pytorch_dataloader():
    # Create async loader
    loader = s3dlio.create_async_loader(
        uri="s3://bucket/training/",
        opts={'batch_size': 64, 'shuffle': True, 'seed': 42}
    )
    
    async for batch in loader:
        tensors = []
        
        for item_bytes in batch:
            # Load NPZ from bytes
            npz = np.load(io.BytesIO(item_bytes))
            array = npz['data']
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(array)
            tensors.append(tensor)
        
        # Stack batch
        batch_tensor = torch.stack(tensors)
        
        # Train step
        train_step(batch_tensor)

# Run training
asyncio.run(pytorch_dataloader())
```

### TensorFlow Integration

```python
import s3dlio
import tensorflow as tf
import numpy as np
import io
import asyncio

async def tensorflow_dataloader():
    loader = s3dlio.create_async_loader(
        uri="s3://bucket/tfrecords/",
        opts={'batch_size': 32}
    )
    
    async for batch in loader:
        for item_bytes in batch:
            # Option 1: Load NPZ
            npz = np.load(io.BytesIO(item_bytes))
            array = npz['features']
            tensor = tf.constant(array)
            
            # Option 2: Parse TFRecord
            # example = tf.train.Example.FromString(item_bytes)
            
            process_tf_tensor(tensor)

asyncio.run(tensorflow_dataloader())
```

### JAX Integration

```python
import s3dlio
import jax.numpy as jnp
import numpy as np
import io
import asyncio

async def jax_dataloader():
    loader = s3dlio.create_async_loader(
        uri="s3://bucket/data/",
        opts={'batch_size': 128}
    )
    
    async for batch in loader:
        arrays = []
        
        for item_bytes in batch:
            # Load NPZ
            npz = np.load(io.BytesIO(item_bytes))
            array = npz['data']
            
            # Convert to JAX array
            jax_array = jnp.array(array)
            arrays.append(jax_array)
        
        # Stack batch
        batch_array = jnp.stack(arrays)
        
        # Process with JAX
        train_jax_step(batch_array)

asyncio.run(jax_dataloader())
```

---

## Checkpoint Operations

### Checkpoint Writer

```python
import s3dlio
import numpy as np

# Create checkpoint writer
writer = s3dlio.create_checkpoint_writer(
    uri="s3://bucket/checkpoints/epoch_0010/"
)

# Save arrays
model_weights = np.random.randn(1000, 1000)
optimizer_state = np.random.randn(500, 500)

writer.save_array("model_weights", model_weights)
writer.save_array("optimizer_state", optimizer_state)

# Finalize checkpoint
writer.finalize()
```

### Checkpoint Reader

```python
import s3dlio

# Create checkpoint reader
reader = s3dlio.create_checkpoint_reader(
    uri="s3://bucket/checkpoints/epoch_0010/"
)

# Load arrays
model_weights = reader.load_array("model_weights")
optimizer_state = reader.load_array("optimizer_state")

print(f"Model shape: {model_weights.shape}")
print(f"Optimizer shape: {optimizer_state.shape}")
```

---

## Advanced Features

### Concurrent Batch Loading (v0.9.0)

**Automatic** in v0.9.0 - no configuration needed!

```python
# This now fetches batches in parallel (3-8x faster)
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'batch_size': 32}
)

# Before v0.9.0: Sequential fetching
# v0.9.0+: Concurrent fetching with automatic parallelization
```

### Custom Object Types

```python
# Generate different data types
s3dlio.put(
    prefix="s3://bucket/data/",
    num=100,
    template="data_{:03d}.npz",
    object_type='npz',      # NumPy compressed
    size=10 * 1024 * 1024,  # 10 MB
    compress_factor=2,       # Compression ratio
    dedup_factor=1          # No deduplication
)

# HDF5 format
s3dlio.put(
    prefix="s3://bucket/hdf5/",
    num=50,
    template="data_{:03d}.h5",
    object_type='hdf5',
    size=50 * 1024 * 1024
)

# TFRecord format
s3dlio.put(
    prefix="s3://bucket/tfrecords/",
    num=200,
    template="train_{:05d}.tfrecord",
    object_type='tfrecord',
    size=5 * 1024 * 1024
)
```

### Environment Variables

Configure S3dlio via environment variables:

```bash
# AWS S3
export AWS_ENDPOINT_URL="https://s3.us-west-2.amazonaws.com"
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Azure Blob Storage
export AZURE_STORAGE_ACCOUNT="myaccount"
export AZURE_STORAGE_KEY="key"

# Backend selection (for testing)
export S3DLIO_BACKEND="native"  # or "arrow"

# HTTP optimization
export S3DLIO_USE_OPTIMIZED_HTTP="true"
```

### Local File Testing

```python
import s3dlio

# Use file:// URIs for local testing
loader = s3dlio.create_async_loader(
    uri="file:///path/to/local/data/",
    opts={'batch_size': 16}
)

# Note: put() requires S3 URIs
# For local file generation, use Python directly:
import numpy as np
for i in range(10):
    data = {'array': np.random.rand(100, 100)}
    np.savez(f"/tmp/data/file_{i:03d}.npz", **data)
```

---

## Migration Guide

### From v0.8.22 to v0.9.0

#### Step 1: Update Installation

```bash
# Update to v0.9.0
pip install --upgrade s3dlio
# or
uv pip install --upgrade s3dlio
```

#### Step 2: Remove Deprecated Functions

**Only if you were using** (unlikely - they were disabled in v0.7.x):

```python
# Old (doesn't work in v0.9.0)
# s3dlio.save_numpy_array(...)  # REMOVED
# s3dlio.load_numpy_array(...)  # REMOVED

# New - Use checkpoint API
writer = s3dlio.create_checkpoint_writer("s3://bucket/ckpt/")
writer.save_array("name", array)
writer.finalize()

# Or direct NPZ handling
import numpy as np
import io

# Save
data = s3dlio.get("s3://bucket/data.npz")
npz = np.load(io.BytesIO(data))
```

#### Step 3: Test Data Loaders

No changes needed, but verify everything works:

```python
import s3dlio
import asyncio

async def test_loader():
    loader = s3dlio.create_async_loader(
        uri="s3://bucket/test/",
        opts={'batch_size': 8}
    )
    
    batch_count = 0
    async for batch in loader:
        batch_count += 1
        print(f"Batch {batch_count}: {len(batch)} items")
        
        # Verify data is bytes
        assert isinstance(batch[0], bytes)
        
        if batch_count >= 3:  # Test first 3 batches
            break
    
    print("✅ Loader works correctly!")

asyncio.run(test_loader())
```

#### Step 4: Optional - Enable Adaptive Tuning

```python
# Add adaptive tuning for automatic optimization
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={
        'batch_size': 32,
        'adaptive': {
            'mode': 'enabled',
            'workload_type': 'auto'
        }
    }
)
```

### Compatibility Checklist

- [ ] Updated to s3dlio v0.9.0
- [ ] Removed any `save_numpy_array` / `load_numpy_array` calls (if used)
- [ ] Tested data loaders - batches return `list[bytes]`
- [ ] Framework integration still works (PyTorch/TensorFlow/JAX)
- [ ] Checkpoint save/load works
- [ ] (Optional) Evaluated adaptive tuning

---

## Common Patterns

### Pattern 1: Batch Processing with PyTorch

```python
import s3dlio
import torch
import numpy as np
import io
import asyncio

class S3Dataset:
    def __init__(self, uri, batch_size=32):
        self.uri = uri
        self.batch_size = batch_size
    
    async def iterate(self):
        loader = s3dlio.create_async_loader(
            uri=self.uri,
            opts={
                'batch_size': self.batch_size,
                'shuffle': True,
                'adaptive': {'mode': 'enabled'}
            }
        )
        
        async for batch in loader:
            tensors = []
            
            for item_bytes in batch:
                npz = np.load(io.BytesIO(item_bytes))
                tensor = torch.from_numpy(npz['data'])
                tensors.append(tensor)
            
            yield torch.stack(tensors)

# Usage
async def train():
    dataset = S3Dataset("s3://bucket/training/", batch_size=64)
    
    async for batch_tensor in dataset.iterate():
        # batch_tensor: torch.Tensor of shape [64, ...]
        loss = train_step(batch_tensor)
        print(f"Loss: {loss:.4f}")

asyncio.run(train())
```

### Pattern 2: Multi-Epoch Training

```python
import s3dlio
import asyncio

async def multi_epoch_training(num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        loader = s3dlio.create_async_loader(
            uri="s3://bucket/training/",
            opts={
                'batch_size': 32,
                'shuffle': True,
                'seed': epoch  # Different shuffle each epoch
            }
        )
        
        batch_count = 0
        async for batch in loader:
            # Process batch
            process_batch(batch)
            batch_count += 1
        
        print(f"  Processed {batch_count} batches")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1)

asyncio.run(multi_epoch_training())
```

### Pattern 3: Data Generation Pipeline

```python
import s3dlio
import numpy as np

def generate_training_data():
    """Generate synthetic training data efficiently"""
    
    # Generate 10,000 NPZ files with random data
    s3dlio.put(
        prefix="s3://bucket/synthetic/train/",
        num=10000,
        template="train_{:06d}.npz",
        object_type='npz',
        size=512 * 1024,  # 512 KB each
        max_in_flight=32
    )
    
    # Generate validation set
    s3dlio.put(
        prefix="s3://bucket/synthetic/val/",
        num=1000,
        template="val_{:05d}.npz",
        object_type='npz',
        size=512 * 1024,
        max_in_flight=32
    )
    
    print("✅ Generated 11,000 training files")

generate_training_data()
```

---

## Performance Tips

### 1. Use Adaptive Tuning for Unknown Workloads

```python
# Let s3dlio optimize automatically
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'adaptive': {'mode': 'enabled'}}
)
```

### 2. Tune Batch Size for Your Hardware

```python
# Small GPU memory - smaller batches
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'batch_size': 16}
)

# Large GPU memory - larger batches
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'batch_size': 128}
)
```

### 3. Increase Concurrency for Small Files

```python
# Many small files - higher concurrency
loader = s3dlio.create_async_loader(
    uri="s3://bucket/small-files/",
    opts={
        'batch_size': 64,
        'max_concurrency': 32  # More parallel downloads
    }
)
```

### 4. Prefetch for Continuous Loading

```python
# Prefetch next batches while processing current
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={
        'batch_size': 32,
        'prefetch': 4  # Keep 4 batches ready
    }
)
```

---

## Troubleshooting

### Issue: Import Error

```python
# Error: ModuleNotFoundError: No module named 's3dlio'

# Solution: Install s3dlio
pip install s3dlio
# or
uv pip install s3dlio
```

### Issue: AWS Credentials Not Found

```python
# Error: NoCredentialsError

# Solution: Set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
```

### Issue: Slow Loading

```python
# Solution 1: Enable adaptive tuning
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'adaptive': {'mode': 'enabled'}}
)

# Solution 2: Increase concurrency
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'max_concurrency': 32}
)

# Solution 3: Larger batch size
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'batch_size': 64}
)
```

### Issue: Out of Memory

```python
# Solution 1: Smaller batch size
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'batch_size': 8}
)

# Solution 2: Less prefetching
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={'prefetch': 1}
)
```

---

## API Reference

### Core Functions

- `get(uri: str) -> bytes`: Read file
- `put(prefix, num, template, **kwargs)`: Generate files
- `list(uri: str) -> list`: List files
- `delete(uri: str)`: Delete file
- `stat(uri: str) -> dict`: Get metadata

### Data Loader Functions

- `create_async_loader(uri, opts) -> PyBytesAsyncDataLoader`: Create async loader
- `create_dataset(uri, opts) -> PyDataset`: Create dataset
- `read_npz(uri: str) -> dict`: Read NPZ file

### Checkpoint Functions

- `create_checkpoint_writer(uri) -> CheckpointWriter`: Create writer
- `create_checkpoint_reader(uri) -> CheckpointReader`: Create reader

### Utility Classes

- `PyDataset`: Dataset wrapper
- `PyBytesAsyncDataLoader`: Async batch loader
- `PyVecDataset`: In-memory dataset

---

## Additional Resources

- **GitHub**: https://github.com/russfellows/s3dlio
- **Rust API Guide**: `docs/api/rust-api-v0.9.0.md`
- **Changelog**: `docs/Changelog.md`
- **Testing Guide**: `docs/TESTING-GUIDE.md`
- **Adaptive Tuning**: `docs/ADAPTIVE-TUNING.md`

---

**Last Updated**: October 2025  
**Version**: 0.9.0  
**Next Release**: v0.9.1 (Internal optimizations)
