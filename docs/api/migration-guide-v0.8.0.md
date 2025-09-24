# S3DLIO Enhanced API - Migration Guide

## Overview

This document describes the enhanced s3dlio API that provides a unified interface for working with multiple storage backends through a generic dataset factory pattern.

## 🚀 What's New

### Bug Fix: PyS3AsyncDataLoader Issue
- **Problem**: `python/s3dlio/torch.py` was calling non-existent `PyS3AsyncDataLoader`
- **Solution**: Updated to use new `create_async_loader()` function
- **Impact**: PyTorch integration now works seamlessly across all backends

### Enhanced Multi-Backend Support
- **File System**: `file://` URIs for local and network file systems
- **Amazon S3**: `s3://` URIs (existing functionality enhanced)
- **Azure Blob**: `az://` URIs (framework ready)
- **Direct I/O**: `direct://` URIs (framework ready)

### Unified API Pattern
- Generic `create_dataset()` and `create_async_loader()` functions
- Consistent options handling across all backends
- Automatic backend selection based on URI scheme

## 📋 Quick Migration Checklist

### If you were using PyTorch integration:
- ✅ **No changes needed** - existing code continues to work
- ✅ **Bug automatically fixed** - no more PyS3AsyncDataLoader errors
- ✅ **New functionality** - can now use `file://` URIs for local testing

### If you were using direct s3dlio APIs:
- ✅ **Existing APIs preserved** - all original functions still work
- ✅ **New generic APIs available** - `create_dataset()`, `create_async_loader()`
- ✅ **Enhanced error handling** - better error messages and validation

### If you want to use new file system support:
- ✅ **Use `file://` URIs** - works with both files and directories
- ✅ **Same API pattern** - consistent with S3 usage
- ✅ **Automatic scanning** - directories are recursively processed

## 🔧 API Reference

### Core Functions

#### `create_dataset(uri: str, options: dict = None) -> PyDataset`
Creates a dataset for the given URI with optional configuration.

```python
import s3dlio

# File system dataset
dataset = s3dlio.create_dataset("file:///path/to/data")

# S3 dataset  
dataset = s3dlio.create_dataset("s3://bucket/prefix/")

# With options
dataset = s3dlio.create_dataset("file:///data", {
    "batch_size": 32,
    "shuffle": True,
    "num_workers": 4
})
```

#### `create_async_loader(uri: str, options: dict = None) -> PyBytesAsyncDataLoader`
Creates an async data loader for streaming access to data.

```python
import s3dlio
import asyncio

async def process_data():
    loader = s3dlio.create_async_loader("file:///path/to/data")
    async for item in loader:
        # Process each data item
        print(f"Got {len(item)} bytes")

asyncio.run(process_data())
```

### URI Schemes

#### File System (`file://`)
```python
# Single file
s3dlio.create_dataset("file:///home/user/data.txt")

# Directory (recursive)
s3dlio.create_dataset("file:///home/user/dataset/")

# Network path
s3dlio.create_dataset("file:///mnt/shared/data/")
```

#### Amazon S3 (`s3://`)
```python
# Bucket prefix
s3dlio.create_dataset("s3://my-bucket/train-data/")

# Single object
s3dlio.create_dataset("s3://my-bucket/model.bin")

# With S3-specific options
s3dlio.create_dataset("s3://my-bucket/data/", {
    "part_size": 8388608,  # 8MB chunks
    "max_concurrent": 10
})
```

#### Azure Blob Storage (`az://`)
```python
# Container prefix  
s3dlio.create_dataset("az://container/path/")

# Note: Full implementation pending
```

#### Direct I/O (`direct://`)
```python
# Direct file access with custom I/O
s3dlio.create_dataset("direct:///dev/nvme0n1")

# Note: Full implementation pending
```

### Options Dictionary

Common options that work across all backends:

```python
options = {
    "batch_size": 32,        # Items per batch
    "shuffle": True,         # Randomize order
    "num_workers": 4,        # Parallel workers
    "prefetch": 8,           # Prefetch buffer size
}
```

Backend-specific options:
```python
# S3 specific
s3_options = {
    "part_size": 8388608,    # S3 multipart size
    "max_concurrent": 10,    # Concurrent S3 requests
}

# File system specific  
fs_options = {
    "recursive": True,       # Scan subdirectories
    "pattern": "*.txt",      # File pattern filter
}
```

## 📚 Examples

### Example 1: PyTorch DataLoader (Fixed)
```python
from s3dlio.torch import S3IterableDataset
from torch.utils.data import DataLoader

# This now works with any URI scheme!
dataset = S3IterableDataset("file:///local/training/data/")
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    # Train your model
    pass
```

### Example 2: Async Data Processing
```python
import s3dlio
import asyncio

async def process_large_dataset():
    loader = s3dlio.create_async_loader("s3://huge-bucket/dataset/")
    
    count = 0
    async for item in loader:
        # Process each item asynchronously
        result = await process_item(item)
        count += 1
        
        if count % 1000 == 0:
            print(f"Processed {count} items")

asyncio.run(process_large_dataset())
```

### Example 3: Multi-Backend Pipeline
```python
import s3dlio

def create_pipeline(input_uri: str, output_uri: str):
    """Process data from any source to any destination."""
    
    # Input can be file://, s3://, az://, etc.
    input_dataset = s3dlio.create_dataset(input_uri)
    
    # Process data...
    processed_data = process(input_dataset)
    
    # Output to any backend
    if output_uri.startswith("s3://"):
        writer = s3dlio.create_s3_writer(output_uri)
    elif output_uri.startswith("file://"):
        writer = s3dlio.create_filesystem_writer(output_uri)
    
    writer.write(processed_data)

# Works with any combination
create_pipeline("file:///local/input/", "s3://bucket/output/")
create_pipeline("s3://input-bucket/", "file:///local/results/")
```

### Example 4: Error Handling
```python
import s3dlio

def safe_dataset_creation(uri: str):
    try:
        dataset = s3dlio.create_dataset(uri)
        return dataset
    except Exception as e:
        if "unsupported" in str(e).lower():
            print(f"URI scheme not supported: {uri}")
        elif "not found" in str(e).lower():
            print(f"Path not found: {uri}")
        else:
            print(f"Other error: {e}")
        return None

# Test different URIs
dataset = safe_dataset_creation("file:///nonexistent/path")  # Handles gracefully
dataset = safe_dataset_creation("ftp://invalid/scheme")      # Proper error
dataset = safe_dataset_creation("file:///valid/path")        # Works normally
```

## 🔄 Backward Compatibility

All existing s3dlio code continues to work unchanged:

```python
# These still work exactly as before
import s3dlio

# Original functions
data = s3dlio.get("s3://bucket/key")
s3dlio.put("s3://bucket/key", data)
keys = s3dlio.list("s3://bucket/prefix/")

# Original PyTorch integration
from s3dlio.torch import S3IterableDataset
dataset = S3IterableDataset("s3://bucket/data/")  # Fixed internally

# Original classes  
if hasattr(s3dlio, 'PyS3Dataset'):
    old_dataset = s3dlio.PyS3Dataset("s3://bucket/data/")
```

## ⚠️ Breaking Changes

**None!** This is a fully backward-compatible enhancement.

## 🚧 Future Enhancements

- **Azure Blob Storage**: Complete `az://` implementation
- **Direct I/O**: Complete `direct://` implementation  
- **More URI schemes**: `gs://` for Google Cloud, `hdfs://` for Hadoop
- **Advanced options**: Schema validation, type hints, async context managers

## 🐛 Troubleshooting

### "module 's3dlio' has no attribute 'create_dataset'"
This is a linter/IDE issue. The functions ARE available at runtime:

```python
# Works despite linter warnings
import s3dlio
dataset = s3dlio.create_dataset("file:///path/to/data")  # This works!

# Or use getattr if needed
create_dataset = getattr(s3dlio, 'create_dataset')
dataset = create_dataset("file:///path/to/data")
```

### File system datasets return no items
- Check that files exist in the directory
- Verify file permissions are readable
- Use absolute paths in URIs: `file:///absolute/path`

### S3 access errors
- Verify AWS credentials are configured
- Check bucket permissions
- Ensure bucket region is accessible

## 📞 Support

For issues with the enhanced API:
1. Check that your URI scheme is supported (`file://`, `s3://`, `az://`, `direct://`)
2. Verify the path exists and is accessible
3. Check the error message for specific guidance
4. Use backward-compatible APIs if needed

The enhanced s3dlio maintains full compatibility while providing a clean, extensible foundation for multi-backend data access. 🚀