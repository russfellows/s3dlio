# S3DLIO Enhanced API Documentation

## Table of Contents

1. [Core Functions](#core-functions)
2. [Classes](#classes)
3. [URI Schemes](#uri-schemes)
4. [Options](#options)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)

## Core Functions

### `create_dataset(uri: str, options: dict = None) -> PyDataset`

Creates a dataset for the specified URI with optional configuration.

**Parameters:**
- `uri` (str): URI specifying the data location and backend
- `options` (dict, optional): Configuration options for the dataset

**Returns:**
- `PyDataset`: Generic dataset object that works with any backend

**Raises:**
- `ValueError`: Invalid URI format or unsupported scheme
- `FileNotFoundError`: Path does not exist (for file:// URIs)
- `PermissionError`: Access denied to the specified location
- `RuntimeError`: Backend-specific configuration errors

**Example:**
```python
import s3dlio

# Basic usage
dataset = s3dlio.create_dataset("file:///data/training")

# With options
dataset = s3dlio.create_dataset("s3://bucket/prefix", {
    "batch_size": 64,
    "prefetch": 16
})
```

### `create_async_loader(uri: str, options: dict = None) -> PyBytesAsyncDataLoader`

Creates an asynchronous data loader for streaming access to data.

**Parameters:**
- `uri` (str): URI specifying the data location and backend
- `options` (dict, optional): Configuration options for the loader

**Returns:**
- `PyBytesAsyncDataLoader`: Async iterator for streaming data access

**Raises:**
- Same as `create_dataset()`

**Example:**
```python
import s3dlio
import asyncio

async def process_data():
    loader = s3dlio.create_async_loader("file:///large/dataset")
    
    async for item in loader:
        # Process each data item
        result = await process_item(item)
        yield result

# Usage
async for processed_item in process_data():
    print(f"Processed: {len(processed_item)} bytes")
```

### Legacy Functions (Preserved)

All existing s3dlio functions remain available and unchanged:

- `get(uri: str) -> bytes`: Fetch single object
- `put(uri: str, data: bytes)`: Store single object  
- `list(uri: str) -> List[str]`: List objects under prefix
- `stat(uri: str) -> dict`: Get object metadata
- `delete(uri: str)`: Delete object

## Classes

### `PyDataset`

Generic dataset class that holds backend-specific dataset implementations.

**Methods:**
- `__iter__()`: Synchronous iteration over dataset items
- `__len__()`: Number of items in dataset (if known)

**Properties:**
- `backend`: Backend type (e.g., "filesystem", "s3", "azure")
- `uri`: Original URI used to create the dataset

**Example:**
```python
dataset = s3dlio.create_dataset("file:///data")
print(f"Backend: {dataset.backend}")
print(f"URI: {dataset.uri}")

for item in dataset:
    print(f"Item size: {len(item)} bytes")
```

### `PyBytesAsyncDataLoader`

Asynchronous data loader for streaming access to datasets.

**Methods:**
- `__aiter__()`: Async iteration over dataset items
- `clone()`: Create a copy of the loader

**Example:**
```python
async def stream_data():
    loader = s3dlio.create_async_loader("s3://bucket/data")
    
    async for chunk in loader:
        yield chunk

# Multiple concurrent streams
loader1 = s3dlio.create_async_loader("file:///data")
loader2 = loader1.clone()

# Process in parallel
import asyncio
results = await asyncio.gather(
    process_stream(loader1),
    process_stream(loader2)
)
```

## URI Schemes

### File System (`file://`)

Access local and network file systems.

**Format:** `file:///absolute/path`

**Supported Paths:**
- Single files: `file:///path/to/file.txt`
- Directories: `file:///path/to/directory/`
- Network mounts: `file:///mnt/network/share/`

**Behavior:**
- Single files: Returns the file content as one item
- Directories: Recursively scans for all files, returns each as separate item
- Symlinks: Followed if they point to readable files/directories

**Examples:**
```python
# Single file
dataset = s3dlio.create_dataset("file:///home/user/data.bin")

# Directory with recursive scanning  
dataset = s3dlio.create_dataset("file:///home/user/training_data/")

# Network mounted storage
dataset = s3dlio.create_dataset("file:///mnt/vast1/shared_dataset/")
```

### Amazon S3 (`s3://`)

Access Amazon S3 buckets and objects.

**Format:** `s3://bucket/prefix`

**Authentication:** Uses AWS credentials from:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM roles (when running on EC2)

**Examples:**
```python
# Bucket prefix
dataset = s3dlio.create_dataset("s3://my-bucket/training-data/")

# Single object
dataset = s3dlio.create_dataset("s3://my-bucket/model.bin")

# With region-specific endpoint
dataset = s3dlio.create_dataset("s3://bucket/data/", {
    "region": "us-west-2"
})
```

### Azure Blob Storage (`az://`)

Access Azure Blob Storage containers.

**Format:** `az://container/prefix`

**Authentication:** Uses Azure credentials from:
- Environment variables (`AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`)
- Azure CLI authentication
- Managed identity (when running on Azure)

**Status:** Framework implemented, full functionality pending

**Examples:**
```python
# Container prefix
dataset = s3dlio.create_dataset("az://container/training-data/")

# Single blob
dataset = s3dlio.create_dataset("az://container/model.bin")
```

### Direct I/O (`direct://`)

Direct access to block devices and special files.

**Format:** `direct:///path/to/device`

**Use Cases:**
- NVMe drives: `direct:///dev/nvme0n1`
- RAM disks: `direct:///dev/ram0`  
- Custom block devices

**Status:** Framework implemented, full functionality pending

**Examples:**
```python
# Direct NVMe access
dataset = s3dlio.create_dataset("direct:///dev/nvme0n1")

# Memory-mapped file
dataset = s3dlio.create_dataset("direct:///dev/shm/large_file")
```

## Options

### Universal Options

These options work with all backend types:

```python
options = {
    "batch_size": 32,           # Items per batch (default: 1)
    "shuffle": True,            # Randomize item order (default: False)
    "num_workers": 4,           # Parallel processing workers (default: 1)
    "prefetch": 8,              # Items to prefetch (default: 2)
    "buffer_size": 1048576,     # Buffer size in bytes (default: 1MB)
}
```

### Backend-Specific Options

#### File System Options
```python
fs_options = {
    "recursive": True,          # Scan subdirectories (default: True)
    "follow_symlinks": False,   # Follow symbolic links (default: False)  
    "pattern": "*.txt",         # File pattern filter (default: None)
    "max_files": 10000,         # Maximum files to process (default: unlimited)
}
```

#### S3 Options
```python
s3_options = {
    "part_size": 8388608,       # Multipart upload size (default: 8MB)
    "max_concurrent": 10,       # Concurrent S3 requests (default: 5)
    "region": "us-east-1",      # AWS region (default: auto-detect)
    "endpoint_url": None,       # Custom S3 endpoint (default: AWS)
    "use_ssl": True,            # Use HTTPS (default: True)
}
```

#### Azure Options  
```python
azure_options = {
    "chunk_size": 4194304,      # Download chunk size (default: 4MB)
    "max_concurrent": 8,        # Concurrent requests (default: 4)
    "timeout": 30,              # Request timeout seconds (default: 30)
}
```

### Option Validation

All options are validated at creation time:

```python
try:
    dataset = s3dlio.create_dataset("file:///data", {
        "batch_size": -1  # Invalid: negative value
    })
except ValueError as e:
    print(f"Invalid option: {e}")

try:  
    dataset = s3dlio.create_dataset("file:///data", {
        "unknown_option": "value"  # Warning: unknown option ignored
    })
except ValueError as e:
    print(f"Warning: {e}")
```

## Error Handling

### Common Exceptions

#### `ValueError`
Raised for invalid parameters or configuration:
- Malformed URIs
- Unsupported URI schemes  
- Invalid option values

```python
try:
    dataset = s3dlio.create_dataset("invalid-uri")
except ValueError as e:
    print(f"Invalid URI: {e}")
```

#### `FileNotFoundError`
Raised when paths don't exist (file:// schemes):

```python
try:
    dataset = s3dlio.create_dataset("file:///nonexistent/path")
except FileNotFoundError as e:
    print(f"Path not found: {e}")
```

#### `PermissionError`  
Raised for access permission issues:

```python
try:
    dataset = s3dlio.create_dataset("file:///root/private")
except PermissionError as e:
    print(f"Access denied: {e}")
```

#### `RuntimeError`
Raised for backend-specific issues:

```python
try:
    dataset = s3dlio.create_dataset("s3://bucket/data")
except RuntimeError as e:
    print(f"S3 configuration error: {e}")
```

### Error Recovery Patterns

#### Graceful Degradation
```python
def create_dataset_with_fallback(primary_uri: str, fallback_uri: str):
    try:
        return s3dlio.create_dataset(primary_uri)
    except Exception as e:
        print(f"Primary failed ({e}), trying fallback...")
        return s3dlio.create_dataset(fallback_uri)

# Try S3 first, fallback to local
dataset = create_dataset_with_fallback(
    "s3://bucket/data",
    "file:///local/backup/data"
)
```

#### Retry Logic
```python
import time

def create_dataset_with_retry(uri: str, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            return s3dlio.create_dataset(uri)
        except RuntimeError as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Best Practices

### URI Construction
```python
# ✅ Good: Use absolute paths
dataset = s3dlio.create_dataset("file:///absolute/path/to/data")

# ❌ Avoid: Relative paths can be ambiguous  
dataset = s3dlio.create_dataset("file://relative/path")

# ✅ Good: Include trailing slash for directories
dataset = s3dlio.create_dataset("s3://bucket/prefix/")

# ✅ Good: Escape special characters
from urllib.parse import quote
path = quote("/path with spaces/")
dataset = s3dlio.create_dataset(f"file://{path}")
```

### Resource Management
```python
# ✅ Good: Use async context managers when available
async def process_large_dataset():
    async with s3dlio.create_async_loader("s3://huge-bucket/") as loader:
        async for item in loader:
            await process_item(item)

# ✅ Good: Explicit cleanup for long-running processes
def batch_processing():
    dataset = s3dlio.create_dataset("file:///data")
    try:
        for batch in dataset:
            process_batch(batch)
    finally:
        # Cleanup if needed
        del dataset
```

### Performance Optimization
```python
# ✅ Good: Tune options for your workload
options = {
    "batch_size": 128,      # Larger batches for throughput
    "prefetch": 32,         # More prefetch for pipeline efficiency  
    "num_workers": 8,       # Match your CPU cores
}

dataset = s3dlio.create_dataset("s3://bucket/data", options)

# ✅ Good: Use async for I/O-bound workloads
async def high_throughput_processing():
    loader = s3dlio.create_async_loader("s3://data", {
        "max_concurrent": 20,   # High concurrency for S3
        "prefetch": 64         # Large prefetch buffer
    })
    
    tasks = []
    async for item in loader:
        task = asyncio.create_task(process_item_async(item))
        tasks.append(task)
        
        if len(tasks) >= 100:  # Process in batches
            await asyncio.gather(*tasks)
            tasks.clear()
```

### Testing and Development
```python
# ✅ Good: Use file:// URIs for local testing
def create_test_dataset():
    if os.getenv("TESTING"):
        return s3dlio.create_dataset("file:///tmp/test_data")
    else:
        return s3dlio.create_dataset("s3://production-bucket/data")

# ✅ Good: Validate URIs before processing
def validate_uri(uri: str) -> bool:
    try:
        s3dlio.create_dataset(uri)
        return True
    except Exception:
        return False

# ✅ Good: Use appropriate error handling
def robust_data_processing(uri: str):
    try:
        dataset = s3dlio.create_dataset(uri)
        return process_dataset(dataset)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Access error: {e}")
        raise  
    except RuntimeError as e:
        logger.warning(f"Backend error, retrying: {e}")
        return retry_with_backoff(lambda: robust_data_processing(uri))
```

This enhanced API provides a solid foundation for scalable, multi-backend data access while maintaining full backward compatibility with existing s3dlio code. 🚀