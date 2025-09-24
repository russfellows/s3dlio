# s3dlio Python Library Reference - LEGACY v0.7.x

**⚠️ LEGACY DOCUMENTATION WARNING**
- **This document**: For s3dlio v0.7.x and earlier (2024-early 2025)
- **Status**: OUTDATED - Use only for older releases  
- **Current API**: See `python-api-v0.8.0-current.md` for latest version
- **Known Issues**: PyTorch integration was broken in these versions

---

This document outlines the Python interface provided by the `s3dlio` package v0.7.x, providing high-performance storage operations across multiple backends with streaming and compression capabilities.

---

## Installation

After building the wheel with maturin:

```bash
maturin develop --features extension-module
```

Or install from a wheel:

```bash
pip install s3dlio-<version>-*.whl
```

Then, in Python:

```python
import s3dlio
```

---

## Core Features

### Multi-Backend Support

s3dlio supports multiple storage backends through a unified API:

- **Filesystem**: Local and network file systems (`file://` URIs)
- **Azure Blob Storage**: Azure cloud storage (`az://` URIs)
- **Direct I/O**: High-performance O_DIRECT filesystem access (`file://` URIs)
- **Amazon S3**: AWS S3 storage (`s3://` URIs) - requires credentials

---

## Streaming API (Production Ready)

The streaming API provides memory-efficient uploads with optional compression across all backends.

### PyWriterOptions

Configuration object for streaming writers:

```python
options = s3dlio.PyWriterOptions()
options.with_compression('zstd', 6)  # Optional zstd compression, level 1-22
options.with_buffer_size(8192)      # Optional buffer size
```

### PyObjectWriter

Streaming writer object for efficient uploads:

```python
writer = s3dlio.create_filesystem_writer('file:///tmp/data.txt', options)
writer.write_chunk(b'Hello World!')
stats = writer.finalize()  # Returns (bytes_written, compressed_bytes)

# Access statistics
bytes_written = writer.bytes_written()  # Available after finalization
compressed_bytes = writer.compressed_bytes()  # Available after finalization
checksum = writer.checksum()  # CRC32C checksum if available
```

### Creator Functions

#### Filesystem Writer

```python
writer = s3dlio.create_filesystem_writer(uri, options)
```

- **URI Format**: `file:///path/to/file.txt`
- **Compression**: Automatic `.zst` extension when enabled
- **Use Cases**: Local storage, network file systems

#### Azure Blob Writer

```python
writer = s3dlio.create_azure_writer(uri, options)
```

- **URI Format**: `az://account/container/blob`
- **Environment Variables**: Requires `AZURE_BLOB_ACCOUNT`, optionally `AZURE_BLOB_CONTAINER`
- **Use Cases**: Azure cloud storage

#### Direct I/O Writer

```python
writer = s3dlio.create_direct_filesystem_writer(uri, options)
```

- **URI Format**: `file:///path/to/file.txt`
- **Requirements**: Data must be 4KB-aligned for optimal performance
- **Use Cases**: High-performance storage systems

#### S3 Writer

```python
writer = s3dlio.create_s3_writer(uri, options)
```

- **URI Format**: `s3://bucket/key`
- **Requirements**: AWS credentials configured
- **Use Cases**: Amazon S3 storage

---

## Compression Support

s3dlio provides excellent compression using Zstandard (zstd):

```python
options = s3dlio.PyWriterOptions()
options.with_compression('zstd', 9)  # Level 1-22, higher = better compression

writer = s3dlio.create_filesystem_writer('file:///tmp/data.txt', options)
writer.write_chunk(b'Repetitive data! ' * 1000)
stats = writer.finalize()

# File automatically gets .zst extension
# Typical compression ratios: 10x-100x for repetitive data
```

**Compression Features:**
- Automatic file extension handling (`.zst`)
- Configurable compression levels (1-22)
- Excellent compression ratios (often 10x-100x)
- Works across all storage backends

---

## Checkpoint System (Production Ready)

Distributed checkpointing system with optional compression:

### PyCheckpointStore

```python
# Basic checkpoint store
store = s3dlio.PyCheckpointStore('file:///tmp/checkpoints')

# With compression
store = s3dlio.PyCheckpointStore('file:///tmp/checkpoints', compression_level=6)

# Save checkpoint
test_data = b'Model weights or state'
store.save(epoch=1, step=0, name='model', data=test_data, metadata=None)

# Load latest checkpoint
loaded_data = store.load_latest()
assert loaded_data == test_data  # Data integrity preserved
```

**Checkpoint Features:**
- Version management with epoch/step tracking
- Optional zstd compression for storage efficiency
- Robust serialization/deserialization
- Metadata support for additional context

---

## S3-Specific Operations (Credential Required)

These functions work with S3 URIs and require AWS credentials:

### Core Operations

```python
# List objects
keys = s3dlio.list('s3://bucket/prefix/')

# Get object
data = s3dlio.get('s3://bucket/key')

# Upload/download (S3 only)
s3dlio.upload(['local_file.txt'], 's3://bucket/prefix/')
s3dlio.download('s3://bucket/prefix/', '/local/dir/')

# Bucket operations
s3dlio.create_bucket('my-bucket')
s3dlio.delete_bucket('my-bucket')
```

### Multipart Upload

For large S3 uploads:

```python
writer = s3dlio.MultipartUploadWriter('bucket-name', 'object-key')
# Requires S3 credentials and proper configuration
```

---

## Utility Functions

### Logging

```python
s3dlio.init_logging('info')     # Enable info-level logging
s3dlio.init_logging('debug')    # Enable debug-level logging
```

---

## Error Handling

All functions raise `RuntimeError` with descriptive messages:

```python
try:
    writer = s3dlio.create_filesystem_writer('invalid-uri', options)
except RuntimeError as e:
    print(f"Error: {e}")
```

---

## Best Practices

### Streaming Large Files

```python
options = s3dlio.PyWriterOptions()
options.with_compression('zstd', 6)
options.with_buffer_size(16384)

writer = s3dlio.create_filesystem_writer('file:///tmp/large_file.txt', options)

# Write in chunks to minimize memory usage
for chunk in large_data_source:
    writer.write_chunk(chunk)

stats = writer.finalize()
print(f"Wrote {stats[0]} bytes, compressed to {stats[1]} bytes")
```

### Cross-Backend Compatibility

```python
def create_writer_for_uri(uri):
    options = s3dlio.PyWriterOptions()
    options.with_compression('zstd', 6)
    
    if uri.startswith('file://'):
        return s3dlio.create_filesystem_writer(uri, options)
    elif uri.startswith('az://'):
        return s3dlio.create_azure_writer(uri, options)
    elif uri.startswith('s3://'):
        return s3dlio.create_s3_writer(uri, options)
    else:
        raise ValueError(f"Unsupported URI: {uri}")
```

### Direct I/O Optimization

```python
# For Direct I/O, align data to 4KB boundaries
def write_aligned_data(writer, data):
    chunk_size = 4096
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            # Pad final chunk to 4KB
            chunk += b'\x00' * (chunk_size - len(chunk))
        writer.write_chunk(chunk)
```

---

## Production Readiness Status

✅ **Production Ready:**
- Streaming API (all backends)
- Compression system
- Checkpoint system
- Python integration (no async/await required)

⚠️ **Requires Configuration:**
- S3 operations (need AWS credentials)
- Azure operations (need environment variables)

This API provides a stable, high-performance foundation for AI/ML storage workflows across multiple cloud and local storage systems.
