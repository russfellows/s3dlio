# s3dlio Performance Optimization - 3-Phase Implementation Guide

## Overview

This document outlines a comprehensive 3-phase plan to optimize s3dlio performance from ~5.5 GB/s to 8-10 GB/s, based on analysis of your current codebase and bottlenecks.

## Performance Goals

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| **GET Throughput** | 5.5 GB/s | 8-10 GB/s | 8-10 GB/s | 8-10 GB/s |
| **PUT Throughput** | ~3 GB/s | ~3 GB/s | 6-8 GB/s | 6-8 GB/s |
| **Memory Usage** | Object size | Object size | ~16MB constant | ~16MB constant |
| **Python Overhead** | High copies | High copies | Medium copies | Near zero-copy |

## Phase Independence

- **Phase 1**: Completely independent - can be implemented first
- **Phase 2**: Independent of Phase 1, but beneficial to combine  
- **Phase 3**: Benefits from Phase 2's streaming writers, includes fallback for compatibility

## Quick Start - Which Phase to Implement First?

### For Maximum GET Performance Immediately → **Start with Phase 1**
- Biggest single impact on current 5.5 GB/s bottleneck
- Minimal code changes, mostly configuration and new GET path
- Zero breaking changes

### For Memory-Efficient PUT Operations → **Start with Phase 2**  
- Eliminates object-size memory usage
- Streaming uploads start immediately
- New API, existing PUT unchanged

### For Python Framework Integration → **Start with Phase 3**
- Requires Phase 2 for optimal results (but includes fallback)
- Transforms Python experience for ML workflows

## Implementation Phases

### Phase 1: Runtime & Concurrent GET Optimization 
**Target: 5.5 GB/s → 8-10 GB/s for large objects**

#### Key Changes:
1. **Runtime Scaling** (`src/s3_client.rs`)
   - Environment-tunable worker threads (default: `max(8, cores*2)`)
   - Replace hard-coded 2 threads

2. **HTTP Pool Optimization** (`src/s3_client.rs`)
   - Configurable connection pool (default: 512 connections)
   - Tunable timeouts and keep-alive settings

3. **Concurrent Range GET** (`src/s3_utils.rs`)  
   - Pre-allocated buffer concurrent range requests
   - Intelligent size thresholds (32MB+)
   - Environmental tuning for part size and concurrency

4. **ObjectStore Enhancement** (`src/object_store.rs`)
   - New `get_optimized()` method with auto-optimization
   - Backwards compatible with existing `get()`

#### Environment Variables:
```bash
export S3DLIO_RT_THREADS=16
export S3DLIO_HTTP_MAX_CONNS=512
export S3DLIO_GET_PART_SIZE=8388608     # 8MB
export S3DLIO_GET_INFLIGHT=64
export S3DLIO_GET_THRESHOLD=33554432    # 32MB
```

#### Files Modified:
- `src/s3_client.rs` - Runtime and HTTP optimization
- `src/s3_utils.rs` - Concurrent range GET implementation  
- `src/object_store.rs` - Enhanced GET interface
- `src/bin/cli.rs` - CLI integration
- `Cargo.toml` - Add `num_cpus` dependency

---

### Phase 2: Streaming Multipart PUT Optimization
**Target: Eliminate buffering, reduce copies, 2-3x PUT throughput**

#### Key Changes:
1. **Zero-Copy Multipart** (`src/multipart.rs`)
   - Replace `Vec<u8>` with `BytesMut/Bytes`
   - Stream parts as data arrives, not on finish()
   - `split_to()` for zero-copy part creation

2. **Streaming Writer Trait** (`src/object_store.rs`)
   - Universal `ObjectWriter` interface
   - `write_chunk()` and `write_owned()` methods
   - Support for all backends (S3, FS, Azure)

3. **S3 Streaming Implementation**
   - Real-time multipart uploads
   - Configurable part size and concurrency
   - Semaphore-controlled in-flight limits

4. **FileSystem Streaming**
   - Temp file with atomic rename
   - Streaming writes with immediate flushing

#### Environment Variables:
```bash
export S3DLIO_PUT_PART_SIZE=8388608     # 8MB parts  
export S3DLIO_PUT_INFLIGHT=64           # High concurrency
```

#### Files Modified:
- `src/multipart.rs` - Zero-copy streaming implementation
- `src/object_store.rs` - Streaming writer trait and S3 impl
- `src/file_store.rs` - FileSystem streaming writer
- `src/bin/cli.rs` - Streaming PUT command

#### Benefits:
- **Memory**: Constant ~16MB instead of full object size
- **Latency**: First byte uploads immediately
- **Throughput**: 2-3x improvement from streaming + concurrency

---

### Phase 3: Python Zero-Copy Streaming API  
**Target: Eliminate Python/Rust boundary copies, framework integration**

#### Key Changes:
1. **Python Streaming Writer** (`src/python_api/python_core_api.rs`)
   - File-like `PyObjectStream` class
   - Zero-copy via Python buffer protocol
   - Context manager support (`with` statements)

2. **Zero-Copy GET** (`src/python_api/python_core_api.rs`)
   - `get_bytes_view()` returning `memoryview`
   - Optional `writable=True` for mutable access
   - Concurrent GET integration

3. **Framework Integration** (`python/s3dlio/streaming.py`)
   - `save_torch_state()`, `load_torch_state()`
   - `save_pickle()`, `load_pickle()` 
   - `save_numpy()`, `load_numpy()`
   - Direct streaming to `torch.save()`, `pickle.dump()`

4. **Enhanced Module Registration** (`src/python_api.rs`)
   - Register streaming classes and functions
   - Maintain backwards compatibility

#### Python Usage:
```python
import s3dlio
from s3dlio.streaming import save_torch_state, load_torch_state

# Streaming save (no large Python buffers)
metadata = save_torch_state("s3://bucket/model.pth", state_dict)

# Zero-copy load (memoryview of Rust data)  
state_dict = load_torch_state("s3://bucket/model.pth")

# Manual streaming
with s3dlio.open_stream("s3://bucket/data.bin") as stream:
    stream.write(large_numpy_array)  # Zero-copy from NumPy
```

#### Files Modified:
- `src/python_api/python_core_api.rs` - Streaming classes
- `src/python_api.rs` - Module registration
- `python/s3dlio/streaming.py` - Framework helpers
- `python/tests/test_streaming_api.py` - Comprehensive tests

#### Benefits:
- **Python → Rust**: Near zero-copy via buffer protocol
- **Rust → Python**: True zero-copy via memoryview
- **Framework Support**: Direct integration with PyTorch, JAX, TensorFlow
- **Memory**: Constant usage regardless of object size

---

## Testing Strategy

### Phase 1 Testing:
```bash
# Large object GET performance
time ./target/release/s3-cli get s3://bucket/large-file.bin /tmp/test.bin

# Compare single vs concurrent  
S3DLIO_GET_INFLIGHT=1 time ./target/release/s3-cli get s3://bucket/large-file.bin /tmp/single.bin
S3DLIO_GET_INFLIGHT=128 time ./target/release/s3-cli get s3://bucket/large-file.bin /tmp/concurrent.bin
```

### Phase 2 Testing:
```bash
# Memory usage monitoring during large PUT
/usr/bin/time -v ./target/release/s3-cli put-streaming s3://bucket/large.bin /path/to/large-file.bin

# Throughput comparison
time ./target/release/s3-cli put s3://bucket/buffered.bin large-file.bin
time ./target/release/s3-cli put-streaming s3://bucket/streaming.bin large-file.bin
```

### Phase 3 Testing:
```python
# Framework integration tests
import time
start = time.time()
metadata = save_torch_state("s3://bucket/model.pth", large_state_dict)
print(f"Streaming save: {time.time() - start:.2f}s, {metadata['size']/1e9:.2f} GB")

# Memory usage validation
import psutil
process = psutil.Process()
before_mem = process.memory_info().rss
state_dict = load_torch_state("s3://bucket/model.pth")  
after_mem = process.memory_info().rss
print(f"Memory delta: {(after_mem - before_mem)/1e6:.1f} MB")
```

## Expected Performance Results

### Local S3 (MinIO) - Same as warp testing:
- **GET**: 8-12 GB/s for large objects (64GB+)
- **PUT**: 6-10 GB/s with streaming multipart
- **Memory**: <50MB peak usage for any object size
- **Python**: 90%+ of native Rust performance

### Production S3:
- **GET**: Limited by bandwidth, but optimal utilization
- **PUT**: Improved latency to first byte, better concurrency
- **Cost**: Reduced due to better connection reuse

## Migration Path

1. **Phase 1**: Drop-in performance improvement, no API changes
2. **Phase 2**: New streaming APIs alongside existing buffered APIs
3. **Phase 3**: Enhanced Python capabilities, backwards compatible

Each phase provides immediate value and can be deployed independently.
