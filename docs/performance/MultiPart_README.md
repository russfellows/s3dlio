# s3dlio Multipart Upload — Streaming, Zero‑Copy

This document explains how to use the **streaming, concurrent multipart upload (MPU)** feature in s3dlio from both **Rust** and **Python**, with a focus on **zero‑copy** data paths and high throughput.

> TL;DR — For best performance, use the **`reserve()` / `commit()`** API in Python (true zero‑copy into Rust), or call `write_blocking(&[u8])` / `write_owned_blocking(Vec<u8>)` in Rust. Close with `close()` / `finish_blocking()` to complete the MPU.

---

## Table of Contents

* [Overview](#overview)
* [S3 Requirements & Constraints](#s3-requirements--constraints)
* [Configuration](#configuration)
* [Rust API](#rust-api)

  * [Quick start](#quick-start)
  * [Full example](#full-example)
* [Python API](#python-api)

  * [Quick start (zero‑copy)](#quick-start-zero-copy)
  * [Writing from bytes / memoryview / NumPy](#writing-from-bytes--memoryview--numpy)
  * [PyTorch / JAX / TensorFlow notes](#pytorch--jax--tensorflow-notes)
* [Advanced](#advanced)

  * [Flushing partial parts](#flushing-partial-parts)
  * [Aborting an MPU](#aborting-an-mpu)
  * [Content type / metadata](#content-type--metadata)
  * [Throughput tuning](#throughput-tuning)
* [Large Object Downloads](#large-object-downloads)
* [Operational guidance](#operational-guidance)
* [FAQ](#faq)

---

## Overview

`s3dlio` implements a **streaming, concurrent** S3 Multipart Upload pipeline. Large payloads are automatically split into parts and uploaded in parallel with bounded concurrency. The design goal is **multiple GB/s** of sustained throughput to local S3-compatible stores (e.g., MinIO or scale‑out S3 gateways).

Key properties:

* **Streaming**: no staging to temp files by default.
* **Concurrent**: multiple in‑flight `UploadPart` calls, limited by a semaphore.
* **Zero‑copy (Python)**: `reserve()` exposes a Rust‑owned buffer as a writable Python `memoryview`; `commit()` hands the buffer to the uploader without copying.
* **Robust finish**: `close()` / `finish_blocking()` joins outstanding part tasks and issues `CompleteMultipartUpload`.

---

## S3 Requirements & Constraints

* **Minimum part size**: 5 MiB for all parts except the final part.
* **Number of parts**: up to 10,000 parts per object (standard S3 limits).
* **Part ordering**: parts are numbered starting at 1; s3dlio handles ordering.
* **ETag**: the final ETag often reflects a multipart hash; don’t treat it as MD5 of the whole object.

---

## Configuration

`MultipartUploadConfig` (Rust) / keyword args (Python):

* `part_size: usize` — Target size per part in bytes (default: 16 MiB). **Must be ≥ 5 MiB**.
* `max_in_flight: usize` — Max concurrent `UploadPart` requests (default: 16).
* `abort_on_drop: bool` — Best‑effort abort if the writer is dropped without finishing (default: true).
* `content_type: Option<String>` — Optional `Content-Type` set at `CreateMultipartUpload`.

---

## Rust API

### Quick start

```rust
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};

let cfg = MultipartUploadConfig { part_size: 32<<20, max_in_flight: 32, ..Default::default() };
let mut w = MultipartUploadSink::new("my-bucket", "path/to/obj.bin", cfg)?;

// Stream in chunks from your source
for chunk in source_iter() {
    w.write_blocking(&chunk)?;        // &[u8]
}

// Or hand over owned buffers to avoid extra copies in Rust
// w.write_owned_blocking(chunk_vec)?; // Vec<u8>

let info = w.finish_blocking()?;      // Complete MPU
println!("uploaded {} bytes in {} parts", info.total_bytes, info.parts);
```

### Full example

```rust
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};

fn upload_example() -> anyhow::Result<()> {
    let cfg = MultipartUploadConfig {
        part_size: 32 * 1024 * 1024,   // 32 MiB
        max_in_flight: 16,
        content_type: Some("application/octet-stream".into()),
        ..Default::default()
    };
    let mut w = MultipartUploadSink::new("my-bucket", "dir/checkpoint.bin", cfg)?;

    // Write 65 MiB in 8 MiB slices; last < part_size is fine as final part
    let block = 8 * 1024 * 1024;
    let total = (64 * 1024 * 1024) + (1 * 1024 * 1024);
    let pattern = vec![0xABu8; block];

    let mut remaining = total;
    while remaining >= block {
        w.write_blocking(&pattern)?;
        remaining -= block;
    }
    if remaining > 0 { w.write_blocking(&vec![0xAB; remaining])?; }

    let info = w.finish_blocking()?;
    println!("ETag={:?}, parts={}, bytes={}", info.e_tag, info.parts, info.total_bytes);
    Ok(())
}
```

---

## Python API

### Quick start (zero‑copy)

```python
from s3dlio import MultipartUploadWriter

w = MultipartUploadWriter.from_uri(
    "s3://my-bucket/dir/checkpoint.bin",
    part_size=32<<20,
    max_in_flight=16,
)

N = 64 * 1024 * 1024
mv = w.reserve(N)      # Rust‑owned buffer → writable memoryview
mv[:] = b"\xAB" * N   # Fill from Python without copying
w.commit(N)            # Hand ownership to Rust uploader

del mv                 # Don’t use mv after commit

info = w.close()       # Complete MPU
print(info)
```

### Writing from bytes / memoryview / NumPy

* `write(data)` accepts any **bytes‑like** object (e.g., `bytes`, `bytearray`, `memoryview`, NumPy buffer). It copies **once** into Rust for ownership, then streams.
* Large buffers (≥ 8 MiB) use an owned fast path internally to avoid additional copies; small buffers use a borrowed slice path.

```python
# bytes / bytearray
w.write(b"\xCD" * (4 << 20))

# memoryview on NumPy (C‑contiguous) — convenient, still one owning copy in Rust
import numpy as np
arr = np.zeros(32 << 20, dtype=np.uint8)
w.write(memoryview(arr))
```

### PyTorch / JAX / TensorFlow notes

* **Best performance** still comes from `reserve()` / `commit()`.
* For **PyTorch (CPU, contiguous)**: `tensor.numpy()` is zero‑copy into NumPy; then use `memoryview` on that array.

  ```python
  import torch, numpy as np
  t = torch.full((32 << 20,), 0xEE, dtype=torch.uint8, device='cpu').contiguous()
  arr = t.numpy()  # view, no copy (CPU/contiguous)
  w.write(memoryview(arr))
  ```
* **GPU tensors** require staging to host memory first (`.cpu()`), which will copy; prefer `reserve()/commit()` to avoid extra allocations.
* **JAX/TF**: Ensure you have a CPU contiguous buffer (e.g., NumPy view) before calling `write()`.

---

## Advanced

### Flushing partial parts

`flush()` forces any buffered bytes to upload as a (possibly short) final part **without** finishing the MPU. Typically you don’t need this; `close()` will upload any tail automatically.

### Aborting an MPU

`abort()` cancels the upload and best‑effort cleans up the server state. After abort, the writer is unusable.

### Content type / metadata

You can set `content_type` on creation. Future versions may expose additional headers/metadata if needed.

### Throughput tuning

* **`part_size`**: 16–64 MiB works well; too small increases overhead, too large reduces concurrency benefits.
* **`max_in_flight`**: 8–64 typical for fast local S3 backends. Watch endpoint’s concurrency limits.
* Ensure your client and endpoint are on the same high‑bandwidth network; enable jumbo frames as appropriate.

---
## Large Object Downloads

While this document focuses on **multipart uploads**, s3dlio also supports **parallel range downloads** for large objects.

### Overview

For large S3 objects (> 64 MB), s3dlio can automatically split the download into concurrent range requests, significantly improving throughput by hiding network latency.

**Default behavior:**
- Disabled by default to avoid HEAD request overhead on small objects
- Must be explicitly enabled via environment variable

**When to enable:**
- Large ML checkpoints (> 100 MB)
- Video files, large datasets, archives
- Workloads where objects are consistently large

**When to keep disabled:**
- Many small objects (< 64 MB)
- Latency-sensitive workloads
- Mixed object sizes

### Python Example

```python
import os
# Enable parallel range downloads for large objects
os.environ['S3DLIO_ENABLE_RANGE_OPTIMIZATION'] = '1'
os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '64'  # Objects ≥ 64 MB use parallel ranges

import s3dlio

# Large checkpoint automatically uses parallel download
data = s3dlio.get("s3://bucket/checkpoint-148mb.bin")  # 25-50% faster than single GET
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_ENABLE_RANGE_OPTIMIZATION` | `0` (disabled) | Enable parallel range downloads |
| `S3DLIO_RANGE_THRESHOLD_MB` | `64` | Minimum size to trigger parallel download |
| `S3DLIO_RANGE_CONCURRENCY` | Auto (8-32) | Number of parallel range requests |
| `S3DLIO_CHUNK_SIZE` | Auto (1-8 MB) | Size of each range chunk |

### Performance Example

**Actual benchmark results (16x 148 MB objects, MinIO):**

| Threshold | Time | Throughput | Speedup |
|-----------|------|------------|---------|
| Disabled (baseline) | 5.52s | 429 MB/s (0.42 GB/s) | 1.00x |
| 8 MB | 3.50s | 676 MB/s (0.66 GB/s) | **1.58x** (58% faster) |
| 16 MB | 3.27s | 725 MB/s (0.71 GB/s) | **1.69x** (69% faster) |
| 32 MB | 3.23s | 732 MB/s (0.71 GB/s) | **1.71x** (71% faster) |
| **64 MB (default)** | **3.14s** | **755 MB/s (0.74 GB/s)** | **1.76x (76% faster)** 🏆 |
| 128 MB | 3.22s | 735 MB/s (0.72 GB/s) | **1.71x** (71% faster) |

**Key findings:**
- **64 MB threshold (default) is optimal**, achieving 76% improvement over single GET
- Sweet spot: 16-64 MB thresholds provide 69-76% faster downloads
- Even aggressive 8 MB threshold shows 58% improvement
- 128 MB threshold still performs well (71% faster) but slightly suboptimal when object size is just above threshold
- HEAD overhead: ~10-20ms (well amortized over parallel download)

**Conservative tuning:**
```bash
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=1
export S3DLIO_RANGE_THRESHOLD_MB=64
```

**Aggressive tuning (for very large objects > 500 MB):**
```bash
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=1
export S3DLIO_RANGE_THRESHOLD_MB=128
export S3DLIO_RANGE_CONCURRENCY=32
export S3DLIO_CHUNK_SIZE=16777216  # 16 MB chunks
```

**See also:** [Environment Variables Reference](../api/Environment_Variables.md)

---
## Operational guidance

* **Don’t reuse** the memoryview returned by `reserve()` after `commit()`.
* Ensure **contiguous** and **CPU‑resident** buffers when using `write()` with framework tensors.
* Buckets/objects are created in whatever region/endpoint your s3dlio client is configured for (see project README for env vars).

---

## FAQ

**Q: Do I have to use `reserve()`?**
A: No. It’s the fastest path, but `write()` works with any bytes‑like object and is simpler.

**Q: What happens if a part upload fails?**
A: `close()` returns an error. You can call `abort()` to clean up. (A future version may auto‑abort on the first failure.)

**Q: Can I stream more data after `close()`?**
A: No. Create a new writer for another object.

**Q: Minimum part size?**
A: 5 MiB (S3 rule) for all but the final part.

---

*Last updated:* this document accompanies s3dlio ≥ 0.4.4 with streaming multipart upload support.

