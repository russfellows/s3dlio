# S3DLIO API Documentation

## Current Documentation (Recommended)

### API v0.9.0 (October 2025) - **✅ LATEST**

- **[Rust API Guide v0.9.0](rust-api-v0.9.0.md)** - **✅ CURRENT**
  - Complete Rust library reference
  - **"What's Changed Since v0.8.22"** migration section
  - Breaking change: `ObjectStore::get()` returns `Bytes` instead of `Vec<u8>`
  - Adaptive tuning documentation
  - Storage backends, data loaders, checkpoints
  - Performance tips and complete examples

- **[Python API Guide v0.9.0](python-api-v0.9.0.md)** - **✅ CURRENT**
  - Complete Python library reference
  - **"What's Changed Since v0.8.22"** migration section
  - Minimal breaking changes (removed deprecated functions)
  - Framework integration (PyTorch, TensorFlow, JAX)
  - Quick start, common patterns, troubleshooting
  - Migration checklist

- **[Adaptive Tuning Guide](../ADAPTIVE-TUNING.md)** - Optional performance optimization
  - Auto-tuning for different workload types
  - Usage examples and customization
  - When to use adaptive tuning

### Previous Versions

#### Python API v0.8.0+ (September 2025)
- **[Python API Reference v0.8.0](python-api-v0.8.0-current.md)** - **⚠️ SUPERSEDED by v0.9.0**
  - Complete API reference with working status for each function
  - Enhanced multi-backend support (file://, s3://, az://, direct://)
  - Fixed PyTorch integration bug
  - Backward compatible with all older versions

- **[Enhanced API Guide v0.8.0](enhanced-api-v0.8.0.md)** - Technical deep-dive
  - Detailed function signatures and parameters
  - Error handling patterns and best practices
  - Performance optimization guidelines

- **[Migration Guide v0.8.0](migration-guide-v0.8.0.md)** - Upgrade instructions
  - How to migrate from v0.7.x to v0.8.0
  - New features overview
  - Compatibility notes

## Legacy Documentation (Historical)

### Python API v0.7.x (2024-Early 2025)  
- **[Python API Reference v0.7.x](python-api-legacy-v0.7.md)** - **⚠️ LEGACY ONLY**
  - Original streaming API documentation  
  - **Known Issue**: PyTorch integration was broken
  - Use only for understanding older releases

## Other Documentation

### Environment & Setup
- **[Environment Variables](Environment_Variables.md)** - Configuration reference

---

## Quick Start (Current Version)

For **s3dlio v0.9.0**, use the latest API:

**Rust**:
```rust
use s3dlio::api::*;
use bytes::Bytes;

// Create storage backend (auto-detects from URI)
let store = store_for_uri("s3://bucket/prefix/")?;

// Read file (returns Bytes, not Vec<u8>)
let data: Bytes = store.get("s3://bucket/file.bin")?;

// Adaptive tuning (optional)
let opts = LoaderOptions::default()
    .with_batch_size(32)
    .with_adaptive();  // Auto-optimize

// Create async loader
let loader = store.create_async_loader("s3://bucket/data/", opts)?;
```

**Python**:
```python
import s3dlio

# Read file
data = s3dlio.get("s3://bucket/file.bin")  # Returns bytes

# Create async loader with adaptive tuning
loader = s3dlio.create_async_loader(
    uri="s3://bucket/data/",
    opts={
        'batch_size': 32,
        'adaptive': {'mode': 'enabled'}  # Auto-optimize
    }
)

# PyTorch integration
import torch
import numpy as np
import io

async for batch in loader:
    for item_bytes in batch:
        npz = np.load(io.BytesIO(item_bytes))
        tensor = torch.from_numpy(npz['data'])
```

## Version Support

| Version | Status | Rust API Doc | Python API Doc | Notes |
|---------|--------|--------------|----------------|-------|
| v0.9.0  | ✅ Current | `rust-api-v0.9.0.md` | `python-api-v0.9.0.md` | Bytes migration, adaptive tuning |
| v0.8.0+ | ⚠️ Superseded | N/A | `python-api-v0.8.0-current.md` | Enhanced API, PyTorch fixed |
| v0.7.x  | ⚠️ Legacy | N/A | `python-api-legacy-v0.7.md` | PyTorch broken |
| v0.6.x- | 🚫 Unsupported | N/A | Contact maintainers | Very old |

---

**💡 Recommendation**: Always use the current v0.9.0 API documentation for new development. See migration guides for upgrade instructions from v0.8.x.