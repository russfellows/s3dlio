# S3DLIO API Documentation

## Current Documentation (Recommended)

### Python API v0.8.0+ (September 2025)
- **[Python API Reference v0.8.0](python-api-v0.8.0-current.md)** - **✅ CURRENT** 
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

For **s3dlio v0.8.0+**, use the enhanced API:

```python
import s3dlio

# ✅ NEW: Multi-backend dataset creation
dataset = s3dlio.create_dataset("file:///path/to/data")
dataset = s3dlio.create_dataset("s3://bucket/prefix/")

# ✅ NEW: Async data loading  
loader = s3dlio.create_async_loader("file:///path/to/data")

# ✅ FIXED: PyTorch integration now works
from s3dlio.torch import S3IterableDataset
dataset = S3IterableDataset("file:///training/data/")

# ✅ LEGACY: Still works unchanged
data = s3dlio.get("s3://bucket/object")
s3dlio.put("s3://bucket/object", data)
```

## Version Support

| Version | Status | Python API Doc | Notes |
|---------|--------|----------------|-------|  
| v0.8.0+ | ✅ Current | `python-api-v0.8.0-current.md` | Enhanced API, PyTorch fixed |
| v0.7.x  | ⚠️ Legacy | `python-api-legacy-v0.7.md` | PyTorch broken |
| v0.6.x- | 🚫 Unsupported | Contact maintainers | Very old |

---

**💡 Recommendation**: Always use the current v0.8.0+ API documentation for new development. Legacy docs are provided only for understanding older releases.