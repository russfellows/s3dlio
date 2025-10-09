# Deprecation Notice: S3-Specific Python API Functions (v0.9.4)

**Status**: Deprecated as of v0.9.4  
**Removal Target**: v1.0.0  
**Scope**: Python API only (Rust internal APIs unchanged)

## Overview

Two S3-specific Python API functions have been deprecated in favor of universal URI-based alternatives. This ensures consistent API across all 5 storage backends (S3, Azure, GCS, file://, direct://).

**IMPORTANT**: This deprecation **ONLY affects the Python API**. Rust crates and internal APIs are unchanged and remain fully supported.

**NOTE**: `create_bucket()` and `delete_bucket()` are NOT deprecated - they will be made universal in future releases.

## Deprecated Functions

### 1. `list_objects(bucket, prefix, recursive)` → `list(uri, recursive, pattern)`

**OLD (Deprecated)**:
```python
import s3dlio
objects = s3dlio.list_objects("my-bucket", "data/", recursive=True)
```

**NEW (Universal)**:
```python
import s3dlio
objects = s3dlio.list("s3://my-bucket/data/", recursive=True)
```

**Migration Notes**:
- ✅ Works with S3, Azure, GCS, local files, DirectIO
- ✅ Supports optional regex pattern filtering
- ✅ Same performance characteristics

---

### 2. `get_object(bucket, key, offset, length)` → `get(uri)` or `get_range(uri, offset, length)`

**OLD (Deprecated)**:
```python
import s3dlio
data = s3dlio.get_object("my-bucket", "data/file.bin", offset=1024, length=4096)
```

**NEW (Universal)**:
```python
import s3dlio
# Full object retrieval
data = s3dlio.get("s3://my-bucket/data/file.bin")

# Range retrieval (v0.9.3+ with RangeEngine)
data = s3dlio.get_range("s3://my-bucket/data/file.bin", offset=1024, length=4096)
```

**Migration Notes**:
- ✅ Works with S3, Azure, GCS, local files, DirectIO
- ✅ `get_range()` uses concurrent RangeEngine for large files (30-50% faster)
- ✅ Automatic optimization based on file size

---

### 3. `create_bucket(bucket_name)` - NOT DEPRECATED

**Status**: ACTIVE - Will be made universal in future releases

```python
import s3dlio
s3dlio.create_bucket("my-new-bucket")  # Still supported
```

---

### 4. `delete_bucket(bucket_name)` - NOT DEPRECATED

**Status**: ACTIVE - Will be made universal in future releases

```python
import s3dlio
s3dlio.delete_bucket("my-bucket")  # Still supported
```

---

## Rust Crate API: NO CHANGES

**IMPORTANT**: If you're using s3dlio as a Rust dependency, **no changes are required**. The Rust API remains stable:

### Rust Internal APIs (Still Supported)

```rust
// These Rust functions are NOT deprecated:
use s3dlio::s3_utils::{list_objects, create_bucket, delete_bucket};
use s3dlio::s3_copy::s3_get_range;

// Example: Still works as expected
let objects = list_objects("my-bucket", "prefix/", true)?;
```

### Rust Universal APIs (Recommended)

```rust
// For new code, consider the universal ObjectStore trait:
use s3dlio::api::{store_for_uri, ObjectStore};

let store = store_for_uri("s3://my-bucket/prefix/")?;
let objects = store.list("", true, None).await?;
```

**Why the distinction?**
- Python API: Public-facing, follows universal design principles
- Rust internal APIs: Implementation details, performance-optimized, stable

---

## Timeline

- **v0.9.4 (Current)**: Deprecation warnings added
  - Functions still work
  - Compile-time warnings with `#[deprecated]`
  - Runtime warnings to stderr when called
  
- **v0.9.x → v1.0.0-rc**: Functions remain available with warnings
  - Gives users time to migrate
  - All existing code continues working
  
- **v1.0.0**: Complete removal
  - Functions removed from Python API
  - Calling them will result in `AttributeError`
  - Rust internal APIs unchanged

---

## Testing Your Migration

```bash
# Run your existing tests - you'll see warnings
python -m pytest tests/

# Expected output:
# WARNING: list_objects() is deprecated and will be removed in v1.0.0. Use list(uri, recursive, pattern) instead.
# WARNING: get_object() is deprecated and will be removed in v1.0.0. Use get(uri) or get_range(uri, offset, length) instead.
```

Update your code to use the universal APIs to silence warnings.

---

## Benefits of Migration

1. **Universal Backend Support**: Code works with S3, Azure, GCS, file://, direct://
2. **Better Performance**: RangeEngine provides 30-50% faster downloads for large files
3. **Consistent API**: Same patterns across all storage backends
4. **Future-Proof**: Aligned with s3dlio's long-term architecture

---

## Questions?

If you rely on s3dlio's Rust crates in other projects:
- ✅ **No immediate action required** - Rust APIs are stable
- ✅ Consider migrating to universal `ObjectStore` trait for new code
- ✅ S3-specific internal functions remain supported indefinitely

If you use s3dlio's Python API:
- ⚠️ Update to universal functions before v1.0.0
- ⚠️ Test migration in v0.9.x with warnings enabled
- ⚠️ Contact maintainer if migration blockers exist

---

## Related Documentation

- [v0.9.3 Release Notes](./v0.9.3-RELEASE-NOTES.md)
- [Universal API Reference](./api/UNIVERSAL-API.md)
- [RangeEngine Performance Guide](./performance/RANGE-ENGINE.md)
- [Changelog](./Changelog.md)
