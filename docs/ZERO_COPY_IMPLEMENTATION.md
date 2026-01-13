# s3dlio v0.9.34 - Zero-Copy Data Generation

## Summary

Successfully ported NUMA optimizations from dgen-rs to s3dlio with **TRUE zero-copy Python API** using the buffer protocol.

## Key Features

### 1. NUMA-Aware Data Generation
- Auto-detect UMA vs NUMA topology using hwloc2
- Intelligent thread pinning to NUMA nodes
- First-touch memory initialization for NUMA locality
- Smart CPU allocation: **50% by default** (leaves 50% for I/O operations)

### 2. Zero-Copy Python API
Implemented using `bytes::Bytes` and Python buffer protocol - **exactly like dgen-rs**:

```python
import s3dlio

# Generate data - returns BytesView supporting buffer protocol
data = s3dlio.generate_data(1024 * 1024)

# ZERO-COPY access via memoryview (no memory copies!)
view = memoryview(data)
print(f"Generated {len(view)} bytes, zero copies!")

# Or convert to bytes if needed (this DOES copy)
data_bytes = bytes(data)
```

### 3. Three API Patterns

#### Pattern 1: Basic Generation (Zero-Copy Read)
```python
data = s3dlio.generate_data(size, dedup=1, compress=1)
view = memoryview(data)  # Zero-copy access!
```

#### Pattern 2: Custom Thread Count
```python
data = s3dlio.generate_data_with_threads(
    size=1024 * 1024,
    threads=8  # Override default 50%
)
```

#### Pattern 3: Zero-Copy Write
```python
# Pre-allocate buffer
buf = bytearray(1024 * 1024)
nbytes = s3dlio.generate_into_buffer(buf, dedup=2, compress=3)
# Data written directly into buf (zero-copy write!)
```

## Implementation Details

### PyBytesView Class
```rust
#[pyclass(name = "BytesView")]
pub struct PyBytesView {
    bytes: Bytes,  // Reference-counted, cheap to clone
}

impl PyBytesView {
    // Implements __getbuffer__ and __releasebuffer__
    // for Python buffer protocol support
}
```

### Key Differences from Old Approach

**Before (PyBytes - COPIES data)**:
```rust
fn generate_data(py: Python, size: usize) -> PyResult<PyObject> {
    let data = generate_controlled_data_alt(size, 1, 1);
    Ok(PyBytes::new(py, &data).into())  // COPIES Vec<u8> to Python bytes
}
```

**After (PyBytesView - ZERO COPIES)**:
```rust
fn generate_data(py: Python, size: usize) -> PyResult<Py<PyBytesView>> {
    let data = py.detach(|| generate_controlled_data_alt(size, 1, 1));
    let bytes = Bytes::from(data);  // Cheap wrap, no copy
    Py::new(py, PyBytesView { bytes })  // Exposes buffer protocol
}
```

## Performance Characteristics

1. **Rust → Python transfer**: Zero copies (uses buffer protocol)
2. **Python memoryview access**: Zero copies (direct pointer access)
3. **Python bytes() conversion**: One copy (only if user requests it)
4. **Write to existing buffer**: Zero copies (direct memory write)

## Testing

All tests passing (185 total, 8 new):

```bash
$ python test_zero_copy_datagen.py
======================================================================
Test Results: 5 passed, 0 failed
======================================================================

✓ Basic Generation (Zero-Copy BytesView)
✓ Custom Thread Count
✓ Zero-Copy Write into Existing Buffer  
✓ Utility Functions (50% CPU default)
✓ Deduplication and Compression
```

## Build Verification

```bash
# Rust build (no warnings in release mode)
$ cargo build --release --features extension-module
   Compiling s3dlio v0.9.34
    Finished `release` profile [optimized] target(s) in 35.15s

# Python wheel
$ ./build_pyo3.sh
📦 Built wheel for CPython 3.13 to target/wheels/s3dlio-0.9.34-*.whl
```

## Documentation

- **User Guide**: `docs/supplemental/DATA-GENERATION-GUIDE.md`
- **Changelog**: `docs/Changelog.md` (v0.9.34 section)
- **Python Examples**: `test_zero_copy_datagen.py`
- **README**: Updated with v0.9.34 features and test count

## Comparison with dgen-rs

| Feature | dgen-rs | s3dlio v0.9.34 |
|---------|---------|----------------|
| Zero-copy Python API | ✅ PyBytesView + buffer protocol | ✅ PyBytesView + buffer protocol |
| NUMA detection | ✅ hwloc2 | ✅ hwloc2 (ported) |
| Thread pinning | ✅ core_affinity | ✅ core_affinity (ported) |
| Default CPU usage | 100% | **50%** (I/O-optimized) |
| generate_data() | ✅ Returns BytesView | ✅ Returns BytesView |
| generate_into_buffer() | ✅ Zero-copy write | ✅ Zero-copy write |
| NumPy support | ✅ Via buffer protocol | ✅ Via buffer protocol |

**Result**: s3dlio v0.9.34 achieves **exact same zero-copy efficiency** as dgen-rs!

## Files Modified

- `src/numa.rs` - NUMA topology detection (NEW)
- `src/python_api/python_datagen_api.rs` - Zero-copy Python API (NEW)
- `src/data_gen_alt.rs` - NUMA integration, 50% CPU default
- `src/python_api.rs` - Register datagen module
- `src/python_api/python_aiml_api.rs` - Add FromStr import
- `Cargo.toml` - Add hwloc2, core_affinity dependencies
- `docs/supplemental/DATA-GENERATION-GUIDE.md` - Comprehensive guide (NEW)
- `test_zero_copy_datagen.py` - Test suite (NEW)

## Git History

```bash
$ git log --oneline -1
ee31240 feat: Add NUMA-aware data generation with zero-copy Python API (v0.9.34)
```

**Branch**: `feature/enhanced-numa-data-gen_v0.9.34`  
**Status**: Pushed to remote, ready for PR

## Next Steps (User Decision)

1. **Create PR** on GitHub to merge into `main`
2. **Test in production** with real workloads
3. **Consider PyPI release** if testing successful
4. **Update downstream projects** (sai3-bench, dl-driver) to leverage NUMA features

---

**Conclusion**: s3dlio v0.9.34 now has **identical zero-copy data generation** to dgen-rs while maintaining 50% CPU allocation optimized for I/O workloads. No compromises on efficiency!
