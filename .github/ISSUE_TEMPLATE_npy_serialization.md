# Add In-Memory NPY Serialization Support

## Summary
Add zero-copy in-memory .npy serialization to complement existing NPZ/HDF5/TFRecord format support.

## Motivation
- **Current limitation**: ndarray-npy 0.9+ only writes to file paths, not in-memory buffers
- **Use case**: AI/ML tools (dl-driver, sai3-bench) need to generate .npy data in memory for ZIP archives or streaming
- **Performance**: Zero-copy serialization avoids temp files and intermediate buffers

## Proposed Implementation

Add function to `src/data_formats/npz.rs`:

```rust
/// Serialize ndarray to .npy format in memory with zero-copy semantics
/// Implements NPY 1.0 format: magic (6B) + version (2B) + header_len (2B) + header + data
pub fn array_to_npy_bytes<S, D>(array: &ArrayBase<S, D>) -> Result<Vec<u8>>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    // NPY 1.0 magic + version
    // Build header dict: descr, fortran_order, shape
    // Pad header to 64-byte alignment
    // Zero-copy data if contiguous (as_slice_memory_order)
    // Fallback to iteration for non-contiguous arrays
}
```

**Reference**: Working implementation in dl-driver `crates/formats/src/npz.rs::NpzFormat::array_to_npy_bytes()` (48 lines, tested, zero-copy)

## Benefits
1. **Reusability**: dl-driver, sai3-bench, and other tools can use shared implementation
2. **Python bindings**: Expose via PyO3 for numpy interop
3. **Performance**: Zero-copy when possible, no temp files
4. **Consistency**: Complements existing format support (HDF5, TFRecord)

## Technical Details
- **Format**: NPY 1.0 (magic `\x93NUMPY`, version `[1, 0]`, header dict, data)
- **Zero-copy path**: Use `as_slice_memory_order()` + `std::slice::from_raw_parts()` for contiguous arrays
- **Fallback**: Iterator-based copy for non-contiguous layouts
- **Memory**: Pre-allocate buffer with exact size to avoid reallocations

## Dependencies
- Requires `ndarray` (already present)
- No additional dependencies needed

## Testing
- Verify output matches ndarray-npy file format
- Test zero-copy path with contiguous arrays
- Test fallback path with non-contiguous arrays
- Benchmark against temp file approach

## Priority
**Medium** - Not blocking current functionality, but enables future optimizations and reduces code duplication across tools.
