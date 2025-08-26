# Python API Versions Documentation

This directory contains multiple versions of `python_api.rs` to track development progress and enable easy switching between implementations.

## File Descriptions

### `src/python_api.rs` (CURRENT)
- **Status**: Active development version
- **Features**: v0.7.1 with compression_level parameter support
- **Description**: Current implementation with compression parameter wired (needs CheckpointConfig integration)
- **Compression**: Parameter accepted but not yet fully integrated
- **NumPy Functions**: Temporarily disabled (PyO3/numpy compatibility issues)

### `src/python_api.rs.v071_with_compression` (BACKUP)
- **Status**: Identical to current (backup copy)
- **Features**: Same as current version
- **Purpose**: Preserve v0.7.1 compression work

### `src/python_api.rs.current` (PRE-COMPRESSION)
- **Status**: Working baseline before compression changes
- **Features**: All core functionality working, no compression parameter
- **Description**: Stable version with working PyCheckpointStore, Writer, Reader, Stream
- **Use Case**: Fallback if compression integration causes issues

### `src/python_api.rs.git_previous` (ORIGINAL)
- **Status**: Original git version before our changes
- **Features**: Base implementation
- **Description**: Starting point with 3 disabled functions already present
- **Purpose**: Reference for understanding what changed

## Development Strategy

1. **Current Work**: Integrate compression_level parameter with CheckpointConfig
2. **Fallback**: Use `python_api.rs.current` if compression integration fails
3. **Testing**: All versions preserve core PyCheckpointStore functionality
4. **Future**: Re-enable NumPy functions once PyO3/numpy compatibility is resolved

## Key Changes by Version

### v0.7.1 Features Added:
- `compression_level` parameter in PyCheckpointStore constructor
- Updated all internal function calls to include new parameter
- Proper PyO3 signature with optional parameters

### Core Features (All Versions):
- PyCheckpointStore with strategy and multipart_threshold
- PyCheckpointWriter with distributed shard saving
- PyCheckpointStream for streaming writes
- PyCheckpointReader with manifest loading
- Framework tagging support (JAX, PyTorch, TensorFlow)

### Temporarily Disabled (All Versions):
- save_numpy_array() - PyO3/numpy compatibility
- load_numpy_array() - PyO3/numpy compatibility  
- PyValidatedCheckpointReader - PyO3/numpy compatibility

## Testing Status

- ✅ All core Python API functionality tested and working
- ✅ JAX/PyTorch/TensorFlow framework compatibility validated
- ⚠️ Compression parameter accepted but needs backend integration
- ❌ NumPy functions disabled pending compatibility fixes

## Next Steps

1. Integrate compression_level with CheckpointConfig in checkpoint module
2. Test end-to-end compression in Python API
3. Investigate PyO3/numpy compatibility for disabled functions
4. Consider enabling compression by default for optimal performance
