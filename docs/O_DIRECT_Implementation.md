# O_DIRECT File I/O Implementation for s3dlio v0.5.1

## Overview

This implementation adds O_DIRECT file I/O support to s3dlio for high-performance AI/ML workloads that need to bypass the OS page cache. The implementation provides graceful fallback to regular I/O when O_DIRECT is not supported.

## Key Features

### 1. Configurable Direct I/O
- `FileSystemConfig` struct with options for:
  - `direct_io`: Enable/disable O_DIRECT
  - `alignment`: System page size alignment (auto-detected)
  - `min_io_size`: Minimum size for direct I/O operations
  - `sync_writes`: Enable O_SYNC for guaranteed durability

### 2. System Page Size Detection
- Automatic detection using `libc::sysconf(_SC_PAGESIZE)`
- Fallback to 4096 bytes if detection fails
- Supports page sizes from 512 bytes to 64KB

### 3. Graceful Fallback
- Automatic fallback to regular I/O when:
  - Filesystem doesn't support O_DIRECT
  - File size is below `min_io_size` threshold
  - O_DIRECT operations fail for any reason

### 4. Factory Functions
- `direct_io_store_for_uri()`: Basic direct I/O configuration
- `high_performance_store_for_uri()`: Optimized for large files

## Implementation Details

### Files Modified/Added
- `src/file_store_direct.rs`: Complete O_DIRECT implementation
- `src/object_store.rs`: Enhanced factory functions
- `src/lib.rs`: Module exports
- `Cargo.toml`: Added `libc` dependency
- `tests/test_direct_io.rs`: Comprehensive test suite

### Key Methods
- `ConfigurableFileSystemObjectStore::new(config)`
- `test_direct_io_support()`: Filesystem compatibility check
- `try_write_file_direct()` / `try_read_range_direct()`: O_DIRECT implementations
- `get_system_page_size()`: Dynamic page size detection

## Usage Examples

```rust
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::{direct_io_store_for_uri, high_performance_store_for_uri};

// Basic direct I/O configuration
let config = FileSystemConfig::direct_io();
let store = ConfigurableFileSystemObjectStore::new(config);

// Or use factory functions
let store = direct_io_store_for_uri("file:///path/to/data")?;
let hp_store = high_performance_store_for_uri("file:///path/to/large/files")?;

// All ObjectStore methods work with automatic fallback
store.put(&uri, data).await?;
let data = store.get(&uri).await?;
let range = store.get_range(&uri, offset, length).await?;
```

## Testing

The implementation includes a comprehensive test suite (`tests/test_direct_io.rs`) with 10 tests covering:

1. **Basic Operations**: Core put/get/delete with O_DIRECT and fallback
2. **Range Reads**: Aligned and unaligned range operations
3. **Large Files**: Multi-megabyte file handling
4. **Alignment**: Various data sizes and alignment scenarios
5. **Page Size Detection**: Cross-platform page size detection
6. **Fallback Mechanisms**: Graceful degradation when O_DIRECT fails
7. **Factory Functions**: Convenience constructors
8. **Configuration**: Direct I/O vs regular I/O configuration
9. **Baseline Testing**: Regular filesystem operations as reference
10. **High Performance**: Optimized configurations for large workloads

## Performance Benefits

For AI/ML workloads, O_DIRECT provides:
- **Cache Bypass**: Avoids polluting OS page cache with large training data
- **Predictable Performance**: Eliminates cache eviction effects
- **Memory Efficiency**: Reduces memory pressure from cached file data
- **Latency Consistency**: More predictable I/O latency patterns

## Platform Compatibility

- **Unix/Linux**: Full O_DIRECT support with `libc` flags
- **Other Platforms**: Automatic fallback to regular I/O
- **Filesystems**: Works with ext4, xfs, and other O_DIRECT-compatible filesystems
- **Alignment**: Automatically detects and uses system page size

## Error Handling

The implementation handles various error conditions gracefully:
- Filesystem doesn't support O_DIRECT → fallback to regular I/O
- Alignment issues → automatic padding and trimming
- Permission issues → standard filesystem error reporting
- Invalid file paths → standard path validation errors

This implementation successfully addresses the user's requirement to "find a way that we can pass an option that will make all I/O be direct" while providing robust fallback mechanisms for maximum compatibility.
