# S3DL-IO Rust API Design Document

## Current State Analysis

### Currently Exposed Public API (from lib.rs):

**Core Object Store Interface:**
- `ObjectStore` trait - Main abstraction for cloud storage
- `ObjectWriter` trait - Streaming write interface 
- `S3ObjectStore`, `FileSystemObjectStore` - Concrete implementations
- `store_for_uri()`, `direct_io_store_for_uri()` - Factory functions
- `WriterOptions`, `CompressionConfig` - Configuration types
- `Scheme`, `infer_scheme()` - URI scheme detection

**Data Loading Interface:**
- `DataLoader<D>` - Main data loader struct
- `Dataset` trait - Data source abstraction
- `LoaderOptions`, `ReaderMode`, `LoadingMode` - Configuration
- `AsyncPoolDataLoader`, `UnifiedDataLoader` - Advanced loaders

**Utility Functions:**
- `get_object_uri()`, `get_objects_parallel()` - Object retrieval
- `list_objects()`, `list_buckets()` - Listing operations
- `delete_objects()`, `stat_object_uri()` - Management operations
- `parse_s3_uri()` - URI parsing

**Checkpoint System:**
- `CheckpointStore`, `CheckpointConfig` - State management
- `CheckpointInfo` - Metadata types

**Multipart Operations:**
- `MultipartUploadConfig`, `MultipartUploadSink` - Upload management

## Proposed Clean API Structure

### Tier 1: Essential Public API (Stable)
Items that external users should rely on and we commit to maintaining.

### Tier 2: Advanced API (Stable but Optional)
More complex functionality for power users.

### Tier 3: Internal/Experimental (Unstable)
Implementation details that may change.

## Recommendations

1. **Create a clean facade** - New public interface that hides complexity
2. **Version the API** - Use semantic versioning and deprecation warnings
3. **Minimize surface area** - Only expose what's truly needed
4. **Document everything** - Comprehensive API docs with examples
5. **Add stability guarantees** - Clear compatibility promises

## Next Steps

1. Design the clean API facade
2. Implement backward compatibility
3. Add comprehensive documentation
4. Create migration guide
5. Update Python bindings to match
