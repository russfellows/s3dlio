# s3dlio ObjectStore Metadata Extensions Design

## Goal
Add filesystem-style metadata operations to ObjectStore trait for comprehensive storage testing.

## New Methods to Add to ObjectStore Trait

### 1. Directory/Prefix Operations

```rust
/// Create a directory (POSIX) or prefix marker (cloud).
/// 
/// - file://: Creates actual directory with tokio::fs::create_dir_all()
/// - direct://: Creates actual directory  
/// - s3://,gs://,az://: Creates empty .keep marker object
async fn mkdir(&self, uri: &str) -> Result<()> {
    bail!("mkdir not implemented for this backend")
}

/// Remove directory (POSIX) or prefix (cloud).
/// 
/// - file://,direct://: Removes directory (must be empty unless recursive)
/// - s3://,gs://,az://: Deletes all objects under prefix
async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
    bail!("rmdir not implemented for this backend")
}
```

### 2. Metadata Update Operations

```rust
/// Update object metadata (cloud-specific).
/// 
/// For cloud backends, this typically requires copying the object with new metadata.
/// S3: CopyObject with x-amz-metadata-directive: REPLACE
/// GCS: rewrite() with new metadata
/// Azure: Copy with new metadata headers
/// 
/// For file backends, this is not applicable (use update_properties for content-type).
async fn update_metadata(&self, uri: &str, metadata: &HashMap<String, String>) -> Result<()> {
    bail!("update_metadata not supported for this backend")
}

/// Update object properties (content-type, cache-control, etc).
/// 
/// For cloud: Requires object copy with new properties
/// For file: Limited support (can set content-type via xattrs on some systems)
async fn update_properties(&self, uri: &str, properties: &ObjectProperties) -> Result<()> {
    bail!("update_properties not supported for this backend")
}
```

### 3. New ObjectProperties Struct

```rust
#[derive(Debug, Clone, Default)]
pub struct ObjectProperties {
    pub content_type: Option<String>,
    pub cache_control: Option<String>,
    pub content_encoding: Option<String>,
    pub content_language: Option<String>,
    pub content_disposition: Option<String>,
    pub expires: Option<String>,
    pub storage_class: Option<String>,  // For tier changes
}
```

## Implementation Strategy

### Phase 1: Add trait methods with default implementations (bail!)
- Add methods to ObjectStore trait in `src/object_store.rs`
- Default implementations throw "not implemented" errors
- This allows backends to opt-in incrementally

### Phase 2: Implement for FileSystemObjectStore
- Full POSIX support in `src/file_store.rs`
- mkdir: tokio::fs::create_dir_all()
- rmdir: tokio::fs::remove_dir() or remove_dir_all()
- update_properties: No-op or xattr-based (optional)

### Phase 3: Implement for DirectIOStore
- Same as FileSystemObjectStore (POSIX semantics)
- In `src/file_store_direct.rs`

### Phase 4: Implement for S3ObjectStore (optional for now)
- mkdir: PUT empty object with key ending in "/.keep"
- rmdir: List + delete all under prefix
- update_metadata: CopyObject with REPLACE directive
- In `src/object_store.rs` (S3 impl section)

### Phase 5: Update sai3-bench
- Use ObjectStore methods directly
- Remove metadata_ops.rs, fs_metadata.rs, cloud_metadata.rs
- Add OpSpec variants that call ObjectStore methods
- Much simpler!

## Benefits

1. **Cleaner Architecture**: Metadata operations are part of storage abstraction, not benchmarking tool
2. **Reusability**: Other projects can use these methods
3. **Better Testing**: s3dlio can test against real backends
4. **Incremental**: Backends implement only what makes sense for them
5. **Consistent API**: Follows existing ObjectStore patterns

## File Changes in s3dlio

- `src/object_store.rs`: Add methods to trait, add ObjectProperties struct
- `src/file_store.rs`: Implement mkdir/rmdir for FileSystemObjectStore
- `src/file_store_direct.rs`: Implement mkdir/rmdir for DirectIOStore
- (Optional) S3/GCS/Azure implementations

## File Changes in sai3-bench

- Remove: `src/metadata_ops.rs`, `src/fs_metadata.rs`, `src/cloud_metadata.rs`
- Update: `src/config.rs` - Add OpSpec::Mkdir, OpSpec::Rmdir
- Update: `src/workload.rs` - Call store.mkdir(), store.rmdir() directly
- Simpler, cleaner code!
