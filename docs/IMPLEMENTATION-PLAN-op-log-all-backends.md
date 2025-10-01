# Implementation Plan: Op-Log Support for All Backends

**Issue**: [ISSUE-op-log-backend-support.md](./ISSUE-op-log-backend-support.md)  
**Target Version**: v0.8.11  
**Priority**: Medium  
**Complexity**: Low-Medium  
**Estimated Effort**: 3-4 hours

## Overview

Extend op-log (trace logging) facility to support all storage backends (file://, az://, direct://) using a clean wrapper pattern. Currently only S3 operations are traced.

## Current Architecture

```
┌─────────────┐
│   CLI Args  │
│  --op-log   │
└──────┬──────┘
       │
       ├─ S3-specific commands (get, put, list, etc.)
       │  └─> S3Ops struct → Logger ✅ WORKS
       │
       └─ Generic commands (download, upload, ls)
          └─> ObjectStore trait → ❌ NO LOGGING
```

## Proposed Architecture (Wrapper Pattern)

```
┌─────────────┐
│   CLI Args  │
│  --op-log   │
└──────┬──────┘
       │
       ├─ init_op_logger() → Creates global logger
       │
       └─ store_for_uri(uri, logger)
          └─> LoggedObjectStore {
                inner: Arc<dyn ObjectStore>,  // File/S3/Azure/Direct
                logger: Option<Logger>
              }
              └─> Wraps all methods with logging ✅ WORKS FOR ALL
```

## Implementation Steps

### Step 1: Create Wrapper Module (`src/object_store_logger.rs`)

**New File**: `src/object_store_logger.rs` (~200-300 lines)

**Key Components**:

1. **LoggedObjectStore struct**:
```rust
pub struct LoggedObjectStore {
    inner: Arc<dyn ObjectStore>,
    logger: Logger,
}
```

2. **Implement ObjectStore trait**:
   - Wrap every method (get, put, list, delete, etc.)
   - Record timing (start, end, duration)
   - Extract operation type and endpoint from URI
   - Log success or error
   - Delegate to inner store

3. **Helper functions**:
```rust
fn extract_endpoint(uri: &str) -> String {
    // "s3://bucket/key" → "s3://bucket"
    // "file:///path/to/file" → "file://"
    // "az://container/blob" → "az://container"
}

fn extract_file_path(uri: &str) -> String {
    // Full URI for logging
}

fn operation_type_from_method(method: &str) -> &str {
    // "get" → "GET"
    // "put" → "PUT"
    // "list" → "LIST"
}
```

4. **Logging wrapper pattern**:
```rust
async fn get(&self, uri: &str) -> Result<Vec<u8>> {
    let start = SystemTime::now();
    let thread_id = /* get thread ID */;
    
    let result = self.inner.get(uri).await;
    
    let end = SystemTime::now();
    let bytes = result.as_ref().map(|d| d.len() as u64).unwrap_or(0);
    let error = result.as_ref().err().map(|e| e.to_string());
    
    self.logger.log(LogEntry {
        idx: 0,  // Set by logger
        thread_id,
        operation: "GET".to_string(),
        client_id: String::new(),  // Could use username
        num_objects: 1,
        bytes,
        endpoint: extract_endpoint(uri),
        file: uri.to_string(),
        error,
        start_time: start,
        first_byte_time: None,  // Not tracked for ObjectStore
        end_time: end,
    });
    
    result
}
```

### Step 2: Update API Module (`src/api.rs`)

**Modifications**:

1. **Add logger parameter to `store_for_uri()`**:
```rust
// Before:
pub fn store_for_uri(uri: &str) -> Result<Arc<dyn ObjectStore>>

// After:
pub fn store_for_uri(uri: &str) -> Result<Arc<dyn ObjectStore>> {
    store_for_uri_with_logger(uri, None)
}

pub fn store_for_uri_with_logger(
    uri: &str, 
    logger: Option<Logger>
) -> Result<Arc<dyn ObjectStore>>
```

2. **Wrap stores when logger present**:
```rust
pub fn store_for_uri_with_logger(uri: &str, logger: Option<Logger>) -> Result<Arc<dyn ObjectStore>> {
    let scheme = infer_scheme(uri);
    
    let store: Arc<dyn ObjectStore> = match scheme {
        Scheme::S3 => {
            #[cfg(feature = "native-backends")]
            return Ok(Arc::new(S3Store::new()?));
            // ... other backends
        },
        Scheme::File => Arc::new(FileSystemObjectStore::new()),
        Scheme::Azure => Arc::new(AzureStore::new()?),
        Scheme::Direct => Arc::new(ConfigurableFileSystemObjectStore::new(/*...*/)),
        // ...
    };
    
    // Wrap with logger if present
    if let Some(logger) = logger {
        Ok(Arc::new(LoggedObjectStore::new(store, logger)))
    } else {
        Ok(store)
    }
}
```

3. **Maintain backward compatibility**:
   - Keep `store_for_uri()` without logger
   - Add `store_for_uri_with_logger()` variant

### Step 3: Update CLI (`src/bin/cli.rs`)

**Modifications**:

1. **Pass logger to store creation**:
```rust
// After initializing op-logger:
let logger = global_logger();

// When creating stores:
let store = store_for_uri_with_logger(&uri, logger.clone())?;
```

2. **Update command handlers**:
   - `download_cmd()` - Pass logger
   - `upload_cmd()` - Pass logger  
   - `ls_cmd()` - Pass logger
   - Any other commands using ObjectStore

3. **Example**:
```rust
async fn download_cmd(from: &str, to: &str, logger: Option<Logger>) -> Result<()> {
    let store = store_for_uri_with_logger(from, logger)?;
    let data = store.get(from).await?;
    // ... write to local file
}
```

### Step 4: Update Module Declarations (`src/lib.rs`)

```rust
// Add new module
pub mod object_store_logger;

// Re-export if needed for external use
pub use object_store_logger::LoggedObjectStore;
```

### Step 5: Testing

**Test Cases**:

1. **File backend logging**:
```bash
s3-cli --op-log file_test.tsv.zst download file:///tmp/test.txt /tmp/dest.txt
zstd -d file_test.tsv.zst -c | grep GET
# Verify: operation logged with endpoint "file://" and correct timing
```

2. **Azure backend logging**:
```bash
s3-cli --op-log azure_test.tsv.zst download az://container/blob /tmp/dest.txt
zstd -d azure_test.tsv.zst -c | grep GET
# Verify: operation logged with endpoint "az://container"
```

3. **Direct backend logging**:
```bash
s3-cli --op-log direct_test.tsv.zst download direct:///mnt/data/file /tmp/dest.txt
zstd -d direct_test.tsv.zst -c | grep GET
# Verify: operation logged
```

4. **Mixed operations**:
```bash
s3-cli --op-log mixed.tsv.zst upload /tmp/file.txt s3://bucket/key
s3-cli --op-log mixed.tsv.zst download s3://bucket/key file:///tmp/file2.txt
# Verify: Both S3 and file operations logged
```

5. **Error handling**:
```bash
s3-cli --op-log error_test.tsv.zst download file:///nonexistent /tmp/dest.txt
zstd -d error_test.tsv.zst -c | tail -1
# Verify: Error message captured in error field
```

6. **TSV format validation**:
   - Verify all fields present
   - Verify timestamps parseable
   - Verify compatible with warp-replay tool (if available)

7. **Performance verification**:
   - Measure overhead of logging wrapper (<1% expected)
   - Test with LOSSLESS mode
   - Test buffer overflow handling

## Technical Considerations

### Thread ID Extraction
```rust
fn get_thread_id() -> usize {
    // Use thread::current().id() and hash to usize
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    hasher.finish() as usize
}
```

### Endpoint Extraction
```rust
fn extract_endpoint(uri: &str) -> String {
    if let Some(pos) = uri.find("://") {
        let scheme_end = pos + 3;
        if let Some(path_start) = uri[scheme_end..].find('/') {
            return uri[..scheme_end + path_start].to_string();
        }
        return uri.to_string();
    }
    "unknown".to_string()
}
```

### First Byte Time
- ObjectStore trait doesn't expose streaming semantics
- Set `first_byte_time: None` for all operations
- Could add later with more invasive changes

### Client ID
- Could use `whoami::username()` crate
- Or leave empty for consistency with S3 ops
- Recommendation: Empty string initially, add later if needed

## Compatibility

### TSV Format
Must remain compatible with existing format:
```
idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns
```

Fields for non-S3 operations:
- `idx`: Sequential counter (auto-assigned by logger)
- `thread`: Thread ID hash
- `op`: GET, PUT, LIST, DELETE, HEAD
- `client_id`: Empty string ""
- `n_objects`: Usually 1 (or count for list operations)
- `bytes`: Bytes transferred
- `endpoint`: "file://", "az://container", "direct://", etc.
- `file`: Full URI
- `error`: Error message or empty
- `start`: RFC3339 timestamp
- `first_byte`: Empty (not tracked)
- `end`: RFC3339 timestamp
- `duration_ns`: Nanoseconds

### Backward Compatibility
- Existing S3-specific commands continue to use S3Ops logging (unchanged)
- Generic commands gain logging capability
- No breaking changes to APIs or file formats
- warp-replay tool should handle new records transparently

## Files to Modify

1. **New File**: `src/object_store_logger.rs` (~250 lines)
2. **Modify**: `src/api.rs` (~30 lines added)
3. **Modify**: `src/lib.rs` (~2 lines added)
4. **Modify**: `src/bin/cli.rs` (~50 lines modified)

**Total**: ~330 lines added/modified

## Success Criteria

✅ All storage backends (file://, s3://, az://, direct://) generate op-log entries  
✅ TSV format compatible with existing warp-replay tool  
✅ Zero compilation warnings  
✅ Performance overhead <1%  
✅ Test coverage for all backends  
✅ Documentation updated

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance overhead | Low | Use try_send (non-blocking), benchmark |
| Format incompatibility | Medium | Validate against warp-replay before commit |
| Thread safety issues | Low | Logger already thread-safe via channels |
| Missing edge cases | Low | Comprehensive test suite |

## Follow-up Enhancements (Future)

- Add `first_byte_time` tracking with streaming semantics
- Add `client_id` from username
- Support filtering by backend type
- Add trace visualization tools
- Performance profiling mode with more metrics

## Documentation Updates

- Update `docs/ISSUE-op-log-backend-support.md` with resolution
- Add usage examples to README
- Update API documentation for `store_for_uri_with_logger()`
- Add to Changelog.md for v0.8.11
