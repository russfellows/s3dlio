# Issue: Op-Log (Trace) Facility Only Works for S3 Backend

## Status
**Open** - Discovered during v0.8.8 testing (October 2025)

## Severity
**Medium** - Functionality limitation, not a regression

## Summary
The `--op-log` facility (operation trace logging) only records operations for S3-backed commands (`s3://` URIs). Generic storage commands using other backends (`file://`, `az://`, `direct://`) via the unified object_store trait do not generate trace records.

## Current Behavior

### ✅ Works (S3-specific commands)
```bash
# These commands write operation records to trace log:
s3-cli --op-log trace.tsv.zst get s3://bucket/key
s3-cli --op-log trace.tsv.zst put s3://bucket/key
s3-cli --op-log trace.tsv.zst delete s3://bucket/key
s3-cli --op-log trace.tsv.zst list s3://bucket/prefix
s3-cli --op-log trace.tsv.zst mp-get s3://bucket/prefix
```

### ❌ Doesn't Work (Generic storage commands)
```bash
# These commands only create header, no operation records:
s3-cli --op-log trace.tsv.zst download file:///path/to/file file:///dest/
s3-cli --op-log trace.tsv.zst upload file:///src/ s3://bucket/prefix/
s3-cli --op-log trace.tsv.zst ls file:///path/
s3-cli --op-log trace.tsv.zst download az://container/path/ /local/dest/
s3-cli --op-log trace.tsv.zst download direct:///mnt/storage/ /local/dest/
```

## Root Cause

### Architecture
The trace logging is implemented in `src/s3_ops.rs` as part of the S3-specific `S3Ops` struct:

```rust
// src/s3_ops.rs line 54-82
impl S3Ops {
    fn log_op(&self, ctx: LogContext, result: &Result<...>, bytes: u64, ...) {
        if let Some(logger) = &self.logger {
            logger.log(entry);  // Records operation to trace
        }
    }
    
    pub async fn get_object(&self, bucket: &str, key: &str) -> Result<Vec<u8>> {
        // ... performs S3 operation ...
        self.log_op(ctx, &result, bytes, first_byte_time);  // ✅ Logs here
    }
}
```

### Generic Backend Path
The unified storage commands (`download`, `upload`, `ls`) use the `ObjectStore` trait:

```rust
// src/api.rs
pub fn store_for_uri(uri: &str) -> Result<Arc<dyn ObjectStore>> {
    if uri.starts_with("s3://") {
        #[cfg(feature = "native-backends")]
        return Ok(Arc::new(S3Store::new()?));  // Different from S3Ops!
        
        #[cfg(feature = "arrow-backend")]
        return Ok(Arc::new(ArrowStore::new(uri)?));
    }
    // ... other backends (Azure, File, DirectIO)
}
```

**Key Issue**: The `ObjectStore` trait implementations don't have access to the global logger or hooks to record operations.

## Expected Behavior

All storage operations should support trace logging regardless of backend:

```bash
# Should all generate operation records:
s3-cli --op-log trace.tsv.zst download s3://bucket/key /local/file
s3-cli --op-log trace.tsv.zst download file:///data/file /dest/file  
s3-cli --op-log trace.tsv.zst download az://container/blob /dest/file
s3-cli --op-log trace.tsv.zst upload /local/file direct:///mnt/storage/file
```

Trace records should include:
- Operation type (GET, PUT, LIST, etc.)
- Backend/endpoint (s3://bucket, file://path, az://container)
- Object/file path
- Bytes transferred
- Timestamps (start, first_byte, end)
- Duration
- Errors (if any)

## Proposed Solution

### Option 1: Add Logging to ObjectStore Trait
Extend the `ObjectStore` trait to support optional operation logging:

```rust
// src/object_store.rs
#[async_trait]
pub trait ObjectStore: Send + Sync {
    async fn get(&self, path: &str) -> Result<Vec<u8>>;
    
    // Add logging hook
    fn with_logger(&mut self, logger: Option<Logger>) {
        // Default no-op, implementations can override
    }
}
```

**Pros**: Clean abstraction, works for all backends
**Cons**: Requires updating all ObjectStore implementations

### Option 2: Wrapper Pattern
Create a logging wrapper around ObjectStore:

```rust
// src/object_store_logger.rs
pub struct LoggedObjectStore {
    inner: Arc<dyn ObjectStore>,
    logger: Option<Logger>,
}

impl LoggedObjectStore {
    pub fn new(store: Arc<dyn ObjectStore>, logger: Option<Logger>) -> Self {
        Self { inner: store, logger }
    }
}

#[async_trait]
impl ObjectStore for LoggedObjectStore {
    async fn get(&self, path: &str) -> Result<Vec<u8>> {
        let start = SystemTime::now();
        let result = self.inner.get(path).await;
        self.log_operation("GET", path, &result, start);
        result
    }
    // ... wrap all methods
}
```

**Pros**: No changes to existing implementations, decorator pattern
**Cons**: Additional layer of indirection

### Option 3: Command-Level Instrumentation
Add logging at the CLI command level before/after calling store methods:

```rust
// src/bin/cli.rs
async fn download_with_logging(store: &dyn ObjectStore, from: &str, to: &str) -> Result<()> {
    let start = SystemTime::now();
    let result = store.get(from).await;
    
    if let Some(logger) = global_logger() {
        logger.log(LogEntry {
            operation: "GET",
            endpoint: extract_scheme(from),  // "file://", "s3://", etc.
            file: from,
            bytes: result.as_ref().map(|b| b.len()).unwrap_or(0) as u64,
            start_time: start,
            end_time: SystemTime::now(),
            error: result.as_ref().err().map(|e| e.to_string()),
            // ...
        });
    }
    result
}
```

**Pros**: Minimal changes, centralized in CLI
**Cons**: Duplicates logging logic, harder to maintain

## Recommendation

**Option 2 (Wrapper Pattern)** is recommended because:
1. Clean separation of concerns
2. No changes needed to existing ObjectStore implementations
3. Easy to test in isolation
4. Can be selectively applied only when `--op-log` is specified
5. Preserves warp-replay compatibility with existing TSV format

## Implementation Checklist

- [ ] Create `src/object_store_logger.rs` with LoggedObjectStore wrapper
- [ ] Implement logging hooks for all ObjectStore methods:
  - [ ] `get()` → log as GET operation
  - [ ] `put()` → log as PUT operation
  - [ ] `list()` → log as LIST operation
  - [ ] `delete()` → log as DELETE operation
  - [ ] `head()` → log as HEAD operation
- [ ] Update `src/api.rs` `store_for_uri()` to wrap stores with logger when `--op-log` is active
- [ ] Extract scheme/endpoint from URI for endpoint field in LogEntry
- [ ] Add integration tests for trace logging with file://, az://, direct:// backends
- [ ] Update documentation to clarify all backends support trace logging
- [ ] Verify warp-replay compatibility with non-S3 trace records

## Testing Requirements

```bash
# Test file:// backend
s3-cli --op-log file_test.tsv.zst download file:///tmp/test.txt /tmp/dest.txt
zstd -d file_test.tsv.zst -c | grep GET  # Should show operation record

# Test Azure backend
s3-cli --op-log azure_test.tsv.zst download az://container/blob /tmp/dest.txt
zstd -d azure_test.tsv.zst -c | grep GET  # Should show operation record

# Test DirectIO backend
s3-cli --op-log direct_test.tsv.zst download direct:///mnt/data/file /tmp/dest.txt
zstd -d direct_test.tsv.zst -c | grep GET  # Should show operation record

# Test mixed operations
s3-cli --op-log mixed_test.tsv.zst upload /tmp/file.txt s3://bucket/key
s3-cli --op-log mixed_test.tsv.zst download s3://bucket/key file:///tmp/file2.txt
# Should show PUT to S3, GET from S3, and PUT to file:// (or copy semantics)
```

## Related Files

- `src/s3_logger.rs` - Current trace logging implementation (lines 1-216)
- `src/s3_ops.rs` - S3-specific operation logging (lines 54-82)
- `src/s3_utils.rs` - Builds S3Ops with global_logger (lines 48-54)
- `src/object_store.rs` - ObjectStore trait definition
- `src/object_store_arrow.rs` - Arrow backend implementation
- `src/file_store.rs` - File backend implementation
- `src/file_store_direct.rs` - DirectIO backend implementation
- `src/azure_client.rs` - Azure backend implementation
- `src/bin/cli.rs` - CLI argument parsing and op-log initialization (lines 369-372, 558-559)

## Compatibility Notes

- Trace file format must remain compatible with warp-replay tool
- TSV schema: `idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns`
- For non-S3 backends:
  - `client_id`: Could use system username or empty string
  - `endpoint`: Should reflect backend type (file://, az://, direct://)
  - `file`: Full URI path
  - All timing and byte fields should work identically

## Priority

**Medium Priority** - This is a feature enhancement, not a critical bug. The existing S3 trace logging works correctly. This issue affects users who want to:
- Compare performance across different storage backends
- Profile local file I/O operations
- Benchmark Azure Blob Storage performance
- Analyze DirectIO performance characteristics

## Discovered In
Version v0.8.8 during testing of tracing migration (October 2025)

## Notes
The trace logging facility itself is fully functional and was correctly migrated from `log` to `tracing` crate in v0.8.8. Only the initialization message uses the tracing framework; core recording logic remains unchanged. This issue is about extending coverage to all storage backends, not fixing broken functionality.
