# Op-Log (Trace) Facility Should Support All Storage Backends

## Issue Summary
The `--op-log` facility only records operations for S3 commands. Generic storage commands using `file://`, `az://`, or `direct://` URIs create the trace file but don't write operation records.

## Current Behavior

**Works ✅ (S3 only):**
```bash
s3-cli --op-log trace.tsv.zst get s3://bucket/key
s3-cli --op-log trace.tsv.zst put s3://bucket/key
```
→ Creates trace file with operation records

**Doesn't Work ❌ (Other backends):**
```bash
s3-cli --op-log trace.tsv.zst download file:///data/file /dest/
s3-cli --op-log trace.tsv.zst upload /src/ az://container/path/
s3-cli --op-log trace.tsv.zst ls direct:///mnt/storage/
```
→ Creates trace file with header only, no operation records

## Expected Behavior
All storage operations should write trace records regardless of backend (S3, Azure, File, DirectIO).

## Root Cause
Trace logging is implemented in `src/s3_ops.rs` and only hooks S3-specific operations. The unified `ObjectStore` trait used by generic commands (`download`, `upload`, `ls`) has no logging integration.

## Proposed Solution
**Option: Wrapper Pattern** (Recommended)

Create a `LoggedObjectStore` wrapper that instruments all ObjectStore operations:

```rust
pub struct LoggedObjectStore {
    inner: Arc<dyn ObjectStore>,
    logger: Option<Logger>,
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

**Benefits:**
- No changes to existing ObjectStore implementations
- Clean decorator pattern
- Works for all backends uniformly
- Easy to test in isolation

## Impact
**Medium Priority** - Feature enhancement, not a bug. Affects users who want to:
- Compare performance across storage backends
- Profile local file I/O or DirectIO performance
- Benchmark Azure Blob Storage operations
- Generate warp-replay compatible traces for non-S3 workloads

## Implementation Checklist
- [ ] Create `src/object_store_logger.rs` with wrapper
- [ ] Hook all ObjectStore trait methods (get, put, list, delete, head)
- [ ] Update `src/api.rs` to wrap stores when `--op-log` is specified
- [ ] Add integration tests for file://, az://, direct:// trace logging
- [ ] Verify warp-replay TSV format compatibility

## Related Files
- `src/s3_logger.rs` - Trace logging implementation
- `src/s3_ops.rs` - Current S3-specific logging hooks
- `src/object_store.rs` - ObjectStore trait definition
- `src/bin/cli.rs` - CLI `--op-log` initialization

## Version
Discovered in v0.8.8 (October 2025)

---

**See also:** `docs/ISSUE-op-log-backend-support.md` for detailed analysis and alternative solutions.
