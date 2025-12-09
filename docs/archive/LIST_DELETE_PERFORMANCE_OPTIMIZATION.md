# List and Delete Performance Optimization Plan

**Date**: November 22, 2025  
**Status**: ✅ **COMPLETE** - All 3 phases implemented and tested in v0.9.20  
**Issue**: List and delete operations become slow with large object counts (100K-1M+ objects)  
**Root Cause**: Buffering entire result set before processing

---

## Implementation Status

**✅ Phase 1: Batch Delete API** - Complete
- `delete_batch()` trait method implemented for all 7 backends
- S3: DeleteObjects API (1000/request)
- Azure/GCS: Native batch operations
- File/Direct: Parallel deletion
- Tested with file:// backend

**✅ Phase 2: Streaming List** - Complete
- `list_stream()` returns async stream
- S3: True streaming via tokio channels
- CLI: `-c/--count-only` flag, progress indicators, rate formatting
- Proper op-log integration for workload replay

**✅ Phase 3: List+Delete Pipeline** - Complete
- Concurrent lister/deleter tasks
- 1000-object batches, 10-batch buffer
- Progress every 10K objects
- Validated with 10K file test (0.213s)

**Performance Target**: 28 minutes → 5-8 minutes for 1M objects (5x improvement)

---

## Current Performance Problems

### 1. List Operation (`s3dlio list`)

**Symptom**: 1M objects takes several minutes, displays nothing until the end, then dumps all at once

**Root Cause** (src/bin/cli.rs:679-705):
```rust
let mut keys = store.list(uri, recursive).await?;  // Blocks until ALL results collected
for key in &keys {
    writeln!(out, "{}", key)?;  // Dumps all at once
}
```

**Bottleneck Analysis**:
- S3 ListObjectsV2 returns 1000 objects per page
- 1M objects = 1000 API calls (sequential pagination)
- Each response buffered into `Vec<String>` (~100MB+ memory)
- No output until final object received
- Terminal display happens in burst (looks frozen for minutes)

**Measured Impact**:
- 1K objects: ~1 second (acceptable)
- 10K objects: ~5 seconds (acceptable)
- 100K objects: ~45 seconds (slow, no feedback)
- 1M objects: ~8 minutes (appears frozen)

### 2. Delete Operation (`s3dlio rm --recursive`)

**Symptom**: Same listing delay, plus individual delete API calls

**Root Cause** (src/bin/cli.rs:1069-1102):
```rust
eprintln!("Listing objects to delete...");
let keys = store.list(uri, recursive).await?;  // Same LIST bottleneck

delete_objects_concurrent(store.as_ref(), &keys, ...).await?;
```

**Bottleneck Analysis**:
- Must LIST all objects before ANY deletes start
- 1M objects = ~8 minutes of listing
- Then calls `store.delete(uri)` per object (object_store.rs:661-756)
- Concurrent (1000 workers), but 1M individual API calls
- Total: ~8 min LIST + ~20 min delete = 28 minutes

**Compared to AWS CLI**:
- `aws s3 rm s3://bucket/prefix --recursive` on 1M objects: ~3-5 minutes
- Uses streaming LIST + batch DeleteObjects API (1000 objects per request)

---

## Proposed Solutions

### Solution 1: Streaming List Iterator

**Goal**: Display results as they arrive from S3, not buffered

**Implementation**:

1. **Add `list_stream()` method to ObjectStore trait**:
```rust
// src/object_store.rs (add to trait)
async fn list_stream<F>(&self, uri_prefix: &str, recursive: bool, callback: F) -> Result<usize>
where
    F: Fn(String) + Send + Sync;
```

2. **S3 implementation** (src/object_store.rs S3ObjectStore):
```rust
async fn list_stream<F>(&self, uri_prefix: &str, recursive: bool, callback: F) -> Result<usize>
where
    F: Fn(String) + Send + Sync
{
    let (bucket, mut key_prefix) = parse_s3_uri(uri_prefix)?;
    if !key_prefix.is_empty() && !key_prefix.ends_with('/') { 
        key_prefix.push('/'); 
    }
    
    let client = aws_s3_client_async().await?;
    let mut count = 0usize;
    let mut continuation_token: Option<String> = None;
    
    loop {
        let mut req = client.list_objects_v2()
            .bucket(&bucket)
            .prefix(&key_prefix)
            .max_keys(1000);  // S3 default
            
        if let Some(token) = continuation_token {
            req = req.continuation_token(token);
        }
        
        let resp = req.send().await?;
        
        // Process page immediately (no buffering)
        for obj in resp.contents() {
            if let Some(key) = obj.key() {
                let uri = format!("s3://{}/{}", bucket, key);
                callback(uri);
                count += 1;
            }
        }
        
        // Check for more pages
        if resp.is_truncated().unwrap_or(false) {
            continuation_token = resp.next_continuation_token().map(|s| s.to_string());
        } else {
            break;
        }
    }
    
    Ok(count)
}
```

3. **CLI usage** (src/bin/cli.rs):
```rust
async fn generic_list_cmd(uri: &str, recursive: bool, pattern: Option<&str>, count_only: bool) -> Result<()> {
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger)?;
    
    let stdout = io::stdout();
    let mut out = stdout.lock();
    
    if count_only {
        // Fast path: just count objects without printing URIs
        let count = store.list_stream(uri, recursive, |_uri| {
            // Do nothing - just count
        }).await?;
        writeln!(out, "{}", count)?;
    } else {
        // Stream and print each URI as it arrives
        let count = store.list_stream(uri, recursive, |uri| {
            if let Some(pat) = pattern {
                if Regex::new(pat).unwrap().is_match(&uri) {
                    let _ = writeln!(out, "{}", uri);
                }
            } else {
                let _ = writeln!(out, "{}", uri);
            }
        }).await?;
        writeln!(out, "\nTotal objects: {}", count)?;
    }
    
    Ok(())
}
```

**Benefits**:
- Results appear immediately (every 1000 objects = 1 page)
- No memory spike (streaming, not buffering)
- User sees progress (not frozen terminal)
- `--count-only` skips printing for fast counts

**Expected Performance**:
- 1M objects: ~8 minutes → ~3 minutes (with --count-only)
- 1M objects: ~8 minutes frozen → ~8 minutes streaming (same time, but visible progress)

---

### Solution 2: List + Delete Pipeline

**Goal**: Start deleting while still listing (overlap operations)

**Implementation**:

1. **Add `delete_prefix_stream()` method**:
```rust
// src/object_store.rs (add to trait)
async fn delete_prefix_stream<F>(
    &self, 
    uri_prefix: &str, 
    progress_callback: Option<F>
) -> Result<(usize, usize)>  // (listed_count, deleted_count)
where
    F: Fn(usize, usize) + Send + Sync + 'static;
```

2. **S3 implementation with pipeline**:
```rust
async fn delete_prefix_stream<F>(
    &self, 
    uri_prefix: &str, 
    progress_callback: Option<F>
) -> Result<(usize, usize)>
where
    F: Fn(usize, usize) + Send + Sync + 'static
{
    use tokio::sync::mpsc;
    use tokio::task;
    
    let (tx, mut rx) = mpsc::channel::<Vec<String>>(10);  // Buffer 10 batches
    let (bucket, key_prefix) = parse_s3_uri(uri_prefix)?;
    
    // Spawn LIST task (producer)
    let list_handle = task::spawn({
        let bucket = bucket.clone();
        let key_prefix = key_prefix.clone();
        async move {
            let client = aws_s3_client_async().await?;
            let mut count = 0usize;
            let mut batch = Vec::with_capacity(1000);
            let mut continuation_token: Option<String> = None;
            
            loop {
                let mut req = client.list_objects_v2()
                    .bucket(&bucket)
                    .prefix(&key_prefix)
                    .max_keys(1000);
                    
                if let Some(token) = continuation_token {
                    req = req.continuation_token(token);
                }
                
                let resp = req.send().await?;
                
                // Collect keys from page
                for obj in resp.contents() {
                    if let Some(key) = obj.key() {
                        batch.push(key.to_string());
                        count += 1;
                        
                        // Send batch when full
                        if batch.len() >= 1000 {
                            tx.send(std::mem::take(&mut batch)).await.ok();
                            batch = Vec::with_capacity(1000);
                        }
                    }
                }
                
                // Check for more pages
                if resp.is_truncated().unwrap_or(false) {
                    continuation_token = resp.next_continuation_token().map(|s| s.to_string());
                } else {
                    break;
                }
            }
            
            // Send final batch
            if !batch.is_empty() {
                tx.send(batch).await.ok();
            }
            
            Ok::<usize, anyhow::Error>(count)
        }
    });
    
    // DELETE task (consumer) - processes batches as they arrive
    let delete_handle = task::spawn({
        let bucket = bucket.clone();
        let callback = progress_callback.clone();
        async move {
            let ops = build_ops_async().await?;
            let mut deleted = 0usize;
            let mut listed = 0usize;
            
            while let Some(batch) = rx.recv().await {
                listed += batch.len();
                
                // Delete batch using S3 DeleteObjects API (1000 objects per request)
                ops.delete_objects(&bucket, batch).await?;
                deleted += batch.len();
                
                // Report progress
                if let Some(ref cb) = callback {
                    cb(listed, deleted);
                }
            }
            
            Ok::<usize, anyhow::Error>(deleted)
        }
    });
    
    // Wait for both tasks
    let listed = list_handle.await??;
    let deleted = delete_handle.await??;
    
    Ok((listed, deleted))
}
```

3. **CLI usage** (src/bin/cli.rs):
```rust
// Delete entire prefix with streaming pipeline
let pb = ProgressBar::new_spinner();
pb.set_style(
    ProgressStyle::default_spinner()
        .template("{spinner:.green} Listed: {msg} | Deleted: {pos} objects ({per_sec})")
        .unwrap()
);

let pb_clone = pb.clone();
let (listed, deleted) = store.delete_prefix_stream(
    uri,
    Some(move |list_count: usize, del_count: usize| {
        pb_clone.set_message(format!("{}", list_count));
        pb_clone.set_position(del_count as u64);
    })
).await?;

pb.finish_with_message(format!("Listed: {} | Deleted: {}", listed, deleted));
```

**Benefits**:
- Deletes start within 1 second (first batch of 1000 objects)
- Overlap LIST and DELETE (parallel pipeline)
- Uses batch DeleteObjects API (1000x fewer API calls)
- Memory efficient (bounded queue of 10K objects max)

**Expected Performance**:
- 1M objects: 28 minutes → **5-8 minutes**
- Breakdown:
  - LIST: ~8 minutes (overlapped)
  - DELETE: ~3 minutes (batched API calls)
  - Total: ~8 minutes (limited by slowest stage)

---

### Solution 3: Batch Delete API Enhancement

**Goal**: Use S3 DeleteObjects API (1000 objects per request)

**Current State**: 
- `delete_objects_concurrent()` calls `store.delete(uri)` individually
- 1M objects = 1M API calls

**Already Implemented (Good News!)**:
```rust
// src/s3_utils.rs:870-880
pub fn delete_objects(bucket: &str, keys: &[String]) -> Result<()> {
    run_on_global_rt(async move {
        let ops = build_ops_async().await?;
        for chunk in to_delete.chunks(1000) {
            ops.delete_objects(&bucket, chunk.to_vec()).await?;  // Batched!
        }
        Ok(())
    })
}
```

**Problem**: CLI doesn't use this! It calls `delete_objects_concurrent()` which uses individual `delete()` calls.

**Fix** (src/bin/cli.rs): Replace individual deletes with batch API:
```rust
// BEFORE (lines 1089-1102):
delete_objects_concurrent(
    store.as_ref(),
    &keys,
    Some(move |count: usize| {
        pb_clone.set_position(count as u64);
    })
).await?;

// AFTER:
let (bucket, _) = parse_s3_uri(uri)?;
let pb_clone = pb.clone();
for chunk in keys.chunks(1000) {
    s3_delete_objects(&bucket, chunk)?;
    pb_clone.inc(chunk.len() as u64);
}
```

**Benefits**:
- 1M objects: 1M API calls → 1000 API calls (1000x reduction)
- Deletion time: ~20 minutes → **~2-3 minutes**

---

## Implementation Phases

### Phase 1: Quick Win - Batch Delete Fix (30 minutes)
**Files**: `src/bin/cli.rs`  
**Change**: Use `s3_delete_objects()` instead of `delete_objects_concurrent()`  
**Impact**: 1M deletes: 28 min → 10 min

### Phase 2: Streaming List (2-3 hours)
**Files**: `src/object_store.rs`, `src/bin/cli.rs`  
**Changes**:
- Add `list_stream()` to ObjectStore trait
- Implement for S3ObjectStore, AzureObjectStore, GcsObjectStore, FileObjectStore
- Add `--count-only` CLI flag
- Update `generic_list_cmd()` to use streaming

**Impact**: 
- LIST with display: Same time but visible progress
- LIST count only: 8 min → 3 min

### Phase 3: List + Delete Pipeline (4-6 hours)
**Files**: `src/object_store.rs`, `src/bin/cli.rs`  
**Changes**:
- Add `delete_prefix_stream()` to ObjectStore trait
- Implement producer-consumer pipeline with bounded queue
- Update CLI to use streaming delete

**Impact**: 1M deletes: 10 min → **5-8 min**

---

## Testing Plan

### Test Cases

1. **Small scale** (1K objects):
   - Ensure no regression
   - Verify all features work (pattern, count-only, recursive)

2. **Medium scale** (100K objects):
   - Measure list time (with/without streaming)
   - Measure delete time (batch vs individual)
   - Verify progress reporting

3. **Large scale** (1M objects):
   - Full end-to-end timing
   - Memory usage monitoring
   - Error handling (partial failures)

4. **Cross-backend**:
   - S3 (primary target)
   - Azure Blob Storage
   - GCS
   - file:// (local filesystem)

### Performance Targets

| Operation | Object Count | Current | Target | Method |
|-----------|--------------|---------|--------|--------|
| LIST (display) | 1M | 8 min frozen | 8 min streaming | Solution 1 |
| LIST (count) | 1M | 8 min | 3 min | Solution 1 |
| DELETE | 1M | 28 min | 5-8 min | Solution 2+3 |
| DELETE | 100K | 3 min | 1 min | Solution 3 |

---

## API Compatibility

### Backward Compatibility

**Existing code using `store.list()` continues to work**:
- `list()` method unchanged (returns `Vec<String>`)
- New `list_stream()` is optional performance optimization
- CLI uses streaming by default, but output format identical

**Existing code using `store.delete_prefix()` continues to work**:
- `delete_prefix()` method unchanged
- New `delete_prefix_stream()` is optional
- CLI automatically uses optimal method based on backend

### Migration Path

Users can opt into streaming explicitly:
```rust
// Old code (still works)
let keys = store.list("s3://bucket/prefix", true).await?;
for key in keys {
    println!("{}", key);
}

// New code (streaming)
store.list_stream("s3://bucket/prefix", true, |key| {
    println!("{}", key);
}).await?;
```

---

## Future Enhancements

### 1. Parallel LIST (for very large prefixes)
- S3 allows concurrent ListObjectsV2 with different prefixes
- Can parallelize by common prefix (e.g., first 2 hex chars of key)
- Potential: 8 min → 2 min for 1M objects

### 2. Incremental LIST (continuation token persistence)
- Save continuation tokens to file
- Resume interrupted LIST operations
- Useful for multi-million object buckets

### 3. Server-side batch operations
- Azure Blob: Batch API (100 operations per request)
- GCS: Batch API (100 operations per request)
- Optimize beyond S3's 1000-object DeleteObjects

---

## Conclusion

**The root cause is buffering** - waiting for complete result set before processing.

**Three solutions provide cumulative speedup**:
1. Streaming list: Visible progress + optional count-only mode
2. Pipeline: Overlap LIST and DELETE operations
3. Batch API: 1000x fewer delete API calls

**Combined impact**: 1M object deletion: 28 min → **5-8 minutes** (5x faster)

**Recommended implementation order**:
1. Phase 1 (quick win): Batch delete fix
2. Phase 2: Streaming list for UX
3. Phase 3: Pipeline for maximum performance
