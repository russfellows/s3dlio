# CancellationToken Infrastructure Implementation Plan

**Date**: October 8, 2025  
**Status**: 🔄 **IN PROGRESS**

## Goal

Add comprehensive CancellationToken support throughout the DataLoader infrastructure to enable:
- Graceful shutdown on Ctrl-C / SIGTERM
- Clean cancellation of in-flight async operations
- No orphaned background tasks
- Cooperative cancellation in prefetch loops

## Current State Analysis

### ✅ Already Has CancellationToken Support

**RangeEngine** (`src/range_engine_generic.rs`):
- Already accepts `Option<CancellationToken>` parameter
- Checks `is_cancelled()` before starting each range download
- Test coverage: `test_cancellation()` passes ✅
- **No changes needed!**

**Status**: ✅ **COMPLETE**

### ❌ Needs CancellationToken Support

1. **DataLoader** (`src/data_loader/dataloader.rs`)
   - `stream()` spawns background tasks
   - No cancellation mechanism
   - Tasks run until channel closes or error

2. **AsyncPoolDataLoader** (`src/data_loader/async_pool_dataloader.rs`)
   - `stream_with_pool()` spawns pool workers
   - No cancellation mechanism
   - Workers run until dataset exhausted

3. **Prefetch Helper** (`src/data_loader/prefetch.rs`)
   - `spawn_prefetch()` creates async prefetch loop
   - No cancellation mechanism
   - Runs until producer errors

## Implementation Strategy

### Phase 1: Add CancellationToken to LoaderOptions

**File**: `src/data_loader/options.rs`

```rust
use tokio_util::sync::CancellationToken;

pub struct LoaderOptions {
    // ... existing fields ...
    
    /// Optional cancellation token for graceful shutdown
    /// When set, all async operations will cooperatively cancel
    pub cancellation_token: Option<CancellationToken>,
}
```

**Benefits**:
- Single token passed through entire loader pipeline
- User controls cancellation from application level
- Opt-in: None = no cancellation overhead

### Phase 2: Update spawn_prefetch Helper

**File**: `src/data_loader/prefetch.rs`

```rust
pub fn spawn_prefetch<F, Fut, T>(
    cap: usize,
    mut producer: F,
    cancel_token: Option<CancellationToken>,  // NEW
) -> Receiver<Result<T, DatasetError>>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, DatasetError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = channel(cap.max(1));
    tokio::spawn(async move {
        loop {
            // Check cancellation before producing
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;  // Clean exit
                }
            }
            
            match producer().await {
                Ok(item) => {
                    if tx.send(Ok(item)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }
    });
    rx
}
```

### Phase 3: Update DataLoader

**File**: `src/data_loader/dataloader.rs`

**Changes**:
1. Extract `cancel_token` from `self.opts.cancellation_token`
2. Pass to spawned background tasks
3. Check token in prefetch loops

```rust
pub fn stream(self) -> ReceiverStream<Result<Vec<D::Item>, DatasetError>> {
    let cancel_token = self.opts.cancellation_token.clone();
    let batch = self.opts.batch_size.max(1);
    // ...
    
    // Map-style dataset
    tokio::spawn(async move {
        let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
        while let Some(idx) = sampler.next_index() {
            // Check cancellation
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;  // Clean exit
                }
            }
            
            match ds_idx.get(idx).await {
                // ... existing logic ...
            }
        }
        // ...
    });
    
    // Iterable-style dataset  
    tokio::spawn(async move {
        let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
        let mut i: usize = 0;
        while let Some(next) = stream.next().await {
            // Check cancellation
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;  // Clean exit
                }
            }
            
            match next {
                // ... existing logic ...
            }
        }
        // ...
    });
}
```

### Phase 4: Update AsyncPoolDataLoader

**File**: `src/data_loader/async_pool_dataloader.rs`

**Changes**:
1. Extract `cancel_token` from options
2. Check before submitting new requests
3. Check in worker loop

```rust
pub fn stream_with_pool(self, pool_config: PoolConfig) -> ReceiverStream<Result<Vec<Vec<u8>>, DatasetError>> {
    let cancel_token = self.options.cancellation_token.clone();
    // ...
    
    tokio::spawn(async move {
        // ... setup ...
        
        loop {
            // Check cancellation
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;  // Clean exit
                }
            }
            
            // Submit requests
            while in_flight.len() < pool_config.pool_size && next_index < dataset_len {
                // Check cancellation before each request
                if let Some(ref token) = cancel_token {
                    if token.is_cancelled() {
                        break;
                    }
                }
                
                // ... submit request ...
            }
            
            // ... process completions ...
        }
    });
}
```

### Phase 5: Integration with RangeEngine

**Files**: `src/file_store.rs`, `src/file_store_direct.rs`

Pass cancellation token from DataLoader through to RangeEngine:

```rust
async fn get_with_range_engine(&self, uri: &str, object_size: u64) -> Result<Bytes> {
    let engine = RangeEngine::new(self.config.range_engine.clone());
    let get_range_fn = move |offset, length| { /* ... */ };
    
    // TODO: Need way to pass cancel_token from DataLoader context
    // For now, RangeEngine uses None (no cancellation)
    let (bytes, stats) = engine.download(object_size, get_range_fn, None).await?;
    
    Ok(bytes)
}
```

**Challenge**: How to pass token from DataLoader -> ObjectStore -> RangeEngine?

**Options**:
1. **Add to ObjectStore trait** (breaking change, affects all implementations)
2. **Thread-local storage** (tokio LocalSet - fragile)
3. **Context parameter** (requires get() signature change)
4. **Config-based** (store token in FileSystemConfig - works!)

**Recommended**: Option 4 - Add optional token to backend configs:

```rust
pub struct FileSystemConfig {
    // ... existing fields ...
    
    /// Optional cancellation token for async operations
    pub cancellation_token: Option<CancellationToken>,
}
```

Then ObjectStore can pass it to RangeEngine when calling `download()`.

## Testing Strategy

### Unit Tests

**Test 1: Prefetch Cancellation**
```rust
#[tokio::test]
async fn test_prefetch_respects_cancellation() {
    let cancel_token = CancellationToken::new();
    let counter = Arc::new(AtomicUsize::new(0));
    
    let counter_clone = counter.clone();
    let producer = move || {
        let counter = counter_clone.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(42)
        }
    };
    
    let mut rx = spawn_prefetch(10, producer, Some(cancel_token.clone()));
    
    // Consume one item
    assert_eq!(rx.recv().await, Some(Ok(42)));
    
    // Cancel
    cancel_token.cancel();
    
    // Give time for producer to notice cancellation
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Should stop producing new items
    let produced = counter.load(Ordering::SeqCst);
    assert!(produced < 5, "Should stop after cancellation, produced {}", produced);
}
```

**Test 2: DataLoader Cancellation**
```rust
#[tokio::test]
async fn test_dataloader_cancellation() {
    use crate::data_loader::*;
    
    let dataset = /* ... */;
    let cancel_token = CancellationToken::new();
    
    let opts = LoaderOptions {
        batch_size: 10,
        cancellation_token: Some(cancel_token.clone()),
        ..Default::default()
    };
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Consume one batch
    assert!(stream.next().await.is_some());
    
    // Cancel
    cancel_token.cancel();
    
    // Stream should end gracefully
    while let Some(batch) = stream.next().await {
        // May get a few more due to buffering
    }
    
    // Should not hang
}
```

**Test 3: AsyncPoolDataLoader Cancellation**
```rust
#[tokio::test]
async fn test_async_pool_dataloader_cancellation() {
    let dataset = /* ... */;
    let cancel_token = CancellationToken::new();
    
    let opts = LoaderOptions {
        batch_size: 10,
        cancellation_token: Some(cancel_token.clone()),
        ..Default::default()
    };
    
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let mut stream = loader.stream_with_pool(PoolConfig::default());
    
    // Consume one batch
    assert!(stream.next().await.is_some());
    
    // Cancel
    cancel_token.cancel();
    
    // Should stop submitting new requests
    let remaining: Vec<_> = stream.collect().await;
    assert!(remaining.len() < 100, "Should cancel quickly");
}
```

### Integration Tests

**Test 4: Ctrl-C Simulation**
```rust
#[tokio::test]
async fn test_ctrl_c_simulation() {
    use tokio::signal;
    
    let cancel_token = CancellationToken::new();
    
    // Simulate signal handler
    let token_clone = cancel_token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        token_clone.cancel();  // Simulate Ctrl-C
    });
    
    // Run DataLoader
    let loader = /* ... with cancel_token ... */;
    let mut stream = loader.stream();
    
    let mut count = 0;
    while let Some(_batch) = stream.next().await {
        count += 1;
        if count > 1000 {
            panic!("Should have cancelled by now!");
        }
    }
    
    // Should exit cleanly
}
```

## Implementation Checklist

### Phase 1: LoaderOptions ✅
- [ ] Add `cancellation_token: Option<CancellationToken>` field
- [ ] Update Default impl
- [ ] Update all constructors
- [ ] Add documentation

### Phase 2: spawn_prefetch ✅
- [ ] Add `cancel_token` parameter
- [ ] Check cancellation in loop
- [ ] Update all call sites
- [ ] Add unit test

### Phase 3: DataLoader ✅
- [ ] Extract token from options
- [ ] Check in map-style loop
- [ ] Check in iterable-style loop
- [ ] Pass to nested spawns
- [ ] Add integration test

### Phase 4: AsyncPoolDataLoader ✅
- [ ] Extract token from options
- [ ] Check before request submission
- [ ] Check in worker loop
- [ ] Clean shutdown of pool
- [ ] Add integration test

### Phase 5: ObjectStore Integration ⚠️
- [ ] Design: Add token to backend configs vs trait change
- [ ] Implement chosen approach
- [ ] Update File backend
- [ ] Update DirectIO backend
- [ ] Wire through to RangeEngine

### Phase 6: Documentation ✅
- [ ] Document cancellation semantics
- [ ] Add examples to README
- [ ] Update API docs
- [ ] Create usage guide

### Phase 7: Testing ✅
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Manual Ctrl-C testing
- [ ] No orphaned tasks (tokio-console)
- [ ] Zero warnings

## Expected Outcome

After implementation:

```rust
use s3dlio::data_loader::*;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    let cancel_token = CancellationToken::new();
    
    // Setup Ctrl-C handler
    let token_clone = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for Ctrl-C");
        println!("Received Ctrl-C, shutting down gracefully...");
        token_clone.cancel();
    });
    
    // Create DataLoader with cancellation support
    let dataset = MultiBackendDataset::from_prefix("s3://mybucket/").await?;
    let opts = LoaderOptions {
        batch_size: 32,
        prefetch: 4,
        cancellation_token: Some(cancel_token),
        ..Default::default()
    };
    
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let mut stream = loader.stream_with_pool(PoolConfig::default());
    
    // Process batches - will stop cleanly on Ctrl-C
    while let Some(batch_result) = stream.next().await {
        match batch_result {
            Ok(batch) => {
                // Process batch
                process_batch(batch).await?;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
    
    println!("Shutdown complete!");
}
```

**Benefits**:
- ✅ Clean Ctrl-C handling
- ✅ No orphaned tasks
- ✅ Graceful shutdown
- ✅ Resource cleanup
- ✅ User-friendly experience

## Timeline

- **Phase 1-2**: 1-2 hours (LoaderOptions + prefetch)
- **Phase 3-4**: 2-3 hours (DataLoader updates)
- **Phase 5**: 1-2 hours (ObjectStore integration design)
- **Phase 6-7**: 1-2 hours (Testing + docs)

**Total**: 5-9 hours

## Next Steps

1. Start with Phase 1: Add field to LoaderOptions
2. Phase 2: Update spawn_prefetch with test
3. Phase 3: Wire through DataLoader
4. Phase 4: Wire through AsyncPoolDataLoader
5. Phase 5: Design ObjectStore integration
6. Testing and validation

Ready to begin implementation!
