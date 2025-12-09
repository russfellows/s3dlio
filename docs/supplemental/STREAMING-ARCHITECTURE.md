# Stream-Based Architecture Guide

## Overview

s3dlio uses **streams as the universal substrate** for all async I/O operations, providing:
- Unified backpressure control
- Cancellation token support
- Configurable concurrency and ordering
- Consistent patterns across all backends

## Core Streaming Patterns

### 1. Range-Based Downloads with Streams

The RangeEngine uses Rust async streams for concurrent range requests:

```rust
use futures::stream::{self, StreamExt};

// Create stream of range requests
let chunks = stream::iter(ranges)
    .map(|(offset, length)| async move {
        // Fetch range asynchronously
        get_range(offset, length).await
    })
    .buffered(max_concurrent);  // Control concurrency
```

**Key Features:**
- `stream::iter()` - Convert collection to stream
- `.buffered(n)` - Execute up to N futures concurrently
- `.buffer_unordered(n)` - Out-of-order completion for max throughput
- Semaphore-based backpressure control

### 2. DataLoader Streaming

DataLoaders use bounded MPSC channels with `ReceiverStream`:

```rust
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

// Bounded channel provides natural backpressure
let (tx, rx) = mpsc::channel(prefetch_capacity);

// Spawn worker that produces data
tokio::spawn(async move {
    for item in dataset {
        tx.send(item).await.ok();
    }
});

// Return as stream
ReceiverStream::new(rx)
```

**Benefits:**
- Prefetch control via channel capacity
- Natural backpressure (producer blocks when full)
- Clean `Stream<Item = Result<T>>` interface

### 3. Concurrent Operations with FuturesUnordered

For pool-based concurrency (out-of-order completion):

```rust
use futures::stream::FuturesUnordered;

let mut pending: FuturesUnordered<_> = FuturesUnordered::new();

// Add requests to pool
for request in requests {
    pending.push(async move { fetch(request).await });
}

// Process completions as they arrive
while let Some(result) = pending.next().await {
    process(result)?;
}
```

**Use Cases:**
- AsyncPoolDataLoader (out-of-order batch formation)
- Batch delete/copy operations
- Maximum throughput scenarios

### 4. Generic RangeEngine Pattern

The RangeEngine is generic over any backend:

```rust
pub async fn download<F, Fut>(
    &self,
    object_size: u64,
    get_range: F,  // Closure: (offset, length) -> Future<Bytes>
    cancel: Option<CancellationToken>,
) -> Result<(Bytes, Stats)>
where
    F: Fn(u64, u64) -> Fut + Clone,
    Fut: Future<Output = Result<Bytes>>,
{
    // Small files: single request
    if object_size < min_split_size {
        return get_range(0, object_size).await;
    }
    
    // Large files: concurrent ranges
    let ranges = calculate_ranges(object_size, chunk_size);
    
    stream::iter(ranges)
        .map(|(offset, len)| {
            let get_range = get_range.clone();
            async move {
                get_range(offset, len).await
            }
        })
        .buffered(max_concurrent)
        .try_collect()  // Ordered assembly
        .await
}
```

**Key Design:**
- Generic over backend (closure-based)
- Works with S3, Azure, GCS, file://
- Automatic small/large file optimization
- Ordered or unordered completion

## Backpressure Control

Multiple mechanisms for flow control:

### 1. Channel Capacity
```rust
let (tx, rx) = mpsc::channel(PREFETCH);  // Bounded
// Producer blocks when channel full
```

### 2. Semaphore Limits
```rust
let sem = Arc::new(Semaphore::new(MAX_CONCURRENT));
let _permit = sem.acquire().await?;  // Blocks if at limit
```

### 3. Stream Buffering
```rust
stream.buffered(n)           // At most N concurrent
stream.buffer_unordered(n)   // Out-of-order, max N
```

## Cancellation Support

### Pattern 1: CancellationToken
```rust
use tokio_util::sync::CancellationToken;

let cancel = CancellationToken::new();

// Pass to workers
tokio::spawn({
    let cancel = cancel.clone();
    async move {
        loop {
            if cancel.is_cancelled() {
                break;
            }
            // Work...
        }
    }
});

// Cancel from anywhere
cancel.cancel();
```

### Pattern 2: Drop-Based Cancellation
```rust
struct CancellableStream<S> {
    inner: S,
    cancel_token: CancellationToken,
}

impl<S> Drop for CancellableStream<S> {
    fn drop(&mut self) {
        // Cancel background tasks when stream dropped
        self.cancel_token.cancel();
    }
}
```

## Configuration Hierarchy

Unified configuration across streaming components:

### RangeEngine Configuration
```rust
pub struct RangeEngineConfig {
    pub chunk_size: usize,              // Size of each range
    pub max_concurrent_ranges: usize,   // Concurrency limit
    pub min_split_size: u64,            // Threshold for range splits
    pub range_timeout: Duration,        // Per-range timeout
}
```

### DataLoader Configuration
```rust
pub struct LoaderOptions {
    pub prefetch: usize,        // Channel capacity (backpressure)
    pub batch_size: usize,      // Items per batch
    // ...
}

pub struct PoolConfig {
    pub pool_size: usize,       // Concurrent requests
    pub readahead_batches: usize,  // Output buffer
    pub batch_timeout: Duration,
}
```

## Ordering Control

### Ordered Completion (FuturesOrdered)
```rust
use futures::stream::FuturesOrdered;

let mut ordered = FuturesOrdered::new();
for req in requests {
    ordered.push_back(async move { fetch(req).await });
}

// Results arrive in insertion order
while let Some(result) = ordered.next().await {
    // Process in order
}
```

**Use Case**: Maintaining data order (sequential batch processing)

### Unordered Completion (FuturesUnordered)
```rust
use futures::stream::FuturesUnordered;

let mut unordered = FuturesUnordered::new();
// Process as completions arrive (max throughput)
```

**Use Case**: Maximum throughput (don't wait for slow requests)

## Backend Integration Pattern

Each backend can opt into RangeEngine:

```rust
impl ObjectStore for MyBackend {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let size = self.stat(uri).await?.size;
        
        if size >= self.config.range_engine_threshold {
            // Use concurrent ranges for large objects
            self.get_with_range_engine(uri, size).await
        } else {
            // Simple single-request for small objects
            self.get_simple(uri).await
        }
    }
    
    async fn get_with_range_engine(&self, uri: &str, size: u64) -> Result<Bytes> {
        let engine = RangeEngine::new(self.config.range_config);
        
        // Create closure capturing self and uri
        let uri = uri.to_string();
        let backend = self.clone();
        let get_range_fn = move |offset, length| {
            let uri = uri.clone();
            let backend = backend.clone();
            async move {
                backend.get_range(&uri, offset, Some(length)).await
            }
        };
        
        let (bytes, stats) = engine.download(size, get_range_fn, None).await?;
        Ok(bytes)
    }
}
```

## Performance Benefits

Stream-based architecture provides:

1. **Controlled Concurrency**: Semaphores + buffered streams prevent overwhelming backends
2. **Natural Backpressure**: Bounded channels prevent unbounded memory growth
3. **Efficient Cancellation**: Drop-based cleanup, no resource leaks
4. **Flexible Ordering**: Choose ordered (correctness) vs unordered (speed)
5. **Unified Patterns**: Same idioms across all backends and components

## Best Practices

1. **Always use bounded channels** for streams (backpressure)
2. **Add cancellation tokens** to long-running operations
3. **Use semaphores** to limit concurrent I/O operations
4. **Choose ordering** based on requirements:
   - Ordered: Data must be processed sequentially
   - Unordered: Maximum throughput, order doesn't matter
5. **Profile concurrency limits**: Too low = underutilized, too high = connection exhaustion
6. **Timeout individual operations**: Don't let one slow request block everything

## References

- Current RangeEngine: `src/range_engine.rs`
- DataLoader streaming: `src/data_loader/dataloader.rs`
- AsyncPoolDataLoader: `src/data_loader/async_pool_dataloader.rs`
- S3 range reader: `src/data_loader/s3_bytes.rs`
- Tokio streams: https://docs.rs/tokio-stream/
- Futures streams: https://docs.rs/futures/
