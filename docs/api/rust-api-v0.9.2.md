# s3dlio Rust Library API Guide - v0.9.2

**Version**: 0.9.2 (API-stable beta)  
**Date**: October 2025  
**Status**: Breaking changes from v0.8.x - see migration section

---

## Table of Contents

- [Overview](#overview)
- [What's New in v0.9.2](#whats-new-in-v092)
- [Configuration Hierarchy](#configuration-hierarchy-understanding-the-three-levels)
- [Breaking Changes from v0.8.22](#breaking-changes-from-v0822)
- [Core API](#core-api)
- [Adaptive Configuration (NEW)](#adaptive-configuration-new)
- [Storage Backends](#storage-backends)
- [Data Loaders](#data-loaders)
- [Advanced Features](#advanced-features)
- [Migration Guide](#migration-guide)

---

## Overview

s3dlio is a high-performance, multi-protocol storage library for AI/ML workloads, designed for:
- **Performance**: 5+ GB/s reads, 2.5+ GB/s writes on high-speed infrastructure
- **Universal access**: S3, Azure, local file systems, DirectIO
- **Zero-copy operations**: Bytes-based API for minimal allocations
- **AI/ML integration**: Built-in data loaders, checkpoint handling, batching

**Key principle**: Designed for efficient data GENERATION (put) and reading (get), not arbitrary uploads.

---

## What's New in v0.9.2

### New Features

1. **CancellationToken Support** 🛑 (v0.9.2)
   - Graceful shutdown for all DataLoader components
   - Clean Ctrl-C handling in training loops
   - Cooperative cancellation across async workers
   - See [Graceful Shutdown](#graceful-shutdown-v092) section

2. **Configuration Hierarchy Documentation** 📚 (v0.9.2)
   - Clear three-level design (LoaderOptions → PoolConfig → RangeEngineConfig)
   - PyTorch-aligned concepts for ML practitioners
   - See [Configuration Hierarchy](#configuration-hierarchy-understanding-the-three-levels) section

3. **PoolConfig Convenience Constructor** 🛠️ (v0.9.2)
   - `PoolConfig::from_loader_options()` to derive from training parameters
   - Simplifies advanced tuning workflows

### From v0.9.0

1. **Zero-Copy Performance** ⚡
   - `ObjectStore::get()` now returns `bytes::Bytes` instead of `Vec<u8>`
   - 10-15% memory reduction on large transfers
   - Enables true zero-copy from S3 → application

2. **Adaptive Tuning** 🎯 (Optional)
   - Automatic optimization based on workload type
   - Smart defaults for small/medium/large files and batch operations
   - Explicit settings always override adaptive values
   - Disabled by default - opt-in only

3. **Improved Data Loaders** 🚀
   - Concurrent batch fetching (3-8x speedup)
   - Better async iteration patterns
   - Enhanced error handling

### Performance Improvements

- **Memory**: 10-15% reduction via Bytes migration
- **Throughput**: Maintained 5+ GB/s reads, 2.5+ GB/s writes
- **Batch loading**: 3-8x faster with concurrent fetching
- **Cancellation**: Zero overhead when not used, clean shutdown when needed

---

## Configuration Hierarchy: Understanding the Three Levels

s3dlio provides three conceptual levels of configuration, aligned with standard AI/ML training frameworks like PyTorch. Understanding this hierarchy helps you configure data loading efficiently without confusion.

### Level 1: LoaderOptions (User-Facing, Training-Centric)

**Analogy**: PyTorch's `DataLoader(batch_size, num_workers, ...)`

**Purpose**: Controls **WHAT** batches to create and **HOW** to iterate the dataset

**Who uses it**: ML practitioners, data scientists, model trainers

**Mental model**: "I'm training a model, I need batches of data"

```rust
use s3dlio::data_loader::LoaderOptions;

let options = LoaderOptions {
    batch_size: 32,              // Items per batch
    drop_last: false,            // Keep incomplete final batch
    shuffle: true,               // Randomize iteration order
    num_workers: 4,              // Parallelism (like PyTorch)
    prefetch: 2,                 // Prefetch depth (like PyTorch's prefetch_factor)
    pin_memory: true,            // GPU optimization
    persistent_workers: true,    // Keep workers alive between epochs
    ..Default::default()
};

let loader = DataLoader::new(dataset, options);
```

**Key principle**: These options control training behavior, not storage implementation details.

---

### Level 2: PoolConfig (Performance Tuning, Optional)

**Analogy**: Internal worker pool management in PyTorch (hidden behind `num_workers`)

**Purpose**: Controls **HOW** data is fetched efficiently to fill batches

**Who uses it**: Performance engineers, infrastructure teams tuning download performance

**Mental model**: "I want to optimize download speed without changing training behavior"

```rust
use s3dlio::data_loader::{AsyncPoolDataLoader, PoolConfig};

// Simple case: Use defaults (recommended for most users)
let loader = AsyncPoolDataLoader::new(dataset, options);
let stream = loader.stream();  // Uses PoolConfig::default()

// Advanced case: Tune download performance
let pool_config = PoolConfig {
    pool_size: 128,              // Concurrent download workers
    readahead_batches: 8,        // Batch prefetch depth
    batch_timeout: Duration::from_secs(60),
    ..Default::default()
};

let stream = loader.stream_with_pool(pool_config);
```

**Convenience constructor**: Derive from LoaderOptions:
```rust
// Scale pool configuration from training parameters
let pool_config = PoolConfig::from_loader_options(&options);
// pool_size = num_workers * 16, readahead_batches = prefetch.max(2)

let stream = loader.stream_with_pool(pool_config);
```

**When to tune**:
- ✅ You have high-bandwidth infrastructure (>10 Gb/s)
- ✅ You're optimizing for specific workload patterns
- ✅ Default performance doesn't meet requirements
- ❌ You're prototyping or just getting started (use defaults)

---

### Level 3: RangeEngineConfig (Internal Optimization, Hidden)

**Analogy**: File I/O internals in PyTorch Dataset implementations (buffering, caching)

**Purpose**: Controls storage-layer optimizations for large object downloads

**Who uses it**: Storage engineers, infrastructure experts debugging edge cases

**Mental model**: "How does s3dlio split large files into parallel range requests?"

```rust
use s3dlio::range_engine_generic::RangeEngineConfig;

// Configuration (typically not exposed in high-level APIs)
let config = RangeEngineConfig {
    chunk_size: 64 * 1024 * 1024,        // 64MB chunks
    max_concurrent_ranges: 32,           // Parallel ranges per object
    min_split_size: 4 * 1024 * 1024,    // 4MB threshold to trigger splitting
    ..Default::default()
};
```

**Backend-specific defaults**:
- **S3**: `min_split_size = 4MB` (efficient for network objects)
- **Local file**: `min_split_size = 4MB` (⚠️ may add overhead, use simple reads for small files)
- **DirectIO**: `min_split_size = 16MB` (alignment-aware, optimized for large files)

**Key principle**: This is **mostly hidden** from user-facing APIs. The RangeEngine is used internally by ObjectStore implementations and automatically configured based on backend type.

**When exposed**: Only in specialized debugging/profiling scenarios, not in DataLoader APIs.

---

### Comparison with PyTorch DataLoader

| s3dlio | PyTorch Equivalent | Visibility | Typical Users |
|--------|-------------------|------------|---------------|
| **LoaderOptions** | `DataLoader(...)` | ✅ Always visible | ML practitioners |
| **PoolConfig** | Worker pool internals | 🟡 Optional (good defaults) | Performance engineers |
| **RangeEngineConfig** | Dataset I/O internals | ❌ Hidden | Storage engineers |

### PyTorch vs s3dlio Side-by-Side

**PyTorch**:
```python
# User configures high-level training options only
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,        # Level 1: Batch construction
    num_workers=4,        # Level 1: Parallelism
    prefetch_factor=2,    # Level 1: Prefetching
    pin_memory=True,      # Level 1: GPU optimization
)
# Level 2 (worker pool) is automatically managed
# Level 3 (file I/O) is hidden in Dataset implementation
```

**s3dlio (Simple - Like PyTorch)**:
```rust
// Level 1 only: Training-focused configuration
let options = LoaderOptions {
    batch_size: 32,
    num_workers: 4,
    prefetch: 2,
    pin_memory: true,
    ..Default::default()
};

let loader = DataLoader::new(dataset, options);
let stream = loader.stream();  // Uses internal defaults for Levels 2 & 3
```

**s3dlio (Advanced - Tune Download Performance)**:
```rust
// Level 1: Training configuration
let options = LoaderOptions { /* ... */ };

// Level 2: Explicit download pool tuning (NOT available in PyTorch)
let pool_config = PoolConfig {
    pool_size: 128,           // More aggressive than default
    readahead_batches: 8,
    ..Default::default()
};

let loader = AsyncPoolDataLoader::new(dataset, options);
let stream = loader.stream_with_pool(pool_config);

// Level 3 (RangeEngine) still hidden, auto-configured by backend
```

---

### Design Philosophy

s3dlio's configuration hierarchy follows these principles:

1. **Progressive complexity**: Simple by default, powerful when needed
2. **PyTorch alignment**: Familiar concepts for ML practitioners
3. **Separation of concerns**: Training logic vs storage optimization
4. **Good defaults**: Most users never touch Level 2 or 3
5. **Expert escape hatches**: Advanced users can tune performance

**Golden rule**: Start with LoaderOptions only. Add PoolConfig tuning only when profiling shows it's beneficial.

---

## Breaking Changes from v0.8.22

### 1. ObjectStore::get() Return Type

**v0.8.22 (Old)**:
```rust
fn get(&self, uri: &str) -> Result<Vec<u8>> {
    // Returns owned vector
}
```

**v0.9.0 (New)**:
```rust
fn get(&self, uri: &str) -> Result<Bytes> {
    // Returns reference-counted bytes
}
```

**Migration**:
```rust
// Before (v0.8.22)
let data: Vec<u8> = store.get("s3://bucket/key")?;
process_data(&data);

// After (v0.9.0) - Option 1: Use Bytes directly
let data: Bytes = store.get("s3://bucket/key")?;
process_data(&data);  // &[u8] deref still works

// After (v0.9.0) - Option 2: Convert to Vec if needed
let data: Bytes = store.get("s3://bucket/key")?;
let vec: Vec<u8> = data.to_vec();  // Allocates copy
```

**Why?**: Zero-copy performance, reduced memory allocations, better for streaming.

### 2. ObjectStore::get_range() Return Type

Same migration as `get()` - now returns `Bytes` instead of `Vec<u8>`.

### 3. Dependencies

**New required dependencies**:
```toml
[dependencies]
bytes = "1.9"  # NEW - required for Bytes type
```

**No changes needed** for existing dependencies (aws-sdk-s3, azure, etc.).

---

## Core API

### Creating a Storage Backend

**Universal factory (recommended)**:
```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

fn main() -> Result<()> {
    // Auto-detects backend from URI scheme
    let store = store_for_uri("s3://my-bucket/prefix/")?;
    
    // Works with any supported scheme
    let local = store_for_uri("file:///tmp/data/")?;
    let azure = store_for_uri("az://container/prefix/")?;
    let direct = store_for_uri("direct:///nvme/data/")?;
    
    Ok(())
}
```

**Backend-specific creation**:
```rust
use s3dlio::api::{S3Config, create_s3_store};

// S3 with custom configuration
let config = S3Config {
    endpoint: Some("https://s3.us-west-2.amazonaws.com".into()),
    region: Some("us-west-2".into()),
    ..Default::default()
};
let store = create_s3_store(config)?;
```

### Basic Operations

#### Get (Read)

```rust
use bytes::Bytes;

// Simple get - returns zero-copy Bytes
let data: Bytes = store.get("s3://bucket/data.bin")?;

// Get with range
let chunk: Bytes = store.get_range(
    "s3://bucket/large.bin",
    0,      // start offset
    1024    // length
)?;

// Use data as &[u8]
println!("Read {} bytes", data.len());
process_bytes(&data);  // Bytes derefs to &[u8]
```

#### Put (Write)

```rust
use s3dlio::api::WriterOptions;

// Simple put - for small data
let data = vec![1, 2, 3, 4];
store.put("s3://bucket/small.bin", &data)?;

// Streaming put - for large data
let mut writer = store.create_writer(
    "s3://bucket/large.bin",
    WriterOptions::default()
)?;

for chunk in data_chunks {
    writer.write_chunk(&chunk)?;
}
writer.finalize()?;
```

#### List

```rust
// List all objects under prefix
let objects = store.list("s3://bucket/prefix/")?;

for obj in objects {
    println!("{}: {} bytes", obj.key, obj.size);
}
```

#### Delete

```rust
// Delete single object
store.delete("s3://bucket/old.bin")?;

// Delete multiple objects
let keys = vec![
    "s3://bucket/file1.bin",
    "s3://bucket/file2.bin",
];
store.delete_batch(&keys)?;
```

#### Stat (Metadata)

```rust
// Get object metadata without downloading
let info = store.stat("s3://bucket/file.bin")?;

println!("Size: {} bytes", info.size);
println!("ETag: {}", info.etag.unwrap_or_default());
println!("Modified: {:?}", info.last_modified);
```

---

## Adaptive Configuration (NEW)

Optional auto-tuning for optimal performance based on workload characteristics.

### Basic Usage

```rust
use s3dlio::api::{AdaptiveConfig, AdaptiveMode, WorkloadType};

// Enable adaptive tuning for data loading
let config = AdaptiveConfig {
    mode: AdaptiveMode::Enabled,
    workload_type: WorkloadType::SmallFile,
    ..Default::default()
};

let opts = LoaderOptions::default()
    .with_adaptive_config(config);

let loader = store.create_loader("s3://bucket/data/", opts)?;
```

### Workload Types

```rust
pub enum WorkloadType {
    SmallFile,   // < 1MB: Higher concurrency, smaller parts
    MediumFile,  // 1MB-64MB: Balanced settings
    LargeFile,   // > 64MB: Larger parts, optimized buffers
    Batch,       // Multiple files: Maximum concurrency
    Unknown,     // Auto-detect based on file sizes
}
```

### Key Principles

1. **Disabled by default**: Opt-in via `AdaptiveMode::Enabled`
2. **Explicit settings override**: Your values always take precedence
3. **Automatic detection**: `WorkloadType::Unknown` analyzes actual files
4. **Bounded optimization**: Never exceeds safe limits

### Examples

**Adaptive with overrides**:
```rust
let config = AdaptiveConfig {
    mode: AdaptiveMode::Enabled,
    workload_type: WorkloadType::LargeFile,
    max_concurrency_override: Some(32),  // Override: Use 32, not adaptive
    ..Default::default()
};

let opts = LoaderOptions::default()
    .with_adaptive_config(config)
    .with_batch_size(16);  // Explicit batch size overrides adaptive

// Result: Uses concurrency=32 (override), batch=16 (explicit),
// but adaptive values for part_size and buffer_size
```

**Disable adaptive**:
```rust
let opts = LoaderOptions::default();
// Adaptive is disabled by default - uses explicit or default values
```

### When to Use Adaptive

✅ **Good use cases**:
- Unknown or varying workloads
- Prototyping and experimentation
- General-purpose tools
- Quick start without manual tuning

❌ **Avoid when**:
- You've already tuned for your specific workload
- Performance is critical and predictable
- You want full control over all parameters

---

## Storage Backends

### S3 (AWS, MinIO, Vast, etc.)

```rust
use s3dlio::api::{S3Config, create_s3_store};

let config = S3Config {
    endpoint: Some("https://minio.local:9000".into()),
    region: Some("us-east-1".into()),
    bucket: "my-data".into(),
    ..Default::default()
};

let store = create_s3_store(config)?;
```

**Environment variables**:
- `AWS_ENDPOINT_URL`: S3 endpoint
- `AWS_REGION`: Region name
- `AWS_ACCESS_KEY_ID`: Credentials
- `AWS_SECRET_ACCESS_KEY`: Credentials

### Azure Blob Storage

```rust
let store = store_for_uri("az://container/prefix/")?;
```

**Environment variables**:
- `AZURE_STORAGE_ACCOUNT`: Account name
- `AZURE_STORAGE_KEY`: Access key

### Local Filesystem

```rust
// Standard file I/O
let store = store_for_uri("file:///path/to/data/")?;

// DirectIO (bypasses page cache)
let store = store_for_uri("direct:///nvme/fast/data/")?;
```

### Google Cloud Storage (Experimental)

```rust
// S3-compatible mode (recommended)
let store = store_for_uri("s3://bucket/")?;  // with GCS endpoint

// Native GCS (experimental, requires gcs-backend feature)
#[cfg(feature = "gcs-backend")]
let store = store_for_uri("gs://bucket/")?;
```

---

## Data Loaders

> **📚 See also**: [Configuration Hierarchy](#configuration-hierarchy-understanding-the-three-levels) for understanding LoaderOptions, PoolConfig, and RangeEngineConfig.

### Basic Usage (Level 1: LoaderOptions Only)

```rust
use s3dlio::data_loader::{DataLoader, LoaderOptions, MultiBackendDataset};
use futures::stream::StreamExt;

// Create dataset from URIs
let uris = vec![
    "s3://bucket/data/sample_000001.bin".to_string(),
    "s3://bucket/data/sample_000002.bin".to_string(),
    // ... more URIs
];
let dataset = MultiBackendDataset::from_uris(uris)?;

// Configure training options (Level 1)
let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token);  // v0.9.2: Graceful shutdown

// Create loader and stream batches
let loader = DataLoader::new(dataset, options);
let mut stream = loader.stream();

while let Some(batch_result) = stream.next().await {
    let batch = batch_result?;  // Vec<Vec<u8>>
    // Process batch for training
    train_step(batch).await?;
}
```

### Advanced Usage (Level 2: PoolConfig Tuning)

For high-performance scenarios, use `AsyncPoolDataLoader` with optional `PoolConfig`:

```rust
use s3dlio::data_loader::{AsyncPoolDataLoader, PoolConfig};

let options = LoaderOptions::default()
    .with_batch_size(32);

let loader = AsyncPoolDataLoader::new(dataset, options);

// Option A: Use defaults (recommended)
let stream = loader.stream();

// Option B: Derive from LoaderOptions
let pool_config = PoolConfig::from_loader_options(&options);
let stream = loader.stream_with_pool(pool_config);

// Option C: Custom tuning for high-bandwidth infrastructure
let pool_config = PoolConfig {
    pool_size: 128,          // More concurrent downloads
    readahead_batches: 8,    // Deeper prefetch queue
    ..Default::default()
};
let stream = loader.stream_with_pool(pool_config);

// Process batches
while let Some(batch_result) = stream.next().await {
    let batch = batch_result?;
    train_step(batch).await?;
}
```

### Dataset Creation

```rust
use s3dlio::api::{LoaderOptions, create_dataset};

let opts = LoaderOptions::default()
    .with_batch_size(32)
    .with_prefetch(3)
    .with_max_concurrency(16);

let dataset = create_dataset("s3://bucket/training/", opts)?;
```

### Async Iteration

```rust
use futures::stream::StreamExt;

let mut loader = store.create_async_loader("s3://bucket/data/", opts)?;

while let Some(batch) = loader.next().await {
    let batch = batch?;  // Vec<Bytes>
    
    for item in batch {
        process_item(item);  // item: Bytes
    }
}
```

### Loader Options

```rust
pub struct LoaderOptions {
    pub batch_size: usize,           // Items per batch (default: 32)
    pub prefetch: usize,             // Batches to prefetch (default: 2)
    pub max_concurrency: usize,      // Parallel downloads (default: 16)
    pub shuffle: bool,               // Shuffle items (default: false)
    pub seed: Option<u64>,           // Shuffle seed (default: None)
    pub part_size: Option<usize>,    // Multipart size (default: 8MB)
    pub adaptive: Option<AdaptiveConfig>,  // Adaptive tuning (v0.9.0)
}
```

**Builder pattern**:
```rust
let opts = LoaderOptions::default()
    .with_batch_size(64)
    .with_prefetch(4)
    .with_shuffle(true)
    .with_seed(42)
    .with_adaptive();  // Enable adaptive tuning
```

### Graceful Shutdown (v0.9.2)

**CancellationToken support** enables clean shutdown of data loading operations:

```rust
use tokio_util::sync::CancellationToken;
use futures::stream::StreamExt;

// Create cancellation token
let cancel_token = CancellationToken::new();

// Configure loader with cancellation support
let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token.clone());

let loader = DataLoader::new(dataset, options);
let mut stream = loader.stream();

// Spawn Ctrl-C handler
let cancel_token_handler = cancel_token.clone();
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    println!("Received Ctrl-C, shutting down gracefully...");
    cancel_token_handler.cancel();
});

// Training loop with graceful shutdown
while let Some(batch_result) = stream.next().await {
    let batch = batch_result?;
    train_step(batch).await?;
}

println!("Training stopped cleanly");
```

**AsyncPoolDataLoader also supports cancellation**:

```rust
let cancel_token = CancellationToken::new();

let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token.clone());

let loader = AsyncPoolDataLoader::new(dataset, options);
let mut stream = loader.stream();  // Cancellation handled automatically

// Set up Ctrl-C handler as above
// ...

while let Some(batch_result) = stream.next().await {
    // Processes batches, stops cleanly on cancellation
}
```

**Behavior on cancellation**:
- ✅ Workers exit cleanly without submitting new requests
- ✅ In-flight requests are allowed to complete (drain pattern)
- ✅ MPSC channels properly closed
- ✅ No orphaned background tasks
- ✅ Zero overhead when token not cancelled

**Multiple loaders, shared token**:
```rust
let cancel_token = CancellationToken::new();

// Both loaders share the same token
let options1 = LoaderOptions::default()
    .with_cancellation_token(cancel_token.clone());
let options2 = LoaderOptions::default()
    .with_cancellation_token(cancel_token.clone());

// Single Ctrl-C cancels both loaders
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    cancel_token.cancel();  // Stops all loaders
});
```

---

## Advanced Features

### Writer Options

```rust
use s3dlio::api::WriterOptions;

let opts = WriterOptions::default()
    .with_part_size(16 * 1024 * 1024)  // 16 MB parts
    .with_max_concurrency(8)            // 8 parallel uploads
    .with_buffer_size(64 * 1024)        // 64 KB buffer
    .with_adaptive();                   // Enable adaptive tuning

let mut writer = store.create_writer("s3://bucket/output.bin", opts)?;
```

### Checkpoint Operations

```rust
use s3dlio::api::{CheckpointWriter, CheckpointReader};

// Save checkpoint
let mut writer = CheckpointWriter::new(&store, "s3://bucket/ckpt/")?;
writer.save_state("model", &model_state)?;
writer.save_state("optimizer", &opt_state)?;
writer.finalize()?;

// Load checkpoint
let reader = CheckpointReader::new(&store, "s3://bucket/ckpt/")?;
let model_state = reader.load_state("model")?;
let opt_state = reader.load_state("optimizer")?;
```

### Data Generation (put)

s3dlio is designed for **efficient data generation**, not arbitrary uploads.

```rust
use s3dlio::api::{DataGenerator, ObjectType};

// Generate synthetic training data
let gen = DataGenerator::new(&store);

gen.put_many(
    "s3://bucket/data/",
    "sample_{:06d}.npz",
    10000,                          // Generate 10,000 files
    ObjectType::Npz,                // NPZ format
    1024 * 1024,                    // 1 MB each
)?;
```

**Supported formats**:
- `ObjectType::Zeros`: All-zero data (fast)
- `ObjectType::Random`: Random bytes
- `ObjectType::Npz`: NumPy compressed arrays
- `ObjectType::Hdf5`: HDF5 files
- `ObjectType::TfRecord`: TensorFlow records

### Page Cache Optimization (v0.8.8+)

```rust
use s3dlio::api::PageCacheMode;

// Configure page cache behavior (Linux/Unix only)
let opts = WriterOptions::default()
    .with_page_cache_mode(PageCacheMode::Sequential);

// Modes:
// - Auto: Sequential for ≥64MB files, Random for smaller
// - Sequential: Read-ahead optimization
// - Random: No read-ahead
// - DontNeed: Immediate cache eviction
// - Normal: System default
```

---

## Migration Guide

### From v0.8.22 to v0.9.0

#### Step 1: Update Dependencies

```toml
[dependencies]
s3dlio = "0.9.0"  # Update version
bytes = "1.9"     # NEW - add this dependency
```

#### Step 2: Update ObjectStore Calls

**Find and replace** all `get()` and `get_range()` calls:

```rust
// Old (v0.8.22)
let data: Vec<u8> = store.get(uri)?;

// New (v0.9.0) - Minimal change
let data: Bytes = store.get(uri)?;
// Note: Bytes derefs to &[u8], so most code works unchanged

// If you need Vec<u8>
let data: Bytes = store.get(uri)?;
let vec = data.to_vec();  // Explicit conversion
```

#### Step 3: Update Type Signatures

```rust
// Old function signatures
fn process_data(data: &Vec<u8>) -> Result<()> { ... }
fn store_result(data: Vec<u8>) -> Result<()> { ... }

// New signatures (two options)

// Option A: Accept Bytes directly
fn process_data(data: &Bytes) -> Result<()> { ... }
fn store_result(data: Bytes) -> Result<()> { ... }

// Option B: Generic over AsRef<[u8]> (works with both)
fn process_data<T: AsRef<[u8]>>(data: T) -> Result<()> {
    let bytes = data.as_ref();  // &[u8]
    ...
}
```

#### Step 4: Test Data Loaders

Data loaders now return `Vec<Bytes>` instead of `Vec<Vec<u8>>`:

```rust
// Old iteration
for batch in loader {  // batch: Vec<Vec<u8>>
    for item in batch {  // item: Vec<u8>
        process(&item);
    }
}

// New iteration (minimal change)
for batch in loader {  // batch: Vec<Bytes>
    for item in batch {  // item: Bytes
        process(&item);  // Bytes derefs to &[u8]
    }
}
```

#### Step 5: Optional - Enable Adaptive Tuning

```rust
// Add adaptive tuning for automatic optimization
let opts = LoaderOptions::default()
    .with_adaptive();  // NEW in v0.9.0

let loader = store.create_loader(uri, opts)?;
```

### Compatibility Checklist

- [ ] Updated `Cargo.toml` dependencies
- [ ] Added `bytes = "1.9"`
- [ ] Changed `Vec<u8>` → `Bytes` in function signatures
- [ ] Tested data loaders with new return types
- [ ] Verified no performance regressions
- [ ] (Optional) Evaluated adaptive tuning for your workload

---

## Performance Tips

### 1. Use Bytes for Zero-Copy

```rust
// ❌ Bad: Unnecessary copy
let data: Bytes = store.get(uri)?;
let vec = data.to_vec();  // Allocates new Vec
process(&vec);

// ✅ Good: Zero-copy
let data: Bytes = store.get(uri)?;
process(&data);  // No allocation
```

### 2. Tune for Your Workload

```rust
// Small files (< 1 MB)
let opts = LoaderOptions::default()
    .with_batch_size(128)        // Larger batches
    .with_max_concurrency(32);   // Higher concurrency

// Large files (> 64 MB)
let opts = LoaderOptions::default()
    .with_part_size(16 * 1024 * 1024)  // 16 MB parts
    .with_max_concurrency(8);           // Moderate concurrency
```

### 3. Use Adaptive for Unknown Workloads

```rust
let opts = LoaderOptions::default()
    .with_adaptive();  // Auto-tunes based on actual data

// Or specify workload type
let opts = LoaderOptions::default()
    .with_adaptive_config(AdaptiveConfig {
        mode: AdaptiveMode::Enabled,
        workload_type: WorkloadType::Batch,
        ..Default::default()
    });
```

### 4. DirectIO for High-Speed Storage

```rust
// Bypasses page cache for maximum throughput
let store = store_for_uri("direct:///nvme/data/")?;

// Best for:
// - NVMe storage
// - Large sequential reads/writes
// - Avoiding page cache pollution
```

---

## Examples

### Complete Data Loading Pipeline

```rust
use s3dlio::api::*;
use bytes::Bytes;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create storage backend
    let store = store_for_uri("s3://my-bucket/training/")?;
    
    // 2. Configure loader with adaptive tuning
    let opts = LoaderOptions::default()
        .with_batch_size(64)
        .with_shuffle(true)
        .with_seed(42)
        .with_adaptive();  // Auto-tune for workload
    
    // 3. Create async loader
    let mut loader = store.create_async_loader(
        "s3://my-bucket/training/",
        opts
    )?;
    
    // 4. Process batches
    use futures::stream::StreamExt;
    
    while let Some(batch) = loader.next().await {
        let batch: Vec<Bytes> = batch?;
        
        // Process items in parallel
        for item in batch {
            tokio::spawn(async move {
                process_training_sample(item).await
            });
        }
    }
    
    Ok(())
}

async fn process_training_sample(data: Bytes) -> Result<()> {
    // Data is zero-copy reference-counted bytes
    // Can be cloned cheaply and sent to other tasks
    let data_clone = data.clone();  // Cheap: just increments refcount
    
    // Process data...
    Ok(())
}
```

### Checkpoint Save/Load

```rust
use s3dlio::api::*;

fn save_checkpoint(
    store: &dyn ObjectStore,
    model: &ModelState,
    optimizer: &OptimizerState,
    epoch: usize,
) -> Result<()> {
    let uri = format!("s3://checkpoints/epoch_{:04d}/", epoch);
    
    let mut writer = CheckpointWriter::new(store, &uri)?;
    writer.save_state("model", model)?;
    writer.save_state("optimizer", optimizer)?;
    writer.finalize()?;
    
    Ok(())
}

fn load_checkpoint(
    store: &dyn ObjectStore,
    epoch: usize,
) -> Result<(ModelState, OptimizerState)> {
    let uri = format!("s3://checkpoints/epoch_{:04d}/", epoch);
    
    let reader = CheckpointReader::new(store, &uri)?;
    let model = reader.load_state("model")?;
    let optimizer = reader.load_state("optimizer")?;
    
    Ok((model, optimizer))
}
```

---

## Additional Resources

- **GitHub**: https://github.com/russfellows/s3dlio
- **Changelog**: `docs/Changelog.md`
- **Testing Guide**: `docs/TESTING-GUIDE.md`
- **Adaptive Tuning**: `docs/ADAPTIVE-TUNING.md`
- **Performance Monitoring**: `docs/PERFORMANCE-MONITORING-v0.8.7.md`

---

**Last Updated**: October 2025  
**Version**: 0.9.0  
**Next Release**: v0.9.1 (Stage 3 - Backend-agnostic range engine)
