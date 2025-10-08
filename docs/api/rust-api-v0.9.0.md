# s3dlio Rust Library API Guide - v0.9.0

**Version**: 0.9.0 (API-stable beta)  
**Date**: October 2025  
**Status**: Breaking changes from v0.8.x - see migration section

---

## Table of Contents

- [Overview](#overview)
- [What's New in v0.9.0](#whats-new-in-v090)
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

## What's New in v0.9.0

### Major Features

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

### Deferred Features

- **Stage 3** (Backend-agnostic range engine): Moved to v0.9.1
  - Non-breaking internal optimization
  - Will improve range request efficiency across all backends

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
