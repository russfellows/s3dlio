# S3DL-IO Rust API Documentation

## Overview

S3DL-IO provides a unified, high-performance interface for cloud storage operations optimized for machine learning and data processing workloads.

## API Stability Guarantee

- **`s3dlio::api`** - Stable public API with semantic versioning compatibility
- **`s3dlio::api::advanced`** - Advanced features with stability guarantees  
- **Internal modules** - Implementation details that may change

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
s3dlio = "0.8"
```

### Basic Usage

```rust
use s3dlio::api::{store_for_uri, WriterOptions, CompressionConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a store for any supported backend
    let store = store_for_uri("s3://my-bucket/data/")?;
    
    // Read an object
    let data = store.get("input.txt").await?;
    println!("Read {} bytes", data.len());
    
    // Write with streaming and compression
    let options = WriterOptions::new()
        .with_compression(CompressionConfig::zstd_default());
    let mut writer = store.create_writer("output.txt.zst", options).await?;
    writer.write_chunk(b"Hello, ").await?;
    writer.write_chunk(b"World!").await?;
    writer.finalize().await?;
    println!("Write completed");
    
    Ok(())
}
```

### Data Loading

```rust
use s3dlio::api::{DataLoader, Dataset, LoaderOptions};

// Custom dataset implementation
struct MyDataset {
    files: Vec<String>,
}

impl Dataset for MyDataset {
    type Item = Vec<u8>;
    
    async fn get_item(&self, index: usize) -> Result<Self::Item, anyhow::Error> {
        // Load and return data for the given index
        todo!()
    }
    
    fn len(&self) -> usize {
        self.files.len()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dataset = MyDataset { files: vec!["file1.dat".to_string()] };
    let options = LoaderOptions::default();
    let loader = DataLoader::new(dataset, options);
    
    // Use the loader in your training loop
    while let Some(batch) = loader.next().await {
        // Process batch
        println!("Loaded batch of {} items", batch.len());
    }
    
    Ok(())
}
```

## Supported Backends

- **Amazon S3** - `s3://bucket/path/`
- **Azure Blob Storage** - `azure://container/path/`  
- **Local Filesystem** - `file:///path/to/dir/`
- **Direct I/O** - Use `high_performance_store_for_uri()` for zero-copy operations

## Core API Reference

### ObjectStore Trait

The main interface for storage operations:

```rust
#[async_trait]
pub trait ObjectStore: Send + Sync {
    async fn get(&self, uri: &str) -> Result<Vec<u8>>;
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()>;
    async fn delete(&self, uri: &str) -> Result<()>;
    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>>;
    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>>;
    // ... additional methods
}
```

### ObjectWriter Trait

Streaming interface for efficient uploads:

```rust
#[async_trait] 
pub trait ObjectWriter: Send + Sync {
    async fn write_chunk(&mut self, data: &[u8]) -> Result<()>;
    async fn write_owned_bytes(&mut self, data: Vec<u8>) -> Result<()>;
    async fn finalize(self: Box<Self>) -> Result<()>;
}
```

### Factory Functions

- `store_for_uri(uri)` - Create appropriate store for URI
- `high_performance_store_for_uri(uri)` - Optimized store with direct I/O
- `infer_scheme(uri)` - Detect URI scheme

### Configuration Types

- `WriterOptions` - Configure compression, buffering, etc.
- `CompressionConfig` - Gzip, Zstd, or None
- `LoaderOptions` - Data loading configuration

## Advanced API

For high-performance scenarios, use the advanced API:

```rust
use s3dlio::api::advanced::{AsyncPoolDataLoader, PoolConfig};

let pool_config = PoolConfig {
    max_connections: 100,
    prefetch_size: 1000,
    ..Default::default()
};
let loader = AsyncPoolDataLoader::new(dataset, pool_config).await?;
```

## Error Handling

All operations return `anyhow::Result<T>` for flexible error handling:

```rust
use s3dlio::api::{S3dlioResult, DatasetError};

fn handle_errors() -> S3dlioResult<()> {
    // S3dlioResult<T> is an alias for anyhow::Result<T>
    let store = store_for_uri("s3://bucket/")?;
    // ... operations
    Ok(())
}
```

## Performance Tuning

### High-Throughput Scenarios

```rust
// Use direct I/O for maximum performance
// Note: O_DIRECT requires compatible filesystems (ext4, xfs, etc.)
// Avoid tmpfs, /tmp, or other memory-based filesystems
let store = match high_performance_store_for_uri("file:///data/bucket/") {
    Ok(store) => {
        println!("Using O_DIRECT optimization");
        store
    }
    Err(e) => {
        println!("O_DIRECT unavailable ({}), using regular store", e);
        store_for_uri("file:///data/bucket/")?
    }
};

// Configure aggressive prefetching
let options = LoaderOptions {
    prefetch_size: 10000,
    num_workers: 32,
    ..Default::default()
};
```

### Memory Optimization

```rust
// Use streaming for large files
let mut writer = store.create_writer("large_file.dat", Default::default()).await?;
for chunk in large_data_chunks {
    writer.write_owned_bytes(chunk).await?; // Zero-copy when possible
}
writer.finalize().await?;
```

## Checkpointing

Save and restore state for long-running operations:

```rust
use s3dlio::api::{CheckpointStore, CheckpointConfig};

let checkpoint_config = CheckpointConfig::new("s3://bucket/checkpoints/");
let checkpoint_store = CheckpointStore::new(checkpoint_config).await?;

// Save state
checkpoint_store.save("model_step_1000", &my_state).await?;

// Load state
let restored_state = checkpoint_store.load("model_step_1000").await?;
```

## Migration Guide

### From v0.7.x

The new API is mostly backward compatible. Update imports:

```rust
// Old
use s3dlio::{ObjectStore, store_for_uri};

// New (recommended)
use s3dlio::api::{ObjectStore, store_for_uri};
```

### Deprecation Warnings

Legacy imports will show deprecation warnings but continue to work until v1.0.

## Thread Safety

All API types are `Send + Sync` and can be safely used across threads:

```rust
use std::sync::Arc;

let store = Arc::new(store_for_uri("s3://bucket/")?);
let handles: Vec<_> = (0..10)
    .map(|i| {
        let store = store.clone();
        tokio::spawn(async move {
            store.get_object(&format!("file_{}.dat", i)).await
        })
    })
    .collect();
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.rs` - Simple read/write operations
- `data_loading.rs` - ML dataset loading patterns
- `high_performance.rs` - Optimized configurations
- `checkpointing.rs` - State management

## Version Compatibility

Check API compatibility programmatically:

```rust
use s3dlio::api::{API_VERSION, is_compatible_version};

assert!(is_compatible_version("0.8.0")); // true
assert!(!is_compatible_version("0.9.0")); // false - minor version too high
```
