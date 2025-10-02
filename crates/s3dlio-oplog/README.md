# s3dlio-oplog

Shared operation log replay library for the s3dlio ecosystem, enabling code reuse across s3-bench (io-bench), dl-driver, and other storage performance tools.

## Overview

This crate provides a universal operation log replay system with:

- **Pluggable Execution**: `OpExecutor` trait for custom backend implementations
- **Format Support**: JSONL and TSV with automatic zstd decompression
- **Timeline Replay**: Microsecond-precision timing with speed multiplier
- **URI Translation**: Cross-backend retargeting (s3:// → az:// → file:// → direct://)
- **Error Handling**: Continue-on-error support for resilient replay
- **Zero Warnings**: Production-ready code quality

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
s3dlio-oplog = { path = "../s3dlio-oplog" }
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Basic Usage

```rust
use s3dlio_oplog::{OpExecutor, OpLogReader, ReplayConfig, replay_workload};
use async_trait::async_trait;
use anyhow::Result;

// Implement OpExecutor trait for your backend
struct MyExecutor;

#[async_trait]
impl OpExecutor for MyExecutor {
    async fn get(&self, uri: &str) -> Result<()> {
        println!("GET {}", uri);
        // Your implementation here
        Ok(())
    }
    
    async fn put(&self, uri: &str, bytes: usize) -> Result<()> {
        println!("PUT {} ({} bytes)", uri, bytes);
        // Your implementation here
        Ok(())
    }
    
    // ... implement other operations
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = ReplayConfig {
        op_log_path: "operations.tsv.zst".into(),
        target_uri: Some("s3://my-bucket".to_string()),
        speed: 10.0,  // 10x faster replay
        continue_on_error: true,
        filter_ops: None,
    };
    
    let executor = std::sync::Arc::new(MyExecutor);
    replay_workload(config, executor).await?;
    
    Ok(())
}
```

## Features

### Supported Formats

- **JSONL**: One JSON object per line
- **TSV**: Tab-separated values with flexible column mapping
- **Zstd Compression**: Automatic decompression for `.zst` files

### Operation Types

- `GET` - Object retrieval
- `PUT` - Object upload
- `DELETE` - Object deletion
- `LIST` - Listing operations
- `STAT` / `HEAD` - Metadata operations (HEAD is alias for STAT)

### URI Translation

Remap operations between storage backends:

```rust
use s3dlio_oplog::translate_uri;

// S3 → Azure
let result = translate_uri("s3://oldbucket/data.bin", "s3://oldbucket", "az://newstorage/container")?;
// Result: "az://newstorage/container/data.bin"

// File → DirectIO
let result = translate_uri("file:///tmp/data.bin", "file:///tmp", "direct:///nvme/data")?;
// Result: "direct:///nvme/data/data.bin"
```

## Compatibility Testing

This crate has been thoroughly tested for compatibility with:

### s3-bench (io-bench)
- ✅ 13-column TSV format
- ✅ HEAD → STAT operation aliasing
- ✅ Speed multiplier timeline scheduling
- ✅ Endpoint prefix stripping + target remapping
- ✅ Zstd compression auto-detection

### dl-driver
- ✅ JSONL format with flexible field mapping
- ✅ Base URI path joining
- ✅ Path remapping for cross-environment replay
- ✅ Field aliases ("operation" vs "op", "t_start_ns" vs "start")
- ✅ Fast mode (skip timing delays)

**Test Coverage**: 33 tests (11 unit + 17 integration + 5 doc) - 100% passing

See [COMPATIBILITY_TESTING.md](docs/COMPATIBILITY_TESTING.md) for detailed compatibility test documentation.

## API Documentation

### Core Types

- `OpType`: Operation enumeration (GET, PUT, DELETE, LIST, STAT)
- `OpLogEntry`: Single operation record
- `OpExecutor`: Trait for pluggable execution backends
- `OpLogReader`: Parser for JSONL/TSV formats
- `ReplayConfig`: Configuration for replay workload

### Main Functions

- `replay_workload(config, executor)`: Execute replay with timeline scheduling
- `translate_uri(source, old_prefix, new_prefix)`: Remap URIs between backends

See [API Documentation](docs/S3DLIO_OPLOG_INTEGRATION.md) for full integration guide.

## Examples

See [examples/oplog_replay_basic.rs](examples/oplog_replay_basic.rs) for a complete working example.

## File Format Examples

### TSV Format (s3-bench compatible)

```tsv
idx	thread	op	client_id	n_objects	bytes	endpoint	file	error	start	first_byte	end	duration_ns
0	1	GET	client-1	1	1048576	http://s3.local:9000	/bucket/data/file1.bin		2025-01-01T00:00:00Z	2025-01-01T00:00:00.5Z	2025-01-01T00:00:01Z	1000000000
1	2	PUT	client-1	1	2097152	http://s3.local:9000	/bucket/data/file2.bin		2025-01-01T00:00:02Z	2025-01-01T00:00:02.5Z	2025-01-01T00:00:03Z	1000000000
```

### JSONL Format (dl-driver compatible)

```jsonl
{"operation": "GET", "file": "/test/file1.dat", "bytes": 1024, "t_start_ns": 1000000000}
{"operation": "PUT", "file": "/test/file2.dat", "bytes": 2048, "t_start_ns": 1500000000}
```

### Minimal TSV Format

```tsv
op	file	bytes	start	duration_ns	error
GET	/data/small.bin	512	2025-01-01T00:00:00Z	1000000	
PUT	/data/large.bin	1048576	2025-01-01T00:00:01Z	5000000	
```

## Migration Guide

### For s3-bench Users

The shared crate is fully compatible with existing s3-bench operation logs:

```rust
use s3dlio_oplog::{OpLogReader, ReplayConfig, replay_workload};

// Read s3-bench 13-column TSV format
let reader = OpLogReader::from_file("s3bench_operations.tsv.zst")?;

// Configure replay with speed multiplier (s3-bench --speed flag)
let config = ReplayConfig {
    op_log_path: "s3bench_operations.tsv.zst".into(),
    target_uri: Some("s3://newbucket".to_string()),  // s3-bench --target flag
    speed: 10.0,  // s3-bench --speed flag
    continue_on_error: true,  // s3-bench --continue-on-error flag
    filter_ops: None,
};

// Your existing s3-bench executor implementation
replay_workload(config, executor).await?;
```

### For dl-driver Users

The shared crate supports dl-driver's JSONL format and path remapping:

```rust
use s3dlio_oplog::{OpLogReader, ReplayConfig, replay_workload};

// Read dl-driver JSONL format
let reader = OpLogReader::from_file("dl_driver_ops.jsonl")?;

// Configure with base URI (dl-driver --base-uri flag)
let config = ReplayConfig {
    op_log_path: "dl_driver_ops.jsonl".into(),
    target_uri: Some("file:///tmp/replay_test".to_string()),
    speed: 1.0,  // Normal speed
    continue_on_error: false,
    filter_ops: None,
};

// Path remapping handled via translate_uri() function
replay_workload(config, executor).await?;
```

## Development

### Running Tests

```bash
# All tests (unit + integration + doc)
cargo test

# Integration tests only
cargo test --test integration_test

# Specific compatibility test
cargo test test_s3_bench_13_column_tsv
```

### Zero Warnings Build

```bash
cargo build --release
# Should produce zero warnings
```

## License

Part of the s3dlio project. See main repository for license details.

## Related Documentation

- [Integration Guide](docs/S3DLIO_OPLOG_INTEGRATION.md) - Detailed API documentation
- [Compatibility Testing](docs/COMPATIBILITY_TESTING.md) - Test coverage details
- [Main s3dlio README](../../README.md) - Parent project documentation
