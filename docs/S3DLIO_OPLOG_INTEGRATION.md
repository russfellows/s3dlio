# s3dlio-oplog Integration Guide

## Overview

The `s3dlio-oplog` crate provides shared operation log replay functionality, eliminating code duplication across the s3dlio ecosystem (s3dlio, s3-bench, dl-driver).

**Location**: `crates/s3dlio-oplog/`  
**Version**: 0.1.0  
**Status**: Production-ready

## Architecture

### Core Components

1. **Format-Tolerant Reader** (from dl-driver design)
   - Auto-detects JSONL and TSV formats
   - Automatic zstd decompression (.zst files)
   - Header-driven column mapping for TSV
   - Handles variations ("operation" vs "op", etc.)

2. **Timeline-Based Replayer** (from s3-bench design)
   - Microsecond-precision scheduling
   - Absolute timeline preservation
   - Speed multiplier support (0.1x to 1000x)
   - Continue-on-error mode

3. **Pluggable Execution** (new trait-based design)
   - `OpExecutor` trait for custom backends
   - Default `S3dlioExecutor` using s3dlio ObjectStore
   - Easy integration with custom storage systems

### Data Types

```rust
pub enum OpType {
    GET,
    PUT,
    DELETE,
    LIST,
    STAT,
}

pub struct OpLogEntry {
    pub idx: Option<u64>,
    pub op: OpType,
    pub bytes: Option<usize>,
    pub endpoint: Option<String>,
    pub file: String,
    pub start: DateTime<Utc>,
    pub duration_ns: Option<i64>,
    pub error: Option<String>,
}
```

## Usage Examples

### Basic Usage with Default Executor

```rust
use s3dlio_oplog::{ReplayConfig, replay_with_s3dlio};
use std::path::PathBuf;

// Replay with default s3dlio executor
let config = ReplayConfig {
    op_log_path: PathBuf::from("workload.tsv.zst"),
    target_uri: Some("s3://my-bucket/data/".to_string()),
    speed: 1.0,  // Real-time speed
    continue_on_error: false,
    filter_ops: None,
};

replay_with_s3dlio(config).await?;
```

### Custom Executor Implementation

```rust
use s3dlio_oplog::{OpExecutor, ReplayConfig, replay_workload};
use async_trait::async_trait;
use anyhow::Result;
use std::sync::Arc;

struct MyCustomExecutor {
    // Your storage backend
}

#[async_trait]
impl OpExecutor for MyCustomExecutor {
    async fn get(&self, uri: &str) -> Result<()> {
        // Custom GET implementation
        Ok(())
    }

    async fn put(&self, uri: &str, bytes: usize) -> Result<()> {
        // Custom PUT implementation
        Ok(())
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        // Custom DELETE implementation
        Ok(())
    }

    async fn list(&self, uri: &str) -> Result<()> {
        // Custom LIST implementation
        Ok(())
    }

    async fn stat(&self, uri: &str) -> Result<()> {
        // Custom STAT implementation
        Ok(())
    }
}

// Use custom executor
let config = ReplayConfig { /* ... */ };
let executor = Arc::new(MyCustomExecutor { /* ... */ });
replay_workload(config, executor).await?;
```

### URI Translation

```rust
use s3dlio_oplog::translate_uri;

// Translate from one backend to another
let original = "http://10.0.0.1:8000/data/file.bin";
let new_uri = translate_uri(
    original,
    "http://10.0.0.1:8000",  // Original endpoint
    "s3://my-bucket"         // Target URI
);
// Result: "s3://my-bucket/data/file.bin"
```

### Operation Filtering

```rust
use s3dlio_oplog::{OpType, OpLogReader};
use std::path::PathBuf;

let reader = OpLogReader::from_file(PathBuf::from("ops.tsv"))?;
let entries = reader.entries();

// Filter to only GET operations
let filter_ops = vec![OpType::GET];
let config = ReplayConfig {
    filter_ops: Some(filter_ops),
    // ...
};
```

## Integration Patterns

### s3-bench Integration

**Before** (local implementation in `src/replay.rs`):
```rust
// s3-bench had its own replay.rs with OpType, OpLogEntry, etc.
```

**After** (using shared crate):
```toml
# Cargo.toml
[dependencies]
s3dlio-oplog = { path = "../s3dlio/crates/s3dlio-oplog" }
```

```rust
// src/main.rs or src/commands/replay.rs
use s3dlio_oplog::{ReplayConfig, replay_with_s3dlio};

match args.command {
    Commands::Replay { op_log, target, speed, .. } => {
        let config = ReplayConfig {
            op_log_path: op_log,
            target_uri: target,
            speed,
            continue_on_error: args.continue_on_error,
            filter_ops: args.filter_ops,
        };
        replay_with_s3dlio(config).await?;
    }
}
```

**Migration Steps**:
1. Remove `src/replay.rs` completely
2. Add `s3dlio-oplog` dependency to `Cargo.toml`
3. Replace replay command implementation with shared crate call
4. Remove redundant type definitions (OpType, OpLogEntry, etc.)

### dl-driver Integration

**Before** (local implementation in `src/oplog_ingest.rs`):
```rust
// dl-driver had OpLogReader with format detection
```

**After** (using shared crate):
```toml
# Cargo.toml
[dependencies]
s3dlio-oplog = { path = "../s3dlio/crates/s3dlio-oplog" }
```

```rust
// src/commands/replay.rs
use s3dlio_oplog::{OpExecutor, ReplayConfig, replay_workload};
use async_trait::async_trait;
use std::sync::Arc;

// Option 1: Use default s3dlio executor
use s3dlio_oplog::replay_with_s3dlio;

match args.command {
    Commands::Replay { op_log, base_uri, remap, .. } => {
        // Pre-translate URIs if using --remap
        let target = if let Some(remap) = remap {
            Some(remap)
        } else {
            base_uri
        };

        let config = ReplayConfig {
            op_log_path: op_log,
            target_uri: target,
            speed: args.speed,
            continue_on_error: true,
            filter_ops: None,
        };
        
        replay_with_s3dlio(config).await?;
    }
}

// Option 2: Custom executor for dl-driver-specific logic
struct DlDriverExecutor {
    // Your custom fields
}

#[async_trait]
impl OpExecutor for DlDriverExecutor {
    async fn get(&self, uri: &str) -> Result<()> {
        // dl-driver specific GET logic
        Ok(())
    }
    // ... other methods
}

let executor = Arc::new(DlDriverExecutor { /* ... */ });
replay_workload(config, executor).await?;
```

**Migration Steps**:
1. Keep or remove `src/oplog_ingest.rs` based on custom logic needs
2. Add `s3dlio-oplog` dependency
3. Use `replay_with_s3dlio()` for simple cases
4. Implement custom `OpExecutor` if special handling needed
5. Leverage `translate_uri()` for endpoint remapping

## Supported Formats

### JSONL Format
```json
{"idx":0,"op":"GET","bytes":1048576,"endpoint":"http://10.0.0.1:8000","file":"/data/file.bin","start":"2024-01-15T10:30:45.123456Z","duration_ns":1234567,"error":null}
{"idx":1,"op":"PUT","bytes":2097152,"endpoint":"http://10.0.0.1:8000","file":"/data/file2.bin","start":"2024-01-15T10:30:45.234567Z","duration_ns":2345678,"error":null}
```

### TSV Format
```tsv
idx	operation	bytes	endpoint	file	start	duration_ns	error
0	GET	1048576	http://10.0.0.1:8000	/data/file.bin	2024-01-15T10:30:45.123456Z	1234567	
1	PUT	2097152	http://10.0.0.1:8000	/data/file2.bin	2024-01-15T10:30:45.234567Z	2345678	
```

**Column Name Variations**:
- Operation: "operation", "op", "Operation", "OP"
- File: "file", "path", "uri", "key"
- Start time: "start", "timestamp", "time"

### Compressed Formats
- `.jsonl.zst` - zstd compressed JSONL
- `.tsv.zst` - zstd compressed TSV
- `.csv.zst` - treated as TSV with compression

Auto-detection based on file extension, automatic decompression.

## Configuration Options

### ReplayConfig Fields

- **op_log_path**: Path to operation log file (PathBuf)
- **target_uri**: Optional target URI for retargeting (Option<String>)
  - If `Some(uri)`, translates all operations to this backend
  - If `None`, uses original URIs from log
- **speed**: Speed multiplier (f64)
  - `0.1` = 10x slower (stress testing)
  - `1.0` = Real-time replay
  - `10.0` = 10x faster (quick validation)
  - `1000.0` = Maximum speed (no delays)
- **continue_on_error**: Error handling (bool)
  - `true` = Log errors and continue
  - `false` = Stop on first error
- **filter_ops**: Operation filter (Option<Vec<OpType>>)
  - `None` = Replay all operations
  - `Some(vec![OpType::GET])` = Only GET operations
  - `Some(vec![OpType::PUT, OpType::DELETE])` = Only writes

## Performance Characteristics

### Timeline Accuracy
- **Microsecond precision**: Uses Tokio's high-resolution timers
- **Absolute scheduling**: Maintains original operation timing relationships
- **Concurrent execution**: Operations spawn independently for true parallel replay

### Scalability
- **Memory efficient**: Streams log entries, doesn't load entire log
- **I/O optimized**: zstd decompression on-the-fly
- **Async design**: Non-blocking execution with Tokio runtime

### Typical Use Cases
1. **Workload replay**: Reproduce production patterns in test environments
2. **Performance testing**: Stress test with speed multipliers
3. **Migration validation**: Verify data integrity across backends
4. **Debugging**: Filter and replay specific operation sequences

## Testing

The crate includes comprehensive tests:

```bash
# Run all tests
cargo test --package s3dlio-oplog

# Expected output:
# - 11 unit tests pass (types, reader, uri, replayer)
# - 5 doc tests pass (examples in documentation)
```

## Error Handling

All functions return `anyhow::Result<T>`:

```rust
use anyhow::{Context, Result};

let config = ReplayConfig { /* ... */ };
replay_with_s3dlio(config)
    .await
    .context("Failed to replay workload")?;
```

Common errors:
- File not found: Invalid op_log_path
- Parse errors: Malformed log entries
- Network errors: Target storage unavailable (if continue_on_error=false)

## Dependencies

```toml
[dependencies]
anyhow = "1.0"
async-trait = "0.1"
chrono = { version = "^0.4", features = ["serde"] }
csv = "1.3"
futures = "0.3"
s3dlio = { path = "../..", default-features = false, features = ["native-backends"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
tracing = "^0.1"
zstd = "0.13"
```

## Version History

### v0.1.0 (Initial Release)
- Format-tolerant reader (JSONL/TSV with zstd)
- Timeline-based replayer with microsecond precision
- OpExecutor trait for pluggable backends
- Default S3dlioExecutor implementation
- URI translation utilities
- Comprehensive test coverage

## See Also

- [s3dlio Main Documentation](../README.md)
- [s3dlio API Reference](api/)
- [Changelog](Changelog.md)
