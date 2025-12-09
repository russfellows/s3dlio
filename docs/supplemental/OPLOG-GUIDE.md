# Operation Log (OpLog) Guide

**Last Updated:** October 9, 2025  
**Version:** v0.9.4  
**Crate Location:** `crates/s3dlio-oplog/`  
**Status:** Production-ready

---

## Overview

The `s3dlio-oplog` crate provides shared operation log replay functionality for the s3dlio ecosystem, eliminating code duplication across s3dlio, s3-bench (io-bench), and dl-driver.

**Key Features:**
- ✅ Format-tolerant parsing (JSONL, TSV, zstd-compressed)
- ✅ Timeline-based replay with microsecond precision
- ✅ Pluggable execution via `OpExecutor` trait
- ✅ URI translation for backend retargeting
- ✅ Speed multiplier support (0.1x to 1000x)
- ✅ Continue-on-error mode for resilience

---

## Architecture

### Core Components

#### 1. Format-Tolerant Reader
**File:** `src/reader.rs`

- Auto-detects JSONL and TSV formats
- Automatic zstd decompression (.zst files)
- Header-driven column mapping for TSV
- Handles column name variations ("operation" vs "op", "file" vs "path", etc.)

**Supported Formats:**
```bash
# Plain JSONL
{"idx":0,"op":"GET","file":"s3://bucket/file.bin","start":"2025-10-09T10:00:00Z","duration_ns":1000000}

# TSV (tab-separated)
idx    op    bytes    endpoint    file    start    duration_ns    error
0      GET   1024     s3://       file1   2025...  1000000       null

# Compressed (.zst extension)
workload.jsonl.zst
workload.tsv.zst
```

#### 2. Timeline-Based Replayer
**File:** `src/replayer.rs`

- Microsecond-precision absolute scheduling
- Preserves original operation timing relationships
- Speed multiplier: 0.1x (slow-mo) to 1000x (stress test)
- Continue-on-error mode for incomplete logs

**Timeline Scheduling:**
```rust
// Calculate absolute schedule time
let schedule_at = base_time + (entry.start - first_timestamp) / speed;

// Sleep until scheduled time
let now = Instant::now();
if schedule_at > now {
    tokio::time::sleep(schedule_at - now).await;
}

// Execute operation
executor.execute(&entry).await?;
```

#### 3. Pluggable Execution
**File:** `src/replayer.rs` (OpExecutor trait)

Default executor uses s3dlio's `ObjectStore` trait for universal backend support:

```rust
#[async_trait]
pub trait OpExecutor: Send + Sync {
    async fn execute(&self, entry: &OpLogEntry) -> Result<()>;
}

pub struct S3dlioExecutor {
    store: Arc<dyn ObjectStore>,
}
```

---

## Data Types

### OpType Enum
```rust
pub enum OpType {
    GET,      // Download object
    PUT,      // Upload object
    DELETE,   // Delete object
    LIST,     // List objects with prefix
    STAT,     // Get object metadata
}
```

### OpLogEntry Struct
```rust
pub struct OpLogEntry {
    pub idx: Option<u64>,           // Sequential operation index
    pub op: OpType,                  // Operation type
    pub bytes: Option<usize>,        // Size in bytes (for GET/PUT)
    pub endpoint: Option<String>,    // Backend endpoint
    pub file: String,                // Object URI/path
    pub start: DateTime<Utc>,        // Operation start timestamp
    pub duration_ns: Option<i64>,    // Duration in nanoseconds
    pub error: Option<String>,       // Error message if failed
}
```

---

## Usage Examples

### Basic Replay with Default Executor

```rust
use s3dlio_oplog::{ReplayConfig, replay_with_s3dlio};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Replay workload at real-time speed
    let config = ReplayConfig {
        op_log_path: PathBuf::from("workload.tsv.zst"),
        target_uri: Some("s3://my-bucket/data/".to_string()),
        speed: 1.0,  // 1.0 = real-time, 2.0 = 2x faster
        continue_on_error: false,
        filter_ops: None,
    };

    replay_with_s3dlio(config).await?;
    Ok(())
}
```

### Advanced: Custom Executor

```rust
use s3dlio_oplog::{OpExecutor, OpLogEntry, ReplayConfig, replay_workload};
use async_trait::async_trait;
use anyhow::Result;
use std::sync::Arc;

struct MyCustomExecutor {
    // Your custom storage client
}

#[async_trait]
impl OpExecutor for MyCustomExecutor {
    async fn execute(&self, entry: &OpLogEntry) -> Result<()> {
        match entry.op {
            OpType::GET => {
                // Implement custom GET logic
                println!("Custom GET: {}", entry.file);
                Ok(())
            }
            OpType::PUT => {
                // Implement custom PUT logic
                println!("Custom PUT: {}", entry.file);
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let executor = Arc::new(MyCustomExecutor {});
    
    let config = ReplayConfig {
        op_log_path: PathBuf::from("workload.jsonl"),
        target_uri: None,
        speed: 10.0,  // 10x faster for stress testing
        continue_on_error: true,
        filter_ops: Some(vec![OpType::GET, OpType::PUT]),
    };

    replay_workload(config, executor).await?;
    Ok(())
}
```

### URI Translation (Backend Retargeting)

```rust
use s3dlio_oplog::translate_uri;

// Translate S3 URIs to Azure
let azure_uri = translate_uri(
    "s3://my-bucket/data/file.bin",
    "az://my-container/data/"
)?;
assert_eq!(azure_uri, "az://my-container/data/file.bin");

// Translate to GCS
let gcs_uri = translate_uri(
    "s3://my-bucket/data/file.bin",
    "gs://my-bucket/data/"
)?;
assert_eq!(gcs_uri, "gs://my-bucket/data/file.bin");

// Translate to local filesystem
let file_uri = translate_uri(
    "s3://my-bucket/data/file.bin",
    "file:///mnt/storage/data/"
)?;
assert_eq!(file_uri, "file:///mnt/storage/data/file.bin");
```

---

## Replay Configuration

### ReplayConfig Fields

```rust
pub struct ReplayConfig {
    /// Path to operation log file (JSONL, TSV, or .zst compressed)
    pub op_log_path: PathBuf,
    
    /// Target URI for backend retargeting (None = use original URIs)
    pub target_uri: Option<String>,
    
    /// Speed multiplier: 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower
    pub speed: f64,
    
    /// Continue executing operations even if some fail
    pub continue_on_error: bool,
    
    /// Filter to specific operation types (None = replay all)
    pub filter_ops: Option<Vec<OpType>>,
}
```

### Common Configurations

#### Production Replay (Exact Timing)
```rust
ReplayConfig {
    op_log_path: PathBuf::from("prod_workload.tsv.zst"),
    target_uri: Some("s3://staging-bucket/".to_string()),
    speed: 1.0,  // Exact timing
    continue_on_error: false,  // Fail on errors
    filter_ops: None,  // All operations
}
```

#### Stress Testing (10x Speed)
```rust
ReplayConfig {
    op_log_path: PathBuf::from("workload.jsonl"),
    target_uri: Some("s3://test-bucket/".to_string()),
    speed: 10.0,  // 10x faster
    continue_on_error: true,  // Keep going despite errors
    filter_ops: None,
}
```

#### Read-Only Analysis
```rust
ReplayConfig {
    op_log_path: PathBuf::from("workload.tsv.zst"),
    target_uri: Some("s3://readonly-bucket/".to_string()),
    speed: 1.0,
    continue_on_error: true,
    filter_ops: Some(vec![OpType::GET, OpType::LIST]),  // Only reads
}
```

---

## Integration with s3dlio Ecosystem

### s3-bench (io-bench) Integration

**Before (duplicated code):**
```rust
// Custom replay logic in s3-bench
// ~500 lines of parsing, scheduling, execution
```

**After (uses shared crate):**
```rust
use s3dlio_oplog::{ReplayConfig, replay_with_s3dlio};

let config = ReplayConfig::from_args(args);
replay_with_s3dlio(config).await?;
```

**Benefits:**
- Eliminated 500+ lines of duplicate code
- Consistent behavior across tools
- Shared bug fixes and improvements

### dl-driver Integration

**Before (custom JSONL parser):**
```rust
// Custom format-tolerant reader
// ~200 lines of parsing logic
```

**After (uses shared crate):**
```rust
use s3dlio_oplog::read_op_log;

let entries = read_op_log(&path).await?;
for entry in entries {
    process_operation(entry).await?;
}
```

**Benefits:**
- Automatic format detection (JSONL/TSV)
- Automatic decompression (.zst)
- Header-driven column mapping

---

## OpLog Format Specification

### JSONL Format
```json
{"idx":0,"op":"GET","bytes":1024,"endpoint":"s3://","file":"s3://bucket/file1.bin","start":"2025-10-09T10:00:00.123456Z","duration_ns":1234567,"error":null}
{"idx":1,"op":"PUT","bytes":2048,"endpoint":"s3://","file":"s3://bucket/file2.bin","start":"2025-10-09T10:00:01.234567Z","duration_ns":2345678,"error":null}
```

### TSV Format
```tsv
idx	op	bytes	endpoint	file	start	duration_ns	error
0	GET	1024	s3://	s3://bucket/file1.bin	2025-10-09T10:00:00.123456Z	1234567	
1	PUT	2048	s3://	s3://bucket/file2.bin	2025-10-09T10:00:01.234567Z	2345678	
```

### Required Fields
- `op` (or `operation`): Operation type (GET, PUT, DELETE, LIST, STAT)
- `file` (or `path`): Object URI or path
- `start` (or `timestamp`): ISO 8601 timestamp with timezone

### Optional Fields
- `idx` (or `index`): Sequential operation number
- `bytes` (or `size`): Object size in bytes
- `endpoint`: Backend endpoint/URI scheme
- `duration_ns`: Operation duration in nanoseconds
- `error`: Error message if operation failed

---

## Performance Characteristics

### Memory Usage
- **Streaming Reader**: Processes log entries one at a time (O(1) memory)
- **Replay Buffer**: Single entry in flight per concurrent operation
- **Large Logs**: 1M+ entries processed with constant memory footprint

### Throughput
- **Parsing**: 100K+ entries/second (JSONL/TSV)
- **Decompression**: Automatic zstd streaming (minimal overhead)
- **Replay**: Limited by executor backend, not replay logic

### Concurrency
- **Default**: Executes operations sequentially in timeline order
- **Custom Executors**: Can implement internal concurrency
- **Thread-Safe**: All components are Send + Sync

---

## Testing

### Test Coverage (16 tests total)

**Unit Tests (11 tests):**
- `types.rs`: OpType parsing and display (2 tests)
- `reader.rs`: JSONL/TSV parsing, format detection (4 tests)
- `uri.rs`: Cross-backend translation (3 tests)
- `replayer.rs`: Mock executor replay (2 tests)

**Doc Tests (5 tests):**
- API usage examples validation

### Running Tests
```bash
# All tests
cargo test -p s3dlio-oplog

# Specific module
cargo test -p s3dlio-oplog reader::

# With output
cargo test -p s3dlio-oplog -- --nocapture
```

---

## CLI Usage Examples

### Generate OpLog from s3dlio Operations

```bash
# Using s3-cli with --op-log flag
s3-cli get s3://bucket/data/*.bin ./local/ \
    --op-log workload.tsv.zst \
    --op-log-format tsv

# Output: workload.tsv.zst with GET operations recorded
```

### Replay Workload

```bash
# Replay at real-time speed
s3-bench replay workload.tsv.zst --target s3://staging-bucket/

# Replay 10x faster
s3-bench replay workload.tsv.zst --target s3://test-bucket/ --speed 10

# Read-only replay (GET/LIST only)
s3-bench replay workload.tsv.zst --filter GET,LIST

# Stress test with continue-on-error
s3-bench replay workload.tsv.zst --speed 100 --continue-on-error
```

---

## Troubleshooting

### Error: "Unsupported format"
**Cause:** File is not valid JSONL or TSV  
**Solution:** Verify file format matches specification above

### Error: "Missing required field: op"
**Cause:** Log file missing operation type column  
**Solution:** Ensure log has `op` or `operation` column

### Error: "Failed to parse timestamp"
**Cause:** Invalid timestamp format  
**Solution:** Use ISO 8601 format with timezone (e.g., `2025-10-09T10:00:00Z`)

### Slow Replay Performance
**Cause:** Backend executor bottleneck, not replay logic  
**Solution:** Tune backend configuration (connection pools, concurrency)

### Memory Growth During Replay
**Cause:** Custom executor holding references  
**Solution:** Ensure executor cleans up resources after each operation

---

## Migration Guide

### From s3-bench Custom Replay

**Before:**
```rust
// Custom replay implementation in s3-bench
let entries = parse_op_log(&path)?;
for entry in entries {
    schedule_operation(entry)?;
}
```

**After:**
```rust
use s3dlio_oplog::{ReplayConfig, replay_with_s3dlio};

let config = ReplayConfig { /* ... */ };
replay_with_s3dlio(config).await?;
```

### From dl-driver Custom Parser

**Before:**
```rust
// Custom format-tolerant reader
let entries = read_jsonl_or_tsv(&path)?;
```

**After:**
```rust
use s3dlio_oplog::read_op_log;

let entries = read_op_log(&path).await?;
```

---

## See Also

- **Main README.md** - Project overview and setup
- **BACKEND-TESTING.md** - Backend testing strategies
- **docs/api/rust-api-v0.9.3-addendum.md** - Rust API reference
- **Source Code**: `crates/s3dlio-oplog/src/`

---

## Version History

- **v0.1.0** (v0.8.13) - Initial release with shared replay functionality
- **v0.9.x** - Continued maintenance and integration improvements
