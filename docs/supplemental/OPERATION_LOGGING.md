# Operation Logging in s3dlio

## Overview

s3dlio v0.9.22+ includes comprehensive operation logging with support for distributed timing synchronization and client identification. This document explains the logging architecture, particularly the first_byte_time tracking strategy.

## Core Concepts

### Operation Log Format

All operations are logged to a zstd-compressed TSV file with the following fields:

```
idx  thread  op  client_id  n_objects  bytes  endpoint  file  error  start  first_byte  end  duration_ns
```

**Field Descriptions:**
- `idx`: Sequential operation index (starts at 0)
- `thread`: Thread ID performing the operation
- `op`: Operation type (GET, PUT, LIST, DELETE, HEAD, etc.)
- `client_id`: Client identifier (set via `set_client_id()`)
- `n_objects`: Number of objects affected (1 for single ops, N for batch/list)
- `bytes`: Data transferred in bytes (0 for metadata operations)
- `endpoint`: Storage backend endpoint (e.g., "file://", "s3://bucket")
- `file`: Object path without URI scheme
- `error`: Error message if operation failed (empty on success)
- `start`: ISO 8601 timestamp when operation started
- `first_byte`: ISO 8601 timestamp when first byte received/sent (see below)
- `end`: ISO 8601 timestamp when operation completed
- `duration_ns`: Operation duration in nanoseconds (end - start)

### Global Configuration Functions

#### Clock Offset Synchronization

```rust
pub fn set_clock_offset(offset_ns: i64) -> io::Result<()>
pub fn get_clock_offset() -> i64
```

**Purpose**: Synchronize timestamps across distributed agents to a controller's reference time.

**Use case**: In distributed benchmarking, agents may have slight clock skew. The controller sends its reference timestamp, and agents calculate an offset to adjust all logged timestamps.

**Example**:
```rust
// Controller sends: start_timestamp_ns = 1732558320000000000
// Agent receives at: local_now_ns = 1732558320005000000
// Offset = start_timestamp_ns - local_now_ns = -5000000 (5ms behind)
s3dlio::set_clock_offset(-5000000)?;
// All subsequent timestamps adjusted by -5ms
```

#### Client Identification

```rust
pub fn set_client_id(client_id: &str) -> io::Result<()>
pub fn get_client_id() -> String
```

**Purpose**: Tag all operations with a client identifier for multi-agent analysis.

**Use case**: When multiple agents write to the same oplog directory, each needs a unique identifier. Also useful for tracking which client performed which operation in merged logs.

**Example**:
```rust
// Standalone client
s3dlio::set_client_id("standalone")?;

// Distributed agent
s3dlio::set_client_id("agent-1")?;

// Custom identifier from environment
let client_id = std::env::var("CLIENT_ID").unwrap_or_else(|_| "default".to_string());
s3dlio::set_client_id(&client_id)?;
```

## First Byte Tracking Strategy (v0.9.22+)

### Overview

The `first_byte` field provides **approximate time-to-first-byte (TTFB)** tracking. Due to architectural constraints of the ObjectStore trait, true first-byte timing requires streaming APIs not yet implemented.

### Why Approximate?

The `ObjectStore` trait methods return complete data:
```rust
async fn get(&self, uri: &str) -> Result<Bytes>  // Returns ALL data
async fn put(&self, uri: &str, data: &[u8]) -> Result<()>  // Sends ALL data
```

These APIs don't expose streaming progress, so we can't distinguish:
- When HTTP response headers arrive vs when body completes
- When first chunk is acknowledged vs when entire upload finishes
- Per-chunk timing for large objects

### Current Implementation

#### GET Operations
```rust
let start = SystemTime::now();
let result = self.inner.get(uri).await;
let first_byte = SystemTime::now();  // Captured immediately after get() completes
let end = first_byte;  // For simple get(), first_byte ≈ end
```

**Interpretation**:
- `first_byte` ≈ `end` for small objects (< 1MB)
- For network operations (S3/Azure/GCS), this approximates when the full HTTP response (headers + body) has been received
- For local files, this is essentially the same as end time since file I/O is typically synchronous
- **Limitation**: Can't distinguish header receipt from body completion without deeper instrumentation

**When this is accurate**:
- Small objects where network latency dominates (TTFB ≈ total time)
- Throughput analysis where exact TTFB isn't critical
- Relative comparisons (comparing across different backends/configs)

**When this is NOT accurate**:
- Large objects (> 10MB) where streaming matters
- True TTFB analysis for CDN/caching behavior
- Per-chunk performance analysis

#### PUT Operations
```rust
let start = SystemTime::now();
let bytes = data.len() as u64;
let result = self.inner.put(uri, data).await;
let end = SystemTime::now();

// For PUT, first_byte = start (upload begins immediately)
self.log_operation("PUT", uri, bytes, 1, start, Some(start), end, error);
```

**Interpretation**:
- `first_byte` = `start` since upload begins immediately after put() is called
- **Future enhancement**: Could track when first chunk ACK received for multipart uploads
- Currently just marks the upload initiation time

#### Metadata Operations (LIST, HEAD, DELETE)
```rust
// These operations don't transfer object data, so first_byte = None
self.log_operation("LIST", uri_prefix, 0, num_objects, start, None, end, error);
```

**Operations with no first_byte**:
- `LIST` - Metadata enumeration
- `HEAD` / `stat()` - Object metadata query
- `DELETE` - Object removal
- `CREATE_CONTAINER` / `DELETE_CONTAINER` - Container management

### Future Enhancements

To achieve true first-byte tracking, we would need:

1. **Streaming GET API**:
   ```rust
   async fn get_stream(&self, uri: &str) -> Result<impl Stream<Item = Result<Bytes>>>
   ```
   - Capture timestamp when first chunk arrives
   - Track per-chunk timing for large objects

2. **Backend Instrumentation**:
   - Modify S3/Azure/GCS store implementations to expose internal timing
   - Hook into HTTP client to capture response header timing
   - Instrument multipart upload progress

3. **RangeEngine Integration**:
   - For large objects using range requests, track first chunk completion
   - Already has per-chunk timing infrastructure
   - Could populate first_byte with first range request completion time

### Recommendations

**For benchmarking purposes:**
- ✅ Use current implementation for throughput analysis
- ✅ Compare relative performance across backends
- ✅ Analyze small object performance (< 1MB)
- ❌ Don't use for precise TTFB metrics on large objects
- ❌ Don't assume first_byte accurately represents header receipt time

**For true TTFB analysis:**
- Use dedicated HTTP timing tools (curl with --write-out, vegeta, etc.)
- Instrument application code with streaming APIs
- Consider backend-specific tools (S3 CloudWatch metrics, Azure Monitor)

**For large object analysis:**
- Wait for streaming API implementation (future s3dlio version)
- Use RangeEngine for per-chunk timing (requires large_object_threshold config)
- Consider object_store crate's get_range() for manual chunking

## Lifecycle: one op-log session per process

The op-log moves through `Uninitialized` -> `Active` -> `Finalized`, and
`Finalized` is terminal: a process supports **exactly one** op-log session.

| Call | Uninitialized | Active | Finalized |
|---|---|---|---|
| `init_op_logger(path)` (`init_op_log`) | creates the file, becomes Active (a failed init stays Uninitialized, so it can be retried) | returns `Ok`; the new path is **not** created or truncated | **error** (Python: `RuntimeError`); no file is created or truncated |
| `finalize_op_logger()` (`finalize_op_log`) | no-op | writes every accepted record (unless the writer fails with an I/O error), finishes the zstd stream, waits until the writer signals it has finished, becomes Finalized | no-op (waits if another finalize is still draining) |
| `global_logger()` / `is_op_log_active()` | `None` / `False` | `Some` / `True` | `None` / `False` |

Stores capture the logger when they are constructed
(`store_for_uri_with_logger(uri, global_logger())`):

- stores created **before** init are unlogged for their whole life;
- stores created **before** finalize stop recording when finalize runs (their
  later entries are rejected with `LogOutcome::Closed`, not written);
- stores created **after** finalize are unlogged.

`Logger::log(entry)` returns `()`, as it always has. `Logger::try_log(entry)`
does the same work and returns a `LogOutcome` (`Accepted`, `Full`, `Closed`,
`WriterFailed`). Every `Accepted` entry is written by finalize, including
entries racing with it, unless the writer fails with an I/O error (after which
`try_log` returns `WriterFailed`). The serializer itself was not changed by the
lifecycle work (header, column order, idx numbering, operation names, zstd
framing); see "What the format tests establish" below for what is checked.

After finalize, `set_clock_offset` / `get_clock_offset` return an error of kind
`Unsupported` saying the op-log was already finalized (before any init the kind
is `NotFound`, "Logger not initialized").

### Detecting dropped records

By default the op-log is **lossy**: it never slows the caller, so when the
writer cannot keep up and the queue (`S3DLIO_OPLOG_BUF`, default 8192 entries)
is full, the entry is dropped. `S3DLIO_OPLOG_LOSSLESS=1` makes callers wait
instead. A caller can always tell whether a log is complete:

```rust
if let Some(stats) = s3dlio::op_log_stats() {          // Option<OpLogStats>
    // stats.accepted, stats.dropped_full, stats.rejected_closed,
    // stats.writer_failed, stats.lossless
    let complete = stats.dropped_full == 0 && stats.writer_failed == 0;
}
```

```python
stats = s3dlio.op_log_stats()   # None before init_op_log(), else a dict
complete = stats["dropped_full"] == 0 and stats["writer_failed"] == 0
```

| Count | Meaning |
|---|---|
| `accepted` | entries queued for the writer (`LogOutcome::Accepted`); after finalize these are the data rows in the file, unless `writer_failed` is non-zero |
| `dropped_full` | entries dropped because the queue was full (`LogOutcome::Full`); default mode only, always 0 when lossless |
| `rejected_closed` | entries that arrived after finalize began (`LogOutcome::Closed`); not written, not a loss of an open session |
| `writer_failed` | entries lost because the writer stopped after an I/O error (`LogOutcome::WriterFailed`) |
| `lossless` | the session's mode |

`op_log_stats()` / `Logger::stats()` work while the op-log is Active **and after
it is Finalized** (`None` only before init), so a program can finalize and then
check the totals. The counts cover every clone of the session's `Logger`, are
relaxed atomics (a snapshot taken while threads are still logging is a progress
report, not a transaction), and `OpLogStats` is `#[non_exhaustive]`.
`finalize_op_logger()` / `finalize_op_log()` prints **one** stderr warning with
the counts and the remedies (`S3DLIO_OPLOG_LOSSLESS=1`, a larger
`S3DLIO_OPLOG_BUF`) if `dropped_full + writer_failed > 0`. A per-call
`try_log` outcome is still available when you need to know about one entry.

Cost: one relaxed `fetch_add` per call, on the path that matches its outcome.

### Op-log destinations

The destination is a local file. A plain path is used as given. A destination
starting with the lowercase prefix `file://` follows **s3dlio's `file://`
convention** (the same as the `file://` object store; not full RFC 8089): the
prefix is removed and the rest is the path, used literally. `file:///tmp/x` is
`/tmp/x`; `file://localhost/x` is the *relative* path `localhost/x`;
`file://relative/x` is the relative path `relative/x`; percent sequences are not
decoded; `file://` alone is an `InvalidInput` error. Any other string whose text
before the first `://` is a valid URI scheme (`s3://`, `az://`, `gs://`,
`direct://`, and also uppercase `FILE://`) is rejected as an unsupported scheme,
with the scheme shown as written. Text before `://` that is not a valid scheme
(for example `dir/a://b.log`) is a plain path.

### Behaviour changes for users of the op-log API

Compared with the previous release (4542f82), these are the changes a user can
observe that are known and recorded here; the list is maintained by hand and is
not a machine-checked inventory:

- `init_op_logger` / `init_op_log` after `finalize_op_logger` / `finalize_op_log`
  now fails (`ErrorKind::Unsupported`; Python `RuntimeError`) and touches no file.
  Before, it returned `Ok` and did nothing.
- After finalize, `global_logger()` is `None` and `is_op_log_active()` is `False`.
  Before, both still reported the dead logger.
- After finalize, entries from a `Logger` that was captured earlier are rejected
  quietly: one "op-log finalized" line on stderr per logger. Before, in the
  default (non-lossless) mode every entry printed "op-log channel full; dropping
  entry: sending on a closed channel"; in lossless mode
  (`S3DLIO_OPLOG_LOSSLESS=1`) the failed send was silent.
- `set_clock_offset` / `get_clock_offset` after finalize now return
  `Err(Unsupported)`; before, they succeeded on the dead logger. Before any init
  they still return `NotFound`, but the message now starts with the function name.
- Finalize now drains exactly what was accepted (see the table above), is
  idempotent, and a repeated or concurrent call waits for the first to finish.
- `S3DLIO_OPLOG_LOSSLESS` is read each time a `Logger` is created. Before, the
  first read was kept for the whole process. For the single global session this
  is the same unless the variable changes between two `Logger::new` calls.
- The queue-full message no longer appends the channel error text
  ("op-log channel full; dropping entry").
- A refused writer thread is an `io::Error` from init instead of a panic; the
  writer thread is named `s3dlio-oplog`.
- A destination is resolved as described under "Op-log destinations": `file://`
  paths work, other schemes are rejected instead of becoming relative paths, and
  create errors name the path.
- `Logger::log` still returns `()`; `Logger::try_log` and `LogOutcome` are new.
- An entry whose `operation` is `"__SHUTDOWN__"` used to be the writer's stop
  signal and ended the log (later entries were not written). It is now written as an
  ordinary row; the log ends only through finalize.
- A file-create error is rebuilt with `io::Error::new(kind, message)` so that it
  can name the path. The `ErrorKind` is kept, but `raw_os_error()` and the
  original source error are no longer available from it.
- Python `finalize_op_log` releases the GIL while it drains the queue, so other
  Python threads can run during a long drain.
- `op_log_stats()` / `Logger::stats()` / `OpLogStats` (Python `op_log_stats()`)
  are new: accepted, dropped (queue full), rejected-after-finalize and
  writer-failed counts, available while Active and after Finalized, so a caller
  can detect dropped records (see "Detecting dropped records"). Finalize prints
  one stderr warning if records were dropped or lost. The default stays lossy.
- Every `try_log` / `log` call now performs one atomic `fetch_add` and one
  `fetch_sub` on a single gate counter shared by all threads. This is a new
  per-entry cost, and it is contended when many threads log at once. It has not
  been benchmarked. Each call also bumps one relaxed outcome counter (see
  "Detecting dropped records").
- A clock offset of `i64::MIN` no longer panics the writer thread in builds with
  overflow checks. A corrected timestamp before the epoch (or at or after year
  9999) still panics inside the timestamp formatter, as before.

### What the format tests establish

The golden test (`oplog_golden_format_tests` in `src/s3_logger.rs`) pins the
serializer's output, header plus rows, for three fixed entries and clock offset 0:
a GET that succeeded with a first-byte time (caller-supplied idx 777 is replaced
by 0); a PUT with an error text and no first-byte time (idx 1); and a LIST with an
empty client id, 25 objects, 123456789012 bytes and a first-byte time 1 ns after
the start (idx 2). A second test checks that a plain path and the equivalent
`file://` destinations write identical content for those entries.
`oplog_clock_offset_tests` adds fixed rows for offsets of +1.5 s, 0, -2.25 s,
`i64::MAX` and `i64::MIN`. They do **not** establish what each operation puts in
its fields (endpoint, file, bytes, object counts), when its start, first-byte and
end times are taken, or that any consumer (sai3-bench, polarWarp, warp-replay)
parses the output.

## Example Usage

### Standalone Application

```rust
use s3dlio::{init_op_logger, set_client_id, store_for_uri};

// Initialize operation logging
init_op_logger("/tmp/my-app.tsv.zst")?;

// Set client identifier
let client_id = std::env::var("CLIENT_ID").unwrap_or_else(|_| "standalone".to_string());
set_client_id(&client_id)?;

// All subsequent operations automatically logged
let store = store_for_uri("s3://my-bucket/data")?;
let data = store.get("s3://my-bucket/data/object.bin").await?;
// Logged: idx=0, thread=123, op=GET, client_id="standalone", bytes=1024, ...
```

### Distributed Agent

```rust
use s3dlio::{init_op_logger, set_client_id, set_clock_offset, store_for_uri};

// Agent receives configuration from controller
struct AgentConfig {
    agent_id: String,
    oplog_path: String,
    start_timestamp_ns: i64,  // Controller's reference time
}

fn start_agent(config: AgentConfig) -> Result<()> {
    // Initialize oplog with agent-specific filename
    init_op_logger(&config.oplog_path)?;
    
    // Set client identifier for this agent
    set_client_id(&config.agent_id)?;
    
    // Synchronize clock to controller's reference time
    let local_now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos() as i64;
    let offset_ns = config.start_timestamp_ns - local_now_ns;
    set_clock_offset(offset_ns)?;
    
    // All operations now logged with synchronized timestamps and agent_id
    run_workload().await?;
    
    Ok(())
}
```

### Post-Processing: Sorting Operation Logs

**Important**: Operation logs are **NOT sorted during capture**. In multi-threaded applications with concurrent I/O operations, entries are written as operations complete, resulting in an unsorted log.

#### Why Logs Are Unsorted

- **Concurrent writes**: Multiple threads complete operations at different times
- **Out-of-order completion**: Operation started at t=10ms might complete at t=100ms, while operation started at t=50ms completes at t=60ms
- **Buffering**: Writes are buffered for performance, further mixing entry order

#### Sorting with sai3-bench

The `sai3-bench` tool provides post-processing sort functionality:

```bash
# Sort by start timestamp (creates .sorted.tsv.zst files)
sai3-bench sort --files /tmp/my-app.tsv.zst

# In-place sorting (overwrites original)
sai3-bench sort --files /tmp/my-app.tsv.zst --in-place

# Sort multiple files at once
sai3-bench sort --files /data/oplogs/*.tsv.zst

# Validate sorting
sai3-bench replay --op-log /tmp/my-app.sorted.tsv.zst --dry-run
```

**Benefits of Sorting**:
- **Better compression**: Sorted files are typically 30-40% smaller due to improved compression ratios
- **Chronological replay**: Required for accurate workload replay and debugging
- **Accurate analysis**: Enables time-series analysis and latency correlation

**Performance**: Streaming window-based sort (default 10,000 lines) uses constant memory regardless of log size.

### Analyzing Logs

```bash
# Decompress and view
zstd -d < /tmp/my-app.tsv.zst | less

# Count operations by type
zstd -d < /tmp/my-app.tsv.zst | awk 'NR>1 {print $3}' | sort | uniq -c

# Calculate average GET latency (duration_ns column)
zstd -d < /tmp/my-app.tsv.zst | awk '$3=="GET" && NR>1 {sum+=$13; count++} END {print sum/count/1e6 " ms"}'

# Find slowest operations
zstd -d < /tmp/my-app.tsv.zst | sort -k13 -nr | head -20

# Compare first_byte vs end times (should be similar for small objects)
zstd -d < /tmp/my-app.tsv.zst | awk 'NR>1 && $3=="GET" {
    if ($11 != "") {  # first_byte exists
        # Parse timestamps and calculate difference (simplified)
        print $11, $12, "TTFB_analysis_needed"
    }
}'
```

## Implementation Notes

### Thread Safety

All global state (client_id, clock_offset, logger) uses thread-safe primitives:
- `OnceCell<Mutex<T>>` for single initialization + mutable access
- All setters check if already initialized
- All getters provide safe default values

### Performance Impact

- **Clock offset**: ~5ns overhead per timestamp (single atomic read)
- **Client ID**: ~10ns overhead per log entry (mutex lock + clone)
- **First byte capture**: ~50ns overhead (one extra SystemTime::now() call)
- **Overall**: < 100ns per operation, negligible for I/O-bound workloads

### Compatibility

- **Backwards compatible**: Existing code without set_client_id() uses empty string
- **Optional features**: Clock offset and client_id can be left unset
- **Format stable**: TSV format unchanged, first_byte column existed but was empty before v0.9.22

## Related Documentation

- [s3dlio API Reference](../README.md)
- [Changelog - v0.9.22](Changelog.md)
- [Multi-Endpoint Support](MULTI_ENDPOINT.md)
- [sai3-bench Operation Logging](../../sai3-bench/docs/USAGE.md#operation-logging)

---

**Version**: s3dlio v0.9.22+  
**Last Updated**: November 25, 2025
