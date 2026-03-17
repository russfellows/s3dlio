# GCS Backend — s3dlio Reference Guide

**Status**: All issues resolved. Production-ready as of v0.9.75.  
**Date**: March 2026  
**Fork**: [`russfellows/google-cloud-rust`](https://github.com/russfellows/google-cloud-rust),
tag `release-20260212-rf-gcsrapid-20260317.1`

---

## Overview

s3dlio uses the **gRPC Storage v2 API** for all Google Cloud Storage operations — reads,
writes, and deletes. The JSON/REST path is not used. This provides:

- Binary protobuf encoding (no Base64/JSON overhead)
- HTTP/2 multiplexed persistent connections
- Bidirectional streaming for chunked uploads
- Native support for RAPID / Hyperdisk ML zonal buckets

The GCS backend requires the `backend-gcs` feature flag (included in `full-backends`):

```bash
cargo build --release --features backend-gcs
```

Authentication is via `GOOGLE_APPLICATION_CREDENTIALS` (service account JSON key) or
Application Default Credentials. `GOOGLE_CLOUD_PROJECT` is required for `list-buckets`.

---

## RAPID / Hyperdisk ML Buckets

GCS RAPID storage (also marketed as **Hyperdisk ML**) is a zonal storage tier for
ultra-low-latency AI/ML workloads. RAPID buckets have two constraints not shared with
ordinary GCS buckets:

1. **Writes must use `appendable=true`** with a two-phase finalize protocol.
2. **Reads require gRPC** — the JSON/REST API returns HTTP 400.

s3dlio handles both automatically. Auto-detection runs `GetStorageLayout` on first access
to each bucket and caches the result for the process lifetime. See [Runtime Configuration](#runtime-configuration) for override options.

---

## Architecture

### Read Path — `BidiReadObject`

All GCS reads use `BidiReadObject` gRPC streaming:

```
Client                              GCS gRPC Server
  │── BidiReadObjectRequest ────────────►│
  │   bucket, object, read_range         │
  │◄── BidiReadObjectResponse ──────────│
  │   checksummed_data (streams…)        │
```

- **Full object**: `ReadRange::all()` — reads the complete object in streaming chunks
- **Byte range**: `ReadRange::segment()` or `ReadRange::offset()` — partial reads

For RAPID buckets, `BidiReadObject` is required for correct size metadata (the Storage
Control API returns stale `size=0` for new objects). s3dlio's `stat()` call uses
`BidiReadObject` when RAPID mode is active or auto-detected.

**Read buffer**: `BytesMut` with exact pre-allocation from the descriptor's `object.size`,
followed by `freeze()` — a zero-copy `BytesMut → Bytes` promotion. No intermediate copies.

### Write Path — `BidiWriteObject`

All GCS writes use `BidiWriteObject` gRPC streaming with a concurrent producer/consumer:

```
Client                              GCS gRPC Server
  │── BidiWriteObjectRequest #1 ────────►│
  │   WriteObjectSpec (+ appendable=true on RAPID)  │
  │   ChecksummedData { chunk_0, crc32c }│
  │── BidiWriteObjectRequest #2..N ─────►│
  │   ChecksummedData { chunk_n, crc32c }│
  │   write_offset: <n>                  │
  │── flush probe (RAPID only) ─────────►│
  │◄── PersistedSize ack ───────────────│
  │── BidiWriteObjectRequest final ─────►│
  │   finish_write: true                 │
  │◄── WriteStatus::Resource(Object) ───│
```

#### RAPID Two-Phase Finalize (v0.9.75)

Standard GCS and RAPID differ in their write commit protocol:

| Mode | Protocol |
|------|----------|
| Standard | Send data chunks, set `finish_write=true` on last chunk |
| RAPID | Send data chunks with `flush=false`, send flush probe (`flush=true, state_lookup=true`), wait for `PersistedSize >= total_len`, then send `finish_write=true` |

The flush-probe step prevents the server from committing partially-flushed data.
The initial `Resource(size=0)` response from RAPID servers is a protocol handshake,
not an error — s3dlio skips it and waits for the final `Resource` after `finish_write`.

**Producer/consumer**: A bounded channel (capacity 8) decouples CRC32C computation from
network I/O. A background `tokio::spawn` task computes checksums and pushes chunks;
the gRPC stream sends them as they arrive. The producer task is awaited (not leaked)
after the stream completes.

### Delete Path — Concurrent Batch

`delete_objects()` dispatches up to 64 concurrent gRPC delete requests using
`FuturesUnordered` + `Semaphore`. Errors are collected and reported after all requests
complete, ensuring partial failures do not abort the batch.

### List Path — Paginated API

`list_objects()` uses the paginated GCS list API. A prefix normalisation fix ensures
exact object names are not mistakenly treated as directory prefixes.

---

## HTTP/2 Flow Control

**This was the root cause of the original gRPC hang observed at high concurrency.**

The tonic/h2 default HTTP/2 stream window is 65,535 bytes. At a same-region RTT of
~1.5 ms this limits each gRPC stream to ~43 MB/s. At 24 parallel streams that caps
throughput at ~1 GB/s — far below a 24 Gbit/s NIC.

s3dlio sets a 128 MiB initial stream and connection window by default, making the
window effectively unlimited at same-region latencies (bandwidth-delay product ≈ 0.25 MB
<< 128 MiB). This is configured in the google-cloud-rust fork at
`src/gax-internal/src/grpc.rs → make_endpoint()`:

```rust
endpoint
    .initial_connection_window_size(window_bytes)
    .initial_stream_window_size(window_bytes)
    .tcp_nodelay(true)
```

Override with `S3DLIO_GRPC_INITIAL_WINDOW_MIB` (set to `0` for the 65 KB protocol default,
which is not recommended for high-throughput workloads).

### Subchannels and Concurrency

The gRPC transport multiplexes all concurrent streams across a pool of TCP connections
(subchannels). Each subchannel has its own HTTP/2 connection-level flow-control window.
Too few subchannels means multiple streams compete for one connection's window.

Priority order for subchannel count (`GcsClient::new()`):

1. `S3DLIO_GCS_GRPC_CHANNELS` env var — explicit override, always wins
2. `set_gcs_channel_count(n)` programmatic API — set before first GCS I/O
3. Auto-fallback: `max(64, cpu_count)`

**Recommended**: match subchannels to your actual job/concurrency count.

---

## Performance Results

Before fixes (v0.9.50, default 65 KB HTTP/2 window, 16 MiB write chunks):

| Operation | Standard bucket | RAPID bucket |
|-----------|----------------|--------------|
| Write     | ~1.4 GB/s      | ~1.8 GB/s    |
| Read      | ~1.3 GB/s      | ~1.0 GB/s    |

After all fixes (v0.9.65+, 128 MiB window, 2 MiB chunks, zero-copy):

| Operation | Result |
|-----------|--------|
| Upload (1,000 × 32 MiB, RAPID, 32 jobs) | **3.83 GB/s** |

The pre-fix RAPID read (1.0 GB/s) was slower than standard (1.3 GB/s) because RAPID
uses gRPC exclusively. Standard reads used the HTTP/reqwest path, which benefits from
OS-level TCP window management. Once the gRPC window was increased, both paths exceed
the old ceiling.

---

## google-cloud-rust Fork

All GCS support depends on our fork of `google-cloud-rust`. The upstream crate does not
include gRPC write support or RAPID-specific protocol handling. Our fork is at:

**Repository**: https://github.com/russfellows/google-cloud-rust  
**Active tag**: `release-20260212-rf-gcsrapid-20260317.1`  
**s3dlio reference** (`Cargo.toml`):
```toml
google-cloud-storage = { git = "https://github.com/russfellows/google-cloud-rust.git",
                         tag = "release-20260212-rf-gcsrapid-20260317.1", optional = true }
google-cloud-gax = { git = "https://github.com/russfellows/google-cloud-rust.git",
                     tag = "release-20260212-rf-gcsrapid-20260317.1", optional = true }
google-cloud-gax-internal = { git = "https://github.com/russfellows/google-cloud-rust.git",
                              tag = "release-20260212-rf-gcsrapid-20260317.1",
                              features = ["_internal-grpc-client"], optional = true }
```

### Commits on Our Fork (our changes only, excluding upstream merges)

| Commit | Description |
|--------|-------------|
| `10ff86a106` | **Initial RAPID support**: BidiWriteObject + HTTP/2 window tuning. Added `gcs_constants.rs` (server message ceiling, chunk size constants, env-var names), `grpc.rs` window and `tcp_nodelay` tuning, `write_object.rs` `set_appendable()` and `send_grpc()` builder methods, full `BidiWriteObject` implementation in `transport.rs` with concurrent producer, CRC32C checksums, chunk streaming. |
| `9d7b23e2d3` | **Constants + reconnect telemetry**: Centralised bidi runtime constants into `src/storage/src/constants.rs` (attempt timeout, channel capacity, recv-many batch size, progress-log interval). Added INFO-level reconnect/backoff telemetry throughout the bidi stack so production operators can observe behavior without verbose tracing. |
| `626dc9f302` | **RAPID PUT truncation fix**: Three issues with the two-phase finalize protocol for appendable/RAPID writes: (1) flush probe waits for `PersistedSize >= total_len` before `finish_write`; (2) skip the initial `Resource(size=0)` spec-ack handshake; (3) producer holds mpsc tx alive until final Resource is confirmed, preventing premature stream teardown. Validated on sig65-rapid1: 8/8 objects at exactly 16 MiB. |
| `0cfa2ab11e` | **BidiWriteObject buffer optimisations**: Lower-level allocation improvements for the write path. |

### Files Modified in the Fork

**`src/gax-internal/src/gcs_constants.rs`** — Protocol constants (new file):
- `GCS_SERVER_MAX_MESSAGE_SIZE = 4 MiB` — raw GCS server message ceiling
- `MAX_GRPC_WRITE_CHUNK_SIZE = 4 MiB − 64 KiB` — max safe data payload (guards against ~89-byte protobuf framing overhead)
- `DEFAULT_GRPC_WRITE_CHUNK_SIZE = 2 MiB` — conservative default
- `DEFAULT_WINDOW_MIB = 128` — HTTP/2 stream and connection window
- All env-var name string constants

**`src/gax-internal/src/grpc.rs`** — `make_endpoint()` patched to read `S3DLIO_GRPC_INITIAL_WINDOW_MIB` and apply `.initial_connection_window_size()`, `.initial_stream_window_size()`, `.tcp_nodelay(true)`.

**`src/storage/src/storage/stub.rs`** — `write_object_grpc()` default stub method added to the `Storage` trait.

**`src/storage/src/storage/write_object.rs`** — Two builder methods: `set_appendable(bool)` and `send_grpc()`.

**`src/storage/src/storage/transport.rs`** — Full `write_object_grpc()` BidiWriteObject implementation, plus RAPID two-phase finalize logic (flush probe, skip initial Resource, producer keep-alive).

**`src/storage/src/constants.rs`** — Shared bidi runtime constants (new file).

**`src/storage/src/storage/bidi/{connector,transport,worker}.rs`** — Reconnect telemetry wired to shared constants.

**`doc/RAPID_PUT_TRUNCATION_FIX_HISTORY.md`** — In-repo history of the RAPID PUT truncation investigation (in the fork, not in s3dlio).

---

## Runtime Configuration

### Programmatic API (set before first GCS I/O)

#### `s3dlio::set_gcs_channel_count(n: usize)`

Pre-configure the number of gRPC subchannels (TCP connections).

```rust
s3dlio::set_gcs_channel_count(my_concurrency);
```

#### `s3dlio::set_gcs_rapid_mode(force: Option<bool>)`

Override RAPID auto-detection:

```rust
s3dlio::set_gcs_rapid_mode(Some(true));   // force RAPID on all buckets
s3dlio::set_gcs_rapid_mode(Some(false));  // force RAPID off all buckets
s3dlio::set_gcs_rapid_mode(None);         // auto-detect per bucket (default)
```

#### `s3dlio::get_gcs_channel_count() -> usize`

Returns the programmatic subchannel setting (`0` if never called; env var takes precedence inside `GcsClient::new()`).

#### `s3dlio::get_gcs_rapid_mode() -> Option<bool>`

Returns the current effective RAPID mode (resolves env var + programmatic setting).
- `Some(true)` — forced on
- `Some(false)` — forced off
- `None` — auto-detect per bucket (default)

#### `s3dlio::query_gcs_rapid_bucket(bucket_or_uri: &str) -> bool` *(async)*

Returns `true` if the bucket is a RAPID/zonal bucket. Calls `GetStorageLayout` on
first access; result is cached per bucket for the process lifetime. Safe to call
concurrently — duplicate calls for the same bucket are deduplicated (one RPC only).

```rust
let is_rapid = s3dlio::query_gcs_rapid_bucket("gs://my-bucket/").await;
```

### Environment Variables

All env vars take precedence over the programmatic API.

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_GCS_RAPID` | `auto` | RAPID mode: `true`/`1`/`yes` force on; `false`/`0`/`no` force off; `auto` or unset = per-bucket detection |
| `S3DLIO_GCS_GRPC_CHANNELS` | auto | Force exact subchannel count; overrides `set_gcs_channel_count()` and auto fallback |
| `S3DLIO_GRPC_INITIAL_WINDOW_MIB` | `128` | HTTP/2 initial stream + connection window in MiB. Set to `0` for the 65 KB protocol default (not recommended). |
| `S3DLIO_GRPC_WRITE_CHUNK_SIZE` | `2097152` | gRPC write chunk size in bytes. Silently clamped to `MAX_GRPC_WRITE_CHUNK_SIZE` (4 MiB − 64 KiB). |
| `GOOGLE_APPLICATION_CREDENTIALS` | *(none)* | Path to service account JSON key |
| `GOOGLE_CLOUD_PROJECT` | *(none)* | GCP project ID (required for `list-buckets`) |

### Python API

```python
import s3dlio

# Setters — must be called before first gs:// operation
s3dlio.gcs_set_channel_count(64)
s3dlio.gcs_set_rapid_mode(True)   # True / False / None (auto)

# Getters
channels = s3dlio.gcs_get_channel_count()   # int; 0 = not set
rapid    = s3dlio.gcs_get_rapid_mode()      # True / False / None

# Bucket query (blocking; cached for process lifetime)
is_rapid = s3dlio.gcs_query_rapid_bucket("gs://my-bucket/")
```

### Debug Log Line

On first client initialisation the library emits at INFO level:
```
GCS gRPC config: subchannels=64 (source=jobs/concurrency, cpus=8), initial_window=128 MiB
```

`source` shows which priority tier was used:
- `env-var` — `S3DLIO_GCS_GRPC_CHANNELS` was set
- `jobs/concurrency` — `set_gcs_channel_count()` was called
- `auto-fallback` — neither set; used `max(64, cpu_count)`

---

## Constants Reference

Defined in the fork at `src/gax-internal/src/gcs_constants.rs` and re-exported from
`src/gcs_constants.rs` in s3dlio.

| Constant | Value | Purpose |
|----------|-------|---------|
| `GCS_SERVER_MAX_MESSAGE_SIZE` | 4,194,304 bytes | Raw GCS server ceiling — complete serialised `BidiWriteObjectRequest` must be below this |
| `MAX_GRPC_WRITE_CHUNK_SIZE` | 4,128,768 bytes | Max safe data payload: server ceiling − 64 KiB guard band (observed framing: ~89 bytes; guard band >> overhead) |
| `DEFAULT_GRPC_WRITE_CHUNK_SIZE` | 2,097,152 bytes | Default chunk size (2 MiB; conservative, well below ceiling) |
| `GCS_MIN_CHANNELS` | 64 | Floor for subchannel auto-tune |
| `GCS_MAX_CONCURRENT_DELETES` | 64 | Semaphore limit for batch delete concurrency |

All chunk sizes are 64 KiB-aligned.

---

## Typical Startup Sequence

```rust
use s3dlio;

// 1. Set subchannel count to match your job concurrency (before any GCS I/O).
s3dlio::set_gcs_channel_count(my_job_count);

// 2. Optionally force RAPID mode (auto-detect is correct for most workloads).
s3dlio::set_gcs_rapid_mode(Some(true));

// 3. Log effective settings for diagnostics.
let chans = s3dlio::get_gcs_channel_count();
let rapid = s3dlio::get_gcs_rapid_mode();
println!("GCS: {chans} subchannels, RAPID = {rapid:?}");

// 4. Optionally query a bucket before I/O.
let is_rapid = s3dlio::query_gcs_rapid_bucket("gs://my-bucket/").await;

// 5. Proceed — GCS client initialises on first use.
let store = s3dlio::store_for_uri("gs://my-bucket/prefix/")?;
```

---

## Common Errors

### "This bucket requires appendable objects"

Target is a RAPID/zonal bucket but RAPID mode is off or undetected.

```bash
export S3DLIO_GCS_RAPID=true
# or use auto-detect (default): unset S3DLIO_GCS_RAPID
```

### "Zonal objects only support gRPC API" (HTTP 400)

s3dlio version predates v0.9.60. Upgrade to v0.9.60+ (built with default features or `backend-gcs`).

### OUT_OF_RANGE on RAPID stat

Pre-v0.9.75 behaviour: `stat()` used Storage Control, which returns stale `size=0` for
recently-written RAPID objects. Fixed in v0.9.75 by routing RAPID stat through `BidiReadObject`.

### "RESOURCE_EXHAUSTED: Received message larger than max"

Pre-v0.9.65 behaviour: write chunk size defaulted to 16 MiB, exceeding the 4 MiB server
ceiling. Fixed in v0.9.65 by reducing default to 2 MiB and clamping env-var overrides.

### Authentication: "insufficient authentication scopes"

Service account lacks required IAM roles. RAPID buckets need:
- `roles/storage.objectAdmin` (or `objectCreator` + `objectViewer`)
- `roles/storage.legacyBucketReader` (for list operations)

---

## Quick Start

```bash
# Standard GCS bucket
./target/release/s3-cli put --size 65536 --count 10 gs://my-bucket/bench/
./target/release/s3-cli list gs://my-bucket/bench/
./target/release/s3-cli get gs://my-bucket/bench/object_0_of_10.dat

# RAPID / Hyperdisk ML bucket (auto-detect handles appendable=true)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
./target/release/s3-cli put --size 33554432 --count 100 --concurrency 32 gs://my-rapid-bucket/bench/
./target/release/s3-cli list gs://my-rapid-bucket/bench/

# Force RAPID mode (skip GetStorageLayout detection RPC)
export S3DLIO_GCS_RAPID=true
./target/release/s3-cli put --size 33554432 --count 100 gs://my-rapid-bucket/bench/
```

```dotenv
# .env file for persistent config
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
GOOGLE_CLOUD_PROJECT=my-gcp-project-id
S3DLIO_GCS_RAPID=auto
```

---

## Implementation History (s3dlio versions)

| Version | What changed |
|---------|-------------|
| v0.9.60 | gRPC universal for all GCS operations; `S3DLIO_GCS_RAPID` env var; multi-protocol `list-buckets` |
| v0.9.65 | HTTP/2 window tuned to 128 MiB (fix for ~1 GB/s ceiling); write chunk size fixed (4 MiB → 2 MiB default, clamped at `MAX_GRPC_WRITE_CHUNK_SIZE`); concurrent write producer (CPU+network overlap); zero-copy read (`BytesMut` pre-alloc + `freeze()`); zero-copy write path (`Bytes` throughout); RAPID auto-detection (`GetStorageLayout`, per-bucket cache); constants centralised |
| v0.9.70 | GCS wrapper caching (process-wide singleton confirmed); RAPID mode log reduced to one init-time message; subchannel count wired from CLI concurrency options |
| v0.9.75 | RAPID `stat` via `BidiReadObject` (fixes stale `size=0` from Storage Control); RAPID GET `OUT_OF_RANGE` bypass; RAPID PUT two-phase finalize (flush probe + skip initial Resource + producer keep-alive); fork pinned at `release-20260212-rf-gcsrapid-20260317.1` |
