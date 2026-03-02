# GCS gRPC Fixes & Performance Tuning — s3dlio v0.9.60+

**Status**: All fixes implemented and verified. Build clean. 10/10 zero-copy unit tests passing.  
**Date**: 2026-03-01  
**Applies to**: s3dlio ≥ v0.9.60 / CLI ≥ v0.1.x

---

## Overview

Six root-cause issues were identified and fixed, transforming GCS throughput from well-below-ceiling
to near line-rate:

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | HTTP/2 flow-control window defaulted to 65 KB | **CRITICAL** | ~1 GB/s ceiling on gRPC reads |
| 2 | Write chunk exceeded server 4 MiB message limit | **CRITICAL** | RESOURCE_EXHAUSTED errors on PUT |
| 3 | Write pipeline fully serialised (all chunks queued before stream opens) | HIGH | CPU+network never overlap |
| 4 | Read buffer no pre-allocation, `Vec<u8>` copy on every extend | MEDIUM | O(log N) reallocs per object |
| 5 | `put_object` took `&[u8]` — forced full heap copy before every write | **CRITICAL** | 100% copy overhead on write path |
| 6 | No RAPID auto-detection — users had to configure manually | MEDIUM | Wrong path selection, errors on mismatch |

---

## Background

Testing from a node with a 24 Gbit/s network connection (theoretical ceiling ~3 GB/s) against
GCS buckets in the same region showed:

| Operation | Standard bucket | RAPID bucket | Expected |
|-----------|----------------|--------------|----------|
| Write     | ~1.4 GB/s      | ~1.8 GB/s    | >2.5 GB/s |
| Read      | ~1.3 GB/s      | ~1.0 GB/s    | >2.5 GB/s |

Two anomalies:
1. **Both paths are well below the 24 Gbit network ceiling.**
2. **RAPID reads are slower than standard reads** — backwards, since RAPID (Hyperdisk ML / zonal
   GCS) has lower storage latency and should be faster.

After all fixes, measured upload throughput (1,000 × 32 MiB objects, RAPID bucket, 32 jobs):
**3.83 GB/s** — a 2.7× improvement over the pre-fix baseline.

---

## Root Cause Analysis

### Issue 1 — CRITICAL: gRPC HTTP/2 Flow-Control Window (65 KB default)

**File:** `vendor/google-cloud-gax-internal/src/grpc.rs` → `make_endpoint()`

The tonic/h2 default HTTP/2 stream window size is **65,535 bytes** (RFC 7540 §6.9.2). Nothing in
the original code overrides this. With a same-region RTT of ~1.5 ms:

$$\text{Max throughput per stream} = \frac{65{,}535}{0.0015} \approx 43 \text{ MB/s}$$

At 24 parallel threads: $24 \times 43 \approx \mathbf{1.0 \text{ GB/s}}$

This exactly matches the measured 1.0 GB/s RAPID read ceiling. RAPID reads use the gRPC path
exclusively; standard reads used the HTTP/reqwest path (OS-level TCP window management, far
larger effective window), which is why standard reads (1.3 GB/s) beat RAPID reads (1.0 GB/s)
despite RAPID storage being lower-latency.

For writes, gRPC is used for both standard and RAPID. The write advantage for RAPID (1.8 vs
1.4 GB/s) comes from lower storage-commit latency, but both were flow-control-limited well below
the network ceiling.

### Issue 2 — CRITICAL: Write Chunk Size Exceeded Server Message Limit

**File:** `vendor/google-cloud-storage/src/storage/transport.rs`

The original `DEFAULT_GRPC_WRITE_CHUNK_SIZE = 16 MiB`. Each `BidiWriteObjectRequest` gRPC
message carries exactly one chunk. The GCS server's inbound message limit is **4,194,304 bytes
(4 MiB exactly)**. 16 MiB chunks exceeded this by 4×, producing:

```
RESOURCE_EXHAUSTED: SERVER: Received message larger than max (16777303 vs. 4194304)
```

Critically, the server limit applies to the **entire serialised protobuf message** — including
field tags, varint-encoded lengths, `write_offset`, `ChecksummedData` wrapper, per-chunk CRC32C,
and (on the first message) `WriteObjectSpec`. Observed framing overhead is ~88–90 bytes for middle
chunks. Setting the data payload equal to the server message ceiling will also exceed it.

### Issue 3 — Write Pipeline Fully Serialised

**File:** `vendor/google-cloud-storage/src/storage/transport.rs` → `write_object_grpc()`

The original pipeline used an unbounded channel large enough to hold every chunk:

```rust
let channel_size = std::cmp::max(num_chunks + 1, 2);  // holds ALL chunks
let (tx, rx) = mpsc::channel(channel_size);
// fills entire channel: CRC32C + proto for every chunk
while offset < total_len { tx.send(request).await; ... }
drop(tx);
// ONLY NOW opens the gRPC stream
bidi_stream_with_status(..., ReceiverStream::new(rx), ...).await
```

All CRC32C computation and protobuf serialisation completed before the network connection was
even established. CPU and network I/O were fully serialised.

### Issue 4 — Read Buffer: No Pre-allocation, `Vec<u8>` Copy On Every Extend

**File:** `src/google_gcs_client.rs` → `get_object_via_grpc()`

```rust
let (_descriptor, mut reader) = ...open_object()...send_and_read(range).await?;
//   ^^^^ discarded — contains object.size!
let mut data = Vec::new();          // starts at zero capacity
while let Some(chunk) = reader.next()... {
    data.extend_from_slice(&chunk); // triggers ~log₂(N) realloc+memcpy cycles
}
```

The `ObjectDescriptor` returned as the first element contains `object().size` — the exact byte
count. Discarding it forced approximately `log₂(object_size / initial_capacity)` reallocations
as the object streamed in.

### Issue 5 — CRITICAL: Zero-Copy Regression in `put_object`

**File:** `src/google_gcs_client.rs` → `put_object()`

```rust
// Before — full heap copy on every write:
pub async fn put_object(&self, bucket: &str, object: &str, data: &[u8]) -> Result<()> {
    let bytes = Bytes::copy_from_slice(data);  // allocates + copies entire object
```

The `&[u8]` signature forced a full copy before every write, destroying all previous zero-copy
work at the API boundary.

### Issue 6 — No RAPID Auto-Detection

A hard-coded `bool` forced users to configure RAPID manually. The wrong setting caused writes to
RAPID buckets to fail silently, or standard buckets to use the slower gRPC read path
unnecessarily.

---

## All Fixes Implemented

### Fix 1 — HTTP/2 Flow-Control Window ✅

**File:** `vendor/google-cloud-gax-internal/src/grpc.rs` — `make_endpoint()`

`make_endpoint()` now reads `S3DLIO_GRPC_INITIAL_WINDOW_MIB` (default **128**) and configures
tonic with explicit window sizes:

```rust
let window_bytes = (window_mib * 1024 * 1024).min(u32::MAX as u64) as u32;
endpoint
    .initial_connection_window_size(window_bytes)
    .initial_stream_window_size(window_bytes)
    .tcp_nodelay(true)
```

128 MiB makes each gRPC stream effectively window-unlimited at same-region latencies
(BDP ≈ 0.25 MB << 128 MiB). Set `S3DLIO_GRPC_INITIAL_WINDOW_MIB=0` to fall back to the
protocol default (65 KB).

### Fix 2 — Write Chunk Size (with protobuf overhead headroom) ✅

The server's 4 MiB limit is for the **whole serialised protobuf message**, not just the data
payload. The fix introduces three distinct constants (see §Constants Architecture below):

```rust
// vendor/google-cloud-gax-internal/src/gcs_constants.rs

/// Raw server message ceiling — the full serialised BidiWriteObjectRequest must be < this.
pub const GCS_SERVER_MAX_MESSAGE_SIZE: usize = 4 * 1024 * 1024;   // 4,194,304 bytes

/// Maximum safe DATA payload: server ceiling minus a 64 KiB guard band.
/// Observed framing overhead is ~89 bytes; 64 KiB >> 89 bytes ensures we never hit the ceiling.
/// Value is 63 × 64 KiB = 4,128,768 bytes (64 KiB-aligned).
pub const MAX_GRPC_WRITE_CHUNK_SIZE: usize = GCS_SERVER_MAX_MESSAGE_SIZE - (64 * 1024);

/// Conservative default: 2 MiB = 32 × 64 KiB. Well below the ceiling; easy to raise via env var.
pub const DEFAULT_GRPC_WRITE_CHUNK_SIZE: usize = 2 * 1024 * 1024;
```

`grpc_write_chunk_size()` reads `S3DLIO_GRPC_WRITE_CHUNK_SIZE` and silently clamps to
`MAX_GRPC_WRITE_CHUNK_SIZE` — impossible to trigger `RESOURCE_EXHAUSTED` via misconfiguration.

### Fix 3 — Concurrent Write Producer ✅

Replaced unbounded pre-fill with a bounded channel (capacity 8) and a `tokio::spawn` producer
that runs CRC32C computation concurrently with gRPC network I/O:

```rust
const PRODUCER_CHANNEL_CAPACITY: usize = 8;
let (tx, rx) = mpsc::channel(PRODUCER_CHANNEL_CAPACITY);
let producer_task = tokio::spawn(async move { /* chunk loop */ });
bidi_stream_with_status(..., ReceiverStream::new(rx), ...).await?;
producer_task.abort();
let _ = producer_task.await;  // properly awaited — no task leak
```

`data.clone()` passed to the spawn closure is an O(1) Arc increment — zero-copy.

### Fix 4 — Read Buffer: `BytesMut` + `freeze()` ✅

Both gRPC and HTTP read paths now use `BytesMut` with pre-allocation:

**gRPC path** — exact pre-allocation from descriptor:
```rust
let (descriptor, mut reader) = ...open_object()...send_and_read(range).await?;
let size_hint = descriptor.object().size as usize;
let mut data = BytesMut::with_capacity(size_hint);  // single allocation
while let Some(chunk) = reader.next()... {
    data.put_slice(&chunk);
}
Ok(data.freeze())  // zero-copy: BytesMut → Bytes, same base pointer
```

**HTTP path** — 1 MiB initial capacity (no Content-Length available):
```rust
let mut data = BytesMut::with_capacity(1024 * 1024);
while let Some(chunk) = reader.next()... { data.put_slice(&chunk); }
Ok(data.freeze())
```

### Fix 5 — Zero-Copy Write Path ✅

`put_object` and `put_object_multipart` now take `Bytes` directly:

```rust
// New — zero-copy:
pub async fn put_object(&self, bucket: &str, object: &str, data: Bytes) -> Result<()> {
    // data is Arc-backed — passed straight to write_object_grpc, no copy
```

| Caller | Before | After |
|--------|--------|-------|
| `GcsObjectStore::put` | `data.as_ref()` → `&[u8]` | `data` passed directly |
| `GcsObjectStore::put_multipart` | `data.as_ref()` → `&[u8]` | `data` passed directly |
| `GcsBufferedWriter::finalise` | `&self.buffer` (`&[u8]`) | `Bytes::from(mem::take(&mut self.buffer))` |
| Integration tests | `&[u8]` literals | `Bytes::from_static(b"...")` / `Bytes::from(string)` |

`Bytes::from(Vec<u8>)` takes ownership without copying. `mem::take` leaves an empty `Vec`
in `self.buffer` (O(1)).

### Fix 6 — RAPID Auto-Detection ✅

`RapidMode` enum (`Auto` | `ForceOn` | `ForceOff`), per-bucket cache, and
`get_storage_layout()` detection on first access:

| `S3DLIO_GCS_RAPID` | Behaviour |
|--------------------|-----------|
| unset / `auto`     | Auto-detect via `get_storage_layout()` on first access (default) |
| `true` / `1` / `yes` | Force RAPID gRPC path on all buckets |
| `false` / `0` / `no` | Force standard HTTP path on all buckets |

Reads route to HTTP for standard buckets; gRPC `BidiReadObject` only for RAPID.

---

## gRPC Constants Architecture

All GCS/gRPC tuning values live in **one place** to prevent scattered magic numbers:

| File | Scope |
|------|-------|
| `vendor/google-cloud-gax-internal/src/gcs_constants.rs` | Protocol-level: server message ceiling, chunk sizes, HTTP/2 window, env-var name strings |
| `src/gcs_constants.rs` | Application-level: re-exports all vendor constants + `GCS_MIN_CHANNELS`, `GCS_MAX_CONCURRENT_DELETES`, `ENV_GCS_GRPC_CHANNELS`, `ENV_GCS_RAPID` |

**Write chunk size constants:**

| Constant | Expression | Bytes | Purpose |
|----------|-----------|-------|---------|
| `GCS_SERVER_MAX_MESSAGE_SIZE` | `4 * 1024 * 1024` | 4,194,304 | Raw server-enforced ceiling (whole message) |
| `MAX_GRPC_WRITE_CHUNK_SIZE` | `GCS_SERVER_MAX_MESSAGE_SIZE - (64 * 1024)` | 4,128,768 | Max safe data payload (63 × 64 KiB; guard band >> ~89-byte framing overhead) |
| `DEFAULT_GRPC_WRITE_CHUNK_SIZE` | `2 * 1024 * 1024` | 2,097,152 | Default (32 × 64 KiB; conservative, ~2 MiB headroom below ceiling) |

All values are 64 KiB-aligned. Env-var overrides are silently clamped to `MAX_GRPC_WRITE_CHUNK_SIZE`.

---

## Subchannel Auto-Tune

### Why subchannel count matters

The gRPC transport multiplexes all concurrent streams over a small number of TCP connections
(subchannels). Each TCP connection shares one HTTP/2 connection-level flow-control window across
every active stream. With the old default of `max(4, cpu_count)` subchannels on a c4-standard-8:

```
64 jobs  ÷  8 subchannels  =  8 streams per connection
256 MiB connection window  ÷  8 streams  =  32 MiB per stream
```

RAPID objects are ~30 MiB each. At 32 MiB per stream the receiver stalls waiting for
WINDOW_UPDATE frames, effectively serialising transfers.

### Priority ladder

`GcsClient::new()` uses the following priority order for subchannel count:

1. **`S3DLIO_GCS_GRPC_CHANNELS`** env var — manual override, always wins.
2. **`set_gcs_channel_count(n)`** API call — programmatic pre-init hook.
3. **`max(64, cpu_count)`** auto fallback — conservative floor on any machine.

The CLI and library both call `set_gcs_channel_count(concurrency)` at startup, so
`--jobs N` automatically translates to N subchannels with zero configuration.

### Debug log line

```
GCS gRPC config: subchannels=64 (source=jobs/concurrency, cpus=8), initial_window=128 MiB
```

`source` reflects which tier of the priority ladder was used.

---

## API Reference

### Rust library

```rust
// Call once, before any GCS I/O, to tune channels to your concurrency level.
// Must be called before the first GCS ObjectStore is constructed (process-wide singleton).
s3dlio::set_gcs_channel_count(my_concurrency);
```

### CLI (automatic)

The `--jobs N` / `-j N` flag on every command (get, put, upload, download) automatically
calls `set_gcs_channel_count(N)` before the first GCS operation. No extra flags needed.

---

## `gcs-official` Feature Flag Removed

Previously `google-cloud-storage` and `google-cloud-gax` were optional Cargo features gated
behind `gcs-official`. As of v0.9.60 they are **always-on** unconditional dependencies:

- `google-cloud-storage` (gRPC RAPID client) — compiled unconditionally.
- `google-cloud-gax` — compiled unconditionally.
- `gcs-community` — still an optional legacy feature (JSON API, no RAPID support).
  Enabling it alongside the default is an error (both define the same `GcsClient` symbol).

Projects depending on s3dlio no longer need `features = ["gcs-official"]` in their `Cargo.toml`.
All `#[cfg(feature = "gcs-official")]` guards have been removed.

---

## Complete Zero-Copy Data Flow

### Get (Read)

```
GCS gRPC stream (BidiReadObject)
  └─ chunks arrive as bytes::Bytes (tonic zero-copy from network buffer)
       └─ BytesMut::with_capacity(size_hint)  ← single allocation, exact size
            └─ put_slice() per chunk           ← one copy per chunk (unavoidable: reassembly)
                 └─ BytesMut::freeze()         ← zero-copy: same base pointer → Bytes
                      └─ Bytes returned to caller  ← Arc clone is O(1)
```

With exact pre-allocation (`BytesMut::with_capacity(size_hint)`) there are **zero reallocations**.
The per-chunk `put_slice` is the only copy and is unavoidable — gRPC streams arrive as separate
fragments that must be reassembled into a single contiguous buffer.

### Put (Write)

```
Caller holds Bytes (Arc-backed)
  └─ put_object(data: Bytes)               ← zero-copy: passes Arc pointer
       └─ data.clone() for spawn task       ← O(1) Arc increment
            └─ data.slice(offset..end)      ← zero-copy sub-view, no allocation
                 └─ BidiWriteObjectRequest { data: ChecksummedData { content: chunk } }
                      └─ gRPC stream sends  ← zero-copy end-to-end
```

**There are zero data copies on the write path** when the caller holds `Bytes`.

---

## Zero-Copy and the Python Interface

### Where copies happen

| Operation | Rust-internal copies | Python boundary copy |
|-----------|---------------------|----------------------|
| **Put (write)** | **0** (Bytes all the way) | 1 — `PyBuffer → Bytes` (unavoidable at GIL boundary) |
| **Get (read)**  | 1 — reassemble gRPC chunks into BytesMut | 1 — `Bytes → PyBytes` (avoidable with custom type) |

The copy at the Python boundary cannot be avoided for writes: Python's GIL-managed memory is
not Arc-compatible, and the GIL cannot be held across `await` points. However, the copy happens
exactly **once** at the language boundary and propagates zero-copy through the entire Rust stack
to the network.

A zero-copy read boundary (`BytesView` via the Python buffer protocol) is theoretically possible
by implementing a custom Python type that holds the `Arc<[u8]>` alive. This is deferred work.

---

## Zero-Copy Unit Tests

**Module:** `google_gcs_client::zero_copy_tests` (10 tests, all passing)

Run with: `cargo test zero_copy`

| Test | What it proves |
|------|---------------|
| `test_bytesmut_freeze_is_zero_copy` | `BytesMut::freeze()` returns same base pointer — no reallocation |
| `test_bytes_clone_is_zero_copy` | `Bytes::clone()` is Arc increment — O(1), same pointer |
| `test_bytes_slice_is_zero_copy` | `data.slice(a..b).as_ptr() == base + a` — sub-view, no copy |
| `test_bytes_from_vec_preserves_pointer` | `Bytes::from(vec)` transfers heap pointer, no copy |
| `test_buffered_writer_finalise_path_is_zero_copy` | `Bytes::from(mem::take(&mut self.buffer))` same pointer |
| `test_grpc_read_accumulation_no_realloc` | Exact pre-alloc + 3-chunk fill + freeze: pointer unchanged |
| `test_http_read_freeze_returns_bytes_not_vec` | HTTP read: exact-fit BytesMut fill, freeze pointer preserved |
| `test_write_producer_chunk_slices_are_zero_copy` | 3×2 MiB slices of 6 MiB Bytes: each `as_ptr() == base + offset` |
| `test_producer_task_data_clone_is_arc_increment` | 32 MiB `data.clone()` for spawn: same pointer |
| `test_put_object_caller_bytes_conversion_is_zero_copy` | `Bytes::from(Vec)` at caller boundary: same pointer through two clones |

---

## Performance Results

### Pre-fix baseline (c4-standard-8, RAPID bucket, 1,000 × ~30 MiB files)

| gRPC subchannels | Effective window/stream | Throughput |
|-----------------|------------------------|------------|
| 8 (old default) | 32 MiB                 | ~845 MiB/s |
| 32              | ~256 MiB               | ~3.1 GiB/s |
| 64 (= jobs)     | 256 MiB (uncontested)  | **3.62 GiB/s** |

### Post-fix measurements

**Upload (PUT): 1,000 × 32 MiB objects, RAPID bucket, 32 jobs: `3.83 GB/s`**

| Operation | Standard bucket | RAPID bucket | Notes |
|-----------|----------------|--------------|-------|
| Write     | ~2.0–2.3 GB/s  | **3.5–3.9 GB/s** | Network-limited; TLS+gRPC framing overhead |
| Read      | ~1.8–2.2 GB/s  | ~2.0–2.5 GB/s | RAPID lower commit latency |

### Recommended `--jobs` by VM size

| VM              | vCPU | NIC        | Recommended `--jobs` | Expected PUT throughput |
|-----------------|------|------------|----------------------|-------------------------|
| c4-standard-8   | 8    | ~16 Gbps   | 32–64                | ~3.5–4 GB/s             |
| c4-standard-32  | 32   | ~50 Gbps   | 128                  | ~8–10 GB/s (estimated)  |
| c4-standard-96  | 96   | ~100 Gbps  | 256                  | ~12 GB/s (estimated)    |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_GCS_RAPID` | `auto` | RAPID mode: `auto`, `true`/`1`/`yes`, `false`/`0`/`no` |
| `S3DLIO_GCS_GRPC_CHANNELS` | auto | Force exact subchannel count; overrides everything |
| `S3DLIO_GRPC_INITIAL_WINDOW_MIB` | `128` | HTTP/2 initial stream + connection window (MiB). Set `0` for protocol default (65 KB). |
| `S3DLIO_GRPC_WRITE_CHUNK_SIZE` | `2097152` | gRPC write DATA payload per `BidiWriteObjectRequest` (bytes); silently clamped to `MAX_GRPC_WRITE_CHUNK_SIZE` (4,128,768). |

---

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | `google-cloud-storage`, `google-cloud-gax`: `optional = true` removed; `gcs-official` removed from `default` and `[features]`; `google-cloud-gax-internal` added as direct dep |
| `vendor/google-cloud-gax-internal/src/gcs_constants.rs` | **NEW** — single source of truth for all GCS/gRPC tuning constants: `GCS_SERVER_MAX_MESSAGE_SIZE`, `MAX_GRPC_WRITE_CHUNK_SIZE`, `DEFAULT_GRPC_WRITE_CHUNK_SIZE`, window defaults, env-var name strings |
| `vendor/google-cloud-gax-internal/src/lib.rs` | `pub mod gcs_constants` added |
| `vendor/google-cloud-gax-internal/src/grpc.rs` | HTTP/2 window constants imported from `gcs_constants`; `make_endpoint()` configures tonic window sizes |
| `vendor/google-cloud-storage/src/storage/transport.rs` | `DEFAULT_GRPC_WRITE_CHUNK_SIZE` and `MAX_GRPC_WRITE_CHUNK_SIZE` imported from `gcs_constants` (via `gaxi` alias); `grpc_write_chunk_size()` clamps env-var values; concurrent bounded-channel producer |
| `src/gcs_constants.rs` | **NEW** — application-layer re-exports + `GCS_MIN_CHANNELS`, `GCS_MAX_CONCURRENT_DELETES`, env-var name constants |
| `src/lib.rs` | `pub mod gcs_constants`; `pub mod google_gcs_client` unconditional; `compile_error!` guards updated |
| `src/google_gcs_client.rs` | `DESIRED_GCS_CHANNELS` AtomicUsize + `set_gcs_channel_count()`; 3-tier priority ladder; `put_object` takes `Bytes`; read path uses `BytesMut` + pre-allocation; 10 zero-copy unit tests; all magic numbers replaced with named constants |
| `src/object_store.rs` | `GcsObjectStore::put`/`put_multipart` pass `Bytes` directly; `GcsBufferedWriter::finalise` uses `mem::take`; `use crate::google_gcs_client::…` now unconditional |
| `src/list_containers.rs` | `list_gcs_async` using official client now unconditional; community variant guarded by `#[cfg(feature = "gcs-community")]` |
| `src/bin/cli.rs` | All 4 command entry points call `set_gcs_channel_count(concurrency)` unconditionally |
| `tests/test_gcs_official.rs` | `#[cfg(feature = "gcs-official")]` guards removed; call sites updated to `Bytes::from_static` / `Bytes::from` |
| `tests/test_gcs_functional.rs` | `use bytes::Bytes` import added; call sites updated |
