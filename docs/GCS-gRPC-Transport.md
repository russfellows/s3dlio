# GCS gRPC Transport in s3dlio

**Applies to:** s3dlio v0.9.60+  
**Backend required:** `gcs-official` (default since v0.9.50)  
**Date:** February 2026

---

## Overview

As of v0.9.60, s3dlio uses the **gRPC Storage v2 API** for all Google Cloud
Storage operations — reads, writes, and deletes. This replaces the JSON/REST
API paths that were used in earlier versions.

The gRPC transport provides:

- **Binary protobuf encoding** — eliminates Base64/JSON overhead
- **HTTP/2 multiplexed connections** — persistent, pooled channels
- **Bidirectional streaming** — native support for chunked uploads
- **Universal compatibility** — works with both standard and RAPID/zonal buckets

### RAPID / Hyperdisk ML Support

GCS RAPID storage (also marketed as **Hyperdisk ML**) is a zonal storage tier
for ultra-low-latency AI/ML inference workloads. RAPID buckets impose two
constraints that differ from ordinary GCS buckets:

1. **Writes must set `appendable=true`** — resumable uploads are rejected.
2. **Reads require gRPC** — the JSON/REST read API returns HTTP 400.

Because s3dlio already uses gRPC universally, constraint #2 is satisfied
automatically. Constraint #1 is enabled by the `S3DLIO_GCS_RAPID` environment
variable, which adds `appendable=true` to the gRPC `WriteObjectSpec`.

---

## Architecture

### Read Path — `BidiReadObject`

All GCS reads use `open_object()` which sends a gRPC `BidiReadObject` request:

```
Client                              GCS gRPC Server
  │                                      │
  │── BidiReadObjectRequest ────────────►│
  │   bucket, object, read_range         │
  │                                      │
  │◄── BidiReadObjectResponse ──────────│
  │   checksummed_data { content, crc }  │
  │   (streams until complete)           │
  ▼                                      ▼
```

- `get_object()` — reads the full object using `ReadRange::all()`
- `get_object_range()` — reads a byte range using `ReadRange::segment()` or
  `ReadRange::offset()`

Both call a shared `get_object_via_grpc()` helper that delegates to
`open_object().send_and_read(range)`.

### Write Path — `BidiWriteObject`

All GCS writes use `send_grpc()` which sends a gRPC `BidiWriteObject` stream:

```
Client                              GCS gRPC Server
  │                                      │
  │── BidiWriteObjectRequest #1 ────────►│
  │   WriteObjectSpec (+ appendable)     │
  │   ChecksummedData { chunk_0, crc }   │
  │   write_offset: 0                    │
  │                                      │
  │── BidiWriteObjectRequest #N ────────►│
  │   ChecksummedData { chunk_n, crc }   │
  │   write_offset: <offset>             │
  │   finish_write: true                 │
  │                                      │
  │◄── BidiWriteObjectResponse ─────────│
  │   WriteStatus::Resource(Object)      │
  ▼                                      ▼
```

- Objects ≤ 2 MiB are sent in a single message (no chunking).
- Larger objects are split into 2 MiB chunks, each with its own CRC32C.
- The final message sets `finish_write: true` and includes whole-object
  `ObjectChecksums`.
- In RAPID mode, the first message includes `appendable: true` in the
  `WriteObjectSpec`.

**Chunk size** defaults to 2 MiB (`DEFAULT_GRPC_WRITE_CHUNK_SIZE`) and can be
overridden via the `S3DLIO_GRPC_WRITE_CHUNK_SIZE` environment variable.

### Delete Path — Concurrent Batch

`delete_objects()` dispatches up to **64 concurrent** gRPC delete requests
using `FuturesUnordered` + `Semaphore`. Errors are collected and reported
after all requests complete, ensuring partial failures don't abort the batch.

### List Path — Paginated JSON API

`list_objects()` uses the standard paginated list API. A prefix normalization
fix ensures exact object names are not incorrectly treated as directory
prefixes (trailing `/` retry logic).

---

## Vendor Patches

s3dlio vendors `google-cloud-storage` v1.8.0 at `vendor/google-cloud-storage/`.
The upstream crate uses gRPC for reads but JSON/REST for writes. The following
patches add gRPC write support:

### 1. `src/storage/stub.rs` — Trait Extension

Added `write_object_grpc()` trait method to the `Storage` trait with a default
`unimplemented_stub` body, following the pattern of existing methods:

```rust
fn write_object_grpc(
    &self,
    _data: bytes::Bytes,
    _spec: crate::model::WriteObjectSpec,
    _options: RequestOptions,
) -> impl std::future::Future<Output = Result<Object>> + Send {
    unimplemented_stub::<Object>()
}
```

### 2. `src/storage/transport.rs` — gRPC Implementation

Full `write_object_grpc()` implementation (~200 lines) that:

- Converts model `WriteObjectSpec` → proto `WriteObjectSpec`
- Splits payloads into chunks at `grpc_write_chunk_size()` boundaries (2 MiB default)
- Computes per-chunk CRC32C via `ChecksummedData`
- Computes whole-object CRC32C via `ObjectChecksums` on the final message
- Sends the first chunk with `FirstMessage::WriteObjectSpec`, subsequent chunks
  with data only
- Sets `finish_write: true` on the last message
- Streams via `bidi_stream_with_status()` from `gaxi::grpc::Client`
- Converts the proto `Object` response back to model `Object`
- Includes TRACE-level logging for chunk size, count, offsets, and CRC32C values

### 3. `src/storage/write_object.rs` — Builder Methods

Two new public methods on the `WriteObject` builder:

- `set_appendable(bool)` — sets `appendable` on the `WriteObjectSpec`
- `send_grpc()` — collects the payload into `Bytes` and delegates to
  `write_object_grpc()`

### 4. `src/storage/perform_upload.rs` — Query Parameter

`apply_preconditions()` patched to emit `appendable=true` as an HTTP query
parameter. This was the original JSON API workaround and remains as a legacy
code path.

All patches are wired via `[patch.crates-io]` in `Cargo.toml`. No changes to
user-visible APIs or build steps are required.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `S3DLIO_GCS_RAPID` | `false` | Enable `appendable=true` on writes for RAPID/zonal buckets |
| `S3DLIO_GRPC_WRITE_CHUNK_SIZE` | `2097152` (2 MiB) | gRPC write chunk size in bytes |
| `GOOGLE_APPLICATION_CREDENTIALS` | *(none)* | Path to GCP service account JSON key |
| `GOOGLE_CLOUD_PROJECT` | *(none)* | GCP project ID (used by `list-buckets`) |

`S3DLIO_GCS_RAPID` accepts `true`, `1`, `yes` (case-insensitive). Any other
value or absence means RAPID mode is off.

---

## Quick Start

```bash
# Standard GCS bucket (gRPC used automatically, no special config)
./target/release/s3-cli put --size 65536 --count 10 gs://my-bucket/bench/
./target/release/s3-cli get gs://my-bucket/bench/object_0_of_10.dat

# RAPID / Hyperdisk ML bucket (add appendable flag)
export S3DLIO_GCS_RAPID=true
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
./target/release/s3-cli put --size 65536 --count 10 gs://my-rapid-bucket/bench/
./target/release/s3-cli get gs://my-rapid-bucket/bench/object_0_of_10.dat

# List objects
./target/release/s3-cli list gs://my-rapid-bucket/bench/

# Delete
./target/release/s3-cli delete gs://my-rapid-bucket/bench/
```

### Using a `.env` File

s3dlio supports `.env` files via `dotenvy` for persistent configuration:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
GOOGLE_CLOUD_PROJECT=my-gcp-project-id
S3DLIO_GCS_RAPID=true
```

Shell environment variables take precedence over `.env` values.

---

## Diagnosing Common Errors

### "This bucket requires appendable objects"

The target is a RAPID/zonal bucket and `S3DLIO_GCS_RAPID` is not set.

```bash
export S3DLIO_GCS_RAPID=true
```

### "Zonal objects only support gRPC API" (HTTP 400)

The binary was compiled with `gcs-community` instead of `gcs-official`, or
an older version of s3dlio is being used that reads via the JSON API.
Ensure you are running v0.9.60+ built with default features.

### Authentication: "insufficient authentication scopes"

The service account lacks required IAM roles. RAPID buckets need:

- `roles/storage.objectAdmin` (or `objectCreator` + `objectViewer`)
- `roles/storage.legacyBucketReader` (for listings)

---

## Implementation History

The gRPC transport was implemented incrementally across several commits:

1. **Vendor fork** (`086c68e`) — Vendored `google-cloud-storage` v1.8.0 with
   `set_appendable()` builder method for RAPID write support.

2. **v0.9.60 base** (`51f8584`) — Switched `gcs-official` to the default backend,
   added `S3DLIO_GCS_RAPID` environment variable, multi-protocol `list-buckets`.

3. **gRPC writes** (`4200a63`) — Implemented `BidiWriteObject` in the vendor crate
   (`write_object_grpc()` + `send_grpc()`). RAPID mode routes through gRPC.

4. **gRPC reads** (`302987e`) — Implemented `BidiReadObject` for RAPID reads via
   `open_object()`. Fixed LIST prefix normalization bug.

5. **gRPC for all GCS** (`70b87b8`) — Made gRPC universal for all GCS operations.
   `S3DLIO_GCS_RAPID` now only controls `appendable=true`. Net -111 lines removed.

6. **Concurrent deletes** (`944516a`) — 64-way parallel delete dispatch using
   `FuturesUnordered` + `Semaphore`. ~35x speedup for batch deletes.
