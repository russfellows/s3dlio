# GCS gRPC v2 Write Path for RAPID Storage — Implementation Plan

**Date:** 2026-02-28
**Author:** GitHub Copilot (Claude Opus 4.6)
**s3dlio Version:** v0.9.60
**Status:** PLANNED — ready for implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Analysis](#3-architecture-analysis)
4. [Existing Infrastructure Inventory](#4-existing-infrastructure-inventory)
5. [Proto Type Reference](#5-proto-type-reference)
6. [Implementation Plan](#6-implementation-plan)
7. [Detailed File Changes](#7-detailed-file-changes)
8. [gRPC Protocol Flow](#8-grpc-protocol-flow)
9. [Chunk Splitting Strategy](#9-chunk-splitting-strategy)
10. [Error Handling](#10-error-handling)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Considerations](#12-performance-considerations)
13. [Reference: Google C++ Implementation](#13-reference-google-c-implementation)
14. [Risks and Mitigations](#14-risks-and-mitigations)
15. [Migration Path](#15-migration-path)

---

## 1. Executive Summary

Writes to GCS RAPID (Hyperdisk ML / zonal) buckets require `appendable=true`
semantics that the JSON API v1 does not fully support. The current HTTP-based
upload path (`uploadType=multipart`) was patched to inject `"appendable": true`
into the JSON metadata body, but the long-term correct solution — matching what
**Go** and **C++** official client libraries do — is to use the **GCS Storage v2
gRPC API**.

This document details the plan to add a `BidiWriteObject` gRPC write path to
the vendored `google-cloud-storage` crate, and expose it to s3dlio's
`google_gcs_client.rs` when `S3DLIO_GCS_RAPID=true`.

### Why gRPC is the Correct Solution

| Aspect | JSON API v1 (current) | gRPC v2 (planned) |
|--------|----------------------|-------------------|
| Write method | `POST /upload/storage/v1/b/{bucket}/o` | `BidiWriteObject` streaming RPC |
| Appendable flag | Must be injected into JSON metadata body (undocumented) | Native `WriteObjectSpec.appendable = true` field |
| Upload type | Multipart only (resumable rejected by zonal) | Stateful bidirectional stream |
| Chunked writes | Not supported in single-shot | Native: send multiple `BidiWriteObjectRequest` messages |
| Append after create | Not available via JSON API | `AppendObjectSpec` in `BidiWriteObjectRequest.first_message` |
| Official support | Workaround — no official documentation for appendable via JSON | Fully supported — proto field in `WriteObjectSpec` |
| Reference impl | gcloud CLI uses apitools (Python) | Go SDK, C++ SDK both use gRPC |

---

## 2. Problem Statement

### Background

GCS RAPID Storage (Hyperdisk ML) uses **zonal buckets** that require:
1. No resumable uploads (`"Zonal buckets do not support resumable upload"`)
2. Appendable objects (`"This bucket requires appendable objects"`)

### What We Proved

Through debugging on `sig65-cntrlr-vm` with bucket `gs://sig65-rapid1/`:

- `uploadType=resumable` → **Rejected** ("Zonal buckets do not support resumable upload")
- `uploadType=media` with `appendable=true` as query param → **Rejected** (HTTP 400 "This bucket requires appendable objects" — media upload ignores query params)
- `uploadType=multipart` with `"appendable": true` in JSON metadata body → **Current workaround** (builds in `rapid_multipart_builder()`)
- gRPC `BidiWriteObject` with `WriteObjectSpec.appendable = Some(true)` → **Target solution**

### Current Workaround Location

The multipart workaround lives in:
- `vendor/google-cloud-storage/src/storage/perform_upload/unbuffered.rs` → `rapid_multipart_builder()`
- `vendor/google-cloud-storage/src/storage/perform_upload.rs` → `apply_preconditions()` adds `appendable=true` query param
- `src/google_gcs_client.rs` → `put_object()` sets `set_appendable(true)` when `rapid_mode = true`

---

## 3. Architecture Analysis

### Current Transport Split in Vendor Crate

```
google-cloud-storage v1.8.0 (vendored)
├── StorageInner
│   ├── client: gaxi::http::ReqwestClient     ← HTTP/REST for writes + simple reads
│   ├── grpc: gaxi::grpc::Client              ← gRPC for bidi reads (open_object)
│   ├── cred: Credentials
│   └── options: RequestOptions
│
├── Reads
│   ├── read_object()     → HTTP REST (GET /storage/v1/b/{bucket}/o/{object})
│   └── open_object()     → gRPC BidiReadObject (bidirectional streaming via tonic)
│
└── Writes
    ├── write_object_buffered()    → HTTP REST multipart (buffered)
    └── write_object_unbuffered()  → HTTP REST multipart/resumable (unbuffered)
                                     └── rapid_multipart_builder() [PATCHED]
```

### Target Architecture

```
google-cloud-storage v1.8.0 (vendored) + RAPID gRPC
├── StorageInner (unchanged — grpc field already present)
│
├── Reads (unchanged)
│
└── Writes
    ├── write_object_buffered()    → HTTP REST (unchanged)
    ├── write_object_unbuffered()  → HTTP REST (unchanged, still has rapid_multipart_builder)
    └── write_object_grpc() [NEW]  → gRPC BidiWriteObject (bidirectional streaming)
         └── For RAPID: WriteObjectSpec with appendable=true
         └── Fallback can also be used for non-RAPID writes
```

---

## 4. Existing Infrastructure Inventory

### Already Available (No Changes Needed)

| Component | Location | Purpose |
|-----------|----------|---------|
| `gaxi::grpc::Client` | `StorageInner.grpc` field | Authenticated gRPC client with TLS, subchannels |
| `bidi_stream_with_status()` | `gaxi::grpc::Client` method | Bidirectional streaming RPC — returns `Streaming<Response>` |
| Proto: `BidiWriteObjectRequest` | `generated/protos/storage/google.storage.v2.rs:739` | Request type with `WriteObjectSpec`, `ChecksummedData`, `finish_write` |
| Proto: `BidiWriteObjectResponse` | `generated/protos/storage/google.storage.v2.rs:790` | Response type with `WriteStatus::Resource(Object)` |
| Proto: `WriteObjectSpec` | `generated/protos/storage/google.storage.v2.rs:618` | Spec with `appendable: Option<bool>` (tag 9) |
| Proto: `AppendObjectSpec` | `generated/protos/storage/google.storage.v2.rs:712` | For appending to existing objects |
| Proto: `ChecksummedData` | `generated/protos/storage/google.storage.v2.rs:1820` | `content: Bytes`, `crc32c: Option<u32>` |
| Proto: `Object` | `generated/protos/storage/google.storage.v2.rs:1908` | Full object metadata |
| Bidi read pattern | `storage/bidi/connector.rs` | Shows exactly how to use `bidi_stream_with_status()` |
| `X_GOOG_API_CLIENT_HEADER` | `storage/info` module | API client header for gRPC requests |
| `format_bucket_name()` | `src/google_gcs_client.rs` | `"my-bucket" → "projects/_/buckets/my-bucket"` |
| `read_rapid_mode()` | `src/google_gcs_client.rs` | Reads `S3DLIO_GCS_RAPID` env var |

### Must Create

| Component | Location | Purpose |
|-----------|----------|---------|
| `write_object_grpc()` | `vendor/.../storage/stub.rs` | New trait method on `Storage` |
| `write_object_grpc()` impl | `vendor/.../storage/transport.rs` | gRPC BidiWriteObject implementation |
| `send_grpc()` | `vendor/.../storage/write_object.rs` | New builder method (public API) |
| gRPC write call in `put_object()` | `src/google_gcs_client.rs` | Call `send_grpc()` when `rapid_mode = true` |

---

## 5. Proto Type Reference

### BidiWriteObjectRequest (the message we send)

```rust
// vendor/google-cloud-storage/src/generated/protos/storage/google.storage.v2.rs:739
pub struct BidiWriteObjectRequest {
    pub write_offset: i64,                                    // tag 3
    pub object_checksums: Option<ObjectChecksums>,            // tag 6
    pub state_lookup: bool,                                   // tag 7
    pub flush: bool,                                          // tag 8
    pub finish_write: bool,                                   // tag 9
    pub common_object_request_params: Option<CommonObjectRequestParams>,  // tag 10
    pub first_message: Option<bidi_write_object_request::FirstMessage>,   // oneof tags 1,2,11
    pub data: Option<bidi_write_object_request::Data>,        // oneof tag 4
}

pub mod bidi_write_object_request {
    pub enum FirstMessage {
        UploadId(String),                        // tag 1 — resume existing upload
        WriteObjectSpec(WriteObjectSpec),         // tag 2 — new object creation
        AppendObjectSpec(AppendObjectSpec),       // tag 11 — append to existing object (RAPID)
    }
    pub enum Data {
        ChecksummedData(ChecksummedData),        // tag 4 — payload bytes + optional CRC32C
    }
}
```

### BidiWriteObjectResponse (the message we receive)

```rust
// vendor/google-cloud-storage/src/generated/protos/storage/google.storage.v2.rs:790
pub struct BidiWriteObjectResponse {
    pub write_handle: Option<BidiWriteHandle>,                 // tag 3
    pub write_status: Option<bidi_write_object_response::WriteStatus>,  // oneof tags 1,2
}

pub mod bidi_write_object_response {
    pub enum WriteStatus {
        PersistedSize(i64),     // tag 1 — intermediate: bytes persisted so far
        Resource(Object),       // tag 2 — final: completed object metadata
    }
}
```

### WriteObjectSpec (for creating new appendable objects)

```rust
pub struct WriteObjectSpec {
    pub resource: Option<Object>,                              // tag 1 — object name/bucket/metadata
    pub predefined_acl: String,                                // tag 7
    pub if_generation_match: Option<i64>,                      // tag 3
    pub if_generation_not_match: Option<i64>,                  // tag 4
    pub if_metageneration_match: Option<i64>,                  // tag 5
    pub if_metageneration_not_match: Option<i64>,              // tag 6
    pub object_size: Option<i64>,                              // tag 8
    pub appendable: Option<bool>,                              // tag 9 ← THE KEY FIELD
}
```

### ChecksummedData (payload container)

```rust
pub struct ChecksummedData {
    pub content: prost::bytes::Bytes,    // tag 1 — raw bytes
    pub crc32c: Option<u32>,             // tag 2 — optional CRC32C checksum
}
```

### AppendObjectSpec (for appending to existing objects — future use)

```rust
pub struct AppendObjectSpec {
    pub bucket: String,                                        // tag 1
    pub object: String,                                        // tag 2
    pub generation: i64,                                       // tag 3
    pub if_metageneration_match: Option<i64>,                  // tag 4
    pub if_metageneration_not_match: Option<i64>,              // tag 5
    pub routing_token: Option<String>,                         // tag 6
    pub write_handle: Option<BidiWriteHandle>,                 // tag 7
}
```

---

## 6. Implementation Plan

### Phase 1: Vendor Crate Changes (Foundation)

#### Step 1.1: Add `write_object_grpc()` to Storage Trait

**File:** `vendor/google-cloud-storage/src/storage/stub.rs`

Add a new trait method with a default "unimplemented" body (matching the
pattern of existing methods):

```rust
/// Implements gRPC-based object write using BidiWriteObject.
/// Used for RAPID (Hyperdisk ML) zonal buckets.
fn write_object_grpc(
    &self,
    _data: bytes::Bytes,
    _spec: crate::model::WriteObjectSpec,
    _options: RequestOptions,
) -> impl std::future::Future<Output = Result<Object>> + Send {
    unimplemented_stub::<Object>()
}
```

#### Step 1.2: Implement in transport.rs

**File:** `vendor/google-cloud-storage/src/storage/transport.rs`

Add the implementation that:
1. Builds a `BidiWriteObjectRequest` with `FirstMessage::WriteObjectSpec` containing `appendable: Some(true)`
2. Attaches `ChecksummedData` with the payload bytes and CRC32C
3. Sets `finish_write: true` for single-shot writes
4. Sends via `self.inner.grpc.bidi_stream_with_status()`
5. Reads the first `BidiWriteObjectResponse` from the stream
6. Extracts the `Object` from `WriteStatus::Resource`

#### Step 1.3: Add `send_grpc()` Builder Method

**File:** `vendor/google-cloud-storage/src/storage/write_object.rs`

Add a `send_grpc()` method to the `WriteObject` builder, analogous to
`send_unbuffered()` and `send()`:

```rust
/// Send the object using gRPC BidiWriteObject streaming.
///
/// This is the recommended path for RAPID (Hyperdisk ML) zonal buckets,
/// which require appendable=true semantics that are natively supported
/// by the gRPC Storage v2 API.
pub async fn send_grpc(self) -> crate::Result<crate::model::Object> { ... }
```

### Phase 2: s3dlio Integration

#### Step 2.1: Update `put_object()` in google_gcs_client.rs

**File:** `src/google_gcs_client.rs`

When `rapid_mode = true`, use the new gRPC path:

```rust
if self.rapid_mode {
    debug!("GCS PUT: RAPID mode — using gRPC BidiWriteObject with appendable=true");
    self.storage
        .write_object(&bucket_name, object, bytes)
        .set_appendable(true)
        .send_grpc()
        .await
        .map_err(|e| anyhow!("GCS gRPC PUT failed for gs://{}/{}: {}", bucket, object, e))?;
    debug!("GCS PUT success (gRPC)");
    return Ok(());
}
```

### Phase 3: Cleanup

#### Step 3.1: Remove Debug Diagnostics

Remove all `eprintln!("[RAPID DEBUG] ...")` statements from:
- `vendor/.../perform_upload.rs` (apply_preconditions)
- `vendor/.../perform_upload/unbuffered.rs` (send_unbuffered, single_shot_attempt, single_shot_builder, rapid_multipart_builder)

#### Step 3.2: Keep JSON API Multipart as Fallback

The `rapid_multipart_builder()` code should be kept as a fallback. The
auto-retry logic in `put_object()` can fall back to multipart if gRPC
fails unexpectedly.

---

## 7. Detailed File Changes

### File 1: `vendor/google-cloud-storage/src/storage/stub.rs`

**Change:** Add `write_object_grpc()` trait method (after `write_object_unbuffered`)

```rust
/// Implements gRPC-based object write using BidiWriteObject.
/// Preferred for RAPID (Hyperdisk ML) zonal buckets where appendable=true
/// is natively supported via the gRPC Storage v2 API.
fn write_object_grpc(
    &self,
    _data: bytes::Bytes,
    _spec: crate::model::WriteObjectSpec,
    _options: RequestOptions,
) -> impl std::future::Future<Output = Result<Object>> + Send {
    unimplemented_stub::<Object>()
}
```

### File 2: `vendor/google-cloud-storage/src/storage/transport.rs`

**Change:** Add `write_object_grpc()` implementation in `impl super::stub::Storage for Storage`

```rust
async fn write_object_grpc(
    &self,
    data: bytes::Bytes,
    spec: crate::model::WriteObjectSpec,
    options: RequestOptions,
) -> Result<Object> {
    use crate::google::storage::v2::{
        BidiWriteObjectRequest, BidiWriteObjectResponse, ChecksummedData,
        Object as ProtoObject,
        bidi_write_object_request, bidi_write_object_response,
    };
    use gaxi::grpc::tonic::{Extensions, GrpcMethod};
    use crate::storage::info::X_GOOG_API_CLIENT_HEADER;

    // Convert the model WriteObjectSpec to proto WriteObjectSpec
    let proto_spec = crate::google::storage::v2::WriteObjectSpec {
        resource: spec.resource.map(|r| ProtoObject {
            name: r.name,
            bucket: r.bucket,
            ..ProtoObject::default()
        }),
        appendable: spec.appendable,
        if_generation_match: spec.if_generation_match,
        if_generation_not_match: spec.if_generation_not_match,
        if_metageneration_match: spec.if_metageneration_match,
        if_metageneration_not_match: spec.if_metageneration_not_match,
        object_size: Some(data.len() as i64),
        ..Default::default()
    };

    // Compute CRC32C for data integrity
    let crc32c = crc32c::crc32c(&data);

    // Build the single BidiWriteObjectRequest (single-shot: spec + data + finish)
    let request = BidiWriteObjectRequest {
        first_message: Some(
            bidi_write_object_request::FirstMessage::WriteObjectSpec(proto_spec),
        ),
        data: Some(
            bidi_write_object_request::Data::ChecksummedData(ChecksummedData {
                content: data,
                crc32c: Some(crc32c),
            }),
        ),
        write_offset: 0,
        finish_write: true,
        flush: false,
        state_lookup: false,
        object_checksums: None,
        common_object_request_params: None,
    };

    // Set up gRPC method metadata
    let bucket_name = spec.resource.as_ref()
        .map(|r| r.bucket.as_str())
        .unwrap_or_default();
    let x_goog_request_params = format!("bucket={bucket_name}");

    let extensions = {
        let mut e = Extensions::new();
        e.insert(GrpcMethod::new("google.storage.v2.Storage", "BidiWriteObject"));
        e
    };
    let path = http::uri::PathAndQuery::from_static(
        "/google.storage.v2.Storage/BidiWriteObject",
    );

    // Send via bidirectional stream (send one request, read one response)
    let (tx, rx) = tokio::sync::mpsc::channel::<BidiWriteObjectRequest>(1);
    tx.send(request).await.map_err(Error::io)?;
    drop(tx); // Close the send side to signal we're done

    let request_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let response = self.inner.grpc
        .bidi_stream_with_status::<BidiWriteObjectRequest, BidiWriteObjectResponse>(
            extensions,
            path,
            request_stream,
            options.gax(),
            &X_GOOG_API_CLIENT_HEADER,
            &x_goog_request_params,
        )
        .await?;

    // Parse the response
    let mut stream = match response {
        Ok(r) => r.into_inner(),
        Err(status) => {
            return Err(Error::service(
                google_cloud_gax::error::rpc::Status::default()
                    .set_message(format!("BidiWriteObject gRPC error: {}", status)),
            ));
        }
    };

    // Read the final response
    use futures::StreamExt;
    match stream.next().await {
        Some(Ok(resp)) => {
            match resp.write_status {
                Some(bidi_write_object_response::WriteStatus::Resource(proto_obj)) => {
                    Ok(Object::from(proto_obj)) // Convert proto Object → model Object
                }
                Some(bidi_write_object_response::WriteStatus::PersistedSize(size)) => {
                    Err(Error::deser(format!(
                        "BidiWriteObject returned PersistedSize({}) instead of Resource",
                        size
                    )))
                }
                None => Err(Error::deser("BidiWriteObject response has no write_status")),
            }
        }
        Some(Err(status)) => Err(Error::service(
            google_cloud_gax::error::rpc::Status::default()
                .set_message(format!("BidiWriteObject stream error: {}", status)),
        )),
        None => Err(Error::deser("BidiWriteObject stream closed without response")),
    }
}
```

> **Note:** The above is a detailed sketch. The actual implementation will need
> to handle the model-to-proto conversion correctly. The vendor crate has
> `gaxi::prost::ToProto`/`FromProto` traits for most types, or we can
> construct the proto types directly since they're in
> `crate::google::storage::v2`.

### File 3: `vendor/google-cloud-storage/src/storage/write_object.rs`

**Change:** Add `send_grpc()` public method to the `WriteObject` builder

```rust
/// Send using gRPC BidiWriteObject (recommended for RAPID/zonal buckets).
pub async fn send_grpc(self) -> crate::Result<crate::model::Object> {
    let (payload_bytes, request) = self.into_parts();
    let data = /* collect payload into Bytes */;
    self.stub.write_object_grpc(data, request.spec, self.options).await
}
```

### File 4: `src/google_gcs_client.rs`

**Change:** Update `put_object()` to use gRPC when `rapid_mode = true`

The first attempt for RAPID mode will use `send_grpc()`. The auto-retry
fallback (for non-RAPID buckets that unexpectedly require appendable) will
continue using the existing `send_unbuffered()` path with the multipart
workaround.

---

## 8. gRPC Protocol Flow

### Single-Shot Write (Primary Use Case)

For s3dlio benchmark writes (typical: 64KB to 100MB objects), a single-shot
write sends one `BidiWriteObjectRequest` with all data:

```
Client                                    GCS gRPC Server
  │                                           │
  │── BidiWriteObjectRequest ────────────────►│
  │   first_message: WriteObjectSpec {        │
  │     resource: { name, bucket },           │
  │     appendable: true,                     │
  │     object_size: 65536,                   │
  │   }                                       │
  │   data: ChecksummedData {                 │
  │     content: <65536 bytes>,               │
  │     crc32c: 0xABCD1234,                   │
  │   }                                       │
  │   write_offset: 0                         │
  │   finish_write: true                      │
  │                                           │
  │   (close send stream)                     │
  │                                           │
  │◄── BidiWriteObjectResponse ──────────────│
  │   write_status: Resource(Object {         │
  │     name: "obj.dat",                      │
  │     size: 65536,                          │
  │     generation: 123456789,                │
  │   })                                      │
  │   write_handle: Some(BidiWriteHandle {    │
  │     handle: <opaque bytes>                │
  │   })                                      │
  │                                           │
  ▼                                           ▼
```

### Multi-Chunk Write (Objects > 2MB)

For large objects, split into multiple requests. gRPC has a default ~4MB
message size limit. Each chunk is a separate `BidiWriteObjectRequest`:

```
Client                                    GCS gRPC Server
  │                                           │
  │── Request 1 (first + data chunk 1) ─────►│
  │   first_message: WriteObjectSpec { ... }  │
  │   data: ChecksummedData { chunk_0 }      │
  │   write_offset: 0                         │
  │   finish_write: false                     │
  │                                           │
  │── Request 2 (data chunk 2) ─────────────►│
  │   data: ChecksummedData { chunk_1 }      │
  │   write_offset: 2097152                   │
  │   finish_write: false                     │
  │                                           │
  │── Request N (final chunk) ──────────────►│
  │   data: ChecksummedData { chunk_n }      │
  │   write_offset: <last_offset>             │
  │   finish_write: true                      │
  │                                           │
  │   (close send stream)                     │
  │                                           │
  │◄── BidiWriteObjectResponse ──────────────│
  │   write_status: Resource(Object { ... })  │
  │                                           │
  ▼                                           ▼
```

---

## 9. Chunk Splitting Strategy

### Default gRPC Message Limits

- **tonic default max message size:** 4 MB (4,194,304 bytes)
- **Recommended chunk size:** 2 MiB (2,097,152 bytes)
  - Matches Google's recommendation and the C++ SDK default
  - Leaves room for proto overhead within the 4MB limit

### Implementation

```rust
const GRPC_WRITE_CHUNK_SIZE: usize = 2 * 1024 * 1024; // 2 MiB

fn chunk_data(data: &[u8]) -> Vec<(usize, &[u8])> {
    data.chunks(GRPC_WRITE_CHUNK_SIZE)
        .enumerate()
        .map(|(i, chunk)| (i * GRPC_WRITE_CHUNK_SIZE, chunk))
        .collect()
}
```

For the first message, include both the `WriteObjectSpec` and the first data
chunk. For subsequent messages, only include data. The last message sets
`finish_write: true`.

### Objects <= 2 MiB

Single message with `WriteObjectSpec` + `ChecksummedData` + `finish_write: true`.
No chunking needed.

---

## 10. Error Handling

### gRPC Status Codes to Handle

| gRPC Code | Meaning | Action |
|-----------|---------|--------|
| `OK` (0) | Success | Extract Object from response |
| `INVALID_ARGUMENT` (3) | Bad request (e.g., missing fields) | Return error, don't retry |
| `NOT_FOUND` (5) | Bucket doesn't exist | Return error, don't retry |
| `ALREADY_EXISTS` (6) | Object or generation conflict | Return error, don't retry |
| `PERMISSION_DENIED` (7) | Auth failure | Return error, don't retry |
| `RESOURCE_EXHAUSTED` (8) | Quota exceeded | Retry with backoff |
| `ABORTED` (10) | Redirect (similar to BidiReadObject) | Handle redirect, retry |
| `UNAVAILABLE` (14) | Transient server error | Retry with backoff |
| `INTERNAL` (13) | Server bug | Retry with backoff |

### Retry Strategy

Use the same retry infrastructure as `BidiReadObject` (already in the vendor
crate):
- `google_cloud_gax::retry_loop_internal::retry_loop()`
- Exponential backoff via `options.backoff_policy`
- Throttling via `options.retry_throttler`
- Idempotency: Single-shot writes with `if_generation_match` preconditions
  are safe to retry.

---

## 11. Testing Strategy

### Unit Tests (in vendor crate)

1. **Proto construction test:** Verify `BidiWriteObjectRequest` is built
   correctly with `WriteObjectSpec.appendable = Some(true)` and
   `ChecksummedData`.

2. **Chunk splitting test:** Verify data > 2 MiB is split into correct chunks
   with proper `write_offset` values and `finish_write` on the last chunk.

3. **Response parsing test:** Verify `BidiWriteObjectResponse` with
   `WriteStatus::Resource(Object)` is correctly converted to the model
   `Object`.

4. **Error handling test:** Verify gRPC status codes are correctly mapped to
   vendor crate `Error` types.

### Integration Tests (on sig65-cntrlr-vm)

```bash
# Test 1: Small object (single-shot, no chunking)
S3DLIO_GCS_RAPID=1 ./target/release/s3-cli -vvv put --size 65536 gs://sig65-rapid1/test_grpc_small.dat

# Test 2: Large object (multi-chunk)
S3DLIO_GCS_RAPID=1 ./target/release/s3-cli -vvv put --size 10485760 gs://sig65-rapid1/test_grpc_large.dat

# Test 3: Benchmark with multiple objects
S3DLIO_GCS_RAPID=1 ./target/release/s3-cli -vvv put --count 100 --size 1048576 gs://sig65-rapid1/bench/

# Test 4: Read-back verification
./target/release/s3-cli -vvv get gs://sig65-rapid1/test_grpc_small.dat | md5sum
```

---

## 12. Performance Considerations

### gRPC vs JSON API Performance

| Factor | JSON API multipart | gRPC BidiWriteObject |
|--------|-------------------|---------------------|
| Connection setup | New HTTP/1.1 per request | Persistent HTTP/2 multiplexed |
| Encoding overhead | Base64 in multipart + JSON metadata | Binary protobuf (zero-copy) |
| Header overhead | ~500 bytes per request | ~50 bytes per message |
| Streaming | Not supported | Native bidirectional |
| Subchannels | N/A | Configurable via `with_grpc_subchannel_count()` |

### Subchannel Configuration

The vendor crate already supports gRPC subchannel pooling:

```rust
let client = Storage::builder()
    .with_grpc_subchannel_count(num_cpus::get() / 2)
    .build()
    .await?;
```

This should be exposed to s3dlio's `GcsClient::new()` for high-throughput
RAPID benchmarks.

### Zonal Endpoints

For maximum performance with zonal RAPID buckets, use the zonal endpoint:

```
https://{zone}-storage.googleapis.com
# Example: https://us-central1-c-storage.googleapis.com
```

This can be configured via the existing `Storage::builder().with_endpoint()`
method. We should add a `S3DLIO_GCS_ENDPOINT` environment variable.

---

## 13. Reference: Google C++ Implementation

### Repository

- **Storage directory:** `https://github.com/googleapis/google-cloud-cpp/tree/main/google/cloud/storage`
- **gRPC internal:** `https://github.com/googleapis/google-cloud-cpp/tree/main/google/cloud/storage/internal/grpc`
- Key files: `stub.cc`, `object_request_parser.cc` — handles `AppendObjectSpec` for RAPID

### Key Patterns from C++ SDK

1. **Single-shot path:** For objects under `max_insert_object_size` (default 32 MiB), send a single `BidiWriteObjectRequest` with all data + `finish_write=true`.

2. **Chunked path:** For larger objects, use `flush=true` periodically to get server acknowledgment (`PersistedSize`), and `finish_write=true` on the last chunk.

3. **Connection reuse:** The C++ SDK maintains a pool of gRPC channels and reuses them across writes. This is analogous to the `grpc_subchannel_count` setting.

4. **CRC32C per-chunk:** Each `ChecksummedData` message includes a `crc32c` for that chunk, not for the whole object. The final `ObjectChecksums` can include a full-object CRC32C and/or MD5.

---

## 14. Risks and Mitigations

### Risk 1: Proto-to-Model Object Conversion

**Risk:** The `Object` type in `crate::google::storage::v2` (proto) differs
from `crate::model::Object` (REST/model). We need to convert correctly.

**Mitigation:** The vendor crate already has `From<v1::Object>` for the REST
response. We need to either add `From<proto::Object>` or use the existing
`gaxi::prost::FromProto` trait. Check if the existing bidi read path already
does this conversion.

### Risk 2: gRPC Max Message Size

**Risk:** Default tonic max message size is 4 MB. Objects larger than ~3.5 MB
(accounting for proto overhead) would fail.

**Mitigation:** Implement chunking for > 2 MiB objects. Also, configure tonic
channel with increased max message size if needed:
```rust
Channel::from_shared(endpoint)?
    .initial_connection_window_size(16 * 1024 * 1024)
    .initial_stream_window_size(16 * 1024 * 1024)
    .connect().await?
```

### Risk 3: Auth Token Propagation

**Risk:** The `gaxi::grpc::Client` handles auth internally, but we need to
verify it works correctly for write operations.

**Mitigation:** The same `gaxi::grpc::Client` instance is already used for
`BidiReadObject` (reads). If reads work, writes should use the same auth path.
Test with service account credentials on `sig65-cntrlr-vm`.

### Risk 4: Bi-directional vs Client-Streaming Semantics

**Risk:** `BidiWriteObject` is technically bidirectional, but for single-shot
writes we use it like a client-streaming RPC (send N messages, read 1 response).
The `bidi_stream_with_status()` method returns `Streaming<Response>` — we only
read the first message.

**Mitigation:** This is exactly how the C++ and Go SDKs use it. The server
only sends a response after `finish_write: true`. We read one message and
close the stream.

### Risk 5: `crc32c` Crate Dependency

**Risk:** We need the `crc32c` crate for computing CRC32C checksums per chunk.

**Mitigation:** The vendor crate already uses CRC32C for checksumming (see
`storage/checksum/`). Check if it's available via `gaxi` or if we need to
add a dependency. Alternatively, the CRC32C can be set to `None` for initial
testing (it's optional in the proto).

---

## 15. Migration Path

### Phase 1: MVP (This Sprint)

1. Implement `write_object_grpc()` in the vendor crate
2. Wire it to `send_grpc()` builder method
3. Call from `google_gcs_client.rs` when `rapid_mode = true`
4. Test on `sig65-cntrlr-vm` with `gs://sig65-rapid1/`
5. Keep JSON API multipart as fallback in auto-retry path

### Phase 2: Hardening

1. Add chunk splitting for objects > 2 MiB
2. Add retry logic with backoff
3. Add CRC32C checksumming per chunk
4. Expose `S3DLIO_GCS_ENDPOINT` for zonal endpoint configuration
5. Expose gRPC subchannel count configuration

### Phase 3: Full gRPC Migration (Future)

1. Consider using gRPC for non-RAPID writes (better performance)
2. Consider using gRPC for reads via `BidiReadObject` (already available
   via `open_object()`)
3. Remove JSON API multipart workaround once gRPC is stable
4. Remove debug `eprintln!` statements from vendor code

---

## Appendix A: Key File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `vendor/google-cloud-storage/src/storage/stub.rs` | 1-109 | Storage trait (add `write_object_grpc`) |
| `vendor/google-cloud-storage/src/storage/transport.rs` | 1-150 | Storage impl (add `write_object_grpc` impl) |
| `vendor/google-cloud-storage/src/storage/write_object.rs` | 1-1636 | WriteObject builder (add `send_grpc`) |
| `vendor/google-cloud-storage/src/storage/bidi.rs` | 1-177 | BidiRead module (reference pattern) |
| `vendor/google-cloud-storage/src/storage/bidi/connector.rs` | 1-822 | BidiRead connector (reference for gRPC setup) |
| `vendor/google-cloud-storage/src/storage/client.rs` | 100-110 | StorageInner with `grpc` field |
| `vendor/google-cloud-storage/src/generated/protos/storage/google.storage.v2.rs` | 618-810 | Proto types for write operations |
| `src/google_gcs_client.rs` | 195-260 | `put_object()` method (update for gRPC) |

## Appendix B: gaxi::grpc::Client Streaming Methods

The `gaxi::grpc::Client` (from `google-cloud-gax-internal` v0.7.9) exposes:

| Method | Type | Used For |
|--------|------|----------|
| `execute()` | Unary | Simple request/response RPCs |
| `bidi_stream()` | Bidirectional | Bidi streaming (converts tonic::Status → gax Error) |
| `bidi_stream_with_status()` | Bidirectional | Bidi streaming (preserves raw tonic::Status) |

**Note:** There is NO `client_streaming()` method. This is why we use
`bidi_stream_with_status()` for `BidiWriteObject` even though writes are
semantically client-streaming. The underlying tonic `streaming()` method
handles both bidirectional and client-streaming patterns.

## Appendix C: Sample Code from User

The user provided a reference implementation sketch using standalone `tonic`
channels and `google-cloud-googleapis` types. Our approach uses the **existing**
`gaxi::grpc::Client` (already initialized with auth and TLS) instead of
creating a separate `StorageClient`. This avoids duplicating auth logic and
leverages the vendor crate's existing infrastructure.

Key differences from the user's sketch:
- **Auth:** Handled by `gaxi::grpc::Client` (not a custom `GcsInterceptor`)
- **Channel:** Reused from `StorageInner.grpc` (not created separately)
- **Proto types:** From `crate::google::storage::v2` (already generated, not from `google-cloud-googleapis`)
- **No `async_trait`:** Using Rust's native async-in-trait (RPITIT)
- **Bucket format:** `projects/_/buckets/{NAME}` (already handled by `format_bucket_name()`)
