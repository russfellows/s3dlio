# GCS RAPID Storage — Bug Analysis and Fix Plan

**Issues:** [#121](https://github.com/russfellows/s3dlio/issues/121) (list-buckets region) and
[#122](https://github.com/russfellows/s3dlio/issues/122) (RAPID appendable writes / GET 400)

**Date:** December 2025 — **Updated:** February 2026  
**Status:** Both fixes implemented (v0.9.50); integration testing against live services pending

---

## Background: s3dlio's Dual GCS Client Architecture

s3dlio maintains two distinct GCS backends selected by Cargo feature flag:

| Feature flag | File | Crate | API style |
|---|---|---|---|
| `gcs-community` (**default**) | `src/gcs_client.rs` | `gcloud-storage ^1.2` | JSON API (HTTP/1.1) |
| `gcs-official` | `src/google_gcs_client.rs` | `google-cloud-storage ^1.8` | gRPC (HTTP/2) |

### Critical crate naming history

The crate that was once called `google-cloud-storage` on crates.io was an independent
community project by @yoshidan (not from Google).  
In 2024/25, Google took over the crate name and replaced it with a completely different
gRPC-based official SDK. @yoshidan's crate continues under the new name **`gcloud-storage`**.

This creates a confusing naming inversion:

```
gcloud-storage    = community JSON API wrapper (old "google-cloud-storage")
google-cloud-storage = NEW official Google gRPC SDK
```

Both are present as dependencies in s3dlio today.

---

## Issue #121 — `list-buckets` always uses `us-east-1`

### Symptoms

```
[ERROR] Failed to list S3 buckets: Error { code: InvalidArgument, ... }
```

Users with `AWS_ENDPOINT_URL=https://storage.googleapis.com` and
`AWS_REGION=us-central1` see this when running `s3dlio list-buckets`.

### Root Cause

`src/s3_utils.rs`, `list_buckets()` (≈ line 292):

```rust
// Force us-east-1 for global ListBuckets operation
let region = RegionProviderChain::first_try(Some(Region::new("us-east-1")));
```

The original comment is correct for **AWS S3**: `ListBuckets` is a global operation
that must hit `us-east-1`. However, when `AWS_ENDPOINT_URL` points to GCS (or any
non-AWS S3-compatible service), forcing `us-east-1` causes the service to reject the
request.

Additionally, GCS's S3-compatible API requires **path-style addressing**
(`https://storage.googleapis.com/my-bucket`), but the SDK defaults to virtual-host
style (`https://my-bucket.storage.googleapis.com`). The `AWS_S3_ADDRESSING_STYLE=path`
env var is not being respected.

### Fix

In `list_buckets()`, replace the hardcoded region with environment-aware logic:

- When `AWS_ENDPOINT_URL` is set (non-AWS endpoint): read `AWS_REGION` from the
  environment instead of forcing `us-east-1`.
- Respect `AWS_S3_ADDRESSING_STYLE=path` by calling `force_path_style(true)` on the
  SDK config.
- Only fallback to `us-east-1` when no custom endpoint is configured (genuine AWS).

This fix is fully independent of the GCS client choice.

---

## Issue #122 — RAPID bucket writes fail with "appendable objects" error

### Symptoms

**PUT:**
```
Error: GCS PUT failed for gs://<bucket>/<object>:
  service error: This bucket requires appendable objects.
  Make sure you use the appendable=true query parameter.
```

**GET (on objects already in RAPID bucket):**
```
HTTP 400 Bad Request
```

### What is GCS RAPID / Hyperdisk ML?

RAPID is a Google Cloud Storage tier designed for high-throughput AI/ML inference
workloads (serving large model weights). Two key constraints apply to all objects in
a RAPID bucket:

1. **Writes must set `appendable=true`** in the write request spec.
2. **Reads use the standard gRPC path** — the JSON API returns 400 for RAPID objects.

Google explicitly documents that appendable objects can only be created using the
**gRPC API**, not the JSON API.

### Analysis: `gcloud-storage` (community JSON API)

`gcloud-storage 1.2.0` (used by the `gcs-community` default feature) wraps the
GCS JSON API (XML / multipart HTTP).

**Confirmed blockers** (verified by grep in the cargo registry):

```
$ grep -rn "appendable" ~/.cargo/registry/src/.../gcloud-storage-1.2.0/
(no output — exit code 1)
```

- `UploadObjectRequest` has **no `appendable` field**.
- The `Object` struct has **no `appendable` field**.
- The JSON API itself does not support appendable object creation.

**Conclusion: `gcs-community` cannot support RAPID buckets at all.** This is a
fundamental API limitation, not a missing implementation detail.

### Analysis: `google-cloud-storage` (official Google gRPC SDK)

`google-cloud-storage 1.8.0` (used by the `gcs-official` feature) talks to GCS via
gRPC, which is the only path that supports appendable objects.

**The model supports appendable:**

```
$ grep -n "appendable" .../google-cloud-storage-1.8.0/src/generated/gapic/model.rs
2646: /// If `true`, the object is created in appendable mode.
2648: pub appendable: std::option::Option<bool>,
2858: /// Sets the value of [appendable][crate::model::WriteObjectSpec::appendable].
2865: pub fn set_appendable<T>(mut self, v: T) -> Self
```

`WriteObjectSpec::set_appendable(true)` exists and works.

**The high-level builder does not expose it:**

The public `WriteObject` request builder in `storage/write_object.rs` wraps a
`WriteObjectRequest { spec: WriteObjectSpec, ... }` struct, but:

- The `request` field is `pub(crate)` — inaccessible from external code.
- None of the ~30 public builder methods on `WriteObject` expose `set_appendable`.

This means that even with the `gcs-official` backend, **calling
`storage.write_object(...).send_unbuffered()` will not set `appendable=true`** unless
we either:

(a) Submit a PR to `google-cloud-storage` upstream to add `.set_appendable()` to
the `WriteObject` builder (trivial ~5-line change).

(b) Detect the error on first write, then use a workaround for RAPID.

---

## Summary Comparison

| Capability | `gcs-community` (`gcloud-storage`) | `gcs-official` (`google-cloud-storage`) |
|---|---|---|
| Standard GCS buckets | ✅ Works | ✅ Works |
| RAPID bucket writes | ❌ Impossible (JSON API limitation) | ⚠️ Supported in model, not yet in builder |
| RAPID bucket reads | ❌ 400 Bad Request | ✅ gRPC reads work |
| `appendable` in spec | ❌ No field | ✅ `WriteObjectSpec::set_appendable(true)` |
| Builder exposes appendable | N/A | ❌ `request` is `pub(crate)` |

---

## Proposed Course of Action

### Issue #121 — Implement immediately (low risk)

**Change:** [src/s3_utils.rs](../src/s3_utils.rs) `list_buckets()`:

```rust
// Use AWS_REGION if set; only default to us-east-1 for genuine AWS endpoints
let region = if std::env::var("AWS_ENDPOINT_URL").is_ok() {
    // Non-AWS endpoint: respect user's region
    RegionProviderChain::default_provider()
} else {
    // Genuine AWS S3: global ListBuckets requires us-east-1
    RegionProviderChain::first_try(Some(Region::new("us-east-1")))
};

// Respect path-style addressing
let use_path_style = std::env::var("AWS_S3_ADDRESSING_STYLE")
    .map(|v| v.to_lowercase() == "path")
    .unwrap_or(false);
```

Then pass `force_path_style(use_path_style)` to the S3 client config.

### Issue #122 — RAPID appendable writes

There are two parallel tracks:

#### Track A — Upstream PR (recommended, ~1-2 day turnaround)

File a PR against `google-cloud-storage` to add `set_appendable()` to the
`WriteObject` builder:

```rust
// In storage/write_object.rs
pub fn set_appendable<V: Into<bool>>(mut self, v: V) -> Self {
    self.request.spec.appendable = Some(v.into());
    self
}
```

Repository: https://github.com/googleapis/google-cloud-rust

Once merged, update to the new version and use it in `google_gcs_client.rs`:

```rust
self.storage
    .write_object(&bucket_name, object, bytes)
    .set_appendable(true)   // ← new method
    .send_unbuffered()
    .await?;
```

#### Track B — RAPID error detection + workaround (available now)

Without an upstream fix, implement error detection in `google_gcs_client.rs`:

1. Attempt a normal `write_object(...).send_unbuffered()`.
2. If the error message contains `"appendable"`, retry with a raw gRPC call
   constructed directly using `model::WriteObjectRequest` and the internal
   gapic transport's `write_object_unbuffered()` — or use reqwest to call the
   gRPC-JSON transcoded endpoint with `?appendable=true`.

This is more complex and couples the library to GCS error strings, so Track A is
strongly preferred.

#### Track C — Explicit RAPID configuration flag

Expose a `gcs_rapid_storage = true` option in the s3dlio URI or environment variable
(e.g., `S3DLIO_GCS_RAPID=true`). When set, always write with `appendable=true` in the
`gcs-official` backend. This avoids error-detection complexity and is appropriate since
RAPID buckets have a distinct purpose that users explicitly provision.

**Recommended**: Combine Tracks A + C — upstream PR for the builder method, plus a
config flag so users can opt into RAPID mode without relying on first-write error
detection.

### Feature flag recommendation

Given that `gcs-official` (gRPC) is the ONLY path that can support RAPID buckets,
consider either:

- Documenting that RAPID requires `--features gcs-official` (no default change needed).
- If RAPID becomes a key use case, switching the default feature from `gcs-community`
  to `gcs-official` in a future minor version.

---

## Implementation Priority

| Priority | Task | Risk | Effort |
|---|---|---|---|
| 1 | Fix issue #121 (list-buckets region + path-style) | Low | 1–2 h |
| 2 | Add `S3DLIO_GCS_RAPID=true` / config flag in `gcs-official` backend | Low | 2–3 h |
| 3 | File upstream PR for `WriteObject::set_appendable()` | Low | 1–2 h |
| 4 | Wire `set_appendable(true)` once upstream merges | Low | 30 min |

---

## Files to Modify

| File | Change |
|---|---|
| [`src/s3_utils.rs`](../src/s3_utils.rs) | Fix `list_buckets()` region and path-style addressing |
| [`src/google_gcs_client.rs`](../src/google_gcs_client.rs) | Add RAPID support once upstream builder method exists |
| [`src/config.rs`](../src/config.rs) | Add optional `gcs_rapid` flag to `BackendConfig` (if Track C chosen) |
| `README.md` / docs | Document RAPID requirements and `gcs-official` feature |
