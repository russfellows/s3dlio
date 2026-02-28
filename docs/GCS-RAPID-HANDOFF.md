# GCS RAPID Storage — Bug Handoff for Opus

**Date:** 2026-02-28  
**Branch:** `feature/v0.9.60-gcs-rapid-list-buckets` (HEAD: `51f8584`)  
**Status:** UNSOLVED — passing to next agent  
**Remote test host:** `sig65-cntrlr-vm` · bucket: `gs://sig65-rapid1/`

---

## The Problem

Writes to GCS RAPID (Hyperdisk ML / zonal) buckets fail. RAPID buckets have two hard
requirements:

1. Upload must use **`uploadType=multipart`** (single-shot). Resumable upload is
   categorically rejected: `"Zonal buckets do not support resumable upload."`
2. The request URL must include **`?appendable=true`** as a query parameter.
   Omitting it: `"This bucket requires appendable objects."`

---

## Error Progression (Chronological)

### Stage 1 — Before any fix (binary built from `main`-era code)

```
S3DLIO_GCS_RAPID=0 → auto-retry with appendable=true + resumable
  Error: Zonal buckets do not support resumable upload.

S3DLIO_GCS_RAPID=1 → appendable=true + forcing resumable
  Error: Zonal buckets do not support resumable upload.
```

Both paths were calling `.with_resumable_upload_threshold(0_usize)`, forcing
`uploadType=resumable`. Resumable is rejected outright.

### Stage 2 — After fix: `threshold=0` → `threshold=usize::MAX`

Commit `51f8584` changed both the explicit RAPID path and the auto-retry path to use
`.with_resumable_upload_threshold(usize::MAX)` (forces single-shot).

```
S3DLIO_GCS_RAPID=1 → appendable=true + forcing single-shot
  Error: This bucket requires appendable objects.

S3DLIO_GCS_RAPID=0 → auto-retry with appendable=true + single-shot
  Error: This bucket requires appendable objects.
```

**The resumable error is gone** — we are now confirmed on the single-shot path.
But `appendable=true` is not reaching the server as a query parameter.

---

## The Vendor Patch

`google-cloud-storage` v1.8.0 is vendored at `vendor/google-cloud-storage/` and wired
via:

```toml
# Cargo.toml
[patch.crates-io]
google-cloud-storage = { path = "vendor/google-cloud-storage" }
```

Two manual patches were applied to the vendor copy:

### Patch 1 — `set_appendable()` builder method

File: [`vendor/google-cloud-storage/src/storage/write_object.rs`](../vendor/google-cloud-storage/src/storage/write_object.rs) line 424

```rust
pub fn set_appendable<V: Into<bool>>(mut self, v: V) -> Self {
    self.request.spec.appendable = Some(v.into());
    self
}
```

Sets `appendable` on the `WriteObjectSpec` inside the builder's `request` struct.

### Patch 2 — `apply_preconditions()` emits the query param

File: [`vendor/google-cloud-storage/src/storage/perform_upload.rs`](../vendor/google-cloud-storage/src/storage/perform_upload.rs) lines 156–157

```rust
match self.spec.appendable {
    Some(true) => builder.query(&[("appendable", "true")]),
    _ => builder,
}
```

This function is called in TWO places:

| Call site | Path |
|---|---|
| `perform_upload.rs:96` | `start_resumable_upload_request()` (NOT our path with `usize::MAX`) |
| `perform_upload/unbuffered.rs:156` | `single_shot_builder()` ← **our path** |

---

## Current s3dlio Code (`src/google_gcs_client.rs`)

Lines 200–245 — the PUT method:

```rust
let write = if self.rapid_mode {
    debug!("GCS PUT: RAPID mode — appendable=true + forcing single-shot for {}", bucket_name);
    write.set_appendable(true).with_resumable_upload_threshold(usize::MAX)
} else {
    trace!("GCS PUT: standard mode — forcing single-shot upload for {}", bucket_name);
    write.with_resumable_upload_threshold(usize::MAX)
};
write.send_unbuffered().await

// ... on error "requires appendable objects":
self.storage
    .write_object(&bucket_name, object, bytes)
    .set_appendable(true)
    .with_resumable_upload_threshold(usize::MAX)
    .send_unbuffered()
    .await
```

---

## Control Flow Through the Vendor Library

```
google_gcs_client.rs:
  write.set_appendable(true)                    →  request.spec.appendable = Some(true)
  .with_resumable_upload_threshold(usize::MAX)  →  options.resumable_upload_threshold = usize::MAX
  .send_unbuffered()

write_object.rs:1003-1005:
  self.stub.write_object_unbuffered(self.payload, self.request, self.options)

transport.rs:96:
  PerformUpload::new(payload, inner, req.spec, req.params, options)
  //                                  ^^^^^^^^^
  //   req.spec IS the WriteObjectSpec with appendable=Some(true)

perform_upload/unbuffered.rs:33-46:
  send_unbuffered() {
      let threshold = self.options.resumable_upload_threshold() as u64;  // usize::MAX
      if hint.upper().is_none_or(|max| max >= threshold) {               // false for 4096
          self.send_unbuffered_resumable(hint).await                       // NOT taken
      } else {
          self.send_unbuffered_single_shot(hint).await                     // TAKEN ✓
      }
  }

perform_upload/unbuffered.rs:137-156:
  single_shot_builder() {
      builder = builder
          .query(&[("uploadType", "multipart")])  // ✓
          .query(&[("name", object)]);
      let builder = self.apply_preconditions(builder);  // ← appendable should be added here
      ...
  }

perform_upload.rs:123-158:
  apply_preconditions() {
      // ... other preconditions ...
      match self.spec.appendable {
          Some(true) => builder.query(&[("appendable", "true")]),  // ← should fire
          _ => builder,
      }
  }
```

On paper every step looks correct. Yet the server still rejects with
`"This bucket requires appendable objects."` — meaning `?appendable=true` is NOT
in the actual HTTP request URL.

---

## Hypotheses (Most to Least Likely)

### H1 — Stale binary on remote host ⚠️ NOT YET RULED OUT

The test command was:
```
rfellows@sig65-cntrlr-vm:~/Documents$ cp ../s3-cli ./target/release/
```

This copies a binary named `s3-cli` from `~/` (home dir) to `./target/release/`. The
binary in `~/s3-cli` may have been built **without** the vendor patch (e.g. built from
`main` before the vendor commit `086c68e`).

**To verify**: Run `cargo build --release` locally after confirming HEAD is `51f8584`,
then copy the fresh binary to the remote and retest.

### H2 — `WriteObjectSpec` type mismatch

The `set_appendable()` patch sets `self.request.spec.appendable` where `self.request`
is a `WriteObjectRequest`. The `PerformUpload` stores its own `spec: crate::model::WriteObjectSpec`
which it receives as `req.spec` from the transport layer. If there are two different
`WriteObjectSpec` types in scope (e.g. one generated, one hand-rolled) and the wrong
one ends up in `PerformUpload`, the `appendable` field would  silently be `None`.

**To verify**: Add a single `eprintln!("apply_preconditions appendable={:?}", self.spec.appendable)`
in `apply_preconditions()` and rebuild. If it prints `None`, H2 is the cause.

### H3 — `with_resumable_upload_threshold` overrides the spec somehow

If the threshold setter somehow re-initializes the `request` struct, `appendable`
could be cleared. Unlikely but easy to check given H2 debugging.

### H4 — `apply_preconditions` is NOT called for unbuffered single-shot

The vendor `unbuffered.rs:156` calls `self.apply_preconditions(builder)`. The
`apply_preconditions` is defined on `impl<S> PerformUpload<S>` in `perform_upload.rs`.
If there were a separate `PerformUpload` for unbuffered that shadows the impl, the
patched function might not be called. Unlikely but the same `eprintln!` would reveal it.

### H5 — The resumable path is still being taken

The size_hint threshold check: `hint.upper().is_none_or(|max| max >= threshold)`.
With `threshold = usize::MAX as u64` and `hint.upper() = Some(4096)`:
`4096 >= 18446744073709551615` is `false`, so `is_none_or(false)` is `false` →
single-shot is chosen. But `usize::MAX` is cast to `u64`: on a 64-bit system
`usize::MAX = u64::MAX = 18446744073709551615` — this is fine.

Unless the payload size_hint returns `None` for upper, in which case
`is_none_or(|max| ...)` returns `true` → **resumable path is taken**.
`BytesSource::size_hint()` returns `SizeHint::with_exact(s)` which sets the upper
bound to `s`. For a `Bytes` payload this should be known. **But if the payload goes
through a wrapping layer that loses the hint**, upper could be `None`.

---

## What Has NOT Been Tried

1. **Intercept the actual HTTP request** — run with `HTTPS_PROXY=http://localhost:8080`
   and `mitmproxy`, or add `eprintln!` in `single_shot_builder()` to print the URL
   before sending. Confirm whether `?appendable=true` is present in the URL.

2. **Add diagnostic print in `apply_preconditions`** — one line in the vendor code:
   ```rust
   eprintln!("[DEBUG] apply_preconditions: appendable={:?}", self.spec.appendable);
   ```
   Rebuild and retest. This will definitively tell us if the spec is reaching the
   function with `Some(true)` or not.

3. **Build fresh on local machine and copy binary** — confirm the binary under test
   was built from HEAD `51f8584` with the vendor patch active.

4. **Try `uploadType=media` (Simple upload) instead of `multipart`** —
   Gemini/external analysis suggests zonal RAPID may require `uploadType=media` (the
   "Simple" single-object upload) rather than `uploadType=multipart` (which wraps
   data in a MIME multipart envelope). The current single-shot path uses
   `multipart`. This has NOT been tried.

5. **Bypass the crate entirely with a raw reqwest call** — construct the HTTP request
   manually in `google_gcs_client.rs` using reqwest, adding `?appendable=true` and
   `uploadType=media` explicitly. This would definitively isolate whether the issue is
   in our s3dlio code or in the vendor library.

---

## Key Files

| File | Purpose |
|---|---|
| [`src/google_gcs_client.rs`](../src/google_gcs_client.rs) | Our GCS client — PUT logic lines 185–245 |
| [`vendor/google-cloud-storage/src/storage/write_object.rs`](../vendor/google-cloud-storage/src/storage/write_object.rs) | Vendor builder — `set_appendable()` at line 424 |
| [`vendor/google-cloud-storage/src/storage/perform_upload.rs`](../vendor/google-cloud-storage/src/storage/perform_upload.rs) | `apply_preconditions()` — appendable query param at line 156 |
| [`vendor/google-cloud-storage/src/storage/perform_upload/unbuffered.rs`](../vendor/google-cloud-storage/src/storage/perform_upload/unbuffered.rs) | Single-shot path — `apply_preconditions` called at line 156 |
| [`vendor/google-cloud-storage/src/storage/transport.rs`](../vendor/google-cloud-storage/src/storage/transport.rs) | `PerformUpload::new(... req.spec ...)` at line 96 |

---

## Recommended First Action for Opus

Add a single diagnostic `eprintln!` to `apply_preconditions` in the vendor code,
rebuild, copy to remote, and run the test. This will immediately reveal whether:

- `appendable = Some(true)` → the URL is somehow wrong despite the value being correct
- `appendable = None` → the spec isn't reaching `apply_preconditions` with the value set

```rust
// In vendor/google-cloud-storage/src/storage/perform_upload.rs
// After the match at line 155 — add one line BEFORE the match:
eprintln!("[RAPID DEBUG] apply_preconditions: appendable={:?}", self.spec.appendable);
match self.spec.appendable {
    Some(true) => builder.query(&[("appendable", "true")]),
    _ => builder,
}
```

That one data point will cut the remaining hypotheses in half.
