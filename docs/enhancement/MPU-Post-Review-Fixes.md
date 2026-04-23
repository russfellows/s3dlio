# Multipart Upload — Post-Review Fixes

**Date**: April 23, 2026  
**Branch**: `feature/connection-pool-autoscale`  
**Version**: v0.9.92  
**Context**: Following the v0.9.92 coordinator-task rewrite a full code review identified four issues. All were fixed in the same session.

---

## Issues Found and Resolved

### 1. Latent Panic — `write()` / `write_owned()` / `flush()` async methods

**Problem**: All three public `async fn` methods on `MultipartUploadSink` delegated directly to their `_blocking()` counterparts, which call `blocking_send()` on the mpsc channel. Tokio panics immediately when `blocking_send` is called from within any Tokio runtime context — not just when the channel is full. Any Rust async caller invoking `sink.write().await` would panic.

**Fix**: Added a private `async fn enqueue_part_async()` that uses `part_tx.send().await`. Rewrote `write()`, `write_owned()`, and `flush()` to use it. Added `async fn finish()` that sends the `Finish` message via `.send().await` and directly `.await`s the coordinator `JoinHandle`, requiring no `run_on_global_rt`. Python hot-path (`write_blocking` → `blocking_send`) is unchanged.

---

### 2. Policy Violation — Two Clippy Warnings

**Problem**: The new coordinator loop in `multipart.rs` used a `loop { match }` pattern instead of `while let`, and `src/bin/cli.rs` had a manual `Option::filter` reimplementation. Both were flagged by clippy, violating the zero-warnings policy.

**Fix**:
- Coordinator loop rewritten as `while let Some(PartMsg::Part { data, part_number }) = part_rx.recv().await { ... }`
- cli.rs pattern replaced with `chars.next().filter(|&ch| chars.next().is_none() && !ch.is_alphanumeric())`

---

### 3. Missing Guard — `MAX_MULTIPART_PARTS` Never Enforced

**Problem**: `src/constants.rs` defines `MAX_MULTIPART_PARTS = 10_000` but `multipart.rs` never imported or checked it. Uploading a very large object (e.g., 200 GiB at 16 MiB parts = ~12,800 parts) would fail with a cryptic `InvalidPart` S3 error at `CompleteMultipartUpload` time rather than a clear message at write time.

**Fix**: `MAX_MULTIPART_PARTS` imported and checked in both `enqueue_part()` and `enqueue_part_async()` before incrementing the part counter:

```rust
if part_number as usize > MAX_MULTIPART_PARTS {
    bail!(
        "exceeded S3 maximum of {MAX_MULTIPART_PARTS} parts; \
         use a larger part_size for objects this large"
    );
}
```

---

### 4. Incorrect Doc Comment — Memory Ceiling Off by ~2×

**Problem**: The `auto_max_in_flight()` doc stated `Memory ceiling: max_in_flight × part_size`. The actual ceiling is approximately **2×** that, because both the bounded channel and the semaphore are each sized `max_in_flight`: the channel can hold up to `max_in_flight` parts queued while the semaphore permits `max_in_flight` upload tasks in-flight simultaneously. At 32 MiB parts the real ceiling is ~2 GiB, not 1 GiB as documented.

**Fix**: Doc updated to `~2 × max_in_flight × part_size` with corrected worked examples.

---

## Test Added

`test_blocking_send_panics_inside_tokio_runtime` was added to `multipart.rs`. It uses `std::panic::catch_unwind` to prove that `blocking_send` from within a `block_on` context panics, documenting the exact bug class that `write()` / `write_owned()` had before the fix.

---

## Verification

```
cargo build      → Finished, 0 warnings
cargo clippy     → Finished, 0 warnings
cargo test --lib → 247/247 passed (1 new test)
```
