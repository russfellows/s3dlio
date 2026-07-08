// tests/test_phase4_retry_fault_injection.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Phase 4 fault-injection RED-then-GREEN test for issue #148
// (docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md §2.4).
//
// The concern: once the reqwest connector streams response bodies
// (Phase 4b) instead of buffering them, the SDK's smithy-level retry
// no longer covers body-transfer failures. We wrap the concurrent
// range-chunk task in `retry_get_body`, and each retry attempt writes
// directly into a shared, pre-allocated `BytesMut` segment (from
// Phase 1's shared-segment-buffer design).
//
// Silent-data-corruption risk (audit §2.4):
//   > If a retry loop re-issues the range GET after a partial failure
//   > but does not reset `written = 0` fresh at the top of every
//   > attempt — or if a later attempt writes fewer bytes than a
//   > previous failed attempt already wrote into `seg` — stale bytes
//   > from the failed attempt could silently remain in
//   > `seg[0..old_written]`, mixed with new bytes from the successful
//   > retry. No error would be raised.
//
// This test locks the invariant in place: the retry helper must
// re-invoke the closure fresh, and the closure body must declare its
// `written` cursor inside the async block so each attempt starts at
// position 0.
//
// The scenario mirrors the exact code shape used inside
// `s3_utils.rs::concurrent_range_get_impl`'s range-chunk task after
// Phase 4 lands: a mutable Arc<Mutex<Vec<u8>>> stands in for the
// shared `BytesMut` segment (BytesMut can't be sent across a
// `FnMut() -> impl Future` boundary without either a Mutex or
// `.split()` gymnastics; the Mutex faithfully models the mid-body
// write path).

use anyhow::{anyhow, Result};
use s3dlio::retry::retry_get_body;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Mutex as AsyncMutex;

/// Serializes tests in this file so their mutations of
/// `S3DLIO_MAX_RETRY_ATTEMPTS` don't race — cargo runs tests within
/// a binary in parallel by default. Any test that sets or removes
/// that env var must take this lock for its whole body. Async-aware
/// so it can be held across the retry_get_body().await point (a std
/// Mutex trips clippy's `await_holding_lock`).
fn env_lock() -> &'static AsyncMutex<()> {
    static ENV_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| AsyncMutex::new(()))
}

/// Phase 4 fault-injection GREEN test — attempt 1 writes 5 of 10
/// bytes with pattern 0xAA and then fails mid-body; attempt 2 writes
/// all 10 bytes cleanly with pattern 0xBB.
///
/// GREEN behavior: retry_get_body re-invokes op(), producing a fresh
/// future whose `written` starts at 0. Attempt 2's 10-byte write
/// overwrites the entire segment. Final content: 10 bytes of 0xBB.
///
/// RED behavior it locks out: if `written` leaked from attempt 1
/// (declared OUTSIDE the async block, captured by mutable reference),
/// attempt 2 would start at position 5 and either overflow, or —
/// worse — write 5 bytes at [5..10], leaving attempt 1's 0xAA at
/// [0..5]. Length would still be 10, but content would be a mix of
/// both attempts (silent data corruption, no error raised).
#[tokio::test]
async fn phase4_range_chunk_retry_resets_written_cursor_on_each_attempt() {
    let _guard = env_lock().lock().await;
    std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "5");

    let seg: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0u8; 10]));
    let attempt_counter = Arc::new(AtomicUsize::new(0));

    let op = {
        let seg = Arc::clone(&seg);
        let attempt_counter = Arc::clone(&attempt_counter);
        move || {
            let seg = Arc::clone(&seg);
            let attempt_counter = Arc::clone(&attempt_counter);
            async move {
                let n = attempt_counter.fetch_add(1, Ordering::Relaxed);

                // CRITICAL invariant: `written` is declared inside the
                // async block, so each retry attempt starts at 0. This
                // mirrors what `concurrent_range_get_impl`'s per-chunk
                // async block does after Phase 4.
                let mut written = 0usize;

                let mut guard = seg.lock().unwrap();
                let seg_len = guard.len();

                if n == 0 {
                    // Attempt 1: write 5 bytes 0xAA at [0..5], then fail
                    let chunk = [0xAAu8; 5];
                    guard[written..written + chunk.len()].copy_from_slice(&chunk);
                    written += chunk.len();
                    // Simulate mid-body stream failure
                    let _ = written; // silence unused_assignments before the return
                    return Err::<(), anyhow::Error>(anyhow!(
                        "simulated mid-body failure after writing {} bytes",
                        written
                    ));
                }

                // Attempt 2+: write full 10 bytes 0xBB from position 0
                let chunk = [0xBBu8; 10];
                if written + chunk.len() > seg_len {
                    return Err(anyhow!(
                        "range chunk overflow on attempt {}: {} + {} > {}",
                        n,
                        written,
                        chunk.len(),
                        seg_len
                    ));
                }
                guard[written..written + chunk.len()].copy_from_slice(&chunk);
                written += chunk.len();
                if written != seg_len {
                    return Err(anyhow!(
                        "short read: wrote {}/{} bytes on attempt {}",
                        written,
                        seg_len,
                        n
                    ));
                }
                Ok(())
            }
        }
    };

    retry_get_body(op)
        .await
        .expect("retry should succeed on attempt 2");

    let final_bytes = seg.lock().unwrap().clone();
    assert_eq!(
        &final_bytes[..],
        &[0xBB; 10][..],
        "each retry attempt must write the full segment from position 0. \
         Observed final segment = {:?}. If it starts with 0xAA bytes then \
         `written` leaked across attempts and Phase 4's retry loop has the \
         silent-data-corruption bug flagged in audit §2.4.",
        final_bytes
    );
    assert_eq!(
        attempt_counter.load(Ordering::Relaxed),
        2,
        "expected exactly 2 attempts (fail + succeed); saw {}",
        attempt_counter.load(Ordering::Relaxed)
    );

    std::env::remove_var("S3DLIO_MAX_RETRY_ATTEMPTS");
}

/// Phase 4 fault-injection GREEN test — retry budget is respected on
/// persistent body-transfer failures. After N failed attempts, the
/// helper propagates the last error rather than retrying indefinitely.
///
/// This is separate from the SDK-level max_attempts test (which covers
/// smithy's own retry loop for send() failures). This confirms the
/// NEW retry path introduced in Phase 4 for body-transfer failures
/// bypasses the SDK budget but respects its own env-configured budget.
#[tokio::test]
async fn phase4_body_transfer_retry_budget_respected() {
    let _guard = env_lock().lock().await;
    std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");

    let attempt_counter = Arc::new(AtomicUsize::new(0));

    let op = {
        let attempt_counter = Arc::clone(&attempt_counter);
        move || {
            let attempt_counter = Arc::clone(&attempt_counter);
            async move {
                attempt_counter.fetch_add(1, Ordering::Relaxed);
                Err::<(), anyhow::Error>(anyhow!("persistent body-transfer failure"))
            }
        }
    };

    let result: Result<()> = retry_get_body(op).await;
    assert!(
        result.is_err(),
        "should propagate error after budget exhausted"
    );
    assert_eq!(
        attempt_counter.load(Ordering::Relaxed),
        3,
        "should have attempted exactly S3DLIO_MAX_RETRY_ATTEMPTS times"
    );

    std::env::remove_var("S3DLIO_MAX_RETRY_ATTEMPTS");
}
