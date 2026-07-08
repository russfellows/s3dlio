// src/retry.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Shared retry helper for body-transfer-phase failures (issue #148,
// audit §3.4). Introduced as part of Phase 4 (streaming connector +
// centralized retry).
//
// Why this exists
// ───────────────
// Once the reqwest connector streams response bodies to the SDK
// (Phase 4b), the SDK's own smithy-level retry no longer covers
// body-transfer failures — smithy considers the request "done" as
// soon as `send()` returns Ok(response), which now resolves at
// response headers rather than after the full body downloads. Errors
// that surface while the caller consumes the streaming body must be
// retried at the caller level.
//
// Before this module, `s3_client.rs`, `python_api/python_core_api.rs`,
// and `google_gcs_client.rs` each hand-rolled their own retry shape
// (see audit §3.4). Phase 4 wraps four s3 body-transfer call sites in
// `retry_get_body` instead of adding a fourth independently-invented
// loop.
//
// Design
// ──────
// - Bounded by `crate::constants::max_retry_attempts()` (env-tunable
//   via `S3DLIO_MAX_RETRY_ATTEMPTS`, default 3).
// - Linear backoff: sleep `100ms * attempt` between attempts. Linear
//   (not exponential) is deliberate — the caller has bounded retries
//   and each failure has already burned network time; keeping the
//   backoff small stays responsive without dogpiling a struggling
//   backend the way `0ms * 3` does.
// - Retries on any error. The caller decides idempotency (GETs are
//   idempotent; the four body-transfer sites this wraps are all GETs).
// - Each attempt re-invokes the `FnMut() -> Fut` closure to produce a
//   FRESH future. Callers relying on shared mutable state (e.g. a
//   pre-allocated `BytesMut` segment) must declare the "cursor" state
//   (like `written = 0`) inside the async block so each attempt starts
//   from a clean position — see the range-chunk retry in
//   `s3_utils.rs::concurrent_range_get_impl` for the canonical shape,
//   and `tests/test_phase4_retry.rs` for the fault-injection regression
//   test that locks this invariant in.

use std::future::Future;
use std::time::Duration;

use crate::constants::max_retry_attempts;

/// Retry a body-transfer-phase GET a bounded number of times with
/// linear backoff.
///
/// Each attempt calls `op()` to produce a NEW future — the closure is
/// `FnMut` so callers may capture and mutate their own bookkeeping
/// (a wall-clock start time, an attempt-counter for logs, etc.).
///
/// Correctness contract for the closure body: if `op`'s future writes
/// into a shared mutable buffer (e.g. streaming into a pre-allocated
/// `BytesMut` segment), the write cursor state (`written`, etc.) must
/// be declared INSIDE the async block so each attempt begins at
/// position 0. Otherwise a partially-completed failed attempt can
/// silently corrupt the shared buffer when the retry writes at an
/// offset carried over from the previous attempt.
///
/// Retries on any error from `op`. Backoff is `100ms * attempt` (no
/// jitter — the retry budget is small and every retry has already
/// burned round-trip time on a real failure).
pub async fn retry_get_body<F, Fut, T, E>(mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let attempts = max_retry_attempts().max(1);
    let mut last_err: Option<E> = None;
    for attempt in 1..=attempts {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                let is_final = attempt >= attempts;
                last_err = Some(e);
                if !is_final {
                    tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                }
            }
        }
    }
    // Loop is entered at least once because attempts >= 1, and every
    // Err branch stores into last_err, so an Err return is well-defined.
    Err(last_err.expect("attempts >= 1 guarantees at least one Err was recorded"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::time::Instant;
    use tokio::sync::Mutex as AsyncMutex;

    /// Serializes tests in this module — cargo runs unit tests in a
    /// binary concurrently, so any test that mutates the process-global
    /// `S3DLIO_MAX_RETRY_ATTEMPTS` env var must hold this lock. Async
    /// so it can be held across `retry_get_body(...).await` without
    /// tripping clippy's `await_holding_lock`.
    fn env_lock() -> &'static AsyncMutex<()> {
        static ENV_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| AsyncMutex::new(()))
    }

    /// Phase 4a — budget is respected: after N failed attempts, the
    /// helper gives up and returns the last error rather than looping
    /// forever.
    #[tokio::test]
    async fn retry_respects_max_attempts_budget() {
        let _guard = env_lock().lock().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        // Force a known small budget.
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "3");

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    call_count.fetch_add(1, Ordering::Relaxed);
                    Err::<(), &'static str>("boom")
                }
            }
        };

        let err = retry_get_body(op).await.expect_err("should fail");
        assert_eq!(err, "boom");
        assert_eq!(call_count.load(Ordering::Relaxed), 3);

        std::env::remove_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS);
    }

    /// Phase 4a — first-success path returns immediately without
    /// invoking op() twice.
    #[tokio::test]
    async fn retry_returns_first_success_without_retrying() {
        let _guard = env_lock().lock().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    call_count.fetch_add(1, Ordering::Relaxed);
                    Ok::<u32, &'static str>(42)
                }
            }
        };

        let v = retry_get_body(op).await.expect("should succeed");
        assert_eq!(v, 42);
        assert_eq!(call_count.load(Ordering::Relaxed), 1);

        std::env::remove_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS);
    }

    /// Phase 4a — mid-loop success stops retrying at the first Ok.
    #[tokio::test]
    async fn retry_stops_on_first_success_mid_loop() {
        let _guard = env_lock().lock().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    let n = call_count.fetch_add(1, Ordering::Relaxed);
                    if n < 2 {
                        Err::<&'static str, &'static str>("try again")
                    } else {
                        Ok("ok")
                    }
                }
            }
        };

        let v = retry_get_body(op).await.expect("should succeed on 3rd try");
        assert_eq!(v, "ok");
        assert_eq!(call_count.load(Ordering::Relaxed), 3);

        std::env::remove_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS);
    }

    /// Phase 4a — linear backoff timing: two failures then a success
    /// should sleep 100ms after attempt 1 and 200ms after attempt 2.
    /// Total elapsed ≥ 300ms.
    #[tokio::test]
    async fn retry_applies_linear_backoff() {
        let _guard = env_lock().lock().await;
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");
        let call_count = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    let n = call_count.fetch_add(1, Ordering::Relaxed);
                    if n < 2 {
                        Err::<(), &'static str>("try again")
                    } else {
                        Ok(())
                    }
                }
            }
        };

        retry_get_body(op).await.expect("should succeed");
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(300),
            "linear backoff should sleep 100ms + 200ms between the 3 attempts; \
             elapsed = {:?}. If this is much lower, the backoff isn't firing.",
            elapsed
        );

        std::env::remove_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS);
    }
}
